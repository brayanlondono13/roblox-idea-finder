#!/usr/bin/env python3
"""
roblox_research.py - Roblox market & competitor research toolkit.

WHAT THIS IS FOR
  Decide WHAT game to build by looking at real, live demand on Roblox:
    - discover existing games in a niche (so you don't rebuild something that exists)
    - see what is HOT right now (trending / up-and-coming feeds)
    - run GAP ANALYSIS on a niche: is it a monolith (one game owns it -> avoid)
      or fragmented with room for a new 1k+ CCU entrant (-> opportunity)?

DATA SOURCE
  Roblox's own public web/JSON APIs. No key, no login. All endpoints below were
  verified live. They are UNOFFICIAL - Roblox can change them without notice.

  CAN get  : live snapshot - CCU ("playing"), visits, favorites, votes, genre,
             created/updated dates, full keyword search, and Roblox's own
             trending / up-and-coming / top-playing-now feeds.
  CANNOT get: historical peak CCU, retention (D1/D7), or playtime curves for
             games you don't own. For history, cron this into a DB yourself, or
             read a tracker (RoMonitor Stats / Rolimon's). "visits_per_day" here
             is a LIFETIME velocity proxy, not a live growth rate.

VERIFIED ENDPOINTS
  place   -> universe : GET apis.roblox.com/universes/v1/places/{placeId}/universe
  details (batch)     : GET games.roblox.com/v1/games?universeIds=a,b,c
  votes   (batch)     : GET games.roblox.com/v1/games/votes?universeIds=a,b,c
  search  (paginated) : GET apis.roblox.com/search-api/omni-search?searchQuery=..&sessionId=..&pageType=all
  trending feeds      : GET apis.roblox.com/explore-api/v1/get-sort-content?sessionId=..&sortId=..
                        sortIds: top-trending up-and-coming top-playing-now fun-with-friends top-revisited
  icons   (batch)     : GET thumbnails.roblox.com/v1/games/icons?universeIds=..&size=512x512&format=Png

INVARIANTS
  - A place id is the integer in a game URL (roblox.com/games/<PLACE_ID>/Name).
  - CCU == the API's `playing` field. rating% == up / (up + down) * 100.
  - The details/votes APIs take UNIVERSE ids, not place ids -> we convert first.
  - omni-search and the explore feeds already return universeId + live CCU/votes,
    so a niche scan needs no place->universe round-trip.

FAILURE MODES
  - 429 rate limit -> exponential backoff; raise --sleep for very large scans.
  - 404 / private / deleted id -> that game is skipped, never fatal.

INSTALL  pip install requests
USAGE    python roblox_research.py --help
         python roblox_research.py selftest
         python roblox_research.py search  "cooking" --pages 3 --csv cooking.csv
         python roblox_research.py trending --sort up-and-coming
         python roblox_research.py niche   "slime rng" --pages 3 --report slime.md
         python roblox_research.py inspect  76558904092080 https://roblox.com/games/123/X
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass, asdict, fields as dataclass_fields
from datetime import datetime, timezone

try:
    import requests
except ImportError:  # pragma: no cover - guidance only
    sys.exit("This tool needs `requests`.  Install it with:  pip install requests")


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
UNIVERSE_URL     = "https://apis.roblox.com/universes/v1/places/{place_id}/universe"
DETAILS_URL      = "https://games.roblox.com/v1/games"
VOTES_URL        = "https://games.roblox.com/v1/games/votes"
ICONS_URL        = "https://thumbnails.roblox.com/v1/games/icons"
OMNI_URL         = "https://apis.roblox.com/search-api/omni-search"
SORTS_URL        = "https://apis.roblox.com/explore-api/v1/get-sorts"
SORT_CONTENT_URL = "https://apis.roblox.com/explore-api/v1/get-sort-content"

TRENDING_SORTS = [
    "top-trending",     # largest relative increase in DAU over the past week
    "up-and-coming",    # published in the last 28 days, sorted by user growth
    "top-playing-now",  # sorted by concurrent users
    "fun-with-friends",
    "top-revisited",
]

WINNER_CCU   = 1000   # a game at/above this CCU "supports 1k+" - the user's bar
FRESH_DAYS   = 90     # "recently launched"
STALE_DAYS   = 60     # leader not updated in this long == potentially vulnerable
DETAILS_CHUNK = 50    # universe ids per details/votes call (API tolerates more; be polite)


# --------------------------------------------------------------------------- #
# HTTP client
# --------------------------------------------------------------------------- #
class RobloxClient:
    """Thin wrapper over Roblox's public JSON APIs with 429 backoff + a cache."""

    def __init__(self, sleep: float = 0.4, retries: int = 7, timeout: int = 20,
                 verbose: bool = False):
        self.s = requests.Session()
        self.s.headers.update({
            "User-Agent": "roblox-research/2.0 (market-research; +local)",
            "Accept": "application/json",
        })
        self.sleep = sleep
        self.retries = retries
        self.timeout = timeout
        self.verbose = verbose
        self.session_id = str(uuid.uuid4())
        self._uni_cache: dict[int, int | None] = {}

    def _log(self, *a):
        if self.verbose:
            print("  .", *a, file=sys.stderr)

    def _get(self, url, params=None):
        """GET -> parsed JSON. None on 404. Exponential backoff on 429, honoring a
        Retry-After header when present."""
        for attempt in range(self.retries):
            r = self.s.get(url, params=params, timeout=self.timeout)
            if r.status_code == 429:
                ra = r.headers.get("Retry-After", "")
                wait = float(ra) if ra.replace(".", "", 1).isdigit() else min(2 ** attempt, 60)
                self._log(f"429 -> sleeping {wait}s")
                time.sleep(wait)
                continue
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        raise RuntimeError(f"rate-limited after {self.retries} tries: {url}")

    # -- discovery -------------------------------------------------------- #
    def resolve_universe(self, place_id: int) -> int | None:
        """One place id -> its universe id. None if deleted / private / bad id."""
        if place_id in self._uni_cache:
            return self._uni_cache[place_id]
        data = self._get(UNIVERSE_URL.format(place_id=place_id))
        uid = data.get("universeId") if data else None
        self._uni_cache[place_id] = uid
        return uid

    def omni_search(self, query: str, pages: int = 2) -> list[dict]:
        """Keyword search across all of Roblox. Returns raw game content dicts
        (already carry universeId, name, description, playerCount, votes)."""
        out: list[dict] = []
        seen: set[int] = set()
        token = None
        for _ in range(max(1, pages)):
            params = {"searchQuery": query, "sessionId": self.session_id, "pageType": "all"}
            if token:
                params["pageToken"] = token
            data = self._get(OMNI_URL, params)
            if not data:
                break
            for group in data.get("searchResults", []):
                if group.get("contentGroupType") != "Game":
                    continue
                for c in group.get("contents", []):
                    uid = c.get("universeId")
                    if c.get("contentType") == "Game" and uid and uid not in seen:
                        seen.add(uid)
                        out.append(c)
            token = data.get("nextPageToken")
            if not token:
                break
            time.sleep(self.sleep)
        return out

    def sort_content(self, sort_id: str) -> list[dict]:
        """One trending feed -> list of raw game dicts (universeId + playerCount + votes)."""
        data = self._get(SORT_CONTENT_URL, {"sessionId": self.session_id, "sortId": sort_id})
        if not data:
            return []
        return data.get("games", [])

    # -- enrichment ------------------------------------------------------- #
    def details(self, universe_ids: list[int]) -> dict[int, dict]:
        """Batch-fetch full game details. {universe_id: detail_dict}."""
        out: dict[int, dict] = {}
        ids = list(dict.fromkeys(universe_ids))  # dedupe, keep order
        for i in range(0, len(ids), DETAILS_CHUNK):
            chunk = ids[i:i + DETAILS_CHUNK]
            try:
                data = self._get(DETAILS_URL, {"universeIds": ",".join(map(str, chunk))})
            except RuntimeError as e:               # rate-limited: skip chunk, keep the rest
                print(f"  ! details chunk skipped ({e})", file=sys.stderr)
                continue
            for g in (data or {}).get("data", []):
                gid = g.get("id")
                if gid is not None:                 # never let one bad row kill the chunk
                    out[gid] = g
            time.sleep(self.sleep)
        return out

    def votes(self, universe_ids: list[int]) -> dict[int, dict]:
        """Batch-fetch up/down votes. {universe_id: {upVotes, downVotes}}."""
        out: dict[int, dict] = {}
        ids = list(dict.fromkeys(universe_ids))
        for i in range(0, len(ids), DETAILS_CHUNK):
            chunk = ids[i:i + DETAILS_CHUNK]
            try:
                data = self._get(VOTES_URL, {"universeIds": ",".join(map(str, chunk))})
            except RuntimeError as e:
                print(f"  ! votes chunk skipped ({e})", file=sys.stderr)
                continue
            for v in (data or {}).get("data", []):
                vid = v.get("id")
                if vid is not None:
                    out[vid] = v
            time.sleep(self.sleep)
        return out

    def icons(self, universe_ids: list[int], size: str = "256x256") -> dict[int, str]:
        """Batch-fetch icon image urls. {universe_id: url}."""
        out: dict[int, str] = {}
        ids = list(dict.fromkeys(universe_ids))
        for i in range(0, len(ids), DETAILS_CHUNK):
            chunk = ids[i:i + DETAILS_CHUNK]
            data = self._get(ICONS_URL, {
                "universeIds": ",".join(map(str, chunk)),
                "size": size, "format": "Png", "isCircular": "false",
            })
            for d in (data or {}).get("data", []):
                tid = d.get("targetId")
                if tid is not None and d.get("state") == "Completed":
                    out[tid] = d.get("imageUrl", "")
            time.sleep(self.sleep)
        return out


# --------------------------------------------------------------------------- #
# Game model + metrics
# --------------------------------------------------------------------------- #
@dataclass
class Game:
    universe_id: int
    place_id: int
    name: str
    creator: str
    ccu: int
    visits: int
    favorites: int
    up: int
    down: int
    rating: float | None        # percentage, 0-100
    genre_l1: str
    genre_l2: str
    max_players: int | None
    created: str
    updated: str
    url: str
    description: str = ""
    # ---- computed ----
    age_days: int = 0
    days_since_update: int = 0
    visits_per_day: float = 0.0      # LIFETIME velocity proxy (visits / age)
    ccu_per_1k_visits: float = 0.0   # stickiness proxy
    source: str = ""                 # which feed/query surfaced this game

    def is_winner(self) -> bool:
        return self.ccu >= WINNER_CCU

    def is_fresh(self) -> bool:
        return 0 <= self.age_days <= FRESH_DAYS

    def is_stale(self) -> bool:
        return self.days_since_update >= STALE_DAYS


_FRAC_RE = re.compile(r"\.(\d+)")


def _parse_dt(s: str | None):
    """Parse a Roblox ISO timestamp. Roblox emits up to 7 fractional-second
    digits (e.g. ...:13.8540241Z); datetime.fromisoformat only accepts <=6 before
    Python 3.11, so trim to 6 first or freshness/staleness silently drop to 0."""
    if not s:
        return None
    s = s.replace("Z", "+00:00")
    s = _FRAC_RE.sub(lambda m: "." + m.group(1)[:6], s, count=1)
    try:
        return datetime.fromisoformat(s)
    except (ValueError, AttributeError):
        return None


def _build_game(uid: int, det: dict | None, vote: dict | None, disc: dict | None,
                now: datetime) -> Game:
    """Merge a details record (preferred) with votes + a discovery fallback."""
    det = det or {}
    vote = vote or {}
    disc = disc or {}

    name = det.get("name") or disc.get("name") or ""
    place_id = det.get("rootPlaceId") or disc.get("rootPlaceId") or 0
    creator = (det.get("creator") or {}).get("name") or disc.get("creatorName") or ""
    ccu = det.get("playing", disc.get("playerCount", 0)) or 0
    visits = det.get("visits", 0) or 0
    favorites = det.get("favoritedCount", 0) or 0
    up = vote.get("upVotes", disc.get("totalUpVotes", 0)) or 0
    down = vote.get("downVotes", disc.get("totalDownVotes", 0)) or 0
    total = up + down
    rating = round(100 * up / total, 1) if total else None
    created = det.get("created", "") or ""
    updated = det.get("updated", "") or ""
    desc = (det.get("description") or disc.get("description") or "").strip()

    g = Game(
        universe_id=uid,
        place_id=place_id,
        name=name,
        creator=creator,
        ccu=ccu,
        visits=visits,
        favorites=favorites,
        up=up,
        down=down,
        rating=rating,
        genre_l1=det.get("genre_l1", "") or "",
        genre_l2=det.get("genre_l2", "") or "",
        max_players=det.get("maxPlayers"),
        created=created,
        updated=updated,
        url=f"https://www.roblox.com/games/{place_id}" if place_id else "",
        description=desc.replace("\n", " ")[:300],
    )

    cdt, udt = _parse_dt(created), _parse_dt(updated)
    if cdt:
        g.age_days = max(0, (now - cdt).days)
    if udt:
        g.days_since_update = max(0, (now - udt).days)
    if g.age_days > 0 and g.visits:
        g.visits_per_day = round(g.visits / g.age_days, 1)
    if g.visits:
        g.ccu_per_1k_visits = round(1000 * g.ccu / g.visits, 4)
    return g


def enrich(client: RobloxClient, raw_items: list[dict]) -> list[Game]:
    """Discovery dicts (must carry universeId) -> fully enriched Game objects."""
    disc_by_id = {}
    for c in raw_items:
        uid = c.get("universeId")
        if uid:
            disc_by_id.setdefault(uid, c)
    ids = list(disc_by_id.keys())
    if not ids:
        return []
    det = client.details(ids)
    vot = client.votes(ids)
    now = datetime.now(timezone.utc)
    games = [_build_game(uid, det.get(uid), vot.get(uid), disc_by_id.get(uid), now)
             for uid in ids]
    games.sort(key=lambda g: g.ccu, reverse=True)
    return games


# --------------------------------------------------------------------------- #
# Gap / opportunity analysis  (the heart of "what should I build?")
# --------------------------------------------------------------------------- #
def _hhi(shares: list[float]) -> int:
    """Herfindahl-Hirschman Index on CCU shares, scaled 0-10000 (antitrust style).
    10000 = one game owns everything (monolith). <1500 = fragmented/competitive."""
    return int(round(sum(s * s for s in shares) * 10000))


_STOP = {"the", "and", "for", "with", "your", "you", "best"}


def _stem(tok: str) -> str:
    """Crude suffix-strip so a query like 'cooking' matches 'Cook Tycoon'."""
    for suf in ("ing", "ers", "er", "ed", "s"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            return tok[:-len(suf)]
    return tok


def _tokens(query: str) -> list[str]:
    return [_stem(t) for t in re.split(r"[^a-z0-9]+", query.lower())
            if len(t) >= 3 and t not in _STOP]


def _on_topic(g: Game, tokens: list[str]) -> bool:
    if not tokens:
        return False
    text = f"{g.name} {g.genre_l2} {g.description}".lower()
    return any(t in text for t in tokens)


def _winner_credit(ccu: int) -> float:
    """Soft 1k bar: 0 below 500 CCU, full credit at/above 1500, linear between -
    avoids a knife-edge at exactly 1000 on a single live snapshot."""
    return min(1.0, max(0.0, (ccu - 500) / 1000))


def _concentration(subset: list[Game]) -> dict:
    """Demand + concentration metrics for any set of games (cluster or focused)."""
    live = [g for g in subset if g.ccu > 0]
    ranked = sorted(live, key=lambda g: g.ccu, reverse=True)
    demand = sum(g.ccu for g in live)
    winners = [g for g in live if g.is_winner()]
    fresh = [g for g in winners if g.is_fresh()]
    leader_share = (ranked[0].ccu / demand) if demand else 0.0
    top3_share = (sum(g.ccu for g in ranked[:3]) / demand) if demand else 0.0
    hhi = _hhi([g.ccu / demand for g in ranked]) if demand else 0
    return {"live": live, "ranked": ranked, "demand": demand, "winners": winners,
            "fresh": fresh, "leader_share": leader_share, "top3_share": top3_share,
            "hhi": hhi}


def _focus_set(live: list[Game], query: str) -> tuple[list[Game], str]:
    """De-contaminate a keyword cluster: keep games that are on-topic by keyword OR
    sit in the niche's dominant genre_l2. Returns (focused_games, dominant_genre)."""
    tokens = _tokens(query)
    seed = [g for g in live if _on_topic(g, tokens)]
    # dominant genre = the genre_l2 holding the most CCU among on-topic seed games
    # (falls back to the whole cluster only if nothing matched the keyword).
    pool = seed if seed else live
    by_genre: dict[str, int] = {}
    for g in pool:
        if g.genre_l2 and g.genre_l2 != "(unknown)":
            by_genre[g.genre_l2] = by_genre.get(g.genre_l2, 0) + g.ccu
    dominant_genre = max(by_genre, key=by_genre.get) if by_genre else ""
    focus = [g for g in live
             if _on_topic(g, tokens) or (dominant_genre and g.genre_l2 == dominant_genre)]
    return (focus or live), dominant_genre


def analyze_niche(games: list[Game], query: str) -> dict:
    """Turn a list of competitor games into an opportunity verdict.

    A raw keyword search mixes genres and is contaminated by mega-titles that rank
    for everything. So concentration + scoring run on the FOCUSED set: games that
    are on-topic by keyword OR sit in the niche's dominant genre_l2. The verdict and
    score reflect the lane you'd actually enter; raw cluster numbers are reported too.
    """
    live = [g for g in games if g.ccu > 0]
    focus, dominant_genre = _focus_set(live, query)

    clu = _concentration(live)
    foc = _concentration(focus)
    contamination = round(100 * (clu["demand"] - foc["demand"]) / clu["demand"], 1) if clu["demand"] else 0.0

    ranked = foc["ranked"]
    leaders = ranked[:5]
    stale_leaders = [g for g in leaders if g.is_stale()]
    ls, hhi = foc["leader_share"], foc["hhi"]

    def _clamp(x, lo=0.0, hi=1.0):
        return max(lo, min(hi, x))

    # ---- recalibrated scoring on the de-contaminated focused set (max 100) ----
    # DEMAND (0-30): log-linear across 1k -> 1M so it keeps discriminating up high.
    demand_pts = 30.0 * _clamp((math.log10(foc["demand"] + 1) - 3) / 3) if foc["demand"] else 0.0
    # OPENNESS (0-30): low concentration == room to enter. Punishes monoliths.
    openness_pts = max(0.0, 30.0 * (1 - max(ls, hhi / 10000)))
    # REACHABILITY (0-20): how many independent ~1k+ winners exist (soft-banded).
    winners_pts = min(20.0, 2.5 * sum(_winner_credit(g.ccu) for g in foc["live"]))
    # FRESHNESS (0-15): niche still launches winners -> a newcomer can too.
    freshness_pts = min(15.0, 5.0 * len(foc["fresh"]))
    # STALENESS (0-5): top games not updated recently == vulnerable incumbents.
    staleness_pts = min(5.0, 2.5 * len(stale_leaders))
    score = round(demand_pts + openness_pts + winners_pts + freshness_pts + staleness_pts, 1)

    # ---- plain-language verdict (on the focused lane) ----
    lname = ranked[0].name if ranked else None
    if not foc["demand"]:
        verdict = "DEAD - no live players found in this lane."
    elif len(foc["winners"]) == 0:
        verdict = "UNPROVEN - no game in this lane holds 1k+ CCU; demand may be too thin."
    elif ls >= 0.6 or hhi >= 5000:
        verdict = (f"MONOLITH - '{lname}' owns {ls*100:.0f}% of live players in the "
                   f"{dominant_genre or 'niche'} lane. Hard to displace head-on; "
                   f"differentiate hard or pick a sub-niche.")
    elif len(foc["fresh"]) >= 1:
        verdict = (f"OPEN & PROVEN - fragmented (top game {ls*100:.0f}%) AND "
                   f"{len(foc['fresh'])} game(s) hit 1k+ in the last {FRESH_DAYS}d. "
                   f"A fresh entrant can realistically reach 1k+.")
    else:
        verdict = (f"OPEN BUT COOLING - fragmented (top game {ls*100:.0f}%), but no NEW "
                   f"1k+ winner in {FRESH_DAYS}d. Demand exists; momentum unclear.")

    return {
        "label": query,
        "query": query,
        "dominant_genre": dominant_genre,
        "games_scanned": len(games),
        "live_games": len(clu["live"]),
        "focus_games": len(foc["live"]),
        "contamination_pct": contamination,       # share of cluster CCU that was off-lane
        # ---- headline = focused / de-contaminated lane ----
        "demand_ccu": foc["demand"],
        "winners_1k": len(foc["winners"]),
        "fresh_winners_1k": len(foc["fresh"]),
        "leader": lname,
        "leader_ccu": ranked[0].ccu if ranked else 0,
        "leader_share_pct": round(ls * 100, 1),
        "top3_share_pct": round(foc["top3_share"] * 100, 1),
        "hhi": hhi,
        "stale_leaders": [g.name for g in stale_leaders],
        # ---- raw keyword cluster (for transparency / contamination check) ----
        "cluster_demand_ccu": clu["demand"],
        "cluster_leader": clu["ranked"][0].name if clu["ranked"] else None,
        "cluster_leader_share_pct": round(clu["leader_share"] * 100, 1),
        "cluster_hhi": clu["hhi"],
        "cluster_winners_1k": len(clu["winners"]),
        "score": score,
        "score_breakdown": {
            "demand": round(demand_pts, 1),
            "openness": round(openness_pts, 1),
            "winners": round(winners_pts, 1),
            "freshness": round(freshness_pts, 1),
            "staleness": round(staleness_pts, 1),
        },
        "verdict": verdict,
        "leaders": [
            {"name": g.name, "ccu": g.ccu, "rating": g.rating, "age_days": g.age_days,
             "days_since_update": g.days_since_update, "genre_l2": g.genre_l2,
             "visits_per_day": g.visits_per_day, "url": g.url}
            for g in leaders
        ],
    }


def genre_breakdown(games: list[Game]) -> list[dict]:
    """Aggregate live CCU by genre_l2 so you can see which sub-niche is the demand."""
    buckets: dict[str, list[Game]] = {}
    for g in games:
        if g.ccu > 0:
            buckets.setdefault(g.genre_l2 or "(unknown)", []).append(g)
    rows = []
    for genre, gs in buckets.items():
        gs.sort(key=lambda g: g.ccu, reverse=True)
        demand = sum(x.ccu for x in gs)
        rows.append({
            "genre_l2": genre,
            "games": len(gs),
            "demand_ccu": demand,
            "winners_1k": sum(1 for x in gs if x.is_winner()),
            "leader": gs[0].name,
            "leader_ccu": gs[0].ccu,
            "leader_share_pct": round(100 * gs[0].ccu / demand, 1) if demand else 0,
        })
    rows.sort(key=lambda r: r["demand_ccu"], reverse=True)
    return rows


# --------------------------------------------------------------------------- #
# Combo gap-finder  (the "find untapped genre/theme combinations" analysis,
# the Roblox analog of the Steam genre-pair opportunity chart)
# --------------------------------------------------------------------------- #
# Roblox has no multi-tag system like Steam's 447 tags - a game gets ONE genre_l2.
# But the current meta encodes its mash-ups right in the titles (Steal a Brainrot,
# Grow a Garden, Fish a Slime). So we mine a curated mechanic/theme vocabulary from
# names + descriptions, add genre_l2 as a tag, and look for pairs of individually
# popular tags that are rarely combined -> white space.

MECHANIC_TAGS = [
    "rng", "gacha", "roll", "unbox", "steal", "rob", "merge", "fuse", "grow", "plant",
    "harvest", "craft", "build", "cook", "bake", "serve", "fish", "mine", "dig", "hatch",
    "breed", "collect", "trade", "sell", "tycoon", "simulator", "clicker", "tap", "idle",
    "incremental", "rebirth", "upgrade", "evolve", "obby", "parkour", "tower", "escape",
    "survive", "defense", "race", "drive", "fly", "fight", "battle", "tag", "hide", "eat",
    "speed", "run", "jump", "kick", "punch", "lift", "lucky", "spin", "pet", "hoard",
]
THEME_TAGS = [
    "slime", "anime", "food", "candy", "chocolate", "dessert", "pizza", "burger", "sushi",
    "fruit", "horror", "zombie", "ninja", "dragon", "monster", "brainrot", "car", "train",
    "plane", "boat", "ocean", "space", "alien", "garden", "kitchen", "restaurant", "cafe",
    "prison", "school", "city", "fantasy", "magic", "wizard", "robot", "mech", "dino",
    "cat", "dog", "animal", "blox", "noob", "baby", "giant",
]
_TAG_KIND = {**{t: "mechanic" for t in MECHANIC_TAGS}, **{t: "theme" for t in THEME_TAGS}}
_TAG_RE = {t: re.compile(r"\b" + re.escape(t) + r"(?:s|es|ing|ed|er|ers)?\b", re.I)
           for t in MECHANIC_TAGS + THEME_TAGS}
MIN_TAG_GAMES = 8     # a tag must appear in >= this many games to count as "popular"


def game_tags(g: Game) -> set[str]:
    """Mechanic/theme tokens found in a game's name+description, plus its genre_l2."""
    text = f"{g.name} {g.description}"
    tags = {t for t, rx in _TAG_RE.items() if rx.search(text)}
    if g.genre_l2:
        tags.add("genre:" + g.genre_l2)
    return tags


def _redundant(a: str, b: str) -> bool:
    """Two tags name the same concept (e.g. 'tycoon' vs 'genre:Tycoon')."""
    ca, cb = a.split("genre:")[-1].lower(), b.split("genre:")[-1].lower()
    wa, wb = set(ca.split()), set(cb.split())
    return bool(wa & wb) or ca in cb or cb in ca


def harvest_corpus(client: RobloxClient, pages: int = 1, extra_seeds: list[str] | None = None) -> list[Game]:
    """Build a broad Roblox 'catalog': union omni-search over the whole mechanic/theme
    vocabulary plus every trending feed, then enrich. Returns deduped Game objects."""
    raw_by_id: dict[int, dict] = {}
    for sid in TRENDING_SORTS:                       # the current hits
        for c in client.sort_content(sid):
            uid = c.get("universeId")
            if uid:
                raw_by_id.setdefault(uid, c)
    seeds = list(dict.fromkeys(MECHANIC_TAGS + THEME_TAGS + (extra_seeds or [])))
    for i, kw in enumerate(seeds, 1):
        for c in client.omni_search(kw, pages=pages):
            uid = c.get("universeId")
            if uid:
                raw_by_id.setdefault(uid, c)
        if i % 15 == 0:
            print(f"  ...{i}/{len(seeds)} seeds scanned, {len(raw_by_id)} unique games", file=sys.stderr)
    print(f"  enriching {len(raw_by_id)} games...", file=sys.stderr)
    return enrich(client, list(raw_by_id.values()))


def load_corpus(path: str) -> list[Game]:
    """Rebuild Game objects from a saved corpus JSON."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    names = {fld.name for fld in dataclass_fields(Game)}
    return [Game(**{k: d.get(k) for k in names}) for d in data.get("games", [])]


def analyze_combos(games: list[Game]) -> dict:
    """Cross-tabulate every tag pair. Surface 'proven & underbuilt' combos (a 1.5k+
    hit exists but few games do it) and 'untapped' combos (both tags popular, never
    combined). Also a scatter row per pair and a 'winning ingredients' tag table."""
    from itertools import combinations

    tag_games: dict[str, list[Game]] = {}
    for g in games:
        for t in game_tags(g):
            tag_games.setdefault(t, []).append(g)
    popular = {t for t, gs in tag_games.items() if len(gs) >= MIN_TAG_GAMES}

    pair_games: dict[tuple, list[Game]] = {}
    for g in games:
        tags = sorted(t for t in game_tags(g) if t in popular)
        for a, b in combinations(tags, 2):
            pair_games.setdefault((a, b), []).append(g)

    scatter, proven, untapped = [], [], []
    seen_pairs = set()
    for (a, b), gs in pair_games.items():
        if _redundant(a, b):
            continue
        seen_pairs.add((a, b))
        n_a, n_b = len(tag_games[a]), len(tag_games[b])
        n_both = len(gs)
        best = max(gs, key=lambda g: g.ccu)
        demand_both = sum(g.ccu for g in gs)
        rated = [g.rating for g in gs if g.rating is not None]
        row = {
            "tag_a": a, "tag_b": b, "n_a": n_a, "n_b": n_b, "n_both": n_both,
            "reach": n_a + n_b - n_both, "demand_both": demand_both,
            "max_ccu": best.ccu, "best_game": best.name, "best_url": best.url,
            "avg_rating": round(sum(rated) / len(rated), 1) if rated else None,
        }
        if 2 <= n_both <= 8 and best.ccu >= 1500:
            row["score"] = round(best.ccu / n_both)
            proven.append(row)
        scatter.append(row)

    # untapped: both tags popular but they NEVER co-occur in the corpus
    pop_list = sorted(popular, key=lambda t: len(tag_games[t]), reverse=True)
    for a, b in combinations(pop_list, 2):
        key = (a, b) if a < b else (b, a)
        if key in pair_games or _redundant(a, b):
            continue
        # demand of each side -> only interesting if both are genuinely big
        da = sum(g.ccu for g in tag_games[a])
        db = sum(g.ccu for g in tag_games[b])
        if len(tag_games[a]) >= 18 and len(tag_games[b]) >= 18:
            untapped.append({
                "tag_a": a, "tag_b": b, "n_a": len(tag_games[a]), "n_b": len(tag_games[b]),
                "demand_a": da, "demand_b": db, "min_pop": min(len(tag_games[a]), len(tag_games[b])),
            })

    proven.sort(key=lambda r: r["score"], reverse=True)
    untapped.sort(key=lambda r: r["min_pop"], reverse=True)

    # winning ingredients: which tags over-index among 1k+ CCU games
    winners = [g for g in games if g.is_winner()]
    n_all, n_win = len(games), max(1, len(winners))
    ingredients = []
    for t, gs in tag_games.items():
        if len(gs) < MIN_TAG_GAMES or t.startswith("genre:"):
            continue
        share_all = len(gs) / n_all
        share_win = sum(1 for g in gs if g.is_winner()) / n_win
        if share_all > 0:
            ingredients.append({
                "tag": t, "kind": _TAG_KIND.get(t, "?"), "games": len(gs),
                "winners": sum(1 for g in gs if g.is_winner()),
                "lift": round(share_win / share_all, 2),
                "demand": sum(g.ccu for g in gs),
            })
    ingredients.sort(key=lambda r: r["lift"], reverse=True)

    return {"scatter": scatter, "proven": proven, "untapped": untapped,
            "ingredients": ingredients, "n_games": len(games), "n_tags": len(popular)}


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #
CSV_FIELDS = ["name", "ccu", "visits", "favorites", "rating", "up", "down",
              "genre_l1", "genre_l2", "age_days", "days_since_update",
              "visits_per_day", "ccu_per_1k_visits", "max_players",
              "creator", "place_id", "universe_id", "created", "updated",
              "source", "url"]


def _fmt(n) -> str:
    if isinstance(n, float):
        return f"{n:,.1f}"
    if isinstance(n, int):
        return f"{n:,}"
    return str(n) if n is not None else ""


def print_games_table(games: list[Game], limit: int = 40):
    if not games:
        print("  (no games)")
        return
    cols = [("name", 34, "<"), ("ccu", 9, ">"), ("rating", 7, ">"),
            ("visits", 14, ">"), ("age_days", 6, ">"), ("upd", 5, ">"),
            ("genre_l2", 24, "<")]
    header = "  ".join(f"{title:{al}{w}}" for title, w, al in cols)
    print(header)
    print("-" * len(header))
    for g in games[:limit]:
        row = {
            "name": (g.name or "")[:34], "ccu": _fmt(g.ccu),
            "rating": f"{g.rating}%" if g.rating is not None else "-",
            "visits": _fmt(g.visits), "age_days": _fmt(g.age_days),
            "upd": _fmt(g.days_since_update), "genre_l2": (g.genre_l2 or "")[:24],
        }
        print("  ".join(f"{str(row[t]):{al}{w}}" for t, w, al in cols))
    if len(games) > limit:
        print(f"  ... and {len(games) - limit} more")


def write_csv(games: list[Game], path: str):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for g in games:
            d = asdict(g)
            w.writerow({k: d.get(k) for k in CSV_FIELDS})
    print(f"  wrote {path}  ({len(games)} rows)")


def write_json(games: list[Game], path: str, extra: dict | None = None):
    payload = {"generated_utc": datetime.now(timezone.utc).isoformat(),
               "count": len(games), "games": [asdict(g) for g in games]}
    if extra:
        payload.update(extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"  wrote {path}")


def write_niche_report(path: str, analysis: dict, breakdown: list[dict], games: list[Game]):
    a = analysis
    sb = a["score_breakdown"]
    lines = [
        f"# Niche opportunity report - \"{a['label']}\"",
        "",
        f"_Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} from live Roblox data._",
        "",
        "## Verdict",
        "",
        f"**Opportunity score: {a['score']}/100**  ", f"{a['verdict']}",
        "",
        f"Analysis is focused on the **{a['dominant_genre'] or 'whole keyword'}** lane "
        f"(the niche's dominant sub-genre), to strip out keyword-search bleed. "
        f"**{a['contamination_pct']}%** of the raw keyword cluster's CCU was off-lane "
        f"(raw cluster demand {a['cluster_demand_ccu']:,}, raw leader "
        f"*{a['cluster_leader']}* {a['cluster_leader_share_pct']}%).",
        "",
        f"- Demand in lane (live CCU): **{a['demand_ccu']:,}**",
        f"- Games holding 1k+ CCU in lane: **{a['winners_1k']}**  "
        f"(of which **{a['fresh_winners_1k']}** launched in the last {FRESH_DAYS} days)",
        f"- Lane leader: **{a['leader']}** at **{a['leader_ccu']:,}** CCU "
        f"= **{a['leader_share_pct']}%** of the lane (top-3 = {a['top3_share_pct']}%)",
        f"- Concentration HHI: **{a['hhi']}** "
        f"({'monolithic' if a['hhi'] >= 5000 else 'concentrated' if a['hhi'] >= 1500 else 'fragmented'})",
        f"- Stale leaders (no update in {STALE_DAYS}d): "
        f"{', '.join(a['stale_leaders']) if a['stale_leaders'] else 'none'}",
        "",
        f"Score = demand {sb['demand']} + openness {sb['openness']} + "
        f"winners {sb['winners']} + freshness {sb['freshness']} + staleness {sb['staleness']}.",
        "",
        "## Top competitors",
        "",
        "| # | Game | CCU | Rating | Age (d) | Updated (d ago) | Sub-genre |",
        "|---|------|-----|--------|---------|-----------------|-----------|",
    ]
    for i, g in enumerate(games[:15], 1):
        lines.append(f"| {i} | [{g.name}]({g.url}) | {g.ccu:,} | "
                     f"{g.rating if g.rating is not None else '-'}% | {g.age_days} | "
                     f"{g.days_since_update} | {g.genre_l2 or '-'} |")
    lines += ["", "## Where the demand sits (by sub-genre)", "",
              "| Sub-genre | Games | Demand CCU | 1k+ games | Leader | Leader share |",
              "|-----------|-------|-----------|-----------|--------|--------------|"]
    for r in breakdown[:12]:
        lines.append(f"| {r['genre_l2']} | {r['games']} | {r['demand_ccu']:,} | "
                     f"{r['winners_1k']} | {r['leader']} ({r['leader_ccu']:,}) | "
                     f"{r['leader_share_pct']}% |")
    lines += ["", "---",
              "_Snapshot only. CCU is live concurrent players; no historical peak or "
              "retention is available from public APIs. Treat as demand signal, not proof._", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  wrote {path}")


# --------------------------------------------------------------------------- #
# Input parsing
# --------------------------------------------------------------------------- #
def parse_place_id(token: str) -> int | None:
    """Accept a bare place id or a roblox game URL -> place id."""
    token = token.strip()
    if token.isdigit():
        return int(token)
    m = re.search(r"/games/(\d+)", token)
    return int(m.group(1)) if m else None


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_search(client: RobloxClient, args):
    print(f"Searching '{args.query}' ({args.pages} page(s))...", file=sys.stderr)
    raw = client.omni_search(args.query, pages=args.pages)
    games = enrich(client, raw)
    if args.min_ccu:
        games = [g for g in games if g.ccu >= args.min_ccu]
    print(f"\n{len(games)} games for '{args.query}':\n")
    print_games_table(games, limit=args.limit)
    if args.csv:
        write_csv(games, args.csv)
    if args.json:
        write_json(games, args.json, extra={"query": args.query})


def cmd_trending(client: RobloxClient, args):
    sorts = TRENDING_SORTS if args.sort == "all" else [args.sort]
    all_games: list[Game] = []
    for sid in sorts:
        print(f"\n=== {sid} ===", file=sys.stderr)
        raw = client.sort_content(sid)
        if not raw:
            print("  (empty / not enabled)")
            continue
        games = enrich(client, raw) if not args.fast else _quick_games(raw)
        for g in games:
            g.source = sid                 # tag feed origin so it survives to CSV/JSON
        print(f"\n{sid}  ({len(games)} games):\n")
        print_games_table(games, limit=args.limit)
        all_games.extend(games)
    if args.csv and all_games:
        write_csv(all_games, args.csv)
    if args.json and all_games:
        write_json(all_games, args.json, extra={"sorts": sorts})


def _quick_games(raw: list[dict]) -> list[Game]:
    """Build Game objects from discovery data only (no enrichment) - fast mode."""
    now = datetime.now(timezone.utc)
    games = [_build_game(c["universeId"], None, None, c, now) for c in raw if c.get("universeId")]
    games.sort(key=lambda g: g.ccu, reverse=True)
    return games


def cmd_niche(client: RobloxClient, args):
    print(f"Scanning niche '{args.query}' ({args.pages} page(s))...", file=sys.stderr)
    raw = client.omni_search(args.query, pages=args.pages)
    games = enrich(client, raw)
    if not games:
        sys.exit("No games found - try a broader keyword.")
    analysis = analyze_niche(games, args.query)
    breakdown = genre_breakdown(games)

    a = analysis
    print(f"\n{'='*70}\nNICHE: {args.query}\n{'='*70}")
    print(f"Opportunity score : {a['score']}/100")
    print(f"Verdict           : {a['verdict']}")
    print(f"Focused lane      : {a['dominant_genre'] or '(whole keyword)'}   "
          f"({a['contamination_pct']}% of raw cluster CCU was off-lane bleed)")
    print(f"Demand in lane    : {a['demand_ccu']:,} CCU across {a['focus_games']} games "
          f"(raw keyword cluster: {a['cluster_demand_ccu']:,} across {a['live_games']})")
    print(f"1k+ winners       : {a['winners_1k']}   (fresh <{FRESH_DAYS}d: {a['fresh_winners_1k']})")
    print(f"Lane leader       : {a['leader']}  {a['leader_ccu']:,} CCU "
          f"({a['leader_share_pct']}% share)   HHI={a['hhi']}")
    focus_games, _ = _focus_set([g for g in games if g.ccu > 0], args.query)
    focus_games.sort(key=lambda g: g.ccu, reverse=True)
    print(f"\nTop competitors (focused lane):\n")
    print_games_table(focus_games, limit=args.limit)
    print(f"\nDemand by sub-genre:")
    for r in breakdown[:10]:
        print(f"  {r['genre_l2'][:28]:<28}  CCU {r['demand_ccu']:>9,}  "
              f"games {r['games']:>3}  1k+ {r['winners_1k']:>2}  "
              f"leader {r['leader'][:24]} ({r['leader_share_pct']}%)")

    if args.csv:
        write_csv(games, args.csv)
    if args.json:
        write_json(games, args.json,
                   extra={"query": args.query, "analysis": analysis, "genre_breakdown": breakdown})
    if args.report:
        write_niche_report(args.report, analysis, breakdown, focus_games)


def cmd_inspect(client: RobloxClient, args):
    universe_ids = []
    if args.universe:
        universe_ids = [int(t) for t in args.targets if t.strip().isdigit()]
    else:
        for t in args.targets:
            pid = parse_place_id(t)
            if pid is None:
                print(f"  ! {t}: not a place id or game url - skipped", file=sys.stderr)
                continue
            uid = client.resolve_universe(pid)
            if uid is None:
                print(f"  ! {pid}: not found / private - skipped", file=sys.stderr)
                continue
            universe_ids.append(uid)
    if not universe_ids:
        sys.exit("Nothing to inspect.")
    raw = [{"universeId": u} for u in universe_ids]
    games = enrich(client, raw)
    print(f"\n{len(games)} game(s):\n")
    for g in games:
        print(f"{g.name}  ({g.url})")
        print(f"   CCU {g.ccu:,}   visits {g.visits:,}   favorites {g.favorites:,}   "
              f"rating {g.rating}%   maxplayers {g.max_players}")
        print(f"   genre {g.genre_l1} / {g.genre_l2}   age {g.age_days}d   "
              f"updated {g.days_since_update}d ago   visits/day {g.visits_per_day:,.0f}")
        print(f"   by {g.creator}   {g.description[:120]}")
        print()
    if args.csv:
        write_csv(games, args.csv)
    if args.json:
        write_json(games, args.json)


def _tag_label(t: str) -> str:
    return t.split("genre:")[-1] + ("°" if t.startswith("genre:") else "")


def write_combos_csv(scatter: list[dict], path: str):
    fields = ["tag_a", "tag_b", "n_a", "n_b", "n_both", "reach", "demand_both",
              "max_ccu", "best_game", "avg_rating", "best_url"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in sorted(scatter, key=lambda r: r["max_ccu"], reverse=True):
            w.writerow(r)
    print(f"  wrote {path}  ({len(scatter)} tag-pairs - scatter reach vs n_both to see gaps)")


def write_combos_report(path: str, res: dict):
    lines = [
        "# Roblox combo gap-finder - untapped game ideas",
        "",
        f"_Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} from a corpus of "
        f"{res['n_games']:,} games / {res['n_tags']} popular tags. (`°` = official genre.)_",
        "",
        "Each opportunity is a **mechanic x theme/genre** pair that is individually popular "
        "but rarely combined - the Roblox analog of \"only 3 platformer auto-battlers exist.\"",
        "",
        "## Proven & underbuilt (a real 1.5k+ hit exists, but few games copy it)",
        "",
        "These are the safest bets: the combo demonstrably works, yet the lane is thin.",
        "",
        "| Combo | Games | Best game (CCU) | Avg rating | Reach |",
        "|-------|-------|-----------------|-----------|-------|",
    ]
    for r in res["proven"][:20]:
        lines.append(f"| **{_tag_label(r['tag_a'])} x {_tag_label(r['tag_b'])}** | {r['n_both']} | "
                     f"[{r['best_game'][:32]}]({r['best_url']}) ({r['max_ccu']:,}) | "
                     f"{r['avg_rating'] if r['avg_rating'] is not None else '-'}% | {r['reach']} |")
    lines += ["", "## Untapped (both tags popular, but never combined in the corpus)",
              "", "Blue-sky: nobody has shipped these together. Higher risk - could be novel, "
              "or could be a combo players don't want. Validate before committing.", "",
              "| Combo | Games tag A | Games tag B |",
              "|-------|------------|------------|"]
    for r in res["untapped"][:20]:
        lines.append(f"| **{_tag_label(r['tag_a'])} x {_tag_label(r['tag_b'])}** | "
                     f"{r['n_a']} | {r['n_b']} |")
    lines += ["", "## Winning ingredients (tags that over-index among 1k+ CCU games)",
              "", "Lift > 1 means the tag appears more often among winners than in the corpus "
              "at large - the mechanics/themes that correlate with hits right now.", "",
              "| Tag | Kind | Lift | Winners / Games | Demand CCU |",
              "|-----|------|------|-----------------|-----------|"]
    for r in res["ingredients"][:25]:
        lines.append(f"| {r['tag']} | {r['kind']} | **{r['lift']}x** | "
                     f"{r['winners']}/{r['games']} | {r['demand']:,} |")
    lines += ["", "---", "_Snapshot only; CCU is live concurrent players. A pair is only as good "
              "as the corpus - rebuild it (`--rebuild`) for fresh data. Tag matching is keyword-"
              "based, so spot-check the best_game before trusting a combo._", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  wrote {path}")


def cmd_harvest(client: RobloxClient, args):
    games = harvest_corpus(client, pages=args.pages)
    out = args.out or os.path.join("data", "corpus.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    write_json(games, out, extra={"corpus": True, "seeds": len(MECHANIC_TAGS) + len(THEME_TAGS)})
    print(f"corpus: {len(games)} games -> {out}")


def cmd_combos(client: RobloxClient, args):
    corpus_path = args.corpus or os.path.join("data", "corpus.json")
    if args.rebuild or not os.path.exists(corpus_path):
        print("Building corpus (this takes a few minutes)...", file=sys.stderr)
        games = harvest_corpus(client, pages=args.pages)
        os.makedirs(os.path.dirname(corpus_path) or ".", exist_ok=True)
        write_json(games, corpus_path, extra={"corpus": True})
    else:
        games = load_corpus(corpus_path)
        print(f"Loaded corpus: {len(games)} games from {corpus_path} "
              f"(use --rebuild to refresh)", file=sys.stderr)

    res = analyze_combos(games)
    print(f"\n{'='*72}\nROBLOX COMBO GAP-FINDER  ({res['n_games']:,} games, {res['n_tags']} popular tags)\n{'='*72}")
    print("\nPROVEN & UNDERBUILT  (a 1.5k+ hit exists, but few games do it):\n")
    print(f"  {'combo':<34}{'#':>3}  {'best game (CCU)':<34}{'reach':>6}")
    print("  " + "-" * 78)
    for r in res["proven"][:args.limit]:
        combo = f"{_tag_label(r['tag_a'])} x {_tag_label(r['tag_b'])}"
        best = f"{r['best_game'][:24]} ({r['max_ccu']:,})"
        print(f"  {combo[:34]:<34}{r['n_both']:>3}  {best:<34}{r['reach']:>6}")
    print("\nUNTAPPED  (both tags popular, never combined - blue-sky):\n")
    for r in res["untapped"][:args.limit]:
        print(f"  {_tag_label(r['tag_a'])} x {_tag_label(r['tag_b'])}"
              f"   ({r['n_a']} + {r['n_b']} games, 0 combined)")
    print("\nWINNING INGREDIENTS  (tags over-indexed among 1k+ CCU games):\n")
    for r in res["ingredients"][:15]:
        print(f"  {r['tag']:<14} {r['kind']:<9} lift {r['lift']:>4}x   "
              f"{r['winners']:>2}/{r['games']:<3} winners   demand {r['demand']:>8,}")

    if args.csv:
        write_combos_csv(res["scatter"], args.csv)
    if args.report:
        write_combos_report(args.report, res)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print(f"  wrote {args.json}")


# --------------------------------------------------------------------------- #
# Selftest  (happy + edge + failure paths against live API)
# --------------------------------------------------------------------------- #
def selftest():
    c = RobloxClient()
    # happy: live game resolves and enriches
    uid = c.resolve_universe(76558904092080)
    assert uid, "expected a universe id for a live place"
    games = enrich(c, [{"universeId": uid}])
    assert games and games[0].visits > 0, "expected visits for a live game"
    # discovery: omni-search returns the obvious result for a famous game
    hits = c.omni_search("slime rng", pages=1)
    assert any("slime" in (h.get("name", "").lower()) for h in hits), "expected Slime RNG in search"
    # trending feed populated
    trend = c.sort_content("top-playing-now")
    assert trend and trend[0].get("playerCount", 0) > 0, "expected a populated trending feed"
    # niche analysis math: synthetic monolith vs fragmented
    now = datetime.now(timezone.utc)
    mono = [_build_game(i, {"playing": p, "rootPlaceId": i, "name": f"g{i}"}, None, None, now)
            for i, p in enumerate([9000, 200, 100, 50], 1)]
    a = analyze_niche(mono, "mono")
    assert a["leader_share_pct"] > 90 and a["hhi"] > 5000, "monolith should score concentrated"
    assert "MONOLITH" in a["verdict"], "expected monolith verdict"
    frag = [_build_game(i, {"playing": p, "rootPlaceId": i, "name": f"g{i}"}, None, None, now)
            for i, p in enumerate([2500, 2200, 2000, 1800, 1500], 1)]
    af = analyze_niche(frag, "frag")
    assert af["winners_1k"] == 5 and af["hhi"] < 5000, "fragmented niche should not read as monolith"
    assert set(af["score_breakdown"]) == {"demand", "openness", "winners", "freshness", "staleness"}
    # de-contamination: an off-topic mega-title must be stripped from the focused lane
    mixed = [
        _build_game(11, {"playing": 5000, "rootPlaceId": 11, "name": "Cook Tycoon", "genre_l2": "Tycoon"}, None, None, now),
        _build_game(12, {"playing": 3000, "rootPlaceId": 12, "name": "Chef Empire", "genre_l2": "Tycoon"}, None, None, now),
        _build_game(13, {"playing": 60000, "rootPlaceId": 13, "name": "Squid Game X", "genre_l2": "1 vs All"}, None, None, now),
    ]
    am = analyze_niche(mixed, "cooking")
    assert am["dominant_genre"] == "Tycoon", "dominant on-topic genre should be Tycoon"
    assert am["demand_ccu"] == 8000, "focused demand must exclude the off-topic 60k padder"
    assert am["cluster_demand_ccu"] == 68000 and am["contamination_pct"] > 80, "cluster must show the bleed"
    assert am["leader"] == "Cook Tycoon", "lane leader is the focused leader, not the padder"
    # helper math
    assert _stem("cooking") == "cook" and _stem("food") == "food", "stemmer"
    assert _winner_credit(400) == 0 and _winner_credit(1000) == 0.5 and _winner_credit(1500) == 1.0, "soft 1k band"
    # combo tagging + gap analysis on a tiny synthetic corpus
    tg = _build_game(21, {"playing": 3000, "rootPlaceId": 21, "name": "Cook a Slime RNG", "genre_l2": "Tycoon"}, None, None, now)
    assert {"cook", "slime", "rng"} <= game_tags(tg), "should mine mechanic/theme tokens from the title"
    assert "genre:Tycoon" in game_tags(tg), "genre_l2 should become a tag"
    assert not _TAG_RE["car"].search("scary card scar"), "short tags must match whole words only"
    assert _redundant("tycoon", "genre:Tycoon"), "synonym tags should be flagged redundant"
    # failure: bad id -> None, empty -> []
    assert c.resolve_universe(0) is None, "invalid place id should resolve to None"
    assert enrich(c, []) == [], "empty discovery should enrich to []"
    assert _hhi([1.0]) == 10000 and _hhi([0.5, 0.5]) == 5000, "hhi math"
    print("selftest passed - live endpoints, enrichment, and gap-analysis math all OK")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Roblox market & competitor research toolkit.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="examples:\n"
               "  python roblox_research.py selftest\n"
               "  python roblox_research.py search \"cooking\" --pages 3 --csv cooking.csv\n"
               "  python roblox_research.py trending --sort up-and-coming\n"
               "  python roblox_research.py niche \"slime rng\" --pages 3 --report slime.md --json slime.json\n"
               "  python roblox_research.py inspect 76558904092080\n")
    p.add_argument("--sleep", type=float, default=0.4, help="politeness delay between calls (s)")
    p.add_argument("--verbose", action="store_true", help="log backoff/retries to stderr")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search", help="keyword-search existing games (so you don't rebuild one)")
    s.add_argument("query")
    s.add_argument("--pages", type=int, default=2, help="omni-search pages (~40 games each)")
    s.add_argument("--min-ccu", type=int, default=0, help="hide games below this live CCU")
    s.add_argument("--limit", type=int, default=40, help="rows to print")
    s.add_argument("--csv"); s.add_argument("--json")
    s.set_defaults(func=cmd_search)

    t = sub.add_parser("trending", help="Roblox's live hot feeds (what's popular/up-and-coming now)")
    t.add_argument("--sort", default="up-and-coming",
                   choices=TRENDING_SORTS + ["all"], help="which feed (default up-and-coming)")
    t.add_argument("--fast", action="store_true", help="skip detail enrichment (no genre/visits)")
    t.add_argument("--limit", type=int, default=40)
    t.add_argument("--csv"); t.add_argument("--json")
    t.set_defaults(func=cmd_trending)

    n = sub.add_parser("niche", help="GAP ANALYSIS: monolith vs open, can it support 1k+ CCU?")
    n.add_argument("query")
    n.add_argument("--pages", type=int, default=3)
    n.add_argument("--limit", type=int, default=25)
    n.add_argument("--csv"); n.add_argument("--json"); n.add_argument("--report",
                   help="write a markdown opportunity report to this path")
    n.set_defaults(func=cmd_niche)

    i = sub.add_parser("inspect", help="deep-dive specific games by place id / url / universe id")
    i.add_argument("targets", nargs="+", help="place ids, game urls, or (with --universe) universe ids")
    i.add_argument("--universe", action="store_true", help="treat targets as universe ids")
    i.add_argument("--csv"); i.add_argument("--json")
    i.set_defaults(func=cmd_inspect)

    h = sub.add_parser("harvest", help="build a broad game 'catalog' corpus for combo analysis")
    h.add_argument("--pages", type=int, default=1, help="omni-search pages per seed term")
    h.add_argument("--out", help="corpus path (default data/corpus.json)")
    h.set_defaults(func=cmd_harvest)

    c = sub.add_parser("combos", help="FIND IDEAS: untapped mechanic x theme/genre combinations")
    c.add_argument("--corpus", help="corpus path (default data/corpus.json; auto-built if missing)")
    c.add_argument("--rebuild", action="store_true", help="re-harvest the corpus before analyzing")
    c.add_argument("--pages", type=int, default=1, help="pages per seed when (re)building corpus")
    c.add_argument("--limit", type=int, default=20, help="rows per section to print")
    c.add_argument("--csv", help="write the full tag-pair scatter (reach vs co-occurrence)")
    c.add_argument("--json"); c.add_argument("--report",
                   help="write a markdown ideas report to this path")
    c.set_defaults(func=cmd_combos)

    sub.add_parser("selftest", help="hit live endpoints + check analysis math")
    return p


def _setup_stdio():
    """Roblox names are full of emoji; Windows' legacy cp1252 console crashes on
    them. Force UTF-8 with replacement so printing never blows up the run."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def main(argv=None):
    _setup_stdio()
    args = build_parser().parse_args(argv)
    if args.cmd == "selftest":
        selftest()
        return
    client = RobloxClient(sleep=args.sleep, verbose=args.verbose)
    args.func(client, args)


if __name__ == "__main__":
    main()
