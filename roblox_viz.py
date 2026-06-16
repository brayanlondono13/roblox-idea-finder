#!/usr/bin/env python3
"""
roblox_viz.py - turn a harvested corpus into an interactive HTML dashboard,
the way the Steam data-analysis YouTube videos present their findings.

It builds ONE self-contained file (no internet, no dependencies to view - just
double-click it) with two interactive, zoom/pan/hover scatter views:

  1. OPPORTUNITY MAP  (the "genre-combination" chart): every mechanic x theme/genre
     pair as a dot. x = reach (how popular the two ingredients are), y = how many
     games already combine them. Bottom-right = popular but rarely built = ideas.

  2. GAME UNIVERSE    (the "gaming map"): every game in the corpus placed by a t-SNE
     embedding of its tags, so similar games cluster. Colour = genre, size = live CCU.

  plus a WINNING INGREDIENTS bar chart (tags that over-index among 1k+ CCU games).

Build a corpus first:   python roblox_research.py harvest
Then:                   python roblox_viz.py            # -> roblox_dashboard.html
                        python roblox_viz.py --corpus data/corpus.json --out dash.html
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime, timezone

import numpy as np

from roblox_research import (load_corpus, load_corpus_meta, analyze_combos, game_tags,
                             occupancy_check, load_synonyms, load_polluters, concept_tokens,
                             WINNER_CCU, OCCUPANCY_FLOOR)


# --------------------------------------------------------------------------- #
# 2-D embedding of games by their tags (t-SNE if available, else PCA)
# --------------------------------------------------------------------------- #
def embed_games(games):
    tagsets = [game_tags(g) for g in games]
    vocab = sorted({t for ts in tagsets for t in ts})
    idx = {t: i for i, t in enumerate(vocab)}
    M = np.zeros((len(games), len(vocab)), dtype=float)
    for i, ts in enumerate(tagsets):
        for t in ts:
            M[i, idx[t]] = 1.0
    # tf-idf weighting so common tags (e.g. genre) don't dominate, then L2-normalise
    df = M.sum(axis=0)
    idf = np.log(len(games) / (1.0 + df)) + 1.0
    M = M * idf
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    M = M / norms

    method = "pca"
    coords = None
    try:
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        # PCA-reduce first for speed/stability, then t-SNE to 2-D
        k = min(30, M.shape[1])
        red = PCA(n_components=k, random_state=42).fit_transform(M)
        perp = max(5, min(40, (len(games) - 1) // 3))
        coords = TSNE(n_components=2, perplexity=perp, init="pca",
                      random_state=42, max_iter=1000).fit_transform(red)
        method = "t-SNE"
    except Exception as e:                       # numpy-only PCA fallback
        print(f"  (sklearn unavailable: {e}; using PCA)", file=sys.stderr)
        Mc = M - M.mean(axis=0)
        U, S, Vt = np.linalg.svd(Mc, full_matrices=False)
        coords = U[:, :2] * S[:2]
    # scale to a tidy 0..100 box
    coords = np.asarray(coords, dtype=float)
    mn, mx = coords.min(axis=0), coords.max(axis=0)
    span = np.where((mx - mn) == 0, 1, mx - mn)
    coords = (coords - mn) / span * 100.0
    return coords, method


# --------------------------------------------------------------------------- #
# Assemble the data the page needs
# --------------------------------------------------------------------------- #
def compute_right_now(games, res):
    """Data-driven signals that auto-refresh every harvest (no LLM needed):
    the most open lanes, the newest 1k+ winners, and the cleanest proven combos."""
    live = [g for g in games if g.ccu > 0]

    # open lanes: aggregate by genre_l2, find big-demand + low-concentration lanes
    from collections import defaultdict
    lanes = defaultdict(list)
    for g in live:
        if g.genre_l2:
            lanes[g.genre_l2].append(g)
    open_lanes = []
    for genre, gs in lanes.items():
        gs.sort(key=lambda g: g.ccu, reverse=True)
        demand = sum(g.ccu for g in gs)
        if demand < 5000 or len(gs) < 4:
            continue
        leader_share = gs[0].ccu / demand
        winners = sum(1 for g in gs if g.is_winner())
        fresh = sum(1 for g in gs if g.is_winner() and g.is_fresh())
        if leader_share <= 0.6 and winners >= 3:
            open_lanes.append({
                "lane": genre, "demand": demand, "games": len(gs), "winners": winners,
                "fresh_winners": fresh, "leader": gs[0].name, "leader_ccu": gs[0].ccu,
                "leader_share": round(leader_share * 100, 1),
                "openness": round(demand * (1 - leader_share)),
            })
    open_lanes.sort(key=lambda r: r["openness"], reverse=True)

    # newest 1k+ winners (proof the meta still mints hits)
    fresh_winners = sorted(
        [g for g in live if g.is_winner() and g.is_fresh()],
        key=lambda g: (g.age_days <= 30, g.ccu), reverse=True)
    fresh_winners = [{"name": g.name, "ccu": g.ccu, "age_days": g.age_days,
                      "genre": g.genre_l2, "rating": g.rating, "url": g.url}
                     for g in sorted(fresh_winners, key=lambda g: g.ccu, reverse=True)[:14]]

    # cleanest proven combos (drop the keyword-bleed mega-title)
    hot_combos = [r for r in res["proven"]
                  if "keyboard" not in (r["best_game"] or "").lower()][:12]

    return {"open_lanes": open_lanes[:10], "fresh_winners": fresh_winners,
            "hot_combos": hot_combos}


# --------------------------------------------------------------------------- #
# Curated-card enrichment: fix encoding, attach REAL live lane numbers
# --------------------------------------------------------------------------- #
# Known UTF-8-decoded-as-Latin-1 mojibake -> the character it should have been. The
# curated cards were synthesised with a few of these baked in (e.g. an em dash showing
# up as "â€"); fix them at build time so the dashboard never renders garbage.
_MOJIBAKE = {
    "â€”": "—",  # em dash —
    "â€“": "–",  # en dash –
    "â€™": "’",  # right single quote ’
    "â€œ": "“",  # left double quote “
    "â€": "”",  # right double quote ”
    "â€¢": "•",  # bullet •
    "Â ": " ",             # stray non-breaking space
    "�": "—",              # lossy replacement char -> em dash (best effort)
}


def _demojibake(s):
    if not isinstance(s, str):
        return s
    for bad, good in sorted(_MOJIBAKE.items(), key=lambda kv: -len(kv[0])):
        if bad in s:
            s = s.replace(bad, good)
    return s


# niche-file keyword -> the earliest one found in a card's lane text wins, so a
# "cooking / food-stand" card maps to cooking and never inherits a blended number.
_NICHE_KEYS = ["cooking", "food", "garden", "incremental", "merge", "pet",
               "restaurant", "rng", "steal", "brainrot", "unboxing"]
_NICHE_ALIAS = {"steal": "steal-brainrot", "brainrot": "steal-brainrot"}


def _lane_state(leader_pct, verdict=""):
    """Map a niche's live leader-share (+ verdict text) to a 3-value lane state. Static
    only — no TIGHTENING, since that would need yesterday's number and we keep no history."""
    v = (verdict or "").upper()
    if any(w in v for w in ("MONOLITH", "DOMINATED", "DEAD")):
        return "DOMINATED"
    if leader_pct is None:
        return "OPEN"
    if leader_pct >= 55:
        return "DOMINATED"
    if leader_pct >= 38:
        return "CONCENTRATED"
    return "OPEN"


def _load_niche_stats():
    """Per-niche live analysis (leader share, winners, contamination, verdict) keyed by
    niche slug, from data/niche_*.json. These are the REAL numbers a curated card's lane
    prose was supposed to quote. Empty dict if no niche files are present."""
    out = {}
    for path in glob.glob(os.path.join("data", "niche_*.json")):
        key = os.path.basename(path)[len("niche_"):-len(".json")]
        try:
            with open(path, encoding="utf-8") as f:
                an = json.load(f).get("analysis", {})
        except (OSError, ValueError):
            continue
        if not an:
            continue
        ls = an.get("leader_share_pct")
        out[key] = {
            "niche": key,
            "leader_pct": ls,
            "winners_1k": an.get("winners_1k"),
            "fresh_winners_1k": an.get("fresh_winners_1k"),
            "contamination_pct": an.get("contamination_pct"),
            "hhi": an.get("hhi"),
            "leader": an.get("leader"),
            "state": _lane_state(ls, an.get("verdict", "")),
        }
    return out


def _match_niche(card, niche_stats):
    """Earliest niche keyword appearing in the card's lane text -> that niche's live
    stats (or None). First-match-wins keeps attribution honest: it never blends two
    lanes' numbers the way the hand-written lane prose did."""
    lane = _demojibake(str(card.get("lane", ""))).lower()
    best_pos, best_key = len(lane) + 1, None
    for kw in _NICHE_KEYS:
        pos = lane.find(kw)
        if pos != -1 and pos < best_pos:
            best_pos, best_key = pos, _NICHE_ALIAS.get(kw, kw)
    return niche_stats.get(best_key) if best_key else None


def _slug(title, idx):
    s = re.sub(r"[^a-z0-9]+", "-", (title or "").lower()).strip("-")
    return s or f"pick-{idx + 1}"


def _card_occupancy(games, card, syn, deny):
    """Strongest live incumbent that shares this card's mechanic x theme concept, found
    via the synonym map. Returns the occupancy dict (with the matched [mechanic, theme]
    pair) or {'none_above_floor': OCCUPANCY_FLOOR}. So no curated 'open' pick renders
    without showing the real incumbent it has to beat."""
    text = " ".join(str(card.get(k, "")) for k in ("theme", "lane", "title", "tagline"))
    mechanics = [m for m in (card.get("mechanics") or []) if m]
    themes = concept_tokens(text, syn, deny)
    pol = load_polluters()
    best = None
    for m in mechanics:
        for t in themes:
            if m == t:
                continue
            occ = occupancy_check(games, m, t, synonyms=syn, denylist=deny, polluters=pol)
            if occ and (best is None or occ["incumbent_ccu"] > best["incumbent_ccu"]):
                best = {**occ, "pair": [m, t]}
    return best or {"none_above_floor": OCCUPANCY_FLOOR}


def build_payload(games, coverage=None):
    res = analyze_combos(games, coverage=coverage)
    proven_keys = {(r["tag_a"], r["tag_b"]) for r in res["proven"]}

    def lbl(t):
        return t.split("genre:")[-1] + ("°" if t.startswith("genre:") else "")

    combos = []
    for r in res["scatter"]:
        key = (r["tag_a"], r["tag_b"])
        if key in proven_keys:
            cat = "proven"
        elif r["n_both"] <= 2 and r["reach"] >= 30:
            cat = "rare"
        else:
            cat = "common"
        combos.append({
            "x": r["reach"], "y": r["n_both"], "ccu": r["max_ccu"],
            "combo": f"{lbl(r['tag_a'])} × {lbl(r['tag_b'])}",
            "best": r["best_game"], "url": r["best_url"],
            "rating": r["avg_rating"], "cat": cat,
        })

    coords, method = embed_games(games)
    # colour the universe by broad genre (genre_l1); keep the top ones, rest -> Other
    from collections import Counter
    g1 = Counter((g.genre_l1 or "Other") for g in games if g.ccu > 0)
    top_genres = [g for g, _ in g1.most_common(11)]
    gmap = {g: g for g in top_genres}

    universe = []
    for g, (x, y) in zip(games, coords):
        universe.append({
            "x": float(round(x, 2)), "y": float(round(y, 2)),
            "ccu": g.ccu, "name": g.name,
            "genre": gmap.get(g.genre_l1 or "Other", "Other") if (g.genre_l1 or "Other") in gmap else "Other",
            "g2": g.genre_l2 or "", "url": g.url,
        })

    ingredients = [{"tag": r["tag"], "kind": r["kind"], "lift": r["lift"],
                    "winners": r["winners"], "games": r["games"], "demand": r["demand"]}
                   for r in res["ingredients"] if r["games"] >= 8][:30]

    # curated recommendation cards (from the LLM synthesis), if present
    recommendations, rec_date = [], None
    rec_path = os.path.join("data", "recommendations.json")
    if os.path.exists(rec_path):
        with open(rec_path, encoding="utf-8") as f:
            rec = json.load(f)
        recommendations = rec.get("cards", [])
        rec_date = rec.get("generated")

    # enrich curated cards: fix encoding, stamp the live incumbent (occupancy), and
    # attach the REAL niche numbers (leader share, winners, contamination) so the hero
    # quotes data instead of the hand-written lane prose that blended lanes together.
    if recommendations:
        syn, deny = load_synonyms()
        niche_stats = _load_niche_stats()
        for idx, c in enumerate(recommendations):
            for k, v in list(c.items()):
                if isinstance(v, str):
                    c[k] = _demojibake(v)
            c["occupancy"] = _card_occupancy(games, c, syn, deny)
            c["lane_label"] = (c.get("lane", "").split("—")[0].strip()
                               or c.get("lane", ""))
            c["niche"] = _match_niche(c, niche_stats)
            c["slug"] = _slug(c.get("title", ""), idx)

    # top standouts to annotate directly on the Opportunity Map: genuine residents of the
    # zone — POPULAR ingredients (high reach) that are barely paired (low n_both) — ranked
    # by the best hit they've produced, deduped by best game (many collapse to one hit).
    reaches_sorted = sorted(c["x"] for c in combos)
    reach_thresh = reaches_sorted[int(len(reaches_sorted) * 0.6)] if reaches_sorted else 30
    map_annot, seen = [], set()
    for d in sorted(combos, key=lambda d: (d["cat"] != "proven", -d["ccu"])):
        if d["y"] > 4 or d["ccu"] <= 0 or d["x"] < reach_thresh:   # under-built AND popular
            continue
        g = (d["best"] or "").strip().lower()
        if not g or g in seen:
            continue
        seen.add(g)
        map_annot.append({"x": d["x"], "y": d["y"], "combo": d["combo"],
                          "best": d["best"], "ccu": d["ccu"]})
        if len(map_annot) >= 5:
            break

    return {
        "combos": combos,
        "map_annot": map_annot,
        "universe": universe,
        "ingredients": ingredients,
        "proven": res["proven"][:25],
        "untapped": res["untapped"][:25],
        "recommendations": recommendations,
        "rec_date": rec_date,
        "right_now": compute_right_now(games, res),
        "n_games": res["n_games"], "n_tags": res["n_tags"],
        "embed_method": method, "winner_ccu": WINNER_CCU,
        "occupancy_floor": res.get("occupancy_floor", OCCUPANCY_FLOOR),
        "genres": list(gmap.keys()) + ["Other"],
    }


# --------------------------------------------------------------------------- #
# HTML template  (self-contained: vanilla JS canvas, zoom/pan/hover, no CDN)
# --------------------------------------------------------------------------- #
HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Roblox Idea Finder</title>
<style>
  :root{--bg:#0b0d12;--panel:#13171f;--raise:#1b212b;--ink:#e6edf3;--mut:#8b949e;--line:#283039;
        --green:#3fb950;--amber:#e0a040;--red:#e5484d;--blue:#58a6ff;--purple:#bc8cff;
        /* semantic aliases: meaning is fixed */
        --open:var(--green);--conc:var(--amber);--dom:var(--red);--risk:#ff6b6f;--accent:var(--blue);
        --mono:ui-monospace,"JetBrains Mono","Cascadia Code","SF Mono",Consolas,monospace}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
       font:14px/1.55 ui-sans-serif,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;-webkit-font-smoothing:antialiased}
  .num{font-family:var(--mono);font-variant-numeric:tabular-nums}
  header{padding:18px 24px 14px;border-bottom:1px solid var(--line)}
  h1{margin:0;font-size:21px;letter-spacing:-.01em;font-weight:700}
  .sub{color:var(--mut);font-size:13px;margin-top:4px;max-width:80ch}
  .pill{font-family:var(--mono);font-size:11px;padding:1px 8px;border-radius:20px;border:1px solid var(--line);color:var(--mut);font-variant-numeric:tabular-nums}
  .freshline{display:flex;align-items:center;gap:8px;margin-top:8px;font-family:var(--mono);font-size:11.5px;color:var(--mut)}
  .freshline .live{width:7px;height:7px;border-radius:50%;background:var(--green)}
  .tabs{display:flex;gap:6px;padding:12px 24px 0;flex-wrap:wrap}
  .tab{padding:8px 14px;border:1px solid var(--line);border-bottom:none;border-radius:8px 8px 0 0;
       background:var(--panel);color:var(--mut);cursor:pointer;font-weight:600;font:inherit;font-weight:600}
  .tab.on{color:var(--ink);background:var(--raise);border-color:var(--accent)}
  .tab:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
  .wrap{padding:0 24px 56px;max-width:1180px}
  .view{display:none} .view.on{display:block;animation:fade .35s ease both}
  @keyframes fade{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:0 10px 10px 10px;padding:16px}
  .legend{display:flex;gap:14px;flex-wrap:wrap;margin:6px 0 10px;color:var(--mut);font-size:12px}
  .legend b{font-weight:600;color:var(--ink)}
  .dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px;vertical-align:-1px}
  canvas{width:100%;height:62vh;display:block;border-radius:8px;background:#080a0e;cursor:grab;touch-action:none}
  .hint{color:var(--mut);font-size:12px;margin-top:8px}
  #tip{position:fixed;pointer-events:none;z-index:9;background:#080a0e;border:1px solid var(--accent);
       border-radius:8px;padding:8px 10px;font-size:12px;max-width:300px;display:none;box-shadow:0 8px 28px #0009}
  #tip b{color:var(--accent)}
  table{border-collapse:collapse;width:100%;font-size:13px;margin-top:8px}
  th,td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--line)}
  td.r,th.r{text-align:right;font-family:var(--mono);font-variant-numeric:tabular-nums}
  th{color:var(--mut);cursor:pointer;user-select:none;position:sticky;top:0;background:var(--panel)}
  tr:hover td{background:var(--raise)} a{color:var(--blue);text-decoration:none} a:hover{text-decoration:underline}
  .grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px} @media(max-width:900px){.grid2{grid-template-columns:1fr}}
  /* ---- state pills: meaning fixed, colour + word always paired ---- */
  .state{font-family:var(--mono);font-size:11px;letter-spacing:.06em;padding:3px 8px;border-radius:6px;border:1px solid;font-weight:600}
  .s-open{color:var(--open);border-color:var(--open);background:rgba(63,185,80,.1)}
  .s-conc{color:var(--conc);border-color:var(--conc);background:rgba(224,160,64,.1)}
  .s-dom{color:var(--risk);border-color:var(--dom);background:rgba(229,72,77,.14)}
  /* ---- badges (category / confidence / scope) ---- */
  .badge{font-size:11px;padding:2px 8px;border-radius:6px;border:1px solid var(--line);color:var(--mut)}
  .b-proven{color:var(--green);border-color:rgba(63,185,80,.5)}
  .b-emerging{color:var(--amber);border-color:rgba(224,160,64,.5)}
  .b-bold{color:var(--purple);border-color:rgba(188,140,255,.5)}
  /* confidence meter (3 segments) */
  .meter{display:inline-flex;gap:3px;vertical-align:-1px}
  .meter i{width:14px;height:6px;border-radius:2px;background:var(--line)}
  .meter.high i{background:var(--green)} .meter.medium i:nth-child(-n+2){background:var(--amber)}
  .meter.low i:nth-child(1){background:var(--red)}
  /* scope segmented S/M/L */
  .seg{display:inline-flex;border:1px solid var(--line);border-radius:6px;overflow:hidden;font-family:var(--mono);font-size:10px}
  .seg span{padding:2px 7px;color:var(--mut)} .seg span.on{background:var(--blue);color:#08090c;font-weight:700}
  .kv{font-family:var(--mono);font-size:11px;color:var(--mut);font-variant-numeric:tabular-nums}
  .kv b{color:var(--ink)}
  /* ---- HERO card (#1) ---- */
  .hero{position:relative;background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--open);
        border-radius:12px;padding:22px;margin:12px 0 18px;display:grid;grid-template-columns:1.6fr 1fr;gap:26px}
  .hero.dom{border-left-color:var(--dom)} .hero.conc{border-left-color:var(--conc)}
  .hero .rank{font-family:var(--mono);font-weight:700;font-size:30px;line-height:1;color:var(--ink)}
  .hero h2{margin:10px 0 6px;font-size:25px;letter-spacing:-.02em;line-height:1.12}
  .hero .tag2{color:var(--mut);font-size:14px;margin-bottom:14px;max-width:48ch}
  .hero .row{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
  .hero .chips{display:flex;gap:7px;flex-wrap:wrap;margin:14px 0}
  .ev{font-family:var(--mono);font-size:11px;padding:5px 9px;border-radius:6px;background:var(--raise);border:1px solid var(--line);color:var(--ink)}
  .ev .g{color:var(--green)} .ev .a{color:var(--amber)}
  .rail{border-left:1px solid var(--line);padding-left:22px;display:flex;flex-direction:column;gap:14px}
  .rail .lab{font-family:var(--mono);font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:var(--mut);margin-bottom:6px}
  .open-cta{display:inline-block;margin-top:4px;font-size:13px;font-weight:600;color:var(--ink);
            background:var(--raise);border:1px solid var(--line);padding:9px 14px;border-radius:8px;cursor:pointer}
  .open-cta:hover{border-color:var(--accent)}
  .warn{font-size:11.5px;color:var(--amber);font-family:var(--mono)}
  /* ---- secondary grid ---- */
  .rec-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:14px;margin-top:4px}
  .rec{background:var(--panel);border:1px solid var(--line);border-radius:10px;
       padding:15px;cursor:pointer;transition:transform .1s ease,border-color .1s}
  .rec:hover{transform:translateY(-2px);border-color:var(--accent)}
  .rec:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
  .rec .rank{font-family:var(--mono);font-size:12px;color:var(--mut)} .rec h3{margin:4px 0 2px;font-size:16px;line-height:1.2}
  .rec .tag2{color:var(--mut);font-size:12.5px;margin-bottom:10px}
  .meta{display:flex;gap:6px;flex-wrap:wrap;margin-top:10px;align-items:center}
  /* ---- modal / brief ---- */
  .modal-bg{position:fixed;inset:0;background:#000b;display:none;z-index:20;overflow:auto}
  .modal{max-width:780px;margin:5vh auto;background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:24px}
  .modal h2{margin:0;font-size:22px} .modal .x{float:right;cursor:pointer;color:var(--mut);font-size:26px;line-height:.7;background:none;border:none}
  .modal .x:focus-visible{outline:2px solid var(--accent)}
  .verdict{display:flex;gap:20px;flex-wrap:wrap;margin:14px 0 4px;padding:14px;background:var(--raise);border-radius:10px;border:1px solid var(--line)}
  .verdict .col .lab{font-family:var(--mono);font-size:9.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--mut);margin-bottom:6px}
  .modal section{margin-top:14px} .modal h4{margin:0 0 3px;color:var(--accent);font-size:11px;text-transform:uppercase;letter-spacing:.05em}
  .modal p{margin:0;color:var(--ink)}
  .riskbox{background:rgba(229,72,77,.07);border:1px solid rgba(229,72,77,.35);border-radius:10px;padding:14px;margin-top:14px}
  .riskbox h4{color:var(--risk)}
  /* ---- right-now strips ---- */
  .strip{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:10px;margin:6px 0 4px}
  .chip{background:var(--raise);border:1px solid var(--line);border-radius:8px;padding:9px 11px;font-size:13px}
  .chip b{color:var(--ink)} .chip .n{color:var(--mut);font-size:11.5px;margin-top:3px;font-family:var(--mono);font-variant-numeric:tabular-nums}
  h2.sec{font-size:15px;margin:22px 0 2px;border-top:1px solid var(--line);padding-top:16px}
  /* ---- lollipop (winning ingredients) ---- */
  .lol-row{display:flex;align-items:center;gap:10px;margin:6px 0}
  .lol-row .nm{width:120px;text-align:right;font-family:var(--mono);font-size:12px}
  .lol-track{flex:1;position:relative;height:14px}
  .lol-base{position:absolute;top:-3px;bottom:-3px;width:0;border-left:1px dashed var(--mut)}
  .lol-line{position:absolute;top:6px;height:2px}
  .lol-dot{position:absolute;top:3px;width:9px;height:9px;border-radius:50%;transform:translateX(-50%)}
  .lol-val{width:120px;font-family:var(--mono);font-size:11px;color:var(--mut);font-variant-numeric:tabular-nums}
  @media(max-width:760px){
    .hero{grid-template-columns:1fr} .rail{border-left:none;border-top:1px solid var(--line);padding-left:0;padding-top:16px}
    .wrap{padding:0 14px 48px}
  }
  @media (prefers-reduced-motion: reduce){
    *{animation:none!important;transition:none!important}
  }
</style></head>
<body>
<header>
  <h1>Roblox Idea Finder <span class="pill">__NGAMES__ games &middot; __NTAGS__ tags</span></h1>
  <div class="sub">Decide what to build &rarr; validate it in the data &rarr; explore the market. Popular ingredients that are rarely combined,
   the way the Steam data videos do.</div>
  <div class="freshline"><span class="live"></span><span class="num">live data &middot; __BUILT__</span><span id="fresh-curated"></span></div>
</header>
<div class="tabs">
  <div class="tab on" data-v="rec">&#9733; Recommended Games</div>
  <div class="tab" data-v="opp">Opportunity Map</div>
  <div class="tab" data-v="uni">Game Universe</div>
  <div class="tab" data-v="ing">Winning Ingredients</div>
  <div class="tab" data-v="tab">Tables</div>
</div>
<div class="wrap">
  <div class="view on" id="v-rec"><div class="card">
    <div class="hint" id="rec-date"></div>
    <div id="hero"></div>
    <h2 class="sec" id="more-head" style="display:none">More picks</h2>
    <div class="rec-grid" id="rec-grid"></div>
    <h2 class="sec">&#9889; Open lanes right now <span class="pill">auto-refreshed from live data</span></h2>
    <div class="hint" style="margin:2px 0 0">Big-demand genres where no single game dominates &mdash; room for a new 1k+ entrant.</div>
    <div class="strip" id="rn-lanes"></div>
    <h2 class="sec">&#127381; Newest 1k+ winners</h2>
    <div class="hint" style="margin:2px 0 0">Fresh games (&lt;90d) already past 1,000 CCU &mdash; proof the meta still mints hits.</div>
    <div class="strip" id="rn-fresh"></div>
    <h2 class="sec">&#128293; Hot proven combos</h2>
    <div class="hint" style="margin:2px 0 0">Mechanic&times;theme pairs with a real hit but few copies (keyword-bleed removed).</div>
    <div class="strip" id="rn-combos"></div>
  </div></div>

  <div class="view" id="v-opp"><div class="card">
    <div class="legend">
      <span><span class="dot" style="background:var(--amber)"></span><b>Opportunity</b> &mdash; popular ingredients, barely paired</span>
      <span><span class="dot" style="background:var(--green)"></span><b>Proven &amp; underbuilt</b> &mdash; a hit exists, few copies</span>
      <span><span class="dot" style="background:var(--mut)"></span>common / saturated</span>
      <span style="margin-left:auto">bigger dot = best-in-combo CCU</span>
    </div>
    <canvas id="c-opp"></canvas>
    <div class="hint">x = <b>reach</b> (how popular the two ingredients are) &rarr; &nbsp;&middot;&nbsp; y = <b>how under-built the pairing is</b> (rare &uarr;).
      The <b style="color:var(--amber)">opportunity zone</b> is now <b>top-right</b> &mdash; popular ingredients almost nobody pairs &mdash; with the standouts labeled. Scroll to zoom &middot; drag to pan &middot; double-click to reset.</div>
  </div></div>

  <div class="view" id="v-uni"><div class="card">
    <div class="legend" id="uni-legend"></div>
    <canvas id="c-uni"></canvas>
    <div class="hint">Each dot is a game, placed so similar games cluster together (by tags). Colour = broad genre, size = live CCU.
      Explore the dense clusters (saturated) and the lonely islands (novel looks).</div>
  </div></div>

  <div class="view" id="v-ing"><div class="card">
    <div class="hint" style="margin:0 0 10px">Tags that appear more among <b>1k+ CCU games</b> than in the catalog overall (lift &gt; 1 = a "winning ingredient" right now).</div>
    <div id="ing-bars"></div>
  </div></div>

  <div class="view" id="v-tab"><div class="grid2">
    <div class="card"><h3 style="margin:4px 0 0">Proven &amp; underbuilt combos</h3>
      <div class="hint">Sorted by best-CCU &divide; #games. Click a header to re-sort.</div>
      <table id="t-proven"><thead><tr>
        <th data-k="combo">Combo</th><th class="r" data-k="n_both">#</th><th class="r" data-k="max_ccu">Best CCU</th><th data-k="best_game">Best game</th>
      </tr></thead><tbody></tbody></table></div>
    <div class="card"><h3 style="margin:4px 0 0">Untapped combos</h3>
      <div class="hint">Both ingredients popular, never <i>literally</i> combined &mdash; then occupancy-checked against synonym-named clones. OCCUPIED = a real incumbent already ships it.</div>
      <table id="t-untapped"><thead><tr>
        <th data-k="combo">Combo</th><th class="r" data-k="n_a">#A</th><th class="r" data-k="n_b">#B</th><th data-k="status">Occupancy</th>
      </tr></thead><tbody></tbody></table></div>
  </div></div>
</div>
<div id="tip"></div>
<div class="modal-bg" id="modal-bg"><div class="modal" id="modal"></div></div>

<script>
const DATA = __DATA__;
const CAT = {proven:'#3fb950', rare:'#e0a040', common:'#6e7681'};
const GPAL = ['#58a6ff','#3fb950','#e0a040','#bc8cff','#e6679a','#56d4dd','#d29922','#8b949e','#ff7b72','#a5d6ff','#7ee787','#ffa657'];
const tip = document.getElementById('tip');
const MONO='ui-monospace,Consolas,monospace', SANS='ui-sans-serif,Segoe UI,sans-serif';

// ---- reusable interactive canvas scatter ----
function Scatter(canvas, pts, opt){
  opt = opt || {};
  const ctx = canvas.getContext('2d');
  let DPR = Math.max(1, window.devicePixelRatio||1), W=0, H=0;
  let view = null;                       // {sx,sy,ox,oy} world->screen
  const pad = 46;
  const xs = pts.map(p=>opt.xLog?Math.log10(p.x+1):p.x);
  const ys = pts.map(p=>opt.yLog?Math.log10(p.y+1):p.y);
  const xmin=Math.min(...xs), xmax=Math.max(...xs), ymin=Math.min(...ys), ymax=Math.max(...ys);
  function fit(){
    const r=canvas.getBoundingClientRect(); W=r.width; H=r.height;
    canvas.width=W*DPR; canvas.height=H*DPR; ctx.setTransform(DPR,0,0,DPR,0,0);
    const sx=(W-2*pad)/((xmax-xmin)||1), sy=(H-2*pad)/((ymax-ymin)||1);
    view={sx, sy, ox:pad - xmin*sx, oy:H-pad + ymin*sy, base:1, panx:0, pany:0, zoom:1};
  }
  function wx(p){return opt.xLog?Math.log10(p.x+1):p.x}
  function wy(p){return opt.yLog?Math.log10(p.y+1):p.y}
  function WXv(w){return (w*view.sx+view.ox)*view.zoom+view.panx}
  function WYv(w){return (view.oy - w*view.sy)*view.zoom+view.pany}
  function X(p){return WXv(wx(p))}
  function Y(p){return WYv(wy(p))}
  function draw(){
    ctx.clearRect(0,0,W,H);
    // opportunity zone (under the dots): a named, shaded rectangle that points at the answer
    if(opt.zone){const z=opt.zone, x0=WXv(z.x0), x1=WXv(z.x1), y0=WYv(z.y0), y1=WYv(z.y1);
      const lx=Math.min(x0,x1), ly=Math.min(y0,y1), zw=Math.abs(x1-x0), zh=Math.abs(y1-y0);
      ctx.save();
      const g=ctx.createLinearGradient(lx,ly,lx,ly+zh);
      g.addColorStop(0,'rgba(224,160,64,0.15)'); g.addColorStop(1,'rgba(224,160,64,0.02)');
      ctx.fillStyle=g; ctx.fillRect(lx,ly,zw,zh);
      ctx.setLineDash([5,4]); ctx.strokeStyle='rgba(224,160,64,0.5)'; ctx.lineWidth=1; ctx.strokeRect(lx,ly,zw,zh);
      ctx.setLineDash([]);
      if(zh>34&&zw>150){ctx.fillStyle='#e0a040'; ctx.font='700 11px '+MONO; ctx.fillText('OPPORTUNITY ZONE', lx+12, ly+20);
        ctx.fillStyle='rgba(224,160,64,0.7)'; ctx.font='10px '+MONO; ctx.fillText('popular · almost nobody pairs them', lx+12, ly+35);}
      ctx.restore();
    }
    // axes
    ctx.strokeStyle='#283039'; ctx.fillStyle='#8b949e'; ctx.font='11px '+SANS; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(pad,H-pad); ctx.lineTo(W-8,H-pad); ctx.moveTo(pad,8); ctx.lineTo(pad,H-pad); ctx.stroke();
    if(opt.xlabel){ctx.fillText(opt.xlabel, W/2-30, H-14);}
    if(opt.ylabel){ctx.save();ctx.translate(14,H/2+30);ctx.rotate(-Math.PI/2);ctx.fillText(opt.ylabel,0,0);ctx.restore();}
    for(const p of pts){
      const x=X(p), y=Y(p); if(x<pad-20||x>W+20||y<-20||y>H-pad+20) continue;
      ctx.beginPath(); ctx.arc(x,y,p._r,0,7); ctx.fillStyle=p._c; ctx.globalAlpha=p._a||.82; ctx.fill();
    }
    ctx.globalAlpha=1;
    // annotated standouts (over the dots): leader line + label, tracks the transform
    if(opt.annot){for(const a of opt.annot){const x=WXv(a.wx), y=WYv(a.wy);
      if(x<pad||x>W-4||y<8||y>H-pad) continue;
      ctx.beginPath(); ctx.arc(x,y,5,0,7); ctx.fillStyle='rgba(224,160,64,0.95)'; ctx.fill();
      ctx.strokeStyle='#fff'; ctx.lineWidth=1.5; ctx.stroke();
      const tx=x-12, ty=y-15;
      ctx.strokeStyle='rgba(224,160,64,0.5)'; ctx.lineWidth=1; ctx.beginPath(); ctx.moveTo(x,y); ctx.lineTo(tx,ty); ctx.stroke();
      ctx.textAlign='right'; ctx.fillStyle='#f4d58a'; ctx.font='600 11px '+MONO; ctx.fillText(a.label, tx, ty);
      if(a.sub){ctx.fillStyle='rgba(244,213,138,0.6)'; ctx.font='9.5px '+MONO; ctx.fillText(a.sub, tx, ty+12);}
      ctx.textAlign='left';
    }}
  }
  function nearest(mx,my){
    let best=null,bd=1e9;
    for(const p of pts){const dx=X(p)-mx,dy=Y(p)-my,d=dx*dx+dy*dy; if(d<bd){bd=d;best=p;}}
    return bd< (opt.hit||120) ? best : null;
  }
  canvas.addEventListener('mousemove',e=>{
    const r=canvas.getBoundingClientRect(); if(drag){ view.panx+=e.clientX-drag.x; view.pany+=e.clientY-drag.y; drag={x:e.clientX,y:e.clientY}; draw(); return;}
    const p=nearest(e.clientX-r.left, e.clientY-r.top);
    if(p){ tip.style.display='block'; tip.style.left=(e.clientX+14)+'px'; tip.style.top=(e.clientY+14)+'px'; tip.innerHTML=opt.tip(p); }
    else tip.style.display='none';
  });
  canvas.addEventListener('mouseleave',()=>tip.style.display='none');
  canvas.addEventListener('wheel',e=>{
    e.preventDefault(); const r=canvas.getBoundingClientRect(), mx=e.clientX-r.left, my=e.clientY-r.top;
    const f=e.deltaY<0?1.15:1/1.15; view.panx=mx-(mx-view.panx)*f; view.pany=my-(my-view.pany)*f; view.zoom*=f; draw();
  },{passive:false});
  let drag=null;
  canvas.addEventListener('mousedown',e=>{drag={x:e.clientX,y:e.clientY}; canvas.style.cursor='grabbing'; tip.style.display='none';});
  window.addEventListener('mouseup',()=>{drag=null; canvas.style.cursor='grab';});
  canvas.addEventListener('dblclick',()=>{fit(); draw();});
  this.render=()=>{fit(); draw();};
  window.addEventListener('resize',()=>{fit&&draw&&(fit(),draw());});
}

// ---- build the three views ----
function radCCU(c){return 2.5 + Math.sqrt(Math.max(0,c))/26}

// Y is re-encoded as "how UNDER-BUILT the pairing is": rare (few games) sits at the TOP,
// so the opportunity corner moves to top-right where the eye looks for winners.
const RY=n=>-Math.log10((n||0)+1), Lx=v=>Math.log10(v+1);
const oppPts = DATA.combos.map(d=>({x:d.x, y:RY(d.y), _r:Math.max(2.5,Math.min(16, 3+Math.sqrt(d.ccu)/55)),
  _c:CAT[d.cat], _a:d.cat==='common'?.35:.9, d}));
const _reaches = DATA.combos.map(d=>d.x).sort((a,b)=>a-b);
const _ymax = oppPts.length?Math.max(...oppPts.map(p=>p.y)):0;
const _maxReach = _reaches[_reaches.length-1]||100;
// the zone ENCLOSES the annotated standouts (popular + barely-built), extended to the
// top-right corner — so every labeled pick provably sits inside the named box.
const _aR = (DATA.map_annot||[]).map(a=>a.x), _aN = (DATA.map_annot||[]).map(a=>a.y);
const _zReach = _aR.length ? Math.min(..._aR) : (_reaches[Math.floor(_reaches.length*0.6)]||30);
const _zN = _aN.length ? Math.max(..._aN) : 2;
const oppZone = {x0:Lx(Math.max(1,_zReach*0.8)), x1:Lx(_maxReach+1)+0.2, y0:RY(_zN+1), y1:_ymax+0.06};
const oppAnnot = (DATA.map_annot||[]).map(a=>({wx:Lx(a.x), wy:RY(a.y),
  label:a.combo, sub:Number(a.ccu).toLocaleString()+' CCU'}));
const opp = new Scatter(document.getElementById('c-opp'), oppPts,
  {xLog:true, yLog:false, xlabel:'reach (popularity of the two ingredients)  →', ylabel:'how under-built (rare)  ↑', hit:160,
   zone:oppZone, annot:oppAnnot,
   tip:p=>{const d=p.d; return `<b>${d.combo}</b><br>${d.y} game(s) do this &middot; reach ${d.x}`+
     `<br>best: ${esc(d.best)} &middot; <b>${d.ccu.toLocaleString()}</b> CCU`+(d.rating!=null?` &middot; ${d.rating}%`:'')+
     `<br><span class=pill>${d.cat}</span>`;}});

const gi = {}; DATA.genres.forEach((g,i)=>gi[g]=GPAL[i%GPAL.length]);
const uniPts = DATA.universe.map(d=>({x:d.x, y:d.y, _r:radCCU(d.ccu), _c:gi[d.genre]||'#6e7681', _a:.72, d}));
const uni = new Scatter(document.getElementById('c-uni'), uniPts,
  {hit:90, tip:p=>{const d=p.d; return `<b>${esc(d.name)}</b><br>${esc(d.g2||d.genre)} &middot; <b>${d.ccu.toLocaleString()}</b> CCU`;}});
document.getElementById('uni-legend').innerHTML = DATA.genres.map(g=>
  `<span><span class="dot" style="background:${gi[g]}"></span>${esc(g)}</span>`).join('');

// ingredients lollipop — anchored on a 1.0x baseline (above = winning ingredient),
// dots faded by sample size so a flashy lift on thin evidence visibly recedes.
(function(){ const lifts=DATA.ingredients.map(d=>d.lift);
  const LMIN=Math.min(0.9,...lifts), LMAX=Math.max(1.8,...lifts);
  const pos=v=>((v-LMIN)/((LMAX-LMIN)||1))*100, base=pos(1.0);
  const head=`<div class="lol-row"><div class="nm"></div><div class="lol-track">`+
    `<div class="lol-base" style="left:${base}%"></div>`+
    `<div style="position:absolute;left:${base}%;top:-15px;transform:translateX(-50%);font-family:var(--mono);font-size:9px;color:var(--mut)">1.0&times; baseline</div>`+
    `</div><div class="lol-val"></div></div>`;
  document.getElementById('ing-bars').innerHTML = head + DATA.ingredients.map(d=>{
    const win=d.lift>=1, col=win?'var(--green)':'var(--red)', a=Math.max(.4,Math.min(1,.45+d.winners/14));
    const p=pos(d.lift), lo=Math.min(base,p), wd=Math.abs(p-base);
    return `<div class="lol-row">
      <div class="nm">${esc(d.tag)} <span class="pill">${esc(d.kind)}</span></div>
      <div class="lol-track">
        <div class="lol-base" style="left:${base}%"></div>
        <div class="lol-line" style="left:${lo}%;width:${wd}%;background:${col};opacity:${a}"></div>
        <div class="lol-dot" style="left:${p}%;background:${col};opacity:${a}"></div>
      </div>
      <div class="lol-val"><b style="color:${col}">${d.lift}&times;</b> &middot; ${d.winners}/${d.games} win &middot; ${d.demand.toLocaleString()}</div>
    </div>`; }).join(''); })();

// tables
function fillTable(id, rows, cols, fmt){
  const tb=document.querySelector('#'+id+' tbody'); let sortk=cols[2]||cols[1], desc=true;
  function paint(){ rows.sort((a,b)=>{const x=a[sortk],y=b[sortk]; return (x>y?1:x<y?-1:0)*(desc?-1:1);});
    tb.innerHTML=rows.map(r=>'<tr>'+fmt(r)+'</tr>').join(''); }
  document.querySelectorAll('#'+id+' th').forEach(th=>th.onclick=()=>{const k=th.dataset.k; desc=(k===sortk)?!desc:true; sortk=k; paint();});
  paint();
}
const L=t=>t.split('genre:').pop()+(t.startsWith('genre:')?'°':'');
fillTable('t-proven', DATA.proven, ['combo','n_both','max_ccu'],
  r=>`<td>${esc(L(r.tag_a))} × ${esc(L(r.tag_b))}</td><td class="r">${r.n_both}</td><td class="r" style="color:var(--green)">${r.max_ccu.toLocaleString()}</td>`+
     `<td><a href="${r.best_url}" target="_blank">${esc((r.best_game||'').slice(0,30))}</a></td>`);
fillTable('t-untapped', DATA.untapped, ['combo','n_a','n_b'],
  r=>{const o=r.occupancy||{}; const s=(r.status==='OCCUPIED')
      ? `<span class="state s-conc">OCCUPIED</span> <span class="kv">${esc((o.incumbent_game||'').slice(0,18))} ${Number(o.incumbent_ccu||0).toLocaleString()}</span>`
      : (o.incumbent_ccu? `<span class="state s-open">OPEN</span> <span class="kv">weak ${Number(o.incumbent_ccu).toLocaleString()}</span>` : `<span class="state s-open">OPEN</span>`);
    return `<td>${esc(L(r.tag_a))} × ${esc(L(r.tag_b))}</td><td class="r">${r.n_a}</td><td class="r">${r.n_b}</td><td>${s}</td>`;});

// ---- Recommended Games (curated cards) + Right Now (live data) ----
const R=DATA.recommendations||[], rn=DATA.right_now||{};
function FLOOR(){return DATA.occupancy_floor||1000}
function num(n){return Number(n||0).toLocaleString()}
// lane state from REAL niche numbers; fall back to the live occupancy check if unmatched.
function laneState(p){ if(p>=55) return ['s-dom','DOMINATED']; if(p>=38) return ['s-conc','CONCENTRATED']; return ['s-open','OPEN']; }
function stateInfo(c){const nq=c&&c.niche, o=(c&&c.occupancy)||{};
  if(nq&&nq.state){let st=nq.state;
    if(st==='OPEN' && o.above_floor) st='CONCENTRATED';   // a real live incumbent => never green-open
    const m={OPEN:['s-open','OPEN'],CONCENTRATED:['s-conc','CONCENTRATED'],DOMINATED:['s-dom','DOMINATED']}[st]||['s-open','OPEN'];
    return {cls:m[0],label:m[1],leader:nq.leader_pct};}
  if(o.above_floor) return {cls:'s-conc',label:'CONCENTRATED',leader:null};
  return {cls:'s-open',label:'OPEN',leader:null};}
function statePill(c){const s=stateInfo(c);return `<span class="state ${s.cls}">${s.label}</span>`;}
function meterHtml(conf){return `<span class="meter ${(conf||'').toLowerCase()}"><i></i><i></i><i></i></span>`;}
function scopeHtml(sc){const s=(sc||'').toUpperCase();
  return `<span class="seg">${['S','M','L'].map(x=>`<span class="${x===s?'on':''}">${x}</span>`).join('')}</span>`;}
function incumbentLine(c){const o=(c&&c.occupancy)||{};
  if(o.above_floor) return `<span class="ev">incumbent <b class="a">${esc((o.incumbent_game||'').slice(0,22))}</b> ${num(o.incumbent_ccu)}</span>`;
  if(o.incumbent_ccu) return `<span class="ev">weak incumbent ${num(o.incumbent_ccu)}</span>`;
  return `<span class="ev g">no incumbent &ge;${num(FLOOR())}</span>`;}
function evChips(c){const nq=c.niche, out=[];
  if(nq){ if(nq.leader_pct!=null) out.push(`<span class="ev">leader <b class="${nq.leader_pct<38?'g':'a'}">${nq.leader_pct}%</b></span>`);
    if(nq.winners_1k!=null) out.push(`<span class="ev">${nq.winners_1k} winners 1k+${nq.fresh_winners_1k?` &middot; ${nq.fresh_winners_1k} fresh`:''}</span>`); }
  out.push(incumbentLine(c));
  if(nq&&nq.contamination_pct!=null&&nq.contamination_pct>=35)
    out.push(`<span class="ev a" title="share of the lane sample that is keyword-contaminated">&#9888; ${Math.round(nq.contamination_pct)}% noisy sample</span>`);
  return out.join('');}
function sec(h,b){ return b?`<section><h4>${h}</h4><p>${esc(b)}</p></section>`:''; }
function occSection(c){ const o=c.occupancy||{};
  if(o.above_floor) return `<section><h4 style="color:var(--amber)">Occupancy (auto-checked)</h4><p>A live game already ships this mechanic &times; theme: <b>${esc(o.incumbent_game)}</b> &mdash; ${num(o.incumbent_ccu)} CCU${o.incumbent_url?` (<a href="${o.incumbent_url}" target="_blank">open</a>)`:''}. The lane is contested by synonym &mdash; differentiate hard.</p></section>`;
  if(o.incumbent_ccu) return `<section><h4>Occupancy (auto-checked)</h4><p>Only a weak incumbent (${esc(o.incumbent_game)}, ${num(o.incumbent_ccu)} CCU) shares this concept &mdash; lane reads open.</p></section>`;
  return `<section><h4>Occupancy (auto-checked)</h4><p>No live incumbent &ge;${num(FLOOR())} CCU shares this mechanic &times; theme in the corpus sample.</p></section>`; }

function renderHero(c,i){const s=stateInfo(c), cls=s.cls==='s-dom'?'dom':s.cls==='s-conc'?'conc':'';
  return `<div class="hero ${cls}">
    <div>
      <div class="row"><span class="rank">#${i+1}</span><span class="badge b-${c.category}">${esc(c.category)}</span>${statePill(c)}<span class="kv">${esc(c.lane_label||'')}</span></div>
      <h2>${esc(c.title)}</h2><div class="tag2">${esc(c.tagline||'')}</div>
      <div class="chips">${evChips(c)}</div>
      <button class="open-cta" onclick="openRec(${i})">Open build brief &rarr;</button>
    </div>
    <div class="rail">
      <div><div class="lab">Confidence</div>${meterHtml(c.confidence)} <span class="kv" style="margin-left:6px">${esc(c.confidence||'')}</span></div>
      <div><div class="lab">Scope</div>${scopeHtml(c.scope)}</div>
      <div><div class="lab">Biggest risk</div><div style="font-size:12.5px">${esc((c.risk||'—').slice(0,150))}${(c.risk||'').length>150?'…':''}</div></div>
    </div>
  </div>`;}
function renderMini(c,i){return `<div class="rec" tabindex="0" role="button" aria-label="${esc(c.title)}" onclick="openRec(${i})" onkeydown="if(event.key==='Enter'||event.key===' '){event.preventDefault();openRec(${i})}">
  <div class="row" style="gap:8px"><span class="rank">#${i+1}</span><span class="badge b-${c.category}">${esc(c.category)}</span>${statePill(c)}</div>
  <h3>${esc(c.title)}</h3><div class="tag2">${esc(c.tagline||'')}</div>
  <div class="meta">${meterHtml(c.confidence)}<span class="kv">${esc(c.confidence||'')}</span><span class="kv">&middot; scope ${esc(c.scope||'')}</span></div>
  <div class="meta" style="margin-top:8px">${incumbentLine(c)}</div></div>`;}
function openRec(i){ const c=R[i]; if(!c)return; const nq=c.niche;
  const ev=(c.evidence_games||[]).map(g=>`&bull; <b>${esc(g.name)}</b> &mdash; <span class="num">${num(g.ccu)}</span> CCU &mdash; ${esc(g.note||'')}`).join('<br>');
  document.getElementById('modal').innerHTML=`<button class="x" onclick="closeRec()" aria-label="Close">&times;</button>
    <div class="kv">#${i+1} &middot; ${esc(c.lane_label||'')}</div>
    <h2>${esc(c.title)}</h2><div class="tag2" style="color:var(--mut);margin-bottom:2px">${esc(c.tagline||'')}</div>
    <div class="verdict">
      <div class="col"><div class="lab">Status</div>${statePill(c)}</div>
      <div class="col"><div class="lab">Confidence</div>${meterHtml(c.confidence)} <span class="kv">${esc(c.confidence||'')}</span></div>
      <div class="col"><div class="lab">Scope</div>${scopeHtml(c.scope)}</div>
      <div class="col"><div class="lab">Category</div><span class="badge b-${c.category}">${esc(c.category)}</span></div>
      ${nq&&nq.leader_pct!=null?`<div class="col"><div class="lab">Lane leader</div><span class="num">${nq.leader_pct}%</span> <span class="kv">&middot; ${nq.winners_1k} winners 1k+</span></div>`:''}
    </div>
    ${nq&&nq.contamination_pct!=null&&nq.contamination_pct>=35?`<div class="warn" style="margin-top:8px">&#9888; ${Math.round(nq.contamination_pct)}% of the ${esc(nq.niche)} sample is keyword-contaminated &mdash; confirm the lane in the genre breakdown before committing.</div>`:''}
    ${sec('Core loop',c.core_loop)}${sec('First 30 seconds (the hook)',c.first_30_seconds)}${sec('Depth — why it stays fun',c.depth)}
    ${sec('Why now (the data)',c.why_now)}${sec('Incumbent to beat',c.incumbent)}${sec('Differentiation',c.differentiation)}
    ${c.risk?('<div class="riskbox"><h4>&#9650; Biggest risk</h4><p>'+esc(c.risk)+'</p></div>'):''}
    ${sec('Retention',c.retention)}${sec('Monetization',c.monetization)}${sec('Virality / social',c.virality)}
    ${sec('Mechanics & theme',((c.mechanics||[]).join(', '))+(c.theme?(' &middot; '+c.theme):''))}
    ${occSection(c)}
    ${ev?('<section><h4>Evidence (real games)</h4><p>'+ev+'</p></section>'):''}`;
  document.getElementById('modal-bg').style.display='block'; }
// paint: hero (#1) + demoted grid (#2..N) + the live "Right Now" strips
(function(){
  document.getElementById('rec-date').innerHTML =
    (R.length? '<b style="color:var(--ink)">'+R.length+' curated picks</b>' : 'No curated picks yet (run the synthesis)')
    + ' &middot; a curated bet synthesized from live data &mdash; a demand signal, not a guarantee. The lanes &amp; charts below refresh automatically.';
  const fc=document.getElementById('fresh-curated'); if(fc&&DATA.rec_date) fc.innerHTML=' &nbsp;&middot;&nbsp; curated '+esc(DATA.rec_date);
  if(R.length){ document.getElementById('hero').innerHTML=renderHero(R[0],0);
    if(R.length>1){ document.getElementById('more-head').style.display='';
      document.getElementById('rec-grid').innerHTML=R.slice(1).map((c,i)=>renderMini(c,i+1)).join(''); } }
  else document.getElementById('rec-grid').innerHTML='<div class="hint">No curated cards in data/recommendations.json yet. The live signals below are computed from the latest data.</div>';
  document.getElementById('rn-lanes').innerHTML = (rn.open_lanes||[]).map(l=>{const ls=laneState(l.leader_share);
    return `<div class="chip"><b>${esc(l.lane)}</b> <span class="state ${ls[0]}">${ls[1]}</span><div class="n">${num(l.demand)} CCU &middot; ${l.winners} games 1k+ &middot; leader ${l.leader_share}%</div></div>`;}).join('') || '<div class="hint">—</div>';
  document.getElementById('rn-fresh').innerHTML = (rn.fresh_winners||[]).map(g=>
    `<div class="chip"><b><a href="${g.url}" target="_blank">${esc((g.name||'').slice(0,30))}</a></b><div class="n">${num(g.ccu)} CCU &middot; ${g.age_days}d old &middot; ${esc(g.genre||'')}</div></div>`).join('') || '<div class="hint">—</div>';
  document.getElementById('rn-combos').innerHTML = (rn.hot_combos||[]).map(c=>
    `<div class="chip"><b>${esc(L(c.tag_a))} &times; ${esc(L(c.tag_b))}</b><div class="n">${c.n_both} games &middot; best <a href="${c.best_url}" target="_blank">${esc((c.best_game||'').slice(0,20))}</a> ${num(c.max_ccu)}</div></div>`).join('') || '<div class="hint">—</div>';
})();
function closeRec(){ document.getElementById('modal-bg').style.display='none'; }
document.getElementById('modal-bg').addEventListener('click',e=>{if(e.target.id==='modal-bg')closeRec();});
document.addEventListener('keydown',e=>{if(e.key==='Escape')closeRec();});

// tabs (keyboard-accessible tablist)
const views={opp,uni}; let drawn={};
const tabEls=[...document.querySelectorAll('.tab')];
const tablist=document.querySelector('.tabs'); if(tablist) tablist.setAttribute('role','tablist');
function activateTab(t){
  tabEls.forEach(x=>{x.classList.remove('on');x.setAttribute('aria-selected','false');x.tabIndex=-1;});
  t.classList.add('on');t.setAttribute('aria-selected','true');t.tabIndex=0;
  document.querySelectorAll('.view').forEach(x=>x.classList.remove('on'));
  const v=t.dataset.v; document.getElementById('v-'+v).classList.add('on');
  if(views[v] && !drawn[v]){ views[v].render(); drawn[v]=1; }
  if(views[v]) setTimeout(()=>views[v].render(),30);
}
tabEls.forEach((t,i)=>{
  const on=t.classList.contains('on');
  t.setAttribute('role','tab'); t.setAttribute('aria-selected',on?'true':'false'); t.tabIndex=on?0:-1;
  t.onclick=()=>activateTab(t);
  t.onkeydown=e=>{
    if(e.key==='Enter'||e.key===' '){e.preventDefault();activateTab(t);}
    else if(e.key==='ArrowRight'||e.key==='ArrowLeft'){e.preventDefault();
      const n=(i+(e.key==='ArrowRight'?1:tabEls.length-1))%tabEls.length; tabEls[n].focus(); activateTab(tabEls[n]);}
  };
});
function esc(s){return (s==null?'':(''+s)).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
</script>
</body></html>"""


def main():
    ap = argparse.ArgumentParser(description="Build an interactive Roblox idea dashboard from a corpus.")
    ap.add_argument("--corpus", default=os.path.join("data", "corpus.json"))
    ap.add_argument("--out", default=os.path.join("docs", "index.html"))
    args = ap.parse_args()

    if not os.path.exists(args.corpus):
        sys.exit(f"No corpus at {args.corpus}. Build one first:  python roblox_research.py harvest")
    games = load_corpus(args.corpus)
    print(f"Loaded {len(games)} games. Computing combos + embedding...", file=sys.stderr)
    payload = build_payload(games, load_corpus_meta(args.corpus))
    html = (HTML
            .replace("__DATA__", json.dumps(payload, ensure_ascii=False))
            .replace("__NGAMES__", f"{payload['n_games']:,}")
            .replace("__NTAGS__", str(payload["n_tags"]))
            .replace("__BUILT__", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {args.out}  ({len(payload['combos'])} combos, {len(payload['universe'])} games, "
          f"{payload['embed_method']} embedding)")
    print(f"Open it in your browser:  {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
