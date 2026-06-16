# Roblox market & competitor research toolkit

**Live dashboard: https://brayanlondono13.github.io/roblox-idea-finder/**
— auto-refreshes daily (ranked game ideas, an opportunity map, a t-SNE game universe, winning ingredients).

A command-line tool for deciding **what Roblox game to build** by looking at real,
live demand — so you don't rebuild something that already exists, and you can spot
niches that are big enough to support a **1,000+ CCU** game but *aren't* owned by a
single monolith.

The dashboard's **Recommended Games** tab is the headline: ranked "build this now" cards
(click for the full brief — core loop, the 30-second hook, why-now evidence, the incumbent
to beat, monetization, virality), plus a live "Right Now" strip (open lanes, newest 1k+
winners, hot combos) that recomputes from fresh data on every refresh.

## Keeping it updated (daily)

Roblox blocks most **datacenter IPs** (incl. GitHub's CI runners), so the reliable refresh
runs on **your machine** (residential IP) and pushes — GitHub Pages serves it within ~1 min.

```powershell
# one-time: schedule a daily refresh at 9am
powershell -ExecutionPolicy Bypass -File install_scheduler.ps1
# or refresh on demand any time:
powershell -ExecutionPolicy Bypass -File update.ps1
```

`update.ps1` re-harvests live data → recomputes combos → rebuilds `docs/index.html` → commits
+ pushes. The data charts and the "Right Now" strip update automatically; the curated
**Recommended Games** cards refresh when you re-run the synthesis (ask Claude, or it stays as
last generated in `data/recommendations.json`). There's also a manual, best-effort GitHub
Action (Actions tab → "Refresh Roblox dashboard" → Run workflow) that only commits if the CI
harvest wasn't IP-blocked.

Built for the workflow: _"I like games like **Slime RNG** and **Craft/Cooking**
games — show me the competitors, the genres, what's hot, and where the open lanes
are."_

---

## Install

```bash
pip install -r requirements.txt        # just `requests`
python roblox_research.py selftest      # hits the live API + checks the math
```

Python 3.9+. No API key, no login.

---

## The four commands

| Command | What it answers |
|---|---|
| `search "<keyword>"` | *Does this already exist? Who's doing it and how big are they?* |
| `trending [--sort ...]` | *What is hot **right now**?* (Roblox's own live feeds) |
| `niche "<keyword>"` | **Gap analysis** — *is this a monolith or an open lane? Can a new game realistically hit 1k+ CCU here?* |
| `harvest` | *Build a broad game "catalog" corpus (for `combos`).* |
| `combos` | **Idea generator** — *which mechanic × theme/genre combinations are popular but rarely built?* |
| `inspect <id/url ...>` | *Deep-dive specific games I already know about.* |

### `search` — find existing games

```bash
python roblox_research.py search "cooking" --pages 3 --csv cooking.csv
python roblox_research.py search "anime rng" --min-ccu 500
```
Keyword-searches all of Roblox, enriches every hit with full stats, and ranks by
live CCU. `--pages` pulls ~40 games per page. `--csv` / `--json` to save.

### `trending` — what's hot now

```bash
python roblox_research.py trending --sort up-and-coming   # newest risers (best for ideas)
python roblox_research.py trending --sort top-trending     # biggest weekly DAU jumps
python roblox_research.py trending --sort all --csv hot.csv
```
Pulls Roblox's own home-page feeds. Sorts: `top-trending`, `up-and-coming`
(published in the last 28 days, by growth — the single best feed for spotting
emerging genres before they saturate), `top-playing-now`, `fun-with-friends`,
`top-revisited`. Add `--fast` to skip detail enrichment (no genre/visits, but quicker).

### `niche` — gap analysis (the important one)

```bash
python roblox_research.py niche "slime rng" --pages 3 --report slime.md --json slime.json
python roblox_research.py niche "restaurant tycoon" --pages 3 --report restaurant.md
```
Scans the niche, then tells you:
- **Demand** — total live CCU across the niche (is the pie big enough?)
- **1k+ winners** — how many games currently hold 1,000+ CCU, and how many of
  those launched in the last 90 days (proof a *newcomer* can still break in)
- **Concentration** — the leader's share and an **HHI** score (monolith vs. fragmented)
- **Sub-genre breakdown** — where the demand actually sits (`genre_l2`)
- An **opportunity score** and a plain-language verdict

`--report file.md` writes a shareable markdown report; `--json file.json` writes the
full enriched dataset + analysis for further crunching.

### `harvest` + `combos` — generate game ideas from gaps

Inspired by data-driven "what game should I make" analyses (e.g. the Steam study that
found *"only 3 platformer auto-battlers exist"*): find **pairs of individually popular
ingredients that are rarely combined**. On Roblox the current meta is mash-ups baked
into titles (*Steal a Brainrot*, *Grow a Garden*, *Fish a Slime*), so the tool mines a
curated **mechanic** (rng, tycoon, steal, merge, cook, fish, obby, …) and **theme**
(slime, anime, food, brainrot, pet, garden, …) vocabulary from game names + descriptions,
adds each game's `genre_l2`, and cross-tabulates every pair.

```bash
python roblox_research.py harvest                     # build data/corpus.json (~once, few min)
python roblox_research.py combos --report ideas.md --csv combos.csv
python roblox_research.py combos --rebuild            # refresh the corpus, then analyze
```

It prints three lists:
- **Proven & underbuilt** — a combo where a real **1.5k+ CCU hit already exists** but only
  a handful of games do it. The safest bets (the combo demonstrably works; the lane is thin).
- **Untapped** — two individually popular tags that **never** co-occur in the corpus.
  Blue-sky: could be genuinely novel, or a combo players don't want — validate first.
- **Winning ingredients** — tags that **over-index among 1k+ CCU games** (lift > 1): the
  mechanics/themes most correlated with hits right now.

The `--csv` output is a scatter (one row per tag-pair): plot **reach** (x) vs **n_both**
(y) and look bottom-right — both popular, rarely combined — exactly like the Steam chart.

> Tag matching is keyword-based, so a pair is only a *lead*: always open the `best_game`
> link to confirm the combo is real before building on it.

### See it visually — the interactive dashboard

`roblox_viz.py` turns the corpus into a **single self-contained HTML file** (no internet
needed to view — just double-click) the way the Steam videos present their data:

```bash
pip install numpy scikit-learn        # scikit-learn is optional (enables the t-SNE map)
python roblox_research.py harvest      # build data/corpus.json first
python roblox_viz.py                   # -> roblox_dashboard.html, then open it
```

Four interactive tabs (hover for details, scroll to zoom, drag to pan):
1. **Opportunity Map** — every mechanic × theme/genre pair as a dot; x = reach, y = times
   combined. Bottom-right = popular but rarely built. (The Steam "genre-combination" chart.)
2. **Game Universe** — every game placed by a **t-SNE** embedding of its tags, so similar
   games cluster; colour = genre, size = live CCU. (The Steam "gaming map.")
3. **Winning Ingredients** — bar chart of tags that over-index among 1k+ CCU games.
4. **Tables** — sortable proven/untapped combo lists.

### `inspect` — known games

```bash
python roblox_research.py inspect 76558904092080 https://www.roblox.com/games/92416421522960/Slime-RNG
python roblox_research.py inspect 9792947201 --universe
```
Accepts place IDs, full game URLs, or (with `--universe`) universe IDs.

---

## How the opportunity score works (no black box)

### De-contamination first (important)

A raw keyword search mixes genres and is polluted by mega-titles that rank for
*everything* (e.g. *+1 Speed Keyboard Escape* shows up as the "leader" of unboxing,
incremental, and steal-a-brainrot searches alike). Computing concentration on that raw
cluster is misleading. So `niche` first **focuses** the result set to the lane you'd
actually enter:

> A game is kept if it's **on-topic** (a query keyword/stem appears in its name, genre,
> or description) **OR** it sits in the niche's **dominant `genre_l2`** (the sub-genre
> holding the most CCU among the on-topic games).

All concentration + scoring then run on this **focused lane**. The tool reports
`contamination_pct` (the share of raw-cluster CCU it stripped as off-lane bleed) and the
raw cluster numbers too, so you can see what was removed. Example: an `unboxing` scan
reports the raw cluster leader at ~87% (the keyboard mega-title) but, after focusing to
the **Tycoon** lane, flags ~93% of that CCU as bleed and shows the *real* lane is
fragmented (top game ~23%).

### The score (out of 100, computed on the focused lane)

| Component | Max | Meaning |
|---|---|---|
| **Demand** | 30 | `30 × clamp((log10(lane CCU) − 3) / 3)` — log-linear from 1k→1M so it keeps discriminating at high demand. |
| **Openness** | 30 | `30 × (1 − max(leader_share, HHI/10000))`. Punishes monoliths; rewards fragmentation. |
| **Reachability** | 20 | `2.5 ×` Σ soft-credit for each ~1k+ winner (partial credit 500→1500 CCU). How many independent winners the lane sustains. |
| **Freshness** | 15 | `5 ×` games that hit 1k+ **and** launched in the last 90 days. Proof the lane still mints winners. |
| **Staleness** | 5 | `2.5 ×` top-5 leaders not updated in 60+ days. Vulnerable incumbents = your opening. |

The soft winner band (500→1500 CCU) avoids a knife-edge at exactly 1,000 on a single
live snapshot. **HHI** (Herfindahl index, 0–10,000) is the antitrust concentration
measure on CCU share: `10000` = one game owns everything (avoid head-on); `<1500` =
fragmented and competitive (room to enter). The verdict bands:

- **MONOLITH** — leader owns ≥60% of live players or HHI ≥5000. Don't fight head-on; differentiate or pick a sub-niche.
- **OPEN & PROVEN** — fragmented **and** a fresh game hit 1k+ recently. Best signal.
- **OPEN BUT COOLING** — fragmented but no *new* 1k+ winner lately. Demand exists, momentum unclear.
- **UNPROVEN** — no game here holds 1k+; demand may be too thin.

---

## What the data can and can't tell you

**CAN** (live snapshot, no login):
- Current CCU (`playing`), visits, favorites, up/down votes, rating%
- Full keyword search and Roblox's own trending feeds
- Genre taxonomy (`genre_l1` / `genre_l2`), created & last-updated dates
- `visits_per_day` — a **lifetime** velocity proxy (`visits ÷ age`)

**CANNOT** (not exposed by public APIs for games you don't own):
- Historical **peak** CCU, retention (D1/D7), session length, revenue
- True *current* growth rate (only lifetime average velocity)

For history, either cron the `--json` output into a database over time, or cross-
reference a tracker like **RoMonitor Stats** or **Rolimon's**. Treat everything here
as a **demand signal**, not proof of a winning game.

These are **unofficial** endpoints (the same ones roblox.com's front end calls).
They work great for research but Roblox can change them without notice — if a command
starts failing, an endpoint moved.

---

## Verified API endpoints (for reference / extending the tool)

| Purpose | Endpoint |
|---|---|
| place → universe | `apis.roblox.com/universes/v1/places/{placeId}/universe` |
| game details (batch) | `games.roblox.com/v1/games?universeIds=a,b,c` |
| votes (batch) | `games.roblox.com/v1/games/votes?universeIds=a,b,c` |
| keyword search (paginated) | `apis.roblox.com/search-api/omni-search?searchQuery=..&sessionId=..&pageType=all` |
| trending feeds | `apis.roblox.com/explore-api/v1/get-sort-content?sessionId=..&sortId=..` |
| icons (batch) | `thumbnails.roblox.com/v1/games/icons?universeIds=..&size=512x512&format=Png` |

`sortId` values: `top-trending`, `up-and-coming`, `top-playing-now`,
`fun-with-friends`, `top-revisited`.
