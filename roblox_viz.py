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
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

from roblox_research import load_corpus, analyze_combos, game_tags, WINNER_CCU


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


def build_payload(games):
    res = analyze_combos(games)
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

    return {
        "combos": combos,
        "universe": universe,
        "ingredients": ingredients,
        "proven": res["proven"][:25],
        "untapped": res["untapped"][:25],
        "recommendations": recommendations,
        "rec_date": rec_date,
        "right_now": compute_right_now(games, res),
        "n_games": res["n_games"], "n_tags": res["n_tags"],
        "embed_method": method, "winner_ccu": WINNER_CCU,
        "genres": list(gmap.keys()) + ["Other"],
    }


# --------------------------------------------------------------------------- #
# HTML template  (self-contained: vanilla JS canvas, zoom/pan/hover, no CDN)
# --------------------------------------------------------------------------- #
HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Roblox Idea Finder</title>
<style>
  :root{--bg:#0e1117;--panel:#161b22;--ink:#e6edf3;--mut:#8b949e;--line:#30363d;
        --green:#3fb950;--orange:#f0883e;--gray:#6e7681;--blue:#58a6ff;--accent:#bc8cff}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial}
  header{padding:18px 24px;border-bottom:1px solid var(--line)}
  h1{margin:0;font-size:20px} .sub{color:var(--mut);font-size:13px;margin-top:4px}
  .tabs{display:flex;gap:6px;padding:12px 24px 0;flex-wrap:wrap}
  .tab{padding:8px 14px;border:1px solid var(--line);border-bottom:none;border-radius:8px 8px 0 0;
       background:var(--panel);color:var(--mut);cursor:pointer;font-weight:600}
  .tab.on{color:var(--ink);background:#1f2630;border-color:var(--accent)}
  .wrap{padding:0 24px 40px}
  .view{display:none} .view.on{display:block}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:0 10px 10px 10px;padding:14px}
  .legend{display:flex;gap:14px;flex-wrap:wrap;margin:6px 0 10px;color:var(--mut);font-size:12px}
  .legend b{font-weight:600;color:var(--ink)}
  .dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px;vertical-align:-1px}
  canvas{width:100%;height:62vh;display:block;border-radius:8px;background:#0b0f14;cursor:grab;touch-action:none}
  .hint{color:var(--mut);font-size:12px;margin-top:8px}
  #tip{position:fixed;pointer-events:none;z-index:9;background:#0b0f14;border:1px solid var(--accent);
       border-radius:8px;padding:8px 10px;font-size:12px;max-width:280px;display:none;box-shadow:0 6px 24px #0008}
  #tip b{color:var(--accent)}
  table{border-collapse:collapse;width:100%;font-size:13px;margin-top:8px}
  th,td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--line)}
  th{color:var(--mut);cursor:pointer;user-select:none;position:sticky;top:0;background:var(--panel)}
  tr:hover td{background:#1c2330} a{color:var(--blue);text-decoration:none} a:hover{text-decoration:underline}
  .grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px} @media(max-width:900px){.grid2{grid-template-columns:1fr}}
  .bar{height:16px;background:linear-gradient(90deg,var(--accent),var(--blue));border-radius:4px}
  .pill{font-size:11px;padding:1px 7px;border-radius:20px;border:1px solid var(--line);color:var(--mut)}
  /* recommendations */
  .rec-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px;margin-top:10px}
  .rec{background:var(--panel);border:1px solid var(--line);border-left:4px solid var(--accent);border-radius:10px;
       padding:14px;cursor:pointer;transition:transform .08s ease,border-color .1s}
  .rec:hover{transform:translateY(-3px);border-color:var(--accent)}
  .rec .rank{font-size:12px;color:var(--mut)} .rec h3{margin:3px 0 2px;font-size:16px}
  .rec .tag2{color:var(--accent);font-size:12px;margin-bottom:8px}
  .rec .lp{color:var(--mut);font-size:13px}
  .meta{display:flex;gap:6px;flex-wrap:wrap;margin-top:10px}
  .badge{font-size:11px;padding:2px 8px;border-radius:20px;border:1px solid var(--line);color:var(--mut)}
  .b-proven{color:var(--green);border-color:var(--green)}
  .b-emerging{color:var(--orange);border-color:var(--orange)}
  .b-bold{color:var(--accent);border-color:var(--accent)}
  .b-high{color:var(--green);border-color:var(--green)} .b-medium{color:var(--orange);border-color:var(--orange)}
  .modal-bg{position:fixed;inset:0;background:#000b;display:none;z-index:20;overflow:auto}
  .modal{max-width:760px;margin:5vh auto;background:var(--panel);border:1px solid var(--accent);border-radius:14px;padding:24px}
  .modal h2{margin:0} .modal .x{float:right;cursor:pointer;color:var(--mut);font-size:24px;line-height:.7}
  .modal section{margin-top:13px} .modal h4{margin:0 0 3px;color:var(--accent);font-size:11px;text-transform:uppercase;letter-spacing:.05em}
  .modal p{margin:0;color:var(--ink)}
  .strip{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:10px;margin:6px 0 4px}
  .chip{background:#1c2330;border:1px solid var(--line);border-radius:8px;padding:8px 11px;font-size:13px}
  .chip b{color:var(--ink)} .chip .n{color:var(--mut);font-size:12px;margin-top:2px}
  h2.sec{font-size:15px;margin:20px 0 2px;border-top:1px solid var(--line);padding-top:16px}
</style></head>
<body>
<header>
  <h1>Roblox Idea Finder <span class="pill">__NGAMES__ games &middot; __NTAGS__ tags &middot; data updated __BUILT__</span></h1>
  <div class="sub">Find game ideas the way the Steam data videos do: popular ingredients that are rarely combined.
   Hover any point for details &middot; scroll to zoom &middot; drag to pan &middot; double-click to reset.</div>
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
      <span><span class="dot" style="background:var(--green)"></span><b>Proven &amp; underbuilt</b> &mdash; a 1.5k+ hit exists, few copies</span>
      <span><span class="dot" style="background:var(--orange)"></span><b>Rare combo</b> &mdash; barely built</span>
      <span><span class="dot" style="background:var(--gray)"></span>common</span>
      <span style="margin-left:auto">bigger dot = bigger best-in-combo CCU</span>
    </div>
    <canvas id="c-opp"></canvas>
    <div class="hint">x = <b>reach</b> (how popular the two ingredients are) &nbsp;&middot;&nbsp; y = <b>how many games already combine them</b>.
      The opportunities live in the <b>bottom-right</b>: popular ingredients, almost nobody pairs them.</div>
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
        <th data-k="combo">Combo</th><th data-k="n_both">#</th><th data-k="max_ccu">Best CCU</th><th data-k="best_game">Best game</th>
      </tr></thead><tbody></tbody></table></div>
    <div class="card"><h3 style="margin:4px 0 0">Untapped combos</h3>
      <div class="hint">Both ingredients popular, never combined in the corpus. Blue-sky &mdash; validate first.</div>
      <table id="t-untapped"><thead><tr>
        <th data-k="combo">Combo</th><th data-k="n_a">#A</th><th data-k="n_b">#B</th>
      </tr></thead><tbody></tbody></table></div>
  </div></div>
</div>
<div id="tip"></div>
<div class="modal-bg" id="modal-bg"><div class="modal" id="modal"></div></div>

<script>
const DATA = __DATA__;
const CAT = {proven:'#3fb950', rare:'#f0883e', common:'#6e7681'};
const GPAL = ['#58a6ff','#3fb950','#f0883e','#bc8cff','#e6679a','#56d4dd','#d29922','#8b949e','#ff7b72','#a5d6ff','#7ee787','#ffa657'];
const tip = document.getElementById('tip');

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
  function X(p){return (wx(p)*view.sx+view.ox)*view.zoom+view.panx}
  function Y(p){return (view.oy - wy(p)*view.sy)*view.zoom+view.pany}
  function draw(){
    ctx.clearRect(0,0,W,H);
    // axes
    ctx.strokeStyle='#22272e'; ctx.fillStyle='#8b949e'; ctx.font='11px Segoe UI'; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(pad,H-pad); ctx.lineTo(W-8,H-pad); ctx.moveTo(pad,8); ctx.lineTo(pad,H-pad); ctx.stroke();
    if(opt.xlabel){ctx.fillText(opt.xlabel, W/2-30, H-14);}
    if(opt.ylabel){ctx.save();ctx.translate(14,H/2+30);ctx.rotate(-Math.PI/2);ctx.fillText(opt.ylabel,0,0);ctx.restore();}
    for(const p of pts){
      const x=X(p), y=Y(p); if(x<pad-20||x>W+20||y<-20||y>H-pad+20) continue;
      ctx.beginPath(); ctx.arc(x,y,p._r,0,7); ctx.fillStyle=p._c; ctx.globalAlpha=p._a||.82; ctx.fill();
    }
    ctx.globalAlpha=1;
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

const oppPts = DATA.combos.map(d=>({x:d.x, y:d.y, _r:Math.max(2.5,Math.min(16, 3+Math.sqrt(d.ccu)/55)),
  _c:CAT[d.cat], _a:d.cat==='common'?.4:.9, d}));
const opp = new Scatter(document.getElementById('c-opp'), oppPts,
  {xLog:true, yLog:true, xlabel:'reach (popularity of the two ingredients)  →', ylabel:'games already combining them  →', hit:160,
   tip:p=>{const d=p.d; return `<b>${d.combo}</b><br>${d.y} game(s) do this &middot; reach ${d.x}`+
     `<br>best: ${esc(d.best)} &middot; <b>${d.ccu.toLocaleString()}</b> CCU`+(d.rating!=null?` &middot; ${d.rating}%`:'')+
     `<br><span class=pill>${d.cat}</span>`;}});

const gi = {}; DATA.genres.forEach((g,i)=>gi[g]=GPAL[i%GPAL.length]);
const uniPts = DATA.universe.map(d=>({x:d.x, y:d.y, _r:radCCU(d.ccu), _c:gi[d.genre]||'#6e7681', _a:.72, d}));
const uni = new Scatter(document.getElementById('c-uni'), uniPts,
  {hit:90, tip:p=>{const d=p.d; return `<b>${esc(d.name)}</b><br>${esc(d.g2||d.genre)} &middot; <b>${d.ccu.toLocaleString()}</b> CCU`;}});
document.getElementById('uni-legend').innerHTML = DATA.genres.map(g=>
  `<span><span class="dot" style="background:${gi[g]}"></span>${esc(g)}</span>`).join('');

// ingredients bars
(function(){ const m=Math.max(...DATA.ingredients.map(d=>d.lift),1.5);
  document.getElementById('ing-bars').innerHTML = DATA.ingredients.map(d=>
   `<div style="display:flex;align-items:center;gap:10px;margin:5px 0">
      <div style="width:120px;text-align:right">${esc(d.tag)} <span class=pill>${d.kind}</span></div>
      <div class="bar" style="width:${(d.lift/m*60)}%"></div>
      <div style="width:150px;color:var(--mut)"><b style="color:var(--ink)">${d.lift}x</b> &middot; ${d.winners}/${d.games} win &middot; ${d.demand.toLocaleString()} CCU</div>
    </div>`).join(''); })();

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
  r=>`<td>${esc(L(r.tag_a))} × ${esc(L(r.tag_b))}</td><td>${r.n_both}</td><td>${r.max_ccu.toLocaleString()}</td>`+
     `<td><a href="${r.best_url}" target="_blank">${esc((r.best_game||'').slice(0,30))}</a></td>`);
fillTable('t-untapped', DATA.untapped, ['combo','n_a','n_b'],
  r=>`<td>${esc(L(r.tag_a))} × ${esc(L(r.tag_b))}</td><td>${r.n_a}</td><td>${r.n_b}</td>`);

// ---- Recommended Games (curated cards) + Right Now (live data) ----
(function(){
  const R=DATA.recommendations||[], rn=DATA.right_now||{};
  document.getElementById('rec-date').innerHTML =
    (R.length? '<b style="color:var(--ink)">'+R.length+' curated picks</b>' : 'No curated picks yet (run the synthesis)')
    + (DATA.rec_date? ' &middot; synthesized '+esc(DATA.rec_date):'')
    + ' &middot; data: '+DATA.n_games.toLocaleString()+' games. Click a card for the full build brief.';
  document.getElementById('rec-grid').innerHTML = R.length
    ? R.map((c,i)=>recCard(c,i)).join('')
    : '<div class="hint">No curated cards in data/recommendations.json yet. The live signals below are computed from the latest data.</div>';
  document.getElementById('rn-lanes').innerHTML = (rn.open_lanes||[]).map(l=>
    `<div class="chip"><b>${esc(l.lane)}</b><div class="n">${l.demand.toLocaleString()} CCU &middot; ${l.winners} games 1k+ &middot; leader ${l.leader_share}%</div></div>`).join('') || '<div class="hint">—</div>';
  document.getElementById('rn-fresh').innerHTML = (rn.fresh_winners||[]).map(g=>
    `<div class="chip"><b><a href="${g.url}" target="_blank">${esc((g.name||'').slice(0,30))}</a></b><div class="n">${g.ccu.toLocaleString()} CCU &middot; ${g.age_days}d old &middot; ${esc(g.genre||'')}</div></div>`).join('') || '<div class="hint">—</div>';
  document.getElementById('rn-combos').innerHTML = (rn.hot_combos||[]).map(c=>
    `<div class="chip"><b>${esc(L(c.tag_a))} &times; ${esc(L(c.tag_b))}</b><div class="n">${c.n_both} games &middot; best <a href="${c.best_url}" target="_blank">${esc((c.best_game||'').slice(0,20))}</a> ${c.max_ccu.toLocaleString()}</div></div>`).join('') || '<div class="hint">—</div>';
})();
function recCard(c,i){ return `<div class="rec" onclick="openRec(${i})">
  <div class="rank">#${i+1} &middot; ${esc(c.lane||'')}</div>
  <h3>${esc(c.title)}</h3><div class="tag2">${esc(c.tagline||'')}</div>
  <div class="lp">${esc((c.core_loop||'').slice(0,108))}${(c.core_loop||'').length>108?'…':''}</div>
  <div class="meta"><span class="badge b-${c.category}">${esc(c.category)}</span>
   <span class="badge b-${c.confidence}">${esc(c.confidence)} confidence</span>
   <span class="badge">scope ${esc(c.scope)}</span></div></div>`; }
function sec(h,b){ return b?`<section><h4>${h}</h4><p>${esc(b)}</p></section>`:''; }
function openRec(i){ const c=(DATA.recommendations||[])[i]; if(!c)return;
  const ev=(c.evidence_games||[]).map(g=>`&bull; <b>${esc(g.name)}</b> &mdash; ${Number(g.ccu).toLocaleString()} CCU &mdash; ${esc(g.note||'')}`).join('<br>');
  document.getElementById('modal').innerHTML=`<span class="x" onclick="closeRec()">&times;</span>
    <h2>#${i+1} ${esc(c.title)}</h2><div class="tag2" style="color:var(--accent);margin-bottom:4px">${esc(c.tagline||'')}</div>
    <div class="meta"><span class="badge b-${c.category}">${esc(c.category)}</span><span class="badge b-${c.confidence}">${esc(c.confidence)} confidence</span><span class="badge">scope ${esc(c.scope)}</span><span class="badge">${esc(c.lane||'')}</span></div>
    ${sec('Core loop',c.core_loop)}${sec('First 30 seconds (the hook)',c.first_30_seconds)}${sec('Depth — why it stays fun',c.depth)}
    ${sec('Why now (the data)',c.why_now)}${sec('Incumbent to beat',c.incumbent)}${sec('Differentiation',c.differentiation)}
    ${sec('Retention',c.retention)}${sec('Monetization',c.monetization)}${sec('Virality / social',c.virality)}
    ${sec('Mechanics & theme',((c.mechanics||[]).join(', '))+(c.theme?(' &middot; '+c.theme):''))}
    ${ev?('<section><h4>Evidence (real games)</h4><p>'+ev+'</p></section>'):''}`;
  document.getElementById('modal-bg').style.display='block'; }
function closeRec(){ document.getElementById('modal-bg').style.display='none'; }
document.getElementById('modal-bg').addEventListener('click',e=>{if(e.target.id==='modal-bg')closeRec();});
document.addEventListener('keydown',e=>{if(e.key==='Escape')closeRec();});

// tabs
const views={opp,uni}; let drawn={};
document.querySelectorAll('.tab').forEach(t=>t.onclick=()=>{
  document.querySelectorAll('.tab').forEach(x=>x.classList.remove('on')); t.classList.add('on');
  document.querySelectorAll('.view').forEach(x=>x.classList.remove('on'));
  const v=t.dataset.v; document.getElementById('v-'+v).classList.add('on');
  if(views[v] && !drawn[v]){ views[v].render(); drawn[v]=1; }
  if(views[v]) setTimeout(()=>views[v].render(),30);
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
    payload = build_payload(games)
    html = (HTML
            .replace("__DATA__", json.dumps(payload, ensure_ascii=False))
            .replace("__NGAMES__", f"{payload['n_games']:,}")
            .replace("__NTAGS__", str(payload["n_tags"]))
            .replace("__METHOD__", payload["embed_method"])
            .replace("__BUILT__", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {args.out}  ({len(payload['combos'])} combos, {len(payload['universe'])} games, "
          f"{payload['embed_method']} embedding)")
    print(f"Open it in your browser:  {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
