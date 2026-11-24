#!/usr/bin/env python3
"""Build one HTML dashboard of every run and evaluation in runs/.

Open the result in a browser: no server, no install, works offline.
Re-run after each `scripts/perun.sh fetch <run-id>` to see new data.

What it shows:
- tick boxes to pick training runs, and to hide the first 10k steps
  (the start-up loss spike flattens the rest of each curve) (colour follows the run, never its order);
- the same curves for every picked run, overlaid, x = env steps:
  episode return, the 8 losses, imagination metrics, speed;
  hover a chart to read every run's value at that step;
- the Task 6 evaluations next to the DreamerV3 paper (v1, Table M.1),
  and the success rate of each achievement;
- the run registry from EXPERIMENTS.md.

Only the standard library. Usage (in Docker or any Python 3):
  python analysis/dashboard.py            # writes dashboard.html
  xdg-open dashboard.html                 # macOS: open, Windows: start
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAX_POINTS = 400   # per curve, keeps the file small and the page fast
WINDOW = 20        # moving average of the episode return, in episodes

# (metrics key, chart title, group). The group is a heading on the page.
CHARTS = [
    ("episode/score", "Episode return (moving average of 20)", "Real environment"),
    ("episode/length", "Episode length (moving average of 20)", "Real environment"),
    ("train/loss/image", "image loss (decoder)", "World model, trained on real data"),
    ("train/loss/rew", "rew loss (reward head)", "World model, trained on real data"),
    ("train/loss/con", "con loss (continuation head)", "World model, trained on real data"),
    ("train/loss/dyn", "dyn loss (prior)", "World model, trained on real data"),
    ("train/loss/rep", "rep loss (posterior)", "World model, trained on real data"),
    ("train/loss/policy", "policy loss (actor)", "Actor and critic, trained in imagination"),
    ("train/loss/value", "value loss (critic)", "Actor and critic, trained in imagination"),
    ("train/loss/repval", "repval loss (critic on real replay)", "Actor and critic, trained in imagination"),
    ("train/ent/action", "actor entropy (nats, uniform = 2.83)", "Actor and critic, trained in imagination"),
    ("train/rew", "imagined reward per step", "Actor and critic, trained in imagination"),
    ("train/val", "critic value (imagined)", "Actor and critic, trained in imagination"),
    ("train/adv_mag", "|advantage| (imagined)", "Actor and critic, trained in imagination"),
    ("fps/policy", "env steps per second", "Machine"),
    ("usage/nvsmi/compute_avg/gpu0", "GPU use (fraction of time busy, 0 to 1)", "Machine"),
]
SMOOTHED = {"episode/score", "episode/length"}

# DreamerV3 paper, arXiv 2301.04104v1, appendix M, Table M.1 (1M env steps,
# 10 seeds). Shown next to our evaluations; see docs/results.md.
PAPER = [
    {"label": "Paper: DreamerV3, 1M steps", "score": 14.5, "ret": 11.7},
    {"label": "Paper: PPO, 1M steps", "score": 4.6, "ret": 4.2},
    {"label": "Paper: random", "score": 1.6, "ret": 2.1},
]


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def moving_average(points, window):
    out, total = [], 0.0
    for i, (x, y) in enumerate(points):
        total += y
        if i >= window:
            total -= points[i - window][1]
        out.append((x, total / min(i + 1, window)))
    return out


def thin(points):
    """Keep at most MAX_POINTS points, always including the last one."""
    step = max(1, len(points) // MAX_POINTS)
    kept = points[::step]
    if points and kept[-1] != points[-1]:
        kept.append(points[-1])
    return [[x, round(y, 5)] for x, y in kept]


def registry(root):
    """Rows of the table in EXPERIMENTS.md, as dicts keyed by the header."""
    path = root / "EXPERIMENTS.md"
    if not path.exists():
        return []
    rows = [line.strip().strip("|").split(" | ")
            for line in path.read_text().splitlines() if line.startswith("| ")]
    header = [h.strip() for h in rows[0]]
    return [dict(zip(header, (c.strip().replace("`", "") for c in r)))
            for r in rows[1:] if not set(r[0]) <= set("-")]


def load(root):
    reg = {r.get("Run ID"): r for r in registry(root)}
    runs, evals = [], []
    for d in sorted(p for p in (root / "runs").iterdir() if p.is_dir()):
        summary = d / "summary.json"
        if summary.exists():
            s = json.loads(summary.read_text())
            agent = s["agent"]
            if agent == "random":
                label, order = "Random agent", 0
            elif "ckpt_early" in agent:
                label, order = f"Early checkpoint ({agent.split('/')[-1]})", 1
            else:
                label, order = f"Trained ({agent.split('/')[-2]})", 2
            evals.append({"id": d.name, "label": label, "order": order,
                          "episodes": s["episodes"], "seed": s["seed"],
                          "score": s["crafter_score"], "ret": s["return_mean"],
                          "ret_std": s["return_std"], "rates": s["success_rates"]})
            continue
        rows = read_jsonl(d / "metrics.jsonl")
        if not rows:
            continue
        series = {}
        for key, _, _ in CHARTS:
            pts = [(r["step"], r[key]) for r in rows if key in r]
            if key in SMOOTHED:
                pts = moving_average(pts, WINDOW)
            if pts:
                series[key] = thin(pts)
        info = reg.get(d.name, {})
        runs.append({"id": d.name, "step": max(r["step"] for r in rows),
                     "status": info.get("Status", "?"),
                     "config": info.get("Configs + flags", ""),
                     "debug": "debug" in info.get("Configs + flags", ""),
                     "series": series})
    evals.sort(key=lambda e: e["order"])
    return runs, evals, list(reg.values())


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DreamerV3 Crafter runs</title>
<style>
:root { color-scheme: light;
  --surface: #fcfcfb; --card: #ffffff; --ink: #0b0b0b; --ink2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --ring: rgba(11,11,11,0.10);
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100;
  --s5: #e87ba4; --s6: #008300; --s7: #6250d6; --s8: #e34948; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark;
  --surface: #1a1a19; --card: #222220; --ink: #f0efec; --ink2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,0.10);
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500;
  --s5: #d55181; --s6: #0ca30c; --s7: #9085e9; --s8: #e66767; } }
:root[data-theme="dark"] { color-scheme: dark;
  --surface: #1a1a19; --card: #222220; --ink: #f0efec; --ink2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,0.10);
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500;
  --s5: #d55181; --s6: #0ca30c; --s7: #9085e9; --s8: #e66767; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--surface); color: var(--ink);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1280px; margin: 0 auto; padding: 20px 16px 60px; }
h1 { font-size: 22px; margin: 0 0 4px; } h2 { font-size: 17px; margin: 32px 0 10px; }
.sub, .note { color: var(--ink2); margin: 0; } .note { font-size: 12px; margin-top: 6px; }
nav { display: flex; gap: 16px; margin: 14px 0 0; flex-wrap: wrap; }
nav a { color: var(--ink2); }
.filters { position: sticky; top: 0; z-index: 5; background: var(--surface);
  padding: 10px 0; border-bottom: 1px solid var(--grid); display: flex; flex-wrap: wrap; gap: 8px 18px; }
.filters label { display: flex; align-items: center; gap: 6px; cursor: pointer; }
.key { width: 16px; height: 0; border-top: 2px solid; display: inline-block; }
.grid { display: grid; gap: 12px; grid-template-columns: repeat(auto-fill, minmax(290px, 1fr)); }
.card { background: var(--card); border-radius: 8px; box-shadow: 0 0 0 1px var(--ring);
  padding: 10px 12px 6px; position: relative; }
.card h3 { font-size: 13px; font-weight: 600; margin: 0 0 4px; }
svg { display: block; width: 100%; height: auto; overflow: visible; }
svg text { fill: var(--muted); font-size: 10px; }
.gl { stroke: var(--grid); } .ax { stroke: var(--axis); }
.tip { position: absolute; pointer-events: none; background: var(--card); color: var(--ink);
  box-shadow: 0 0 0 1px var(--ring), 0 4px 12px rgba(0,0,0,.15); border-radius: 6px;
  padding: 6px 8px; font-size: 12px; white-space: nowrap; display: none; z-index: 3; }
.tip b { font-variant-numeric: tabular-nums; } .tip .row { display: flex; align-items: center; gap: 6px; }
.tip .dim { color: var(--ink2); }
.tiles { display: grid; gap: 12px; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); margin-top: 16px; }
.tile .v { font-size: 26px; font-weight: 650; font-variant-numeric: tabular-nums; }
.tile .l { color: var(--ink2); font-size: 12px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: left; padding: 5px 8px; border-bottom: 1px solid var(--grid); vertical-align: top; }
th { color: var(--ink2); font-weight: 600; } td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; }
.scroll { overflow-x: auto; }
.bar { fill: var(--s1); } .bar.ref { fill: var(--muted); opacity: .55; }
.bar:hover { opacity: .8; }
.heat td.n { min-width: 64px; }
.wide { grid-template-columns: repeat(auto-fill, minmax(min(100%, 520px), 1fr)); }
.skip { margin-left: auto; color: var(--ink2); }
</style></head>
<body><main>
<h1>DreamerV3 on Crafter: runs and results</h1>
<p class="sub">Generated by <code>analysis/dashboard.py</code> from <code>runs/</code> and <code>EXPERIMENTS.md</code>. Re-run it after each fetch.</p>
<nav><a href="#training">Training curves</a><a href="#eval">Evaluation</a><a href="#achievements">Achievements</a><a href="#registry">Run registry</a></nav>
<div class="tiles" id="tiles"></div>
<h2 id="training">Training curves</h2>
<div class="filters" id="filters"></div>
<p class="note">x axis: env steps, shared by all charts. Hover a chart to read each run's value. Debug-config runs (smoke tests) use a tiny model and are not comparable; they start unticked.</p>
<div id="charts"></div>
<h2 id="eval">Evaluation (Task 6) next to the paper</h2>
<p class="sub">Our agents: 50 episodes on the same worlds (seed 100). Paper: DreamerV3 v1, Table M.1, 1M env steps, 10 seeds (grey).</p>
<div class="grid wide" id="evalcharts" style="margin-top:10px"></div>
<div class="scroll" style="margin-top:12px"><table id="evaltable"></table></div>
<h2 id="achievements">Success rate per achievement (%)</h2>
<p class="sub">Share of the 50 evaluation episodes in which the achievement was unlocked at least once. Darker = more often.</p>
<div class="scroll" style="margin-top:10px"><table class="heat" id="heat"></table></div>
<h2 id="registry">Run registry (EXPERIMENTS.md)</h2>
<div class="scroll"><table id="reg"></table></div>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const NS = 'http://www.w3.org/2000/svg';
const el = (tag, attrs = {}, parent) => {
  const e = tag === 'svg' || ['line','path','text','rect','g'].includes(tag)
    ? document.createElementNS(NS, tag) : document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) k === 'text' ? e.textContent = v : e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
};
const fmt = v => Number.isInteger(v) && Math.abs(v) < 1e5 ? v.toLocaleString('en')
  : Math.abs(v) >= 1e5 ? (v / 1e3).toFixed(0) + 'k'
  : Math.abs(v) >= 1000 ? v.toLocaleString('en', {maximumFractionDigits: 0})
  : Math.abs(v) >= 10 ? v.toFixed(1) : Math.abs(v) >= 0.1 ? v.toFixed(2) : v.toPrecision(2);
function ticks(lo, hi, n = 4) {   // round values covering lo..hi
  if (hi <= lo) hi = lo + 1;
  const mag = 10 ** Math.floor(Math.log10((hi - lo) / n));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => (hi - lo) / s <= n);
  const out = []; let t = Math.floor(lo / step) * step;
  while (!out.length || out[out.length - 1] < hi - step * 0.01) { out.push(+t.toFixed(10)); t += step; }
  return out;
}

// Colour follows the run: slot by position in the full run list, never by
// which runs are ticked. 8 slots (validated categorical palette).
D.runs.forEach((r, i) => r.color = `var(--s${i % 8 + 1})`);
const shown = new Set(D.runs.filter(r => !r.debug).map(r => r.id));
const SKIP = 10000;   // the first updates have huge losses that flatten the rest
let skip = true;

// Tiles
const tiles = document.getElementById('tiles');
const tile = (v, l) => { const c = el('div', {class: 'card tile'}, tiles);
  el('div', {class: 'v', text: v}, c); el('div', {class: 'l', text: l}, c); };
D.runs.filter(r => !r.debug).forEach(r => tile(r.step.toLocaleString('en'), `env steps · ${r.id} (${r.status})`));
D.evals.forEach(e => tile(e.score.toFixed(1) + ' %', `Crafter score · ${e.label}`));

// Filters
const filters = document.getElementById('filters');
D.runs.forEach(r => {
  const lab = el('label', {}, filters);
  const box = el('input', {type: 'checkbox'}, lab); box.checked = shown.has(r.id);
  el('span', {class: 'key', style: `border-color:${r.color}`}, lab);
  el('span', {text: r.id, title: r.config}, lab);
  box.addEventListener('change', () => { box.checked ? shown.add(r.id) : shown.delete(r.id); drawCharts(); });
});
const skipLab = el('label', {class: 'skip'}, filters);
const skipBox = el('input', {type: 'checkbox'}, skipLab); skipBox.checked = skip;
el('span', {text: `hide the first ${SKIP / 1000}k steps (start-up spike)`}, skipLab);
skipBox.addEventListener('change', () => { skip = skipBox.checked; drawCharts(); });

// One line chart: every ticked run overlaid, crosshair + tooltip.
const W = 300, H = 170, PL = 44, PR = 10, PT = 8, PB = 22;
function lineChart(card, key, xmax) {
  const x0 = skip ? SKIP : 0;
  const pts = r => r.series[key].filter(p => p[0] >= x0);
  const runs = D.runs.filter(r => shown.has(r.id) && r.series[key] && pts(r).length);
  const svg = el('svg', {viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': key}, card);
  if (!runs.length) { el('text', {x: PL, y: 40, text: 'no data for the ticked runs'}, svg); return; }
  const ys = runs.flatMap(r => pts(r).map(p => p[1]));
  const yt = ticks(Math.min(0, ...ys), Math.max(...ys) * 1.05);
  const y0 = yt[0], y1 = yt[yt.length - 1], iw = W - PL - PR, ih = H - PT - PB;
  const X = v => PL + (v - x0) / (xmax - x0) * iw, Y = v => PT + ih - (v - y0) / (y1 - y0 || 1) * ih;
  yt.forEach(t => { el('line', {x1: PL, x2: PL + iw, y1: Y(t), y2: Y(t), class: 'gl'}, svg);
    el('text', {x: PL - 5, y: Y(t) + 3, 'text-anchor': 'end', text: fmt(t)}, svg); });
  ticks(x0, xmax).filter(t => t >= x0 && t <= xmax).forEach(t =>
    el('text', {x: X(t), y: H - 6, 'text-anchor': 'middle', text: fmt(t)}, svg));
  el('line', {x1: PL, x2: PL + iw, y1: PT + ih, y2: PT + ih, class: 'ax'}, svg);
  runs.forEach(r => el('path', {fill: 'none', stroke: r.color, 'stroke-width': 2, 'stroke-linejoin': 'round',
    d: pts(r).map((p, i) => (i ? 'L' : 'M') + X(p[0]).toFixed(1) + ',' + Y(p[1]).toFixed(1)).join(' ')}, svg));
  const hair = el('line', {y1: PT, y2: PT + ih, class: 'ax', visibility: 'hidden'}, svg);
  const tip = el('div', {class: 'tip'}, card);
  svg.addEventListener('pointermove', ev => {
    const box = svg.getBoundingClientRect();
    const sx = (ev.clientX - box.left) / box.width * W;
    const step = Math.max(x0, Math.min(xmax, x0 + (sx - PL) / iw * (xmax - x0)));
    hair.setAttribute('x1', X(step)); hair.setAttribute('x2', X(step)); hair.setAttribute('visibility', 'visible');
    tip.replaceChildren(el('div', {class: 'dim', text: `env step ${Math.round(step).toLocaleString('en')}`}));
    runs.forEach(r => {   // nearest logged point of each run
      const s = pts(r); let best = s[0];
      for (const p of s) if (Math.abs(p[0] - step) < Math.abs(best[0] - step)) best = p;
      const row = el('div', {class: 'row'}, tip);
      el('span', {class: 'key', style: `border-color:${r.color}`}, row);
      el('b', {text: fmt(best[1])}, row); el('span', {class: 'dim', text: r.id}, row);
    });
    tip.style.display = 'block';
    const left = ev.clientX - card.getBoundingClientRect().left;
    tip.style.left = Math.min(left + 12, card.clientWidth - tip.offsetWidth - 4) + 'px';
    tip.style.top = '30px';
  });
  svg.addEventListener('pointerleave', () => { hair.setAttribute('visibility', 'hidden'); tip.style.display = 'none'; });
}

function drawCharts() {
  const box = document.getElementById('charts'); box.replaceChildren();
  const xmax = Math.max(1, ...D.runs.filter(r => shown.has(r.id)).map(r => r.step));
  let grid = null, group = null;
  for (const [key, title, g] of D.charts) {
    if (g !== group) { group = g; el('h3', {text: g, style: 'margin:18px 0 8px;font-size:14px'}, box);
      grid = el('div', {class: 'grid'}, box); }
    const card = el('div', {class: 'card'}, grid);
    el('h3', {text: title}, card);
    lineChart(card, key, xmax);
  }
}
drawCharts();

// Evaluation: horizontal bars, ours (blue) then paper (grey).
function hbars(title, rows, unit) {
  const card = el('div', {class: 'card'}, document.getElementById('evalcharts'));
  el('h3', {text: title}, card);
  const rowH = 28, LW = 200, h = rows.length * rowH + 24, w = 520;
  const svg = el('svg', {viewBox: `0 0 ${w} ${h}`, role: 'img', 'aria-label': title}, card);
  const xt = ticks(0, Math.max(...rows.map(r => r.v)) * 1.1), xmax = xt[xt.length - 1];
  const X = v => LW + v / xmax * (w - LW - 40);
  xt.forEach(t => { el('line', {x1: X(t), x2: X(t), y1: 0, y2: h - 20, class: 'gl'}, svg);
    el('text', {x: X(t), y: h - 6, 'text-anchor': 'middle', text: fmt(t) + unit}, svg); });
  const tip = el('div', {class: 'tip'}, card);
  rows.forEach((r, i) => {
    const y = i * rowH + 4;
    el('text', {x: LW - 8, y: y + 13, 'text-anchor': 'end', text: r.label, style: 'fill:var(--ink2);font-size:11px'}, svg);
    const bar = el('rect', {x: LW, y, width: Math.max(2, X(r.v) - LW), height: rowH - 8, rx: 3,
      class: r.ref ? 'bar ref' : 'bar'}, svg);
    el('text', {x: X(r.v) + 5, y: y + 13, text: fmt(r.v) + unit + (r.err ? ` ± ${fmt(r.err)}` : ''),
      style: 'fill:var(--ink);font-size:11px'}, svg);
    bar.addEventListener('pointermove', ev => {
      tip.replaceChildren(el('b', {text: fmt(r.v) + unit}), el('span', {class: 'dim', text: '  ' + r.label + (r.note ? ' · ' + r.note : '')}));
      tip.style.display = 'block';
      tip.style.left = Math.min(ev.clientX - card.getBoundingClientRect().left + 12, card.clientWidth - tip.offsetWidth - 4) + 'px';
      tip.style.top = (ev.clientY - card.getBoundingClientRect().top + 12) + 'px';
    });
    bar.addEventListener('pointerleave', () => tip.style.display = 'none');
  });
}
const ours = f => D.evals.map(e => ({label: e.label, v: f(e), note: `${e.id}, ${e.episodes} episodes`,
  err: f === retOf ? e.ret_std : 0}));
const retOf = e => e.ret, scoreOf = e => e.score;
if (D.evals.length) {
  hbars('Crafter score (geometric mean of success rates)', [...ours(scoreOf),
    ...D.paper.map(p => ({label: p.label, v: p.score, ref: true, note: 'arXiv 2301.04104v1, Table M.1'}))], ' %');
  hbars('Episode return (mean ± std)', [...ours(retOf),
    ...D.paper.map(p => ({label: p.label, v: p.ret, ref: true, note: 'arXiv 2301.04104v1, Table M.1'}))], '');
  const t = document.getElementById('evaltable');
  const hr = el('tr', {}, el('thead', {}, t));
  ['Agent', 'Eval run', 'Episodes', 'Return', 'Crafter score'].forEach((h, i) => el('th', {text: h, class: i > 1 ? 'n' : ''}, hr));
  const tb = el('tbody', {}, t);
  D.evals.forEach(e => { const tr = el('tr', {}, tb);
    [e.label, e.id, e.episodes, `${e.ret.toFixed(2)} ± ${e.ret_std.toFixed(2)}`, e.score.toFixed(1) + ' %']
      .forEach((v, i) => el('td', {text: v, class: i > 1 ? 'n' : ''}, tr)); });
  D.paper.forEach(p => { const tr = el('tr', {}, tb);
    [p.label, 'arXiv 2301.04104v1', '10 seeds', p.ret.toFixed(1), p.score.toFixed(1) + ' %']
      .forEach((v, i) => el('td', {text: v, class: i > 1 ? 'n' : ''}, tr)); });

  // Achievements heat table: one hue, darker = higher rate; value printed in each cell.
  const heat = document.getElementById('heat');
  const names = Object.keys(D.evals[0].rates)
    .sort((a, b) => Math.max(...D.evals.map(e => e.rates[b])) - Math.max(...D.evals.map(e => e.rates[a])) || a.localeCompare(b));
  const hh = el('tr', {}, el('thead', {}, heat));
  el('th', {text: 'Achievement'}, hh); D.evals.forEach(e => el('th', {text: e.label, class: 'n'}, hh));
  const hb = el('tbody', {}, heat);
  names.forEach(n => { const tr = el('tr', {}, hb); el('td', {text: n.replaceAll('_', ' ')}, tr);
    D.evals.forEach(e => { const v = Math.round(e.rates[n] * 100);
      el('td', {class: 'n', text: v, title: `${n}: ${v} % of episodes · ${e.label}`,
        style: `background:color-mix(in oklab, var(--s1) ${Math.round(v * 0.85)}%, transparent);` +
               (v > 55 ? 'color:#fff' : '')}, tr); }); });
} else {
  el('p', {class: 'sub', text: 'No evaluation in runs/ yet (runs/<eval-id>/summary.json).'}, document.getElementById('evalcharts'));
}

// Registry table
const reg = document.getElementById('reg');
if (D.registry.length) {
  const cols = ['Run ID', 'Date', 'Configs + flags', 'Hardware / job', 'Wall-clock', 'Status', 'Notes'];
  const hr = el('tr', {}, el('thead', {}, reg)); cols.forEach(c => el('th', {text: c}, hr));
  const tb = el('tbody', {}, reg);
  D.registry.forEach(r => { const tr = el('tr', {}, tb); cols.forEach(c => el('td', {text: r[c] || ''}, tr)); });
}
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=ROOT,
                    help="folder with runs/ and EXPERIMENTS.md (default: this repo)")
    ap.add_argument("--out", type=Path, default=ROOT / "dashboard.html")
    args = ap.parse_args()
    runs, evals, reg = load(args.root)
    data = {"runs": runs, "evals": evals, "paper": PAPER, "registry": reg,
            "charts": CHARTS}
    # "</" inside the JSON would end the <script> tag early.
    text = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    args.out.write_text(PAGE.replace("__DATA__", text))
    print(f"wrote {args.out} ({len(runs)} training runs, {len(evals)} evaluations)")


if __name__ == "__main__":
    main()
