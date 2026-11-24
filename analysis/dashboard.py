#!/usr/bin/env python3
"""Build one HTML dashboard of every run and evaluation in runs/.

Open the result in a browser: no server, no install, works offline.
Re-run after each `scripts/perun.sh fetch <run-id>` to see new data.

Where the data comes from (nothing is typed in by hand except PAPER):
- runs/<run-id>/metrics.jsonl   training curves (DreamerV3 logger)
- runs/<eval-id>/summary.json   evaluations (scripts/eval_agent.py)
- runs/<run-id>/media/          optional frames or videos (png, gif, mp4)
- EXPERIMENTS.md                run registry, run status, colour order
- ROADMAP.md                    ticked items per week and deliverables
- PAPER below                   DreamerV3 paper v1, Table M.1
- docs/glossary.md              the glossary in "How it works"

What it shows: headline numbers; "what the results say", computed from
the data; training curves of every run, overlaid (tick boxes, hover
readout, option to hide the first 10k steps); evaluations next to the
paper; success rate per achievement; frames; roadmap progress; the
registry. Every chart has a "?" button that explains it.

Only the standard library. Usage (in Docker or any Python 3):
  python analysis/dashboard.py            # writes dashboard.html
  xdg-open dashboard.html                 # macOS: open, Windows: start
  python analysis/dashboard.py --site _site   # GitHub Pages build (CI):
      _site/index.html plus copies of the media files
"""
import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAX_POINTS = 400   # per curve, keeps the file small and the page fast
WINDOW = 20        # moving average over episodes
REPO_URL = "https://github.com/DeepLearningTukeDreamerV3/dreamerv3-crafter"
MEDIA = {".png", ".gif", ".jpg", ".jpeg", ".webp", ".mp4", ".webm"}

WM = "World model, trained on real data"
AC = "Actor and critic, trained in imagination"

# One chart per entry. "lines": (metrics key, label); a second line is
# dashed. "info": the text of the "?" button: what it is, what good
# looks like. Keys starting with "real/" are computed in load().
CHARTS = [
    {"group": "Real environment", "title": "Episode return (moving average of 20)",
     "lines": [("episode/score", "return")],
     "info": "Sum of rewards in one Crafter episode played during training. "
             "Reward: +1 the first time each of the 22 achievements is unlocked in "
             "the episode, -0.1 per health point lost and +0.1 per point regained. "
             "So the maximum is about 22. Rising = the agent unlocks more "
             "achievements. Training episodes sample actions (exploration); the "
             "Evaluation section measures the agents on fixed worlds."},
    {"group": "Real environment", "title": "Episode length (moving average of 20)",
     "lines": [("episode/length", "length")],
     "info": "Env steps until the episode ends. Crafter ends an episode when the "
             "player dies, or after 10,000 steps. About 180 steps means the agent "
             "dies early (hunger, thirst, zombies, lava). A flat curve means "
             "survival did not improve."},
    {"group": "Real environment", "title": "Reward per step: real vs imagined",
     "lines": [("real/rew_per_step", "real (return / length)"), ("train/rew", "imagined")],
     "info": "Solid: real reward per env step (return / length, moving average of "
             "20 episodes). Dashed: mean reward the world model predicts in the "
             "imagined rollouts the actor trains on. Dashed above solid = the model "
             "is optimistic, so the actor learns from rewards that do not happen. "
             "Caveat: imagined starts come from replay and follow the current "
             "policy, so the two are close but not identical quantities."},
    {"group": WM, "title": "image loss (decoder)", "lines": [("train/loss/image", "image")],
     "info": "How badly the decoder rebuilds the 64x64 frame from the model state "
             "(squared error summed over pixels). Falls fast: the model learns to "
             "see. A low value means good reconstruction of the current frame, not "
             "good prediction of future frames (that is Task 4). Each point is the "
             "mean over about 1.9k env steps."},
    {"group": WM, "title": "rew loss (reward head)", "lines": [("train/loss/rew", "rew")],
     "info": "Error of the reward head, which predicts each step's reward from the "
             "model state. Rewards are rare, so the value is small; it rises a "
             "little when the agent starts to unlock more achievements."},
    {"group": WM, "title": "con loss (continuation head)", "lines": [("train/loss/con", "con")],
     "info": "Error of the continuation head: does the episode go on (1) or end "
             "(0)? Small and flat, because an episode ends only once in about 180 "
             "steps."},
    {"group": WM, "title": "dyn loss (prior)", "lines": [("train/loss/dyn", "dyn")],
     "info": "KL divergence between the prior (the model's guess of the next state "
             "before seeing the frame) and the posterior (the state after seeing "
             "it). Trains the prior. Lower = the model predicts the next step "
             "better. Clipped at 1 nat (free bits): at exactly 1.0 it is not "
             "trained at all, as in the smoke run. Weight 1.0."},
    {"group": WM, "title": "rep loss (posterior)", "lines": [("train/loss/rep", "rep")],
     "info": "The same KL as dyn, so the curve is identical; the gradient goes to "
             "the encoder instead, to keep the posterior predictable. Weight 0.1."},
    {"group": WM, "title": "Model uncertainty (entropy, nats)",
     "lines": [("train/rep_ent", "posterior"), ("train/dyn_ent", "prior")],
     "info": "Entropy of the 32 categorical latent variables (64 classes each), "
             "summed: maximum 32 x ln 64 = 133 nats. Falling = the model is more "
             "certain about the state. Prior (dashed) above posterior (solid) is "
             "normal: the prior has not seen the frame yet. The gap is the dyn "
             "loss."},
    {"group": WM, "title": "Gradient norm (whole update)", "lines": [("train/opt/grad_norm", "grad norm")],
     "info": "Size of the gradient of one update. Large at the start, then stable. "
             "Spikes later would mean unstable training; DreamerV3 clips each "
             "gradient (adaptive gradient clipping, agc 0.3)."},
    {"group": AC, "title": "policy loss (actor)", "lines": [("train/loss/policy", "policy")],
     "info": "Actor loss on imagined rollouts: minus (advantage x log-probability "
             "of the action) minus an entropy bonus. Not a score: a value near 0 is "
             "normal. Read the entropy and |advantage| charts instead."},
    {"group": AC, "title": "value loss (critic)", "lines": [("train/loss/value", "value")],
     "info": "Critic error on imagined rollouts: distance between its value and the "
             "lambda-return target. The critic predicts a distribution over 255 "
             "bins, so this is a cross-entropy; it does not reach 0."},
    {"group": AC, "title": "repval loss (critic on real replay)", "lines": [("train/loss/repval", "repval")],
     "info": "The same critic, trained on real replayed sequences (weight 0.3), so "
             "it also learns from real returns, not only imagined ones."},
    {"group": AC, "title": "actor entropy (nats, uniform = 2.83)", "lines": [("train/ent/action", "entropy")],
     "info": "Randomness of the actor. 2.83 nats = uniform over the 17 actions "
             "(random agent); 0 = always the same action. Falling = the agent "
             "becomes decisive. Staying at 2.83 = the actor learned nothing (smoke "
             "run). Very low early can mean too little exploration."},
    {"group": AC, "title": "critic value (imagined)", "lines": [("train/val", "value")],
     "info": "The critic's estimate of the discounted future reward from imagined "
             "states (discount 0.997, horizon about 333 steps)."},
    {"group": AC, "title": "|advantage| (imagined)", "lines": [("train/adv_mag", "|advantage|")],
     "info": "Mean size of (lambda-return - value), after return normalisation: "
             "the learning signal of the actor. Large = the critic is often "
             "surprised; small and flat = little left to learn from imagination."},
    {"group": "Machine", "title": "env steps per second", "lines": [("fps/policy", "steps/s")],
     "info": "Env steps per second, training included. Baseline: about 16/s, "
             "because train_ratio 512 trains on 512 replayed steps per env step. "
             "Smoke: about 180/s (tiny model, train_ratio 8, 4 envs)."},
    {"group": "Machine", "title": "GPU use (fraction of time busy, 0 to 1)",
     "lines": [("usage/nvsmi/compute_avg/gpu0", "GPU")],
     "info": "Share of time the GPU was computing (nvidia-smi), averaged between "
             "two log lines. Close to 1 = the run uses the GPU well. Evidence for "
             "deliverable 8."},
]
SMOOTHED = {"episode/score", "episode/length"}
for _c in CHARTS:   # anchor id, e.g. "c-train-loss-image"
    _c["id"] = "c-" + re.sub(r"[^a-z0-9]+", "-", _c["lines"][0][0].lower())

# DreamerV3 paper, arXiv 2301.04104v1, appendix M, Table M.1 (1M env steps,
# 10 seeds). The same numbers are in the table of docs/results.md.
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


def mean(values):
    return sum(values) / len(values) if values else None


def md_cells(line):
    """Cells of one markdown table row. A | inside `code` is not a separator."""
    cells, cur, code = [], "", False
    for ch in line.strip()[1:-1]:
        if ch == "`":
            code = not code
        elif ch == "|" and not code:
            cells.append(cur.strip())
            cur = ""
            continue
        cur += ch
    cells.append(cur.strip())
    return cells


def registry(root):
    """Rows of the run table in EXPERIMENTS.md (the one whose header starts
    with "Run ID"), as dicts keyed by the header, in file order."""
    path = root / "EXPERIMENTS.md"
    if not path.exists():
        return []
    header, rows = None, []
    for line in path.read_text().splitlines():
        if not line.startswith("|"):
            if header:
                break          # the table ended
            continue
        cells = md_cells(line)
        if header is None:
            if cells[0] == "Run ID":
                header = cells
        elif not set(cells[0]) <= set("-: "):
            if len(cells) != len(header):
                print(f"warning: EXPERIMENTS.md row has {len(cells)} cells, "
                      f"header has {len(header)}: {cells[0]}")
            rows.append(dict(zip(header, (c.replace("`", "") for c in cells))))
    return rows


def roadmap(root):
    """Checkbox items of ROADMAP.md, grouped by "## " section."""
    path = root / "ROADMAP.md"
    if not path.exists():
        return []
    sections = []
    for line in path.read_text().splitlines():
        if line.startswith("## "):
            sections.append({"title": line[3:].strip(), "items": []})
        m = re.match(r"- \[([ xX])\] (.*)", line)
        if m and sections:
            text = re.sub(r"[`*]", "", m.group(2)).strip()
            sections[-1]["items"].append({"done": m.group(1) != " ", "text": text})
    return [s for s in sections if s["items"]]


def glossary(root):
    """Terms of docs/glossary.md: [{"section", "term", "text"}].
    Format: "- **Term** (extra) — definition", continued on indented lines."""
    path = root / "docs" / "glossary.md"
    if not path.exists():
        return []
    terms, section = [], ""
    for line in path.read_text().splitlines():
        if line.startswith("## "):
            section = line[3:].strip()
        elif line.startswith("- **"):
            m = re.match(r"- \*\*(.+?)\*\*\s*(.*)", line)
            terms.append({"section": section, "term": m.group(1), "text": m.group(2)})
        elif line.startswith("  ") and terms and line.strip():
            terms[-1]["text"] += " " + line.strip()
    for t in terms:
        # "**Return** / **episode score** — sum ...": the part before the
        # dash belongs to the term, the part after it is the definition.
        m = re.match(r"(.*?)\s*—\s*(.*)", t["text"])
        if m:
            extra = m.group(1).replace("**", "").strip()
            t["term"] = (t["term"] + " " + extra).strip()
            t["text"] = m.group(2)
        t["term"] = t["term"].replace("`", "")
    return terms


def build_stamp(root):
    """When and from which commit the page was built."""
    sha = os.environ.get("GITHUB_SHA")
    if not sha:
        try:
            sha = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                 capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            sha = ""
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return {"time": now, "sha": sha[:7], "url": f"{REPO_URL}/commit/{sha}" if sha else ""}


def load(root, out_dir):
    reg_rows = registry(root)
    reg = {r.get("Run ID"): r for r in reg_rows}
    order = {r.get("Run ID"): i for i, r in enumerate(reg_rows)}
    runs, evals, media = [], [], []
    for d in sorted(p for p in (root / "runs").iterdir() if p.is_dir()):
        for f in sorted((d / "media").glob("*")) if (d / "media").is_dir() else []:
            if f.suffix.lower() in MEDIA:
                media.append({"path": str(f), "run": d.name, "name": f.name, "video": f.suffix.lower() in {".mp4", ".webm"},
                              "src": os.path.relpath(f, out_dir).replace(os.sep, "/")})
        summary = d / "summary.json"
        if summary.exists():
            s = json.loads(summary.read_text())
            agent = s["agent"]
            if agent == "random":
                label, rank = "Random agent", 0
            elif "ckpt_early" in agent:
                label, rank = f"Early checkpoint ({agent.split('/')[-1]})", 1
            else:
                label, rank = f"Trained ({agent.split('/')[-2]})", 2
            evals.append({"id": d.name, "label": label, "order": rank,
                          "episodes": s["episodes"], "seed": s["seed"],
                          "score": s["crafter_score"], "ret": s["return_mean"],
                          "ret_std": s["return_std"], "rates": s["success_rates"]})
            continue
        rows = read_jsonl(d / "metrics.jsonl")
        if not rows:
            continue
        eps = [r for r in rows if "episode/score" in r and "episode/length" in r]
        series = {}
        for chart in CHARTS:
            for key, _ in chart["lines"]:
                if key == "real/rew_per_step":
                    # Per window: total reward / total steps of the last 20
                    # episodes (a ratio of sums, not a mean of ratios).
                    pts = []
                    for i in range(len(eps)):
                        w = eps[max(0, i - WINDOW + 1): i + 1]
                        pts.append((eps[i]["step"], sum(e["episode/score"] for e in w)
                                    / sum(e["episode/length"] for e in w)))
                else:
                    pts = [(r["step"], r[key]) for r in rows if r.get(key) is not None]
                    if key in SMOOTHED:
                        pts = moving_average(pts, WINDOW)
                if pts:
                    series[key] = thin(pts)
        train = [r for r in rows if "train/loss/image" in r]
        last = lambda key, n=10: mean([r[key] for r in train[-n:] if key in r])
        first = lambda key, n=3: mean([r[key] for r in train[:n] if key in r])
        stats = {
            "episodes": len(eps),
            "ret_first": mean([e["episode/score"] for e in eps[:WINDOW]]),
            "ret_last": mean([e["episode/score"] for e in eps[-WINDOW:]]),
            # Trend: mean return of the last 200 episodes vs the 200 before.
            "ret_recent": mean([e["episode/score"] for e in eps[-200:]]),
            "ret_before": mean([e["episode/score"] for e in eps[-400:-200]]),
            "len_last": mean([e["episode/length"] for e in eps[-WINDOW:]]),
            "real_rew": (sum(e["episode/score"] for e in eps[-WINDOW:])
                         / sum(e["episode/length"] for e in eps[-WINDOW:])) if eps else None,
            "imag_rew": last("train/rew"),
            "image_first": first("train/loss/image"), "image_last": last("train/loss/image"),
            "dyn_last": last("train/loss/dyn"),
            "ent_first": first("train/ent/action"), "ent_last": last("train/ent/action"),
            "fps": mean([r["fps/policy"] for r in rows if "fps/policy" in r]),
        }
        info = reg.get(d.name, {})
        runs.append({"id": d.name, "step": max(r["step"] for r in rows),
                     "status": info.get("Status", "not in EXPERIMENTS.md"),
                     "config": info.get("Configs + flags", ""),
                     "notes": info.get("Notes", ""),
                     "debug": "debug" in info.get("Configs + flags", ""),
                     "order": order.get(d.name, len(order)),
                     "series": series, "stats": stats})
    # Colour follows the registry row, so fetching an older run later does
    # not repaint the others.
    runs.sort(key=lambda r: (r["order"], r["id"]))
    evals.sort(key=lambda e: e["order"])
    return runs, evals, reg_rows, media


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
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 17px; margin: 36px 0 10px; display: flex; align-items: center; gap: 8px; }
.sub, .note { color: var(--ink2); margin: 0; } .note { font-size: 12px; margin-top: 6px; }
nav { display: flex; gap: 6px 16px; margin: 14px 0 0; flex-wrap: wrap; }
nav a { color: var(--ink2); }
.filters { position: sticky; top: 0; z-index: 5; background: var(--surface);
  padding: 10px 0; border-bottom: 1px solid var(--grid); display: flex; flex-wrap: wrap; align-items: center; gap: 8px 18px; }
.filters label { display: flex; align-items: center; gap: 6px; cursor: pointer; }
.filters .cfg { color: var(--muted); font-size: 12px; }
.key { width: 16px; height: 0; border-top: 2px solid; display: inline-block; flex: none; }
.key.dash { border-top-style: dashed; }
.grid { display: grid; gap: 12px; grid-template-columns: repeat(auto-fill, minmax(min(100%, 290px), 1fr)); }
.wide { grid-template-columns: repeat(auto-fill, minmax(min(100%, 520px), 1fr)); }
.card { background: var(--card); border-radius: 8px; box-shadow: 0 0 0 1px var(--ring);
  padding: 10px 12px 6px; position: relative; min-width: 0; }
.card h3 { font-size: 13px; font-weight: 600; margin: 0 0 4px; display: flex; align-items: flex-start; gap: 6px; }
.card h3 span { flex: 1; }
.legend { display: flex; gap: 12px; font-size: 11px; color: var(--ink2); margin: 0 0 2px; }
.legend span { display: flex; align-items: center; gap: 4px; }
svg { display: block; width: 100%; height: auto; overflow: visible; }
svg text { fill: var(--muted); font-size: 10px; }
.gl { stroke: var(--grid); } .ax { stroke: var(--axis); }
.tip { position: absolute; pointer-events: none; background: var(--card); color: var(--ink);
  box-shadow: 0 0 0 1px var(--ring), 0 4px 12px rgba(0,0,0,.15); border-radius: 6px;
  padding: 6px 8px; font-size: 12px; white-space: nowrap; display: none; z-index: 3; }
.tip b { font-variant-numeric: tabular-nums; } .tip .row { display: flex; align-items: center; gap: 6px; }
.tip .dim, .dim { color: var(--ink2); }
.info { flex: none; width: 18px; height: 18px; border-radius: 50%; border: 1px solid var(--axis);
  background: var(--card); color: var(--ink2); font: 600 11px/1 system-ui, sans-serif; cursor: help; padding: 0; }
.info:hover, .info:focus-visible { border-color: var(--ink2); color: var(--ink); outline: none; }
.pop { position: absolute; z-index: 20; width: min(340px, calc(100vw - 32px)); background: var(--card);
  color: var(--ink); box-shadow: 0 0 0 1px var(--ring), 0 6px 20px rgba(0,0,0,.18); border-radius: 8px;
  padding: 10px 12px; font: 400 12.5px/1.5 system-ui, sans-serif; display: none; white-space: normal; }
.tiles { display: grid; gap: 12px; grid-template-columns: repeat(auto-fill, minmax(min(100%, 200px), 1fr)); margin-top: 16px; }
.tile .v { font-size: 26px; font-weight: 650; font-variant-numeric: tabular-nums; }
.tile .l { color: var(--ink2); font-size: 12px; }
.find { display: grid; gap: 12px; grid-template-columns: repeat(auto-fill, minmax(min(100%, 380px), 1fr)); }
.find .card { padding: 12px 14px; } .find h3 { font-size: 14px; margin-bottom: 6px; }
.find p { margin: 0; color: var(--ink2); } .find p b { color: var(--ink); font-variant-numeric: tabular-nums; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: left; padding: 5px 8px; border-bottom: 1px solid var(--grid); vertical-align: top; }
th { color: var(--ink2); font-weight: 600; } td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; }
.scroll { overflow-x: auto; }
.bar { fill: var(--s1); } .bar.ref { fill: var(--muted); opacity: .55; }
.bar:hover { opacity: .8; }
.heat td.n { min-width: 64px; }
.skip { margin-left: auto; color: var(--ink2); }
.meter { height: 8px; border-radius: 4px; background: var(--grid); overflow: hidden; min-width: 80px; }
.meter i { display: block; height: 100%; background: var(--s1); border-radius: 4px; }
.items { margin: 4px 0 0; padding-left: 18px; color: var(--ink2); font-size: 12.5px; }
.items li.done { color: var(--muted); text-decoration: line-through; }
details summary { cursor: pointer; }
.intro { max-width: 820px; margin: 10px 0 0; }
.h3 { font-size: 15px; margin: 22px 0 8px; }
.flow { list-style: none; padding: 0; margin: 0; display: grid; gap: 10px;
  grid-template-columns: repeat(auto-fill, minmax(min(100%, 300px), 1fr)); counter-reset: step; }
.flow li { background: var(--card); border-radius: 8px; box-shadow: 0 0 0 1px var(--ring); padding: 10px 12px 10px 44px;
  position: relative; counter-increment: step; color: var(--ink2); font-size: 13px; }
.flow li::before { content: counter(step); position: absolute; left: 12px; top: 10px; width: 22px; height: 22px;
  border-radius: 50%; background: var(--s1); color: #fff; font: 600 12px/22px system-ui, sans-serif; text-align: center; }
.flow li b.t { display: block; color: var(--ink); margin-bottom: 2px; }
.flow .links { display: block; margin-top: 4px; font-size: 12px; }
.flow .links a { margin-right: 10px; color: var(--ink2); }
.search { width: min(100%, 420px); padding: 7px 10px; border-radius: 6px; border: 1px solid var(--axis);
  background: var(--card); color: var(--ink); font: inherit; margin: 8px 0; }
#gloss h4 { font-size: 13px; margin: 14px 0 4px; color: var(--ink2); }
#gloss dl { margin: 0; display: grid; grid-template-columns: minmax(120px, 220px) 1fr; gap: 4px 16px; }
#gloss dt { font-weight: 600; } #gloss dd { margin: 0; color: var(--ink2); }
@media (max-width: 560px) { #gloss dl { grid-template-columns: 1fr; } #gloss dd { margin-bottom: 6px; } }
code { font-size: 12px; background: var(--grid); padding: 0 3px; border-radius: 3px; }
.docs { color: var(--ink2); padding-left: 18px; } .docs a { color: var(--ink); }
.media { display: grid; gap: 12px; grid-template-columns: repeat(auto-fill, minmax(min(100%, 300px), 1fr)); }
.media img, .media video { width: 100%; image-rendering: pixelated; border-radius: 4px; }
</style></head>
<body><main>
<h1>DreamerV3 on Crafter: runs and results</h1>
<p class="sub">Generated by <code>analysis/dashboard.py</code> from <code>runs/</code>, <code>EXPERIMENTS.md</code> and <code>ROADMAP.md</code>. Re-run it after each fetch. Click or hover <b>?</b> for what a chart means.</p>
<p class="sub" id="stamp"></p>
<nav><a href="#how">How it works</a><a href="#findings">What the results say</a><a href="#training">Training curves</a><a href="#eval">Evaluation</a><a href="#achievements">Achievements</a><a href="#frames">Frames</a><a href="#objectives">Objectives</a><a href="#registry">Run registry</a></nav>
<p class="intro">A university group project (TUKE, MSc): we train <b>DreamerV3</b>, an agent that learns a model of its world and practises inside that model, on <b>Crafter</b>, a 2D survival game with 22 achievements. Then we check what the world model learned. New here? Start with <a href="#how">How it works</a>.</p>
<div class="tiles" id="tiles"></div>
<h2 id="findings">What the results say</h2>
<p class="sub">Computed from the data on this page; numbers update on each rebuild. Single training seed: treat differences of a few tenths as noise.</p>
<div class="find" id="find" style="margin-top:10px"></div>
<h2 id="training">Training curves</h2>
<div class="filters" id="filters"></div>
<p class="note">x axis: env steps, shared by all charts. Hover a chart to read each run's value. Loss and imagination points are means over about 1.9k env steps (one log line).</p>
<div id="charts"></div>
<h2 id="eval">Evaluation (Task 6) next to the paper</h2>
<p class="sub">Our agents: 50 episodes on the same worlds (seed 100), one training seed; ± is the spread over episodes. Paper: DreamerV3 v1, Table M.1, 1M env steps, 10 seeds (grey). Achievements are only measured here: the training logs do not record them.</p>
<div class="grid wide" id="evalcharts" style="margin-top:10px"></div>
<div class="scroll" style="margin-top:12px"><table id="evaltable"></table></div>
<h2 id="achievements">Success rate per achievement (%)</h2>
<p class="sub">Share of the 50 evaluation episodes in which the achievement was unlocked at least once. Darker = more often.</p>
<div class="scroll" style="margin-top:10px"><table class="heat" id="heat"></table></div>
<h2 id="frames">Frames: real vs predicted (Task 4)</h2>
<div id="media"></div>
<h2 id="objectives">Objectives (ROADMAP.md)</h2>
<p class="sub">Ticked checkboxes in <code>ROADMAP.md</code>. A tick is only as current as the file: tick items in the PR that finishes them.</p>
<div class="scroll" style="margin-top:10px"><table id="road"></table></div>
<h2 id="how">How it works: concepts and process</h2>
<p class="sub">The ideas behind every number on this page, in the order they happen. Links jump to the matching chart.</p>
<h3 class="h3">1. How DreamerV3 learns (one training loop)</h3>
<ol class="flow" id="loop"></ol>
<h3 class="h3">2. How we run the project</h3>
<ol class="flow" id="process"></ol>
<h3 class="h3">3. Glossary</h3>
<p class="sub">From <code>docs/glossary.md</code>. Type to filter.</p>
<input id="gq" class="search" type="search" placeholder="Filter terms, e.g. KL, posterior, SLURM" aria-label="Filter glossary">
<div id="gloss"></div>
<h3 class="h3">4. Read more</h3>
<ul class="docs" id="docs"></ul>
<h2 id="registry">Run registry (EXPERIMENTS.md)</h2>
<div class="scroll"><table id="reg"></table></div>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const NS = 'http://www.w3.org/2000/svg';
const el = (tag, attrs = {}, parent) => {
  const e = ['svg','line','path','text','rect','g'].includes(tag)
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

// "?" button: shows an explanation on hover, keyboard focus or click.
let openPop = null;
function info(parent, text) {
  const b = el('button', {type: 'button', class: 'info', 'aria-label': 'What does this show?', text: '?'}, parent);
  const pop = el('div', {class: 'pop', role: 'tooltip'}, document.body);
  pop.textContent = text;
  const show = () => {
    if (openPop && openPop !== pop) openPop.style.display = 'none';
    pop.style.display = 'block'; openPop = pop;
    const r = b.getBoundingClientRect();
    const left = Math.max(8, Math.min(r.left + window.scrollX - 8, window.scrollX + document.documentElement.clientWidth - pop.offsetWidth - 8));
    pop.style.left = left + 'px'; pop.style.top = (r.bottom + window.scrollY + 6) + 'px';
  };
  const hide = () => { pop.style.display = 'none'; if (openPop === pop) openPop = null; };
  b.addEventListener('pointerenter', show); b.addEventListener('pointerleave', hide);
  b.addEventListener('focus', show); b.addEventListener('blur', hide);
  b.addEventListener('click', ev => { ev.stopPropagation(); pop.style.display === 'block' ? hide() : show(); });
  return b;
}
document.addEventListener('click', () => { if (openPop) { openPop.style.display = 'none'; openPop = null; } });
function cardTitle(card, title, text) {
  const h = el('h3', {}, card); el('span', {text: title}, h);
  if (text) info(h, text);
}
const sectionInfo = (id, text) => info(document.getElementById(id), text);

// Colour follows the run (registry order), never which runs are ticked.
// Real runs take the first slots, debug runs the next ones.
[...D.runs.filter(r => !r.debug), ...D.runs.filter(r => r.debug)].forEach((r, i) => r.color = `var(--s${i % 8 + 1})`);
const shown = new Set(D.runs.filter(r => !r.debug).map(r => r.id));
const SKIP = 10000;   // the first updates have huge losses that flatten the rest
let skip = true;
const evalOf = rank => D.evals.find(e => e.order === rank);
// The findings describe the run behind the 'Trained' evaluation, else the first real run.
const main = D.runs.find(r => evalOf(2) && evalOf(2).label.includes(r.id)) || D.runs.find(r => !r.debug);

// Tiles
const tiles = document.getElementById('tiles');
const tile = (v, l) => { const c = el('div', {class: 'card tile'}, tiles);
  el('div', {class: 'v', text: v}, c); el('div', {class: 'l', text: l}, c); };
D.runs.filter(r => !r.debug).forEach(r => tile(r.step.toLocaleString('en'), `env steps · ${r.id} (${r.status})`));
D.evals.forEach(e => tile(e.score.toFixed(1) + ' %', `Crafter score · ${e.label}`));

// What the results say: sentences built from the numbers.
const find = document.getElementById('find');
function finding(title, parts) {
  const c = el('div', {class: 'card'}, find); el('h3', {text: title}, c);
  const p = el('p', {}, c);
  parts.forEach(x => Array.isArray(x) ? el('b', {text: x[0]}, p) : p.appendChild(document.createTextNode(x)));
}
const pct = v => v.toFixed(1) + ' %';
(function findings() {
  const tr = evalOf(2), early = evalOf(1), rnd = evalOf(0), dv3 = D.paper[0], ppo = D.paper[1];
  if (tr && rnd) finding('1. The agent learned', [
    'Crafter score ', [pct(tr.score)], ' vs ', [pct(rnd.score)], ' for the random agent (',
    [(tr.score / rnd.score).toFixed(1) + '×'], ')' + (early ? ', and ' : '. '),
    ...(early ? [[pct(early.score)], ' at the early checkpoint: the score grows with training. '] : []),
    'Return ', [tr.ret.toFixed(2)], ' vs ', [rnd.ret.toFixed(2)], '. The return ranges over episodes overlap (± ',
    tr.ret_std.toFixed(1), '), so the average is the claim, not each episode.']);
  if (tr && main) finding('2. Far from the paper: shorter budget, one seed', [
    'Paper DreamerV3 at 1M steps: ', [pct(dv3.score)], '. We stopped at ', [main.step.toLocaleString('en')],
    ' steps (', [Math.round(main.step / 1e4) + ' %'], ' of 1M), one seed. Our score is close to the paper\'s PPO at 1M (',
    pct(ppo.score), '). Mean training return of the last 200 episodes: ', [fmt(main.stats.ret_recent)], ' vs ',
    [fmt(main.stats.ret_before)], ' for the 200 before',
    main.stats.ret_recent > main.stats.ret_before + 0.2 ? ': still rising when we stopped, so our score is a lower bound. '
      : ': nearly flat at the end, so more steps alone may give only slow gains. ',
    'Other differences: evaluation protocol, seeds, code version (docs/results.md).']);
  if (tr) {
    const names = Object.keys(tr.rates), good = names.filter(n => tr.rates[n] >= 0.5),
      never = names.filter(n => D.evals.every(e => e.rates[n] === 0));
    finding('3. What it can and cannot do', [
      'Reliable (≥ 50 % of episodes): ', [good.map(n => n.replaceAll('_', ' ')).join(', ')], '. ',
      'Sometimes: wood pickaxe ', [Math.round((tr.rates.make_wood_pickaxe || 0) * 100) + ' %'], ', stone ',
      [Math.round((tr.rates.collect_stone || 0) * 100) + ' %'], '. ', [String(never.length)],
      ' of 22 achievements are never unlocked by any agent: the tool chain (stone pickaxe → coal, iron → diamond) is not reached yet.']);
  }
  if (main && main.stats.imag_rew != null) {
    const s = main.stats, ratio = s.imag_rew / s.real_rew;
    finding('4. The world model sees well but is optimistic about reward', [
      'Image loss fell from ', [fmt(s.image_first)], ' to ', [fmt(s.image_last)],
      ': it reconstructs frames well. The dyn KL settled at ', [fmt(s.dyn_last)], ' nats, above the 1-nat floor, so next-step prediction still has error. ',
      'Imagined reward per step ', [s.imag_rew.toFixed(3)], ' vs real ', [s.real_rew.toFixed(3)], ' (',
      [ratio.toFixed(1) + '×'], '): the actor partly trains on rewards that do not happen. To test in Task 4 (open-loop) and Task 8.']);
    finding('5. The policy became decisive; episodes did not get longer', [
      'Actor entropy ', [fmt(s.ent_first)], ' → ', [fmt(s.ent_last)], ' nats (2.83 = random). ',
      'Episodes still last about ', [Math.round(s.len_last) + ' steps'],
      ': the agent unlocks more per life but does not survive longer (eat cow ',
      [tr ? Math.round((tr.rates.eat_cow || 0) * 100) + ' %' : '?'], '). Low entropy can also limit the exploration needed for rare achievements.']);
  }
  const smoke = D.runs.find(r => r.debug);
  if (smoke && main) finding('6. Smoke run vs baseline', [
    'The smoke run (', smoke.id, ') uses the debug config: a model thousands of times smaller (deter 8 vs 8192), batch 8×10 instead of 16×64, train ratio 8 instead of 512. ',
    'It only proves the pipeline runs on the GPU. Its actor entropy stays at ', [fmt(smoke.stats.ent_last)],
    ' (random), its dyn loss sits on the 1-nat floor (', fmt(smoke.stats.dyn_last), '), return ', [fmt(smoke.stats.ret_first)],
    ' → ', [fmt(smoke.stats.ret_last)], ' (random level). It runs ', [Math.round(smoke.stats.fps / main.stats.fps) + '× faster'], ' because it trains far less. Not a learning result.']);
})();

// Filters
const filters = document.getElementById('filters');
D.runs.forEach(r => {
  const lab = el('label', {}, filters);
  const box = el('input', {type: 'checkbox'}, lab); box.checked = shown.has(r.id);
  el('span', {class: 'key', style: `border-color:${r.color}`}, lab);
  el('span', {text: r.id}, lab);
  el('span', {class: 'cfg', text: r.config}, lab);
  if (r.notes) info(filters, `${r.id}: ${r.notes}`);
  box.addEventListener('change', () => { box.checked ? shown.add(r.id) : shown.delete(r.id); drawCharts(); });
});
info(filters, 'Tick runs to overlay them. Runs with the debug config (smoke tests) start unticked: ' +
  'their model is tiny (deter 8, 1 layer of 8 units) and they train 64 times less per env step, ' +
  'so their curves only show that the pipeline runs, not how DreamerV3 learns. See finding 6.');
const skipLab = el('label', {class: 'skip'}, filters);
const skipBox = el('input', {type: 'checkbox'}, skipLab); skipBox.checked = skip;
el('span', {text: `hide the first ${SKIP / 1000}k steps (start-up spike)`}, skipLab);
skipBox.addEventListener('change', () => { skip = skipBox.checked; drawCharts(); });

// One line chart: every ticked run x every line key; crosshair + tooltip.
const W = 300, H = 170, PL = 44, PR = 10, PT = 8, PB = 22;
function lineChart(card, chart, xmax) {
  const x0 = skip ? SKIP : 0;
  const pts = (r, k) => (r.series[k] || []).filter(p => p[0] >= x0);
  const lines = [];
  D.runs.filter(r => shown.has(r.id)).forEach(r => chart.lines.forEach(([k, label], j) => {
    if (pts(r, k).length) lines.push({r, k, label, dash: j > 0, p: pts(r, k)}); }));
  if (chart.lines.length > 1) {
    const lg = el('div', {class: 'legend'}, card);
    chart.lines.forEach(([, label], j) => { const s = el('span', {}, lg);
      el('i', {class: 'key' + (j ? ' dash' : ''), style: 'border-color:var(--ink2)'}, s);
      s.appendChild(document.createTextNode(label)); });
  }
  const svg = el('svg', {viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': chart.title}, card);
  if (!lines.length) { el('text', {x: PL, y: 40, text: 'no data for the ticked runs'}, svg); return; }
  const ys = lines.flatMap(l => l.p.map(p => p[1]));
  const yt = ticks(Math.min(0, ...ys), Math.max(...ys) * 1.05);
  const y0 = yt[0], y1 = yt[yt.length - 1], iw = W - PL - PR, ih = H - PT - PB;
  const X = v => PL + (v - x0) / (xmax - x0) * iw, Y = v => PT + ih - (v - y0) / (y1 - y0 || 1) * ih;
  yt.forEach(t => { el('line', {x1: PL, x2: PL + iw, y1: Y(t), y2: Y(t), class: 'gl'}, svg);
    el('text', {x: PL - 5, y: Y(t) + 3, 'text-anchor': 'end', text: fmt(t)}, svg); });
  ticks(x0, xmax).filter(t => t >= x0 && t <= xmax).forEach(t =>
    el('text', {x: X(t), y: H - 6, 'text-anchor': 'middle', text: fmt(t)}, svg));
  el('line', {x1: PL, x2: PL + iw, y1: PT + ih, y2: PT + ih, class: 'ax'}, svg);
  lines.forEach(l => el('path', {fill: 'none', stroke: l.r.color, 'stroke-width': 2, 'stroke-linejoin': 'round',
    'stroke-dasharray': l.dash ? '5 3' : 'none',
    d: l.p.map((p, i) => (i ? 'L' : 'M') + X(p[0]).toFixed(1) + ',' + Y(p[1]).toFixed(1)).join(' ')}, svg));
  const hair = el('line', {y1: PT, y2: PT + ih, class: 'ax', visibility: 'hidden'}, svg);
  const tip = el('div', {class: 'tip'}, card);
  svg.addEventListener('pointermove', ev => {
    const box = svg.getBoundingClientRect();
    const sx = (ev.clientX - box.left) / box.width * W;
    const step = Math.max(x0, Math.min(xmax, x0 + (sx - PL) / iw * (xmax - x0)));
    hair.setAttribute('x1', X(step)); hair.setAttribute('x2', X(step)); hair.setAttribute('visibility', 'visible');
    tip.replaceChildren(el('div', {class: 'dim', text: `env step ${Math.round(step).toLocaleString('en')}`}));
    lines.forEach(l => {   // nearest logged point of each line
      let best = l.p[0];
      for (const p of l.p) if (Math.abs(p[0] - step) < Math.abs(best[0] - step)) best = p;
      const row = el('div', {class: 'row'}, tip);
      el('span', {class: 'key' + (l.dash ? ' dash' : ''), style: `border-color:${l.r.color}`}, row);
      el('b', {text: fmt(best[1])}, row);
      el('span', {class: 'dim', text: l.r.id + (chart.lines.length > 1 ? ' · ' + l.label : '')}, row);
    });
    tip.style.display = 'block';
    const left = ev.clientX - card.getBoundingClientRect().left;
    tip.style.left = Math.max(0, Math.min(left + 12, card.clientWidth - tip.offsetWidth - 4)) + 'px';
    tip.style.top = '30px';
  });
  svg.addEventListener('pointerleave', () => { hair.setAttribute('visibility', 'hidden'); tip.style.display = 'none'; });
}

function drawCharts() {
  const box = document.getElementById('charts'); box.replaceChildren();
  const xmax = Math.max(SKIP * 2, ...D.runs.filter(r => shown.has(r.id)).map(r => r.step));
  let grid = null, group = null;
  for (const chart of D.charts) {
    if (chart.group !== group) { group = chart.group; el('h3', {text: group, style: 'margin:18px 0 8px;font-size:14px'}, box);
      grid = el('div', {class: 'grid'}, box); }
    const card = el('div', {class: 'card', id: chart.id}, grid);
    cardTitle(card, chart.title, chart.info);
    lineChart(card, chart, xmax);
  }
}
drawCharts();

// Evaluation: horizontal bars, ours (blue) then paper (grey).
function hbars(title, text, rows, unit) {
  const card = el('div', {class: 'card'}, document.getElementById('evalcharts'));
  cardTitle(card, title, text);
  const rowH = 28, LW = 200, h = rows.length * rowH + 24, w = 520;
  const svg = el('svg', {viewBox: `0 0 ${w} ${h}`, role: 'img', 'aria-label': title}, card);
  const xt = ticks(0, Math.max(...rows.map(r => r.v)) * 1.1), xmax = xt[xt.length - 1];
  const X = v => LW + v / xmax * (w - LW - 70);
  xt.forEach(t => { el('line', {x1: X(t), x2: X(t), y1: 0, y2: h - 20, class: 'gl'}, svg);
    el('text', {x: X(t), y: h - 6, 'text-anchor': 'middle', text: fmt(t) + unit}, svg); });
  const tip = el('div', {class: 'tip'}, card);
  rows.forEach((r, i) => {
    const y = i * rowH + 4;
    el('text', {x: LW - 8, y: y + 13, 'text-anchor': 'end', text: r.label, style: 'fill:var(--ink2);font-size:11px'}, svg);
    const bar = el('rect', {x: LW, y, width: Math.max(2, X(r.v) - LW), height: rowH - 8, rx: 3,
      class: r.ref ? 'bar ref' : 'bar'}, svg);
    el('text', {x: X(r.v) + 5, y: y + 13, text: r.text, style: 'fill:var(--ink);font-size:11px'}, svg);
    bar.addEventListener('pointermove', ev => {
      tip.replaceChildren(el('b', {text: r.text}), el('span', {class: 'dim', text: '  ' + r.label + ' · ' + r.note}));
      tip.style.display = 'block';
      tip.style.left = Math.max(0, Math.min(ev.clientX - card.getBoundingClientRect().left + 12, card.clientWidth - tip.offsetWidth - 4)) + 'px';
      tip.style.top = (ev.clientY - card.getBoundingClientRect().top + 12) + 'px';
    });
    bar.addEventListener('pointerleave', () => tip.style.display = 'none');
  });
}
const REF = 'arXiv 2301.04104v1, Table M.1, 10 seeds';
if (D.evals.length) {
  hbars('Crafter score', 'Crafter score = exp(mean of ln(1 + 100 × s_i)) − 1, where s_i is the success rate of ' +
    'achievement i over the episodes (22 achievements). A geometric mean: unlocking a rare achievement now and then ' +
    'counts more than doing an easy one more often. Same metric as the paper. 100 % would mean every achievement in every episode.',
    [...D.evals.map(e => ({label: e.label, v: e.score, text: pct(e.score), note: `${e.id}, ${e.episodes} episodes`})),
     ...D.paper.map(p => ({label: p.label, v: p.score, text: pct(p.score), ref: true, note: REF}))], ' %');
  hbars('Episode return (mean ± std)', 'Mean sum of rewards per episode. Our ± is the standard deviation over the 50 ' +
    'evaluation episodes of one trained agent; the paper reports results over 10 training seeds. So our ± is not ' +
    'comparable to a seed-to-seed spread.',
    [...D.evals.map(e => ({label: e.label, v: e.ret, text: `${e.ret.toFixed(2)} ± ${e.ret_std.toFixed(2)}`, note: `${e.id}, ${e.episodes} episodes`})),
     ...D.paper.map(p => ({label: p.label, v: p.ret, text: p.ret.toFixed(1), ref: true, note: REF}))], '');
  const t = document.getElementById('evaltable');
  const hr = el('tr', {}, el('thead', {}, t));
  ['Agent', 'Source', 'Episodes / seeds', 'Return', 'Crafter score'].forEach((h, i) => el('th', {text: h, class: i > 1 ? 'n' : ''}, hr));
  const tb = el('tbody', {}, t);
  D.evals.forEach(e => { const tr = el('tr', {}, tb);
    [e.label, e.id, `${e.episodes} episodes, 1 seed`, `${e.ret.toFixed(2)} ± ${e.ret_std.toFixed(2)}`, pct(e.score)]
      .forEach((v, i) => el('td', {text: v, class: i > 1 ? 'n' : ''}, tr)); });
  D.paper.forEach(p => { const tr = el('tr', {}, tb);
    [p.label, 'arXiv 2301.04104v1', '10 seeds', p.ret.toFixed(1), pct(p.score)]
      .forEach((v, i) => el('td', {text: v, class: i > 1 ? 'n' : ''}, tr)); });

  // Achievements heat table: one hue, darker = higher rate; value printed in each cell.
  sectionInfo('achievements', 'Each cell: % of the 50 evaluation episodes in which that achievement was unlocked. ' +
    'Crafter achievements form a tech tree: wood → table → wood pickaxe → stone → stone pickaxe → coal/iron → ' +
    'furnace → iron tools → diamond. How far down the tree an agent gets is the clearest sign of what it learned.');
  const heat = document.getElementById('heat');
  const best = n => Math.max(...D.evals.map(e => e.rates[n]));
  const names = Object.keys(D.evals[0].rates).sort((a, b) => best(b) - best(a) || a.localeCompare(b));
  const hh = el('tr', {}, el('thead', {}, heat));
  el('th', {text: 'Achievement'}, hh); D.evals.forEach(e => el('th', {text: e.label, class: 'n'}, hh));
  const hb = el('tbody', {}, heat);
  names.filter(n => best(n) > 0).forEach(n => { const tr = el('tr', {}, hb); el('td', {text: n.replaceAll('_', ' ')}, tr);
    D.evals.forEach(e => { const v = Math.round(e.rates[n] * 100);
      el('td', {class: 'n', text: v, title: `${n}: ${v} % of episodes · ${e.label}`,
        style: `background:color-mix(in oklab, var(--s1) ${Math.round(v * 0.85)}%, transparent);` +
               (v > 55 ? 'color:#fff' : '')}, tr); }); });
  const zero = names.filter(n => best(n) === 0);
  if (zero.length) { const tr = el('tr', {}, hb);
    el('td', {text: `0 for every agent (${zero.length}): ` + zero.map(n => n.replaceAll('_', ' ')).join(', '),
      colspan: D.evals.length + 1, class: 'dim'}, tr); }
} else {
  el('p', {class: 'sub', text: 'No evaluation in runs/ yet (runs/<eval-id>/summary.json).'}, document.getElementById('evalcharts'));
}
sectionInfo('eval', 'Task 6: the same 50 Crafter worlds (seed 100) for every agent, so differences come from the agent, ' +
  'not from easier worlds. Random agent = lower bound. Early checkpoint = a copy of the same run taken earlier in training: shows ' +
  'that the score grows with training. Grey bars = the published DreamerV3 numbers.');

// Frames (Task 4)
const mbox = document.getElementById('media');
sectionInfo('frames', 'The core evidence for a world model: real frames next to the frames the model predicts ' +
  '("open loop": after a few real frames, the model continues alone, from its own predictions). Errors grow with ' +
  'the horizon; where they grow is Task 8. Put png, gif or mp4 files in runs/<run-id>/media/ and rebuild.');
if (D.media.length) {
  const g = el('div', {class: 'media'}, mbox);
  D.media.forEach(m => { const c = el('div', {class: 'card'}, g); el('h3', {text: `${m.run} · ${m.name}`}, c);
    m.video ? el('video', {src: m.src, controls: '', loop: '', muted: ''}, c) : el('img', {src: m.src, alt: m.name}, c); });
} else {
  el('p', {class: 'sub', text: 'Not available yet. DreamerV3 renders these "openloop" videos every 300 s into the scope ' +
    'folder of each logdir on PERUN; we have not copied them yet. Task 4 adds them in runs/<run-id>/media/.'}, mbox);
}

// Objectives from ROADMAP.md
sectionInfo('objectives', 'Read from the checkboxes of ROADMAP.md: weeks 1–8 and the 12 deliverables of the ' +
  'assignment. Open a row to see the items. If work is done but not ticked, tick it in ROADMAP.md.');
const road = document.getElementById('road');
if (D.roadmap.length) {
  const hr = el('tr', {}, el('thead', {}, road));
  ['Section', 'Done', '', 'Open items'].forEach((h, i) => el('th', {text: h, class: i === 1 ? 'n' : ''}, hr));
  const tb = el('tbody', {}, road);
  D.roadmap.forEach(s => {
    const done = s.items.filter(i => i.done).length, tr = el('tr', {}, tb);
    el('td', {text: s.title}, tr);
    el('td', {class: 'n', text: `${done} / ${s.items.length}`}, tr);
    const m = el('div', {class: 'meter', title: `${Math.round(done / s.items.length * 100)} %`}, el('td', {}, tr));
    el('i', {style: `width:${done / s.items.length * 100}%`}, m);
    const det = el('details', {}, el('td', {}, tr));
    el('summary', {class: 'dim', text: `${s.items.length - done} open`}, det);
    const ul = el('ul', {class: 'items'}, det);
    s.items.forEach(i => el('li', {class: i.done ? 'done' : '', text: i.text.slice(0, 160)}, ul));
  });
}

// How it works. Text with **bold** and `code`, built as DOM nodes (no innerHTML).
function rich(parent, s) {
  s.split(/(\*\*[^*]+\*\*|`[^`]+`)/).forEach(part => {
    if (part.startsWith('**') && part.endsWith('**') && part.length > 4) el('b', {text: part.slice(2, -2)}, parent);
    else if (part.startsWith('`') && part.endsWith('`') && part.length > 2) el('code', {text: part.slice(1, -1)}, parent);
    else if (part) parent.appendChild(document.createTextNode(part));
  });
}
const chartOf = key => D.charts.find(c => c.lines.some(l => l[0] === key));
function steps(id, list) {
  const ol = document.getElementById(id);
  list.forEach(([title, text, links = []]) => {
    const li = el('li', {}, ol); el('b', {class: 't', text: title}, li); rich(li, text);
    if (links.length) { const s = el('span', {class: 'links'}, li);
      links.forEach(([label, href]) => { const c = href.startsWith('#') ? null : chartOf(href);
        el('a', {href: c ? '#' + c.id : href, text: label + ' →'}, s); }); }
  });
}
steps('loop', [
  ['Act in the real game', 'The **actor** (policy network) picks one of 17 actions from the model state. Crafter returns a 64×64 image, a reward and whether the episode ended.',
    [['Episode return', 'episode/score'], ['Episode length', 'episode/length']]],
  ['Store experience', 'Every step goes into the **replay buffer**. Training samples from it: 16 sequences of 64 steps per batch.'],
  ['Learn the world model', 'The **encoder** turns each image into a vector. The **RSSM** keeps a deterministic state (8192 numbers) and a stochastic state (32 categorical variables of 64 classes). The **posterior** sees the frame; the **prior** predicts it without seeing it. Heads predict the image (decoder), the reward and whether the episode continues.',
    [['image loss', 'train/loss/image'], ['dyn KL', 'train/loss/dyn'], ['uncertainty', 'train/rep_ent']]],
  ['Imagine', 'From each replayed state (16 × 64 = 1024 starts) the model rolls forward **15 steps** with the prior only, no real frames, choosing actions with the actor. Heads predict reward and continuation for each imagined step.',
    [['real vs imagined reward', 'train/rew']]],
  ['Learn actor and critic in imagination', 'The **critic** estimates future reward. The **λ-return** mixes imagined rewards with critic values over the 15 steps (discount 0.997). The actor raises the probability of actions with positive **advantage** (REINFORCE) and keeps some **entropy** so it still explores.',
    [['entropy', 'train/ent/action'], ['|advantage|', 'train/adv_mag'], ['critic loss', 'train/loss/value']]],
  ['Repeat', '**Train ratio 512**: for each real env step, 512 replayed steps are trained on. That is why the baseline runs at about 16 env steps/s on an H200.',
    [['speed', 'fps/policy']]],
  ['Evaluate', 'The trained agent, an early checkpoint and a random agent play the same 50 worlds. We count which of the 22 **achievements** each unlocks and compute the **Crafter score** (a geometric mean of success rates), the paper\'s metric.',
    [['Evaluation', '#eval'], ['Achievements', '#achievements']]],
]);
steps('process', [
  ['One software environment', 'Everything runs in one **Docker** image (Python 3.11, JAX), so results do not depend on whose laptop ran them. `scripts/check_local.py` checks it.'],
  ['GPU cluster (PERUN)', 'Real training runs on TUKE\'s **PERUN** cluster. The Docker image becomes an **Apptainer** file; jobs are queued with **SLURM** (`slurm/*.sbatch`). Nothing heavy runs on the shared login node; a laptop helper script sends only a fixed set of commands.'],
  ['Staged tests', 'Local check → interactive GPU test → **smoke test** (debug config, tiny model, about 11 min) → production run. A smoke run proves the pipeline works; it does not learn.'],
  ['Registered experiments', 'Every run gets an ID `<date>-<name>-s<seed>` and a row in `EXPERIMENTS.md` before it starts; failed runs stay listed. A sensitivity experiment changes exactly one flag (ours: `--agent.imag_length 5` instead of 15).', [['Run registry', '#registry']]],
  ['Logs, not checkpoints, in git', 'DreamerV3 writes `metrics.jsonl` (all metrics), `scores.jsonl`, `config.yaml`. Small files are copied to `runs/<run-id>/`; checkpoints (GBs) stay on the cluster and are saved every 15 min, so a stopped run can resume.'],
  ['Analysis', '`analysis/plot_curves.py` makes the report figures, `scripts/eval_agent.py` the evaluations, `analysis/dashboard.py` this page. Every change goes through a pull request; merging to `main` rebuilds this page.'],
]);
// Glossary
const gq = document.getElementById('gq'), gbox = document.getElementById('gloss');
function drawGloss() {
  const q = gq.value.trim().toLowerCase(); gbox.replaceChildren();
  let section = null, dl = null, n = 0;
  D.glossary.filter(g => !q || (g.term + ' ' + g.text).toLowerCase().includes(q)).forEach(g => {
    if (g.section !== section) { section = g.section; el('h4', {text: section}, gbox); dl = el('dl', {}, gbox); }
    el('dt', {text: g.term}, dl); rich(el('dd', {}, dl), g.text); n++;
  });
  if (!n) el('p', {class: 'sub', text: D.glossary.length ? 'No term matches.' : 'docs/glossary.md not found.'}, gbox);
}
gq.addEventListener('input', drawGloss); drawGloss();
// Docs links and build stamp
const docs = document.getElementById('docs');
[['README.md', 'start page: setup, layout, commands'], ['ROADMAP.md', '8-week plan and deliverables'],
 ['docs/learning-path.md', 'what to read, in order'], ['docs/architecture.md', 'DreamerV3 code map, losses, imagination'],
 ['docs/glossary.md', 'all terms'], ['docs/hpc.md', 'PERUN, SLURM, Apptainer'],
 ['EXPERIMENTS.md', 'every run'], ['docs/git-workflow.md', 'branches and pull requests']].forEach(([f, d]) => {
  const li = el('li', {}, docs); el('a', {href: `${D.repo}/blob/main/${f}`, text: f}, li);
  li.appendChild(document.createTextNode(' — ' + d)); });
const stamp = document.getElementById('stamp');
stamp.appendChild(document.createTextNode(`Built ${D.built.time}`));
if (D.built.sha) { stamp.appendChild(document.createTextNode(' from commit ')); el('a', {href: D.built.url, text: D.built.sha}, stamp); }
stamp.appendChild(document.createTextNode(' · ')); el('a', {href: D.repo, text: 'repository'}, stamp);

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
                    help="folder with runs/, EXPERIMENTS.md, ROADMAP.md (default: this repo)")
    ap.add_argument("--out", type=Path, default=ROOT / "dashboard.html")
    ap.add_argument("--site", type=Path,
                    help="write <site>/index.html and copy the media files next to it (GitHub Pages)")
    args = ap.parse_args()
    out = (args.site / "index.html" if args.site else args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    runs, evals, reg, media = load(args.root, out.parent)
    for m in media:
        if args.site:   # the site cannot reach ../runs, so copy the file in
            dest = out.parent / "media" / m["run"] / m["name"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(m["path"], dest)
            m["src"] = f"media/{m['run']}/{m['name']}"
        del m["path"]
    if args.site:
        (out.parent / ".nojekyll").write_text("")   # serve files as they are
    data = {"runs": runs, "evals": evals, "paper": PAPER, "registry": reg,
            "charts": CHARTS, "media": media, "roadmap": roadmap(args.root),
            "glossary": glossary(args.root), "built": build_stamp(args.root), "repo": REPO_URL}
    # "</" inside the JSON would end the <script> tag early.
    text = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    out.write_text(PAGE.replace("__DATA__", text))
    print(f"wrote {out} ({len(runs)} training runs, {len(evals)} evaluations, "
          f"{len(media)} media files)")


if __name__ == "__main__":
    main()
