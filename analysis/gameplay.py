#!/usr/bin/env python3
"""Build the gameplay comparison page for the demo: gameplay.html.

Same Crafter worlds, one video panel per training stage (random agent,
~50k, ~213k, ~495k steps, ...), all played in step. Below the videos, a
timeline shows when each agent unlocked each achievement.

Reads every runs/<id>/media/gameplay.json and its world<N>.mp4 files
(recorded by scripts/eval_agent.py --video). The label of each panel
comes from the run's row in EXPERIMENTS.md: "(~495k steps", or a random
agent if the row says --random. Every recorded world is on the page:
nothing is selected by hand.

Only the standard library. Usage (in Docker or any Python 3):
  python analysis/gameplay.py              # writes gameplay.html here
  python analysis/gameplay.py --site _site # GitHub Pages build (CI):
      _site/gameplay.html plus copies of the videos
The page works offline; keep it next to runs/ (or use the --site folder).
"""
import argparse
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = Path(__file__).with_name("gameplay_template.html")


def stage(row):
    """(sort key, label) of a recording, from its EXPERIMENTS.md row."""
    if "--random" in row:
        return 0, "Random"
    m = re.search(r"\(~([0-9.–k]+M?) steps", row)
    if not m:
        raise SystemExit(f"No '(~<n>k steps' in the EXPERIMENTS.md row: {row[:80]}")
    number = float(re.findall(r"[0-9.]+", m[1])[-1])
    return number * (1000 if m[1].endswith("M") else 1), "~" + m[1]


def load(root):
    """One entry per recorded agent, least trained first."""
    rows = {}
    for line in (root / "EXPERIMENTS.md").read_text().splitlines():
        if line.startswith("| "):
            rows[line.split("|")[1].strip()] = line
    agents = []
    for f in sorted((root / "runs").glob("*/media/gameplay.json")):
        run = f.parent.parent.name
        key, label = stage(rows.get(run, ""))
        agents.append({"key": key, "label": label, "run": run,
                       "data": json.loads(f.read_text())})
    return sorted(agents, key=lambda a: a["key"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=ROOT,
                    help="folder with runs/ and EXPERIMENTS.md (default: this repo)")
    ap.add_argument("--site", type=Path,
                    help="write <site>/gameplay.html and copy the videos next to it")
    args = ap.parse_args()

    agents = load(args.root)
    if not agents:
        print("No runs/*/media/gameplay.json yet: page not written")
        return
    out = (args.site or args.root) / "gameplay.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    # Videos stay in runs/; for the site they are copied to gameplay/<run>/.
    for a in agents:
        a["folder"] = f"gameplay/{a['run']}" if args.site else f"runs/{a['run']}/media"
        if args.site:
            target = args.site / a["folder"]
            target.mkdir(parents=True, exist_ok=True)
            for video in (args.root / "runs" / a["run"] / "media").glob("world*.mp4"):
                shutil.copy(video, target)
    data = json.dumps(agents, separators=(",", ":")).replace("</", "<\\/")
    out.write_text(TEMPLATE.read_text().replace("/*DATA*/[]", data))
    print(f"Wrote {out}: {len(agents)} agents")


if __name__ == "__main__":
    main()
