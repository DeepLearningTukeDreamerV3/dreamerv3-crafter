#!/usr/bin/env python3
"""Compare two training runs on the same axes (Task 7, sensitivity).

Reads runs/<run-id>/scores.jsonl and metrics.jsonl of both runs and writes
docs/figures/<run-b>-vs-<run-a>.svg with three charts, x axis = env steps:

  episode return   moving average over 20 episodes
  value loss       critic, trained on imagined rollouts
  image loss       world model decoder, trained on real data

Run A is drawn in blue, run B in orange. Only the standard library
(reuses the SVG helpers of plot_curves.py).
Usage (in Docker, or any Python 3):
  python analysis/compare_runs.py 20261123-baseline-s0 20261124-imag5-s0
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_curves import BLUE, ORANGE, ROOT, chart, moving_average, read_jsonl, series, svg  # noqa: E402

CHARTS = [  # (file, key, title, smooth over episodes?)
    ("scores.jsonl", "episode/score", "Episode return (moving average, 20 episodes)", True),
    ("metrics.jsonl", "train/loss/value", "Value loss (critic, imagined data)", False),
    ("metrics.jsonl", "train/loss/image", "Image loss (decoder, real data)", False),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_a", help="reference run (blue)")
    ap.add_argument("run_b", help="variant run (orange)")
    ap.add_argument("--window", type=int, default=20, help="moving-average window (episodes)")
    args = ap.parse_args()

    w, h = 340, 260
    body = ""
    for i, (fname, key, title, smooth) in enumerate(CHARTS):
        lines = []
        for run, colour in ((args.run_a, BLUE), (args.run_b, ORANGE)):
            pts = series(read_jsonl(ROOT / "runs" / run / fname), key)
            lines.append((run, colour, moving_average(pts, args.window) if smooth else pts))
        body += chart(i * w, 0, w, h, title, lines)
    body += (f'<text x="46" y="{h + 14}" class="tick">Blue: {args.run_a}. '
             f'Orange: {args.run_b}. x axis: env steps.</text>')
    out = ROOT / "docs" / "figures" / f"{args.run_b}-vs-{args.run_a}.svg"
    out.write_text(svg(3 * w, h + 24, body, f"{args.run_b} vs {args.run_a}"))
    print("wrote", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
