#!/usr/bin/env python3
"""Compare two training runs on the same axes (Task 7, sensitivity).

Reads runs/<run-id>/scores.jsonl and metrics.jsonl of both runs and writes
docs/figures/<run-b>-vs-<run-a>.svg with three charts, x axis = env steps:

  episode return   moving average over 20 episodes
  value loss       critic, trained on imagined rollouts
  image loss       world model decoder, trained on real data

Run A is drawn in blue, run B in orange. Only the standard library
(reuses the SVG helpers of plot_curves.py).

A side can be several runs (seeds), comma-separated. Then each run's curve
is cut into 10k-step bins and the line is the mean over the seeds, bin by
bin; the file is named after the first run of each side plus "-3seeds".
Usage (in Docker, or any Python 3):
  python analysis/compare_runs.py 20261123-baseline-s0 20261124-imag5-s0
  python analysis/compare_runs.py 20261123-baseline-s0,20261124-baseline-s1,20261124-baseline-s2 \
      20261124-imag5-s0,20261124-imag5-s1,20261124-imag5-s2
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
BIN = 10_000  # env steps per bin when averaging seeds


def seed_mean(curves):
    """Mean over seeds: average each run's points per 10k-step bin, then over runs."""
    bins = {}
    for pts in curves:
        per_run = {}
        for x, y in pts:
            per_run.setdefault(x // BIN, []).append(y)
        for b, ys in per_run.items():
            bins.setdefault(b, []).append(sum(ys) / len(ys))
    # keep only bins that every seed reached
    return [((b + 0.5) * BIN, sum(v) / len(v)) for b, v in sorted(bins.items()) if len(v) == len(curves)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_a", help="reference run (blue); several seeds: comma-separated")
    ap.add_argument("run_b", help="variant run (orange); several seeds: comma-separated")
    ap.add_argument("--window", type=int, default=20, help="moving-average window (episodes)")
    args = ap.parse_args()

    runs_a, runs_b = args.run_a.split(","), args.run_b.split(",")
    multi = len(runs_a) > 1 or len(runs_b) > 1
    w, h = 340, 260
    body = ""
    for i, (fname, key, title, smooth) in enumerate(CHARTS):
        lines = []
        for side, colour in ((runs_a, BLUE), (runs_b, ORANGE)):
            curves = []
            for run in side:
                pts = series(read_jsonl(ROOT / "runs" / run / fname), key)
                curves.append(moving_average(pts, args.window) if smooth else pts)
            name = ", ".join(side) + (" (mean over seeds)" if multi else "")
            lines.append((name, colour, seed_mean(curves) if multi else curves[0]))
        body += chart(i * w, 0, w, h, title, lines)
    mean = f" Mean over {len(runs_a)} vs {len(runs_b)} seeds." if multi else ""
    body += (f'<text x="46" y="{h + 14}" class="tick">Blue: {runs_a[0]}. '
             f'Orange: {runs_b[0]}.{mean} x axis: env steps.</text>')
    suffix = f"-{len(runs_b)}seeds" if multi else ""
    out = ROOT / "docs" / "figures" / f"{runs_b[0]}-vs-{runs_a[0]}{suffix}.svg"
    out.write_text(svg(3 * w, h + 24, body, f"{runs_b[0]} vs {runs_a[0]}{suffix}"))
    print("wrote", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
