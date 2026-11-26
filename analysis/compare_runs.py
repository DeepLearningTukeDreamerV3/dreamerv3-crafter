#!/usr/bin/env python3
"""Compare two or three training runs on the same axes (Task 7, sensitivity).

Reads runs/<run-id>/scores.jsonl and metrics.jsonl of the runs and writes
docs/figures/<run-b>-vs-<run-a>.svg with four charts, x axis = env steps:

  episode return   moving average over 20 episodes
  episode length   steps survived, moving average over 20 episodes
  value loss       critic, trained on imagined rollouts
  image loss       world model decoder, trained on real data

Run A is drawn in blue, run B in orange, the optional run C in green (file
name: <run-c>-vs-<run-b>-vs-<run-a>.svg). Only the standard library (reuses
the SVG helpers of plot_curves.py).

--max-step N ignores everything after env step N, so runs that were resumed
later can still be compared at an equal budget. It adds "-<N/1000>k" to the
file name, so the 500k and 1M figures do not overwrite each other.

A side can be several runs (seeds), comma-separated. Then each run's curve
is cut into 10k-step bins and the line is the mean over the seeds, bin by
bin; the file is named after the first run of each side plus "-3seeds".
Usage (in Docker, or any Python 3):
  python analysis/compare_runs.py 20261123-baseline-s0 20261124-imag5-s0
  python analysis/compare_runs.py 20261123-baseline-s0,20261124-baseline-s1,20261124-baseline-s2 \
      20261124-imag5-s0,20261124-imag5-s1,20261124-imag5-s2
  python analysis/compare_runs.py --max-step 500000 <horizon 15 seeds> <horizon 5 seeds> <horizon 30 seeds>
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_curves import BLUE, GREEN, ORANGE, ROOT, chart, moving_average, read_jsonl, series, svg  # noqa: E402

CHARTS = [  # (file, key, title, smooth over episodes?)
    ("scores.jsonl", "episode/score", "Episode return (moving average, 20 episodes)", True),
    ("metrics.jsonl", "episode/length", "Episode length (moving average, 20 episodes)", True),
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
    ap.add_argument("run_c", nargs="?", help="second variant (green), optional; several seeds: comma-separated")
    ap.add_argument("--max-step", type=int, help="ignore data after this env step")
    ap.add_argument("--window", type=int, default=20, help="moving-average window (episodes)")
    args = ap.parse_args()

    sides = [side.split(",") for side in (args.run_a, args.run_b, args.run_c) if side]
    colours = [("Blue", BLUE), ("Orange", ORANGE), ("Green", GREEN)]
    multi = any(len(side) > 1 for side in sides)
    w, h = 340, 260
    body = ""
    for i, (fname, key, title, smooth) in enumerate(CHARTS):
        lines = []
        for side, (_, colour) in zip(sides, colours):
            curves = []
            for run in side:
                pts = series(read_jsonl(ROOT / "runs" / run / fname), key)
                if args.max_step:
                    pts = [(x, y) for x, y in pts if x <= args.max_step]
                curves.append(moving_average(pts, args.window) if smooth else pts)
            name = ", ".join(side) + (" (mean over seeds)" if multi else "")
            lines.append((name, colour, seed_mean(curves) if multi else curves[0]))
        body += chart(i * w, 0, w, h, title, lines)
    legend = " ".join(f"{name}: {side[0]}." for side, (name, _) in zip(sides, colours))
    mean = f" Mean over {' vs '.join(str(len(side)) for side in sides)} seeds." if multi else ""
    body += f'<text x="46" y="{h + 14}" class="tick">{legend}{mean} x axis: env steps.</text>'
    suffix = f"-{len(sides[1])}seeds" if multi else ""
    if args.max_step:
        suffix += f"-{args.max_step // 1000}k"
    # variants first, reference last: <b>-vs-<a> or <c>-vs-<b>-vs-<a>
    name = "-vs-".join(side[0] for side in reversed(sides))
    out = ROOT / "docs" / "figures" / f"{name}{suffix}.svg"
    out.write_text(svg(len(CHARTS) * w, h + 24, body, f"{name.replace('-vs-', ' vs ')}{suffix}"))
    print("wrote", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
