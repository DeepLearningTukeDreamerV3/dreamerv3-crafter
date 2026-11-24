#!/usr/bin/env python3
"""Failure cases of the world model from one open-loop run (Task 8).

Reads runs/<openloop-id>/openloop.json (written by scripts/openloop.py)
and writes docs/figures/<openloop-id>-failures.svg with two charts:

  left   image error per step ahead, model vs "copy the last real frame"
  right  reward summed over the open-loop half of each window:
         predicted (x) vs real (y), one dot per window

It also prints the numbers used in docs/failures.md.

How the reward windows are counted. In Crafter, unlocking an achievement
gives +1 and each change of health gives +-0.1 (docs/glossary.md). A
window "has an achievement" when its reward sum is at least 0.5:
  hit          predicted >= 0.5 and real >= 0.5
  false alarm  predicted >= 0.5, real < 0.5 (model invents a reward)
  missed       predicted < 0.5, real >= 0.5 (sparse reward not seen)

Only the standard library (reuses the SVG helpers of plot_curves.py).
Usage (in Docker, or any Python 3):
  python analysis/failures.py 20261124-openloop-s0
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_curves import BLUE, ORANGE, ROOT, chart, svg  # noqa: E402

THRESHOLD = 0.5  # reward sum that counts as "an achievement happened"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("openloop_id")
    args = ap.parse_args()

    d = json.loads((ROOT / "runs" / args.openloop_id / "openloop.json").read_text())
    ctx = d["context"]  # index of the first predicted step
    mse, copy = d["mse_by_step"], d["copy_last_mse_by_step"]

    # Step k ahead = index ctx - 1 + k. k = 0 is the last real frame.
    model = [(k, mse[ctx - 1 + k]) for k in range(0, len(mse) - ctx + 1)]
    base = [(k, copy[ctx - 1 + k]) for k in range(0, len(copy) - ctx + 1)]
    beats = [k for (k, m), (_, c) in zip(model, base) if k > 0 and m < c]

    print(f"{d['windows']} windows, {ctx} context + {len(mse) - ctx} open-loop steps")
    print("steps ahead | model MSE | copy-last MSE")
    for k in (1, 2, 4, 8, 16, 24, 32):
        if k < len(model):
            print(f"{k:11d} | {model[k][1]:9.4f} | {base[k][1]:13.4f}")
    print(f"model below copy-last at {len(beats)} of {len(model) - 1} steps, first at step {beats[0] if beats else '-'}")

    pairs = d["reward_sum_openloop"]  # [predicted, real] per window
    hit = sum(p >= THRESHOLD and r >= THRESHOLD for p, r in pairs)
    false_alarm = sum(p >= THRESHOLD and r < THRESHOLD for p, r in pairs)
    missed = sum(p < THRESHOLD and r >= THRESHOLD for p, r in pairs)
    neg = [r for p, r in pairs if r < 0]
    neg_pred = [p for p, r in pairs if r < 0]
    print(f"reward windows: hit {hit}, false alarm {false_alarm}, missed {missed}, "
          f"neither {len(pairs) - hit - false_alarm - missed}")
    print(f"mean reward sum: predicted {sum(p for p, _ in pairs) / len(pairs):.2f}, "
          f"real {sum(r for _, r in pairs) / len(pairs):.2f}")
    if neg:
        print(f"{len(neg)} windows lose health (real sum < 0, mean {sum(neg) / len(neg):.2f}); "
              f"predicted there: mean {sum(neg_pred) / len(neg_pred):.2f}")
    print(f"continuation probability: min {min(d['continue_prob_by_step']):.3f}, "
          f"windows with an episode end: {d['windows_with_episode_end']}")

    w, h = 420, 300
    body = chart(0, 0, w, h, "Image error vs steps ahead (MSE)",
                 [("copy last real frame", ORANGE, base), ("world model", BLUE, model)])
    top = max(max(p, r) for p, r in pairs)
    body += chart(w, 0, w, h, "Reward per window: predicted (x) vs real (y)",
                  [("predicted = real", "#8a8984", [(0, 0), (top, top)])],
                  dots=[("window", BLUE, [(p, r) for p, r in pairs])])
    body += (f'<text x="46" y="{h + 14}" class="tick">Left: blue world model, orange copy-last baseline; '
             f'mean over {d["windows"]} windows, step 0 = last real frame.</text>'
             f'<text x="46" y="{h + 30}" class="tick">Right: one dot per window, sum over the '
             f'{len(mse) - ctx} open-loop steps. Grey line: perfect prediction. {args.openloop_id}.</text>')
    out = ROOT / "docs" / "figures" / f"{args.openloop_id}-failures.svg"
    out.write_text(svg(2 * w, h + 40, body, "Open-loop failure cases"))
    print("wrote", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
