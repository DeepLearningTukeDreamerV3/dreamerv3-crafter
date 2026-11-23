#!/usr/bin/env python3
"""Plot episode return and world-model losses of one run (Task 3).

Reads runs/<run-id>/scores.jsonl and metrics.jsonl and writes two SVG
files to docs/figures/:

  <run-id>-return.svg   episode return vs env steps (dots + moving average)
  <run-id>-losses.svg   the 8 losses of Agent.loss vs env steps

Only the standard library: the Docker image has no matplotlib. SVG opens
in any browser and renders on GitHub. Which network each loss trains:
docs/architecture.md, section "Losses".

Usage (in Docker, or any Python 3):
  python analysis/plot_curves.py 20261123-baseline-s0
"""
import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BLUE, ORANGE = "#2a78d6", "#eb6834"  # validated pair (dataviz palette)

# (metrics key, title). Grouped: world model first, then actor-critic.
LOSSES = [
    ("train/loss/image", "image (decoder)"),
    ("train/loss/rew", "rew (reward head)"),
    ("train/loss/con", "con (continuation head)"),
    ("train/loss/dyn", "dyn (prior)"),
    ("train/loss/rep", "rep (posterior)"),
    ("train/loss/policy", "policy (actor, imagined)"),
    ("train/loss/value", "value (critic, imagined)"),
    ("train/loss/repval", "repval (critic, real replay)"),
]

# Colours follow the light/dark mode of the viewer.
STYLE = """<style>
svg { --bg:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#8a8984; --line:#e4e3de; }
@media (prefers-color-scheme: dark) {
  svg { --bg:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --muted:#8f8e86; --line:#33332f; }
}
.bg { fill:var(--bg); }
text { font-family:system-ui,sans-serif; fill:var(--ink2); font-size:12px; }
.title { fill:var(--ink); font-size:14px; font-weight:600; }
.tick { fill:var(--muted); font-size:11px; }
.grid { stroke:var(--line); stroke-width:1; }
.axis { stroke:var(--muted); stroke-width:1; }
.label { fill:var(--ink); }
</style>"""


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def series(rows, key):
    """[(step, value)] for the rows that contain key."""
    return [(r["step"], r[key]) for r in rows if key in r and r[key] is not None]


def moving_average(points, window):
    out = []
    for i in range(len(points)):
        chunk = points[max(0, i - window + 1): i + 1]
        out.append((points[i][0], sum(v for _, v in chunk) / len(chunk)))
    return out


def nice_ticks(lo, hi, n=4):
    """Round axis values (0, 0.5, 1, ...) covering lo..hi."""
    if hi <= lo:
        hi = lo + 1
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        step = m * mag
        if (hi - lo) / step <= n:
            break
    t, ticks = (lo // step) * step, []
    while t <= hi + step * 0.01:
        ticks.append(round(t, 10))
        t += step
    return ticks


def fmt(v):
    if abs(v) >= 1000:
        return f"{v / 1000:g}k"
    return f"{v:.3g}"


def chart(x, y, w, h, title, lines, dots=()):
    """One chart as an SVG group at (x, y). lines/dots: [(name, colour, points)]."""
    pl, pr, pt, pb = 46, 12, 26, 30
    pts = [p for _, _, ps in list(lines) + list(dots) for p in ps]
    if not pts:
        return f'<text x="{x}" y="{y + 18}" class="title">{title}: no data</text>'
    x1 = max(p[0] for p in pts) or 1
    ys = [p[1] for p in pts]
    yt = nice_ticks(min(0, min(ys)), max(ys) * 1.05)
    y0, y1 = yt[0], yt[-1]
    iw, ih = w - pl - pr, h - pt - pb

    def X(v):
        return x + pl + v / x1 * iw

    def Y(v):
        return y + pt + ih - (v - y0) / ((y1 - y0) or 1) * ih

    out = [f'<text x="{x + pl}" y="{y + 16}" class="title">{title}</text>']
    for t in yt:
        out.append(f'<line x1="{x + pl}" x2="{x + pl + iw}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" class="grid"/>'
                   f'<text x="{x + pl - 6}" y="{Y(t) + 4:.1f}" class="tick" text-anchor="end">{fmt(t)}</text>')
    for t in nice_ticks(0, x1):
        if t <= x1:
            out.append(f'<text x="{X(t):.1f}" y="{y + pt + ih + 16}" class="tick" '
                       f'text-anchor="middle">{fmt(t)}</text>')
    out.append(f'<line x1="{x + pl}" x2="{x + pl + iw}" y1="{y + pt + ih}" y2="{y + pt + ih}" class="axis"/>')
    for name, colour, ps in dots:
        for px, py in ps:
            out.append(f'<circle cx="{X(px):.1f}" cy="{Y(py):.1f}" r="2.2" fill="{colour}" opacity="0.4">'
                       f'<title>{name}, step {px:,}: {py:.2f}</title></circle>')
    for name, colour, ps in lines:
        d = " ".join(f"{'M' if i == 0 else 'L'}{X(px):.1f},{Y(py):.1f}" for i, (px, py) in enumerate(ps))
        out.append(f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="2" stroke-linejoin="round">'
                   f'<title>{name}</title></path>')
    return "".join(out)


def svg(width, height, body, label):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'width="{width}" height="{height}" role="img" aria-label="{label}">'
            f'{STYLE}<rect class="bg" width="100%" height="100%"/>{body}</svg>\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_id")
    ap.add_argument("--window", type=int, default=20, help="moving-average window (episodes)")
    args = ap.parse_args()

    run = ROOT / "runs" / args.run_id
    out_dir = ROOT / "docs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    scores = series(read_jsonl(run / "scores.jsonl"), "episode/score")
    metrics = read_jsonl(run / "metrics.jsonl")
    last = max((r["step"] for r in metrics), default=0)
    note = f"{args.run_id}, up to env step {last:,}. x axis: env steps."

    # Episode return
    body = chart(0, 0, 720, 300, "Episode return (sum of rewards per episode)",
                 lines=[(f"moving average ({args.window} episodes)", BLUE,
                         moving_average(scores, args.window))],
                 dots=[("one episode", ORANGE, scores)])
    body += (f'<text x="46" y="322" class="tick">{note} Dots: single episodes. '
             f'Line: moving average of {args.window}.</text>')
    path = out_dir / f"{args.run_id}-return.svg"
    path.write_text(svg(720, 332, body, "Episode return vs env steps"))
    print("wrote", path.relative_to(ROOT))

    # Losses: 4 columns x 2 rows, raw value per log line (each is already
    # an average over the steps since the previous log line).
    cw, ch = 250, 190
    body = "".join(
        chart((i % 4) * cw, (i // 4) * ch, cw, ch, title, [(key, BLUE, series(metrics, key))])
        for i, (key, title) in enumerate(LOSSES))
    body += (f'<text x="46" y="{2 * ch + 14}" class="tick">{note} image to rep: world model, real data. '
             f'policy, value: imagined data. repval: critic on real data. See docs/architecture.md.</text>')
    path = out_dir / f"{args.run_id}-losses.svg"
    path.write_text(svg(4 * cw, 2 * ch + 24, body, "Training losses vs env steps"))
    print("wrote", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
