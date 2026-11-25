#!/usr/bin/env python3
"""GPU utilisation and speed of training runs (deliverable 8, issue #26).

Upstream logs hardware use into metrics.jsonl every few minutes
(`run.usage` in configs.yaml):

  usage/nvsmi/compute_avg/gpu0   share of time a GPU kernel was running
                                 (nvidia-smi "GPU Utilization", 0..1)
  usage/nvsmi/memory_avg/gpu0    share of time GPU memory was being read
                                 or written (bandwidth, not memory filled)
  usage/psutil/proc_ram_gb       RAM of the training process (replay buffer)
  fps/policy                     env steps per second (real game)
  fps/train                      replayed steps trained per second

Prints one Markdown table row per run (mean over the run), and writes
docs/figures/gpu-usage.svg: GPU utilisation and env steps/s against env
steps for the first two runs given (blue, orange). Standard library only.

Usage (in Docker, or any Python 3):
  python analysis/gpu_usage.py 20261123-baseline-s0 20261124-imag5-s0 \
      20261124-baseline-s1 20261124-baseline-s2 20261124-imag5-s1 20261124-imag5-s2
"""
import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_curves import BLUE, ORANGE, ROOT, chart, read_jsonl, series, svg  # noqa: E402

GPU = "usage/nvsmi/compute_avg/gpu0"
MEM = "usage/nvsmi/memory_avg/gpu0"
RAM = "usage/psutil/proc_ram_gb"
ENV_FPS = "fps/policy"
TRAIN_FPS = "fps/train"


def values(rows, key):
    return [v for _, v in series(rows, key)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_ids", nargs="+", help="runs/<run-id> folders; the first two are plotted")
    args = ap.parse_args()

    print("| Run | GPU busy | GPU memory bandwidth | Env steps/s | Trained steps/s | RAM (GB) | Hours for 500k at this speed |")
    print("|---|---|---|---|---|---|---|")
    for run in args.run_ids:
        rows = read_jsonl(ROOT / "runs" / run / "metrics.jsonl")
        gpu, mem = values(rows, GPU), values(rows, MEM)
        env, train = values(rows, ENV_FPS), values(rows, TRAIN_FPS)
        ram = values(rows, RAM)
        # Median speed: the first log rows are slow (JAX compiles, replay fills).
        env_fps = statistics.median(env)
        print(f"| `{run}` | {statistics.mean(gpu):.0%} | {statistics.mean(mem):.0%} | "
              f"{env_fps:.1f} | {statistics.median(train):.0f} | {statistics.mean(ram):.1f} | "
              f"{500_000 / env_fps / 3600:.1f} |")

    w, h = 340, 260
    body = ""
    charts = [(GPU, "GPU utilisation (share of time busy)", 1),
              (ENV_FPS, "Env steps per second", 1)]
    for i, (key, title, _) in enumerate(charts):
        lines = []
        for run, colour in zip(args.run_ids[:2], (BLUE, ORANGE)):
            lines.append((run, colour, series(read_jsonl(ROOT / "runs" / run / "metrics.jsonl"), key)))
        body += chart(i * w, 0, w, h, title, lines)
    names = args.run_ids[:2]
    legend = f"Blue: {names[0]}." + (f" Orange: {names[1]}." if len(names) > 1 else "")
    body += f'<text x="46" y="{h + 14}" class="tick">{legend} x axis: env steps.</text>'
    out = ROOT / "docs" / "figures" / "gpu-usage.svg"
    out.write_text(svg(2 * w, h + 24, body, "GPU utilisation and env steps per second"))
    print("\nwrote", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
