# GPU execution evidence (deliverable 8)

What ran on the PERUN cluster, how long it took, and how busy the GPU was.
Scripts: `slurm/` (how to use them: [`hpc.md`](hpc.md)). Every job below
has a row in [`EXPERIMENTS.md`](../EXPERIMENTS.md) with its commit and
flags.

All jobs: one NVIDIA H200, Apptainer image `dreamerv3.sif` built from our
`Dockerfile` (Python 3.11, `jax[cuda]` 0.5.0), code of the listed commit
bind-mounted over `/app` (`slurm/common.sh`).

## 1. Staged workflow, with job IDs

| Stage | Script | Job(s) | Result |
|---|---|---|---|
| 1. Local test | `scripts/check_local.py` (Docker, laptop CPU) | – | passes; also the local CPU debug run `20261001-debug-cpu` |
| 2. Interactive `srun` test | `slurm/00_interactive.sh` | not recorded | see "Gaps" below |
| 3. Smoke test | `slurm/01_smoke.sbatch` (debug config) | 148884, 172451 | 148884 hit the 30 min limit (step-limit flag not read); fixed, 172451 stopped by itself at 100k steps in ~11 min |
| 4. Production | `slurm/02_train.sbatch` (official `crafter` config) | 172453, 173386, 172542, 173387, 173430–173433 | six runs to ~495k env steps |
| Evaluation | `slurm/03_eval.sbatch` | 172543–172545, 173381, 175203–175208 | 50 episodes each, < 7 min |
| Open loop | `slurm/04_openloop.sbatch` | 172547 (crash), 172548, 175209 | < 15 min |

The smoke test found the step-limit bug, and the first open-loop job
found a GPU-only crash. Both were fixed before any long run was lost
([`failures.md`](failures.md), section 5).

## 2. Wall-clock time

From the `EXPERIMENTS.md` rows (start and end times in the job logs).

| Run | Jobs | Wall-clock |
|---|---|---|
| `20261123-baseline-s0` | 172453 (to 213k) + 173386 (resume to 500k) | 3 h 46 min + ~5 h |
| `20261124-imag5-s0` | 172542 (to 213k) + 173387 (resume to 500k) | ~2 h 20 min + ~4.5 h |
| `20261124-baseline-s1`, `-s2` | 173430, 173431 | ~9 h each |
| `20261124-imag5-s1`, `-s2` | 173432, 173433 | ~8 h each |

About 50 GPU-hours of training in total, plus about 2 GPU-hours of
evaluation and open-loop jobs.

## 3. GPU utilisation and speed

Upstream DreamerV3 logs hardware use into `metrics.jsonl` every few
minutes (about 230–260 samples per run). To recompute the table and the
figure:

```sh
python analysis/gpu_usage.py 20261123-baseline-s0 20261124-imag5-s0 \
    20261124-baseline-s1 20261124-baseline-s2 20261124-imag5-s1 \
    20261124-imag5-s2 20261123-smoke-j172451
```

| Run | GPU busy | GPU memory bandwidth | Env steps/s | Trained steps/s | RAM (GB) | Hours for 500k at this speed |
|---|---|---|---|---|---|---|
| `20261123-baseline-s0` | 88% | 28% | 15.9 | 8131 | 20.4 | 8.7 |
| `20261124-imag5-s0` | 87% | 28% | 17.8 | 9109 | 20.5 | 7.8 |
| `20261124-baseline-s1` | 88% | 28% | 15.9 | 8166 | 19.9 | 8.7 |
| `20261124-baseline-s2` | 89% | 28% | 16.0 | 8192 | 20.0 | 8.7 |
| `20261124-imag5-s1` | 88% | 28% | 17.9 | 9156 | 19.9 | 7.8 |
| `20261124-imag5-s2` | 87% | 28% | 17.7 | 9074 | 20.0 | 7.8 |
| `20261123-smoke-j172451` | 15% | 0% | 180.6 | 1448 | 2.4 | 0.8 |

Columns:

- **GPU busy**: mean of `usage/nvsmi/compute_avg/gpu0`, the share of
  time at least one GPU kernel was running (nvidia-smi "GPU Utilization").
- **GPU memory bandwidth**: mean of `usage/nvsmi/memory_avg/gpu0`, the
  share of time GPU memory was being read or written. It is not how
  full the memory is.
- **Env steps/s**: median of `fps/policy`, real Crafter steps per second.
  **Trained steps/s**: median of `fps/train`, replayed steps the model
  trains on per second. Their ratio is about 512, the `train_ratio`.
- **RAM**: memory of the training process on the host, mostly the
  replay buffer.

![GPU utilisation and env steps per second, baseline seed 0 vs horizon 5 seed 0](figures/gpu-usage.svg)

What the numbers show:

- **The GPU is the bottleneck.** It is busy 87–89 % of the time in every
  production run, from the first log row to the last. With
  `train_ratio 512`, each real step is followed by 512 replayed steps of
  training, so the game itself costs almost nothing. A faster CPU or
  more parallel game copies would not make the run faster.
- **Horizon 5 is about 12 % faster** (17.8 vs 15.9 env steps/s). The
  actor-critic update imagines 5 steps instead of 15. The world-model
  update, which is the same in both, is most of the work.
- **The smoke test barely uses the GPU** (15 %): the `debug` config is
  a tiny model, so the H200 waits for the game. Its speed says nothing
  about the real config.
- **Seeds agree.** The three seeds of each setting are within 1 % of each
  other in speed and GPU use, as expected on the same hardware.

## 4. Environment record (`env.txt`)

`scripts/env_snapshot.sh` writes `runs/<run-id>/env.txt` at the start of
each job. In every PERUN job up to 175209 it recorded the **host's**
Python 3.9 without JAX and an empty Git section: `slurm/common.sh` did
not `export SIF`, so the snapshot script never ran inside the image.
Fixed in PR #62. For those runs, the versions are the image's
(`Dockerfile`) and the commit is in `EXPERIMENTS.md`. The slurm `.out`
files on the cluster also start with the image's `pip freeze` (JAX and
NVIDIA packages) and `nvidia-smi`, printed by `upstream/entrypoint.sh`.

## Gaps

- **No interactive `srun` record.** `slurm/00_interactive.sh` exists but
  no run of it was registered. The smoke test (stage 3) checked the same
  things (GPU visible, JAX on CUDA, model built, training steps, logs
  written) in a short batch job.
- **No `sacct` output.** `scripts/perun.sh status` does not get
  accounting data on PERUN, so wall-clock times come from the job logs
  and `EXPERIMENTS.md`, rounded.
