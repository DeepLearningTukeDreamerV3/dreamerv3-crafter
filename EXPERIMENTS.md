# Experiment registry

Add a row **before** a run starts. Update status when it ends.
Never delete a row. Failed runs are evidence (Task 8).

Status: `planned`, `running`, `done`, `failed`, `killed`.

| Run ID | Date | Commit | Configs + flags | Seed | Hardware / job | Wall-clock | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| 20261001-debug-cpu | 2026-10-01 | e3f0224 | `crafter debug`, `--jax.platform cpu` | 0 | local CPU, Docker | ~3 h | done | First local smoke run, `logdir/20261001T153148` |
| 20261109-smoke-j148884 | 2026-11-09 | 4206577 | `crafter debug`, `--jax.platform cuda`, `--run.duration 1200` | 0 | PERUN `gpu_short`, H200, Apptainer, job 148884 | 30 min (limit) | timeout | Stage 3 smoke test, first run with Apptainer after the user-lookup fix. Apptainer, GPU, training and checkpoints all worked (~200 env steps/s). Killed by the 30 min time limit: `run.duration` is not read by upstream `e3f0224`, so the run never stopped. Fixed: smoke now uses `--run.steps 100000` |
| 20261124-openloop-s0 | 2026-11-24 | 7bf2bb4 | `scripts/openloop.py`, run `20261123-baseline-s0`, `ckpt` (~213k steps), 10 episodes, 3 windows each, 64 steps (32 context + 32 open loop), world seed 100 | 0 | PERUN `gpu_short`, H200, Apptainer, job 172548 (job 172547 failed) | < 15 min | done | Task 4: 30 windows. Image MSE 0.009 in context, 0.010 one step ahead, 0.032 after 32 steps; below the copy-last-frame baseline from step 4 on (0.023 vs 0.030 at step 8), close to it at step 32 (0.032 vs 0.034). Frames: player and terrain kept, trees/sand/cows fade. Reward over the 32 open-loop steps: predicted 0.53, real 0.26 per window (2x too high). Continuation ≥ 0.997, no death in the windows. Job 172547 crashed after recording the episodes: implicit GPU-to-host copy refused by upstream transfer guard; fixed in 7bf2bb4 |
