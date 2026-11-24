# Experiment registry

Add a row **before** a run starts. Update status when it ends.
Never delete a row. Failed runs are evidence (Task 8).

Status: `planned`, `running`, `done`, `failed`, `killed`.

| Run ID | Date | Commit | Configs + flags | Seed | Hardware / job | Wall-clock | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| 20261001-debug-cpu | 2026-10-01 | e3f0224 | `crafter debug`, `--jax.platform cpu` | 0 | local CPU, Docker | ~3 h | done | First local smoke run, `logdir/20261001T153148` |
| 20261109-smoke-j148884 | 2026-11-09 | 4206577 | `crafter debug`, `--jax.platform cuda`, `--run.duration 1200` | 0 | PERUN `gpu_short`, H200, Apptainer, job 148884 | 30 min (limit) | timeout | Stage 3 smoke test, first run with Apptainer after the user-lookup fix. Apptainer, GPU, training and checkpoints all worked (~200 env steps/s). Killed by the 30 min time limit: `run.duration` is not read by upstream `e3f0224`, so the run never stopped. Fixed: smoke now uses `--run.steps 100000` |
| 20261123-smoke-j172451 | 2026-11-23 | 6f0d23d | `crafter debug`, `--jax.platform cuda`, `--run.steps 100000` | 0 | PERUN `gpu_short`, H200, Apptainer, job 172451 | ~11 min | done | Stage 3 smoke test after the step-limit fix. Stopped by itself at 100k steps, checkpoint written, ~185 env steps/s (policy), ~1500 train fps |
| 20261123-baseline-s0 | 2026-11-23 | 6f0d23d | `crafter` (official, code unchanged) | 0 | PERUN `gpu_long`, H200, Apptainer, job 172453 | 3 h 46 min | done | Official Crafter baseline (config `run.steps` 1.1M). Stopped by hand at 213,620 env steps (`scripts/perun.sh cancel`, 2026-11-24 01:30), right after a checkpoint save, see ROADMAP week 4. Last checkpoint `ckpt` ≈ 213k steps. 1136 episodes; mean return of the last 20: 4.4. Resume with the same run ID if more steps are needed. Early checkpoint for Task 6: `ckpt_early_52100`, copied 22:39 when the log showed step 52,100; it holds the last checkpoint saved before then (saved every 15 min, so ~40–52k steps) |
| 20261124-imag5-s0 | 2026-11-24 | 46fdc3c | `crafter`, `--agent.imag_length 5`, `--run.steps 213620` | 0 | PERUN `gpu_long`, H200, Apptainer, job 172542 | | running | Task 7 sensitivity: imagination horizon 5 instead of 15, the only algorithm change. `run.steps` set to the step where `20261123-baseline-s0` stopped, so both runs get the same budget |
| 20261124-eval-final-s0 | 2026-11-24 | 46fdc3c | `scripts/eval_agent.py`, run `20261123-baseline-s0`, `ckpt` (~213k steps), 50 episodes, world seed 100 | 0 | PERUN `gpu_short`, H200, Apptainer, job 172543 | | running | Task 6: trained agent, eval mode |
| 20261124-eval-early-s0 | 2026-11-24 | 46fdc3c | `scripts/eval_agent.py`, run `20261123-baseline-s0`, `ckpt_early_52100` (~40–52k steps), 50 episodes, world seed 100 | 0 | PERUN `gpu_short`, H200, Apptainer, job 172544 | | running | Task 6: early checkpoint, same worlds as the trained agent |
| 20261124-eval-random | 2026-11-24 | 46fdc3c | `scripts/eval_agent.py`, `--random`, 50 episodes, world seed 100 | – | PERUN `gpu_short`, H200, Apptainer, job 172545 | | running | Task 6: random-agent baseline, same worlds |
