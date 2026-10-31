# Roadmap

8 weeks, **Mon 19 Oct → Fri 11 Dec 2026**. Milestones M0–M5 come from the
assignment (§10). Each milestone ends with a team review: every member
checks the "Done when" line before the next milestone starts (assignment
§8: "every major phase must be validated").

**Training runs on the PERUN cluster** (TUKE, SLURM + Apptainer, H200
GPUs), with the official `crafter` config: see `docs/hpc.md`. Never
work on the login node. Laptops are for debug runs, analysis and
writing.

Tasks are not assigned here. Put your name on the matching GitHub issue.

## Overview

| Week | Dates | Milestone | Focus |
|---|---|---|---|
| 1 | 19–23 Oct | M0 | Setup, debug run on every machine, learning: DL + RL basics |
| 2 | 26–30 Oct | M0 → M1 | PERUN setup, staged tests, launch official Crafter baseline |
| 3 | 2–6 Nov | M2 | Learning: world models; trace data flow; architecture diagram |
| 4 | 9–13 Nov | M1 / M3 | Baseline done; world-model losses and learning curves |
| 5 | 16–20 Nov | M3 | Prediction / imagination evidence |
| 6 | 23–27 Nov | M4 | Random + early-checkpoint baselines; launch sensitivity run |
| 7 | 30 Nov–4 Dec | M4 / M5 | Sensitivity results; failure analysis |
| 8 | 7–11 Dec | M5 | Reproducibility check, report, demo |

Long GPU jobs run in the background while the team works on the
analysis tasks. Launch them early.

## Week 1 — M0: setup and basics

- [x] Create GitHub org, repo, project board (see `README.md`).
- [ ] Every member: install Docker, clone, run `./setup.sh` (builds the image, runs `scripts/check_local.py`).
- [ ] Every member: run the CPU debug config (`docker compose --profile cpu up train-cpu`).
- [ ] Commit an environment snapshot (`scripts/env_snapshot.sh`).
- [ ] Find out which members have an NVIDIA GPU and how much memory it has.
- [ ] Learning: `docs/learning-path.md` steps 1–2 (neural networks, RL basics).

**Done when:** every member has run the debug config and the snapshot is
committed.

## Week 2 — M0 → M1: cluster setup and baseline launch

- [x] Smoke test on a local GPU (`20261017-smoke-s0`): ~56 h for 1.1M
      steps and ~23 GB replay RAM, too much for a laptop. So the baseline
      runs on PERUN.
- [ ] Build `dreamerv3.sif` and upload it to PERUN (`docs/hpc.md`).
- [ ] Staged tests on PERUN: `slurm/00_interactive.sh` in an `srun` GPU
      shell, then `sbatch slurm/01_smoke.sbatch`. Note steps per minute.
- [ ] Register the run in `EXPERIMENTS.md`, then launch the **Crafter
      baseline**: official config, seed 0, code unchanged:

      ```sh
      sbatch slurm/02_train.sbatch 0
      ```

- [ ] Optional local GPU run (laptop, non-official smaller model):

      ```sh
      docker compose --profile gpu run --rm train python upstream/dreamerv3/main.py \
        --logdir /logdir/<run-id> --configs crafter size12m --seed 0
      ```

      Do not use `docker compose up train` for this: it forces
      `--run.train_ratio 32` (the `crafter` config uses 512) and writes to
      a new `{timestamp}` folder on every start, so it cannot resume.
- [ ] Plan early checkpoint copies (needed in week 6, see below).
- [ ] Learning: `docs/learning-path.md` step 3 (world models).

**Done when:** smoke test passes and the baseline run is going.

**Early checkpoints.** DreamerV3 only keeps the latest checkpoint in
`<logdir>/ckpt`. Copy it at an early step (e.g. ~50k env steps) to
`<logdir>/ckpt_early/` before it is overwritten. Task 6 needs it.

## Week 3 — M2: understand the architecture

- [ ] Learning: DreamerV3 paper (`docs/learning-path.md` step 4).
- [ ] Task 2: trace one batch through `upstream/dreamerv3/agent.py` and `upstream/dreamerv3/rssm.py`.
      Record name and shape of: observation, action, encoder tokens,
      deterministic and stochastic latent state, predicted reward,
      continuation, actor output, critic value. Fill `docs/architecture.md`.
- [ ] Draw the original data-flow diagram. Mark where real data stops and
      imagination begins.
- [ ] If a second GPU is free: launch seeds 1 and 2 of the baseline.

**Done when:** the team has reviewed `docs/architecture.md` and the diagram.

## Week 4 — M1 / M3: baseline and world-model losses

- [ ] Baseline finishes (1.1M env steps). If the smoke test shows it
      cannot finish in time, stop at a fixed step count and say so.
- [ ] Task 3: map each loss in `metrics.jsonl` to its component
      (`train/loss/image`, `rew`, `con`, `dyn`, `rep`, `policy`, `value`, `repval`).
- [ ] Plot: episode return vs env steps; world-model losses vs train steps.
- [ ] Compare the return curve with the published reference in `upstream/scores/`
      (same config as the paper, so scores should be close; explain any gap).

**Done when:** baseline reproduced, curves committed, M1 approved.

## Week 5 — M3: prediction and imagination

- [ ] Task 4: open-loop prediction from a real trajectory prefix. Use the
      `openloop/*` videos from the run report (`agent.py`, `report()`), and
      compare predicted vs observed reward / continuation over k future steps.
- [ ] Task 5: document how actor and critic train on imagined rollouts
      (`imag_length: 15`, `horizon: 333` in `configs.yaml`). Plot return
      next to world-model loss over training.

**Done when:** prediction figure and imagination section done, M3 approved.

## Week 6 — M4: baselines and sensitivity launch

- [ ] Task 6: random agent (`scripts/random_agent.py`) and early checkpoint
      (`--script eval_only --run.from_checkpoint ...`), same seeds and
      episode count as the trained agent.
- [ ] Task 7: launch one sensitivity run. Suggested: `--agent.imag_length 5`
      vs default 15. Change nothing else.
- [ ] Fill the evaluation table (assignment A.7) except the sensitivity row.

## Week 7 — M4 / M5: sensitivity results and failures

- [ ] Sensitivity results vs default; explain the effect.
- [ ] Task 8: failure cases: error growth over the open-loop horizon,
      missed sparse rewards, reward-model errors, crashed or failed runs
      from `EXPERIMENTS.md`.
- [ ] GPU utilisation and wall-clock time of the main run.

**Done when:** M4 approved, failure section drafted.

## Week 8 — M5: final

- [ ] Fresh clone on a clean machine; follow `README.md` only; debug run works.
- [ ] Final README, technical report (`docs/report/`).
- [ ] 10–15 min demo: environment behaviour + internal world-model workflow.
- [ ] Tick all 12 deliverables below.
- [ ] GPU execution evidence for deliverable 8: job IDs, `sacct` output,
      `runs/<run-id>/env.txt` of the PERUN runs.

No new experiments this week. It is the buffer for reruns.

## Deliverables (assignment §7)

- [ ] 1. Working installation and documented environment
- [ ] 2. Reproduced baseline run (official config)
- [ ] 3. Original architecture / data-flow diagram
- [ ] 4. Training logs and learning curves
- [ ] 5. Prediction / imagination experiment
- [ ] 6. Random-agent and early-checkpoint baselines
- [ ] 7. One controlled sensitivity experiment
- [ ] 8. SLURM scripts and GPU execution evidence (PERUN)
- [ ] 9. Error analysis with failure cases
- [ ] 10. Reproducible repository with README
- [ ] 11. Final technical report
- [ ] 12. 10–15 min demo

## Risks

- **Job hits the 4-day limit or crashes.** DreamerV3 resumes from
  `<logdir>/ckpt` when started with the same logdir:
  `RUN_ID=<run-id> sbatch slurm/02_train.sbatch 0`.
- **Scratch purge.** `/mnt/scratch` deletes files 60 days after last
  access. Copy small logs to `runs/<run-id>/` when a run ends.
- **Baseline too slow.** `train_ratio: 512` is heavy. Measure speed in the
  smoke test, extrapolate, and reduce the number of seeds if needed.
- **Lost early checkpoint.** Copy it in week 2–3, see above.
