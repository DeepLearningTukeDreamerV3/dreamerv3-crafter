# Roadmap

8 weeks, **Mon 19 Oct → Fri 11 Dec 2026**. Milestones M0–M5 come from the
assignment (§10). Each milestone ends with a supervisor checkpoint with
Prof. Sinčák. Do not start a milestone before the previous one is approved
(assignment §8: "every major phase must be validated").

Tasks are not assigned here. Put your name on the matching GitHub issue.

## Overview

| Week | Dates | Milestone | Focus |
|---|---|---|---|
| 1 | 19–23 Oct | M0 | Setup, debug run on every machine, learning: DL + RL basics |
| 2 | 26–30 Oct | M0 → M1 | HPC access, smoke test, launch official Crafter baseline |
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
- [ ] Every member: install Docker, build image, run `scripts/check_local.py`.
- [ ] Every member: run the CPU debug config (`docker compose --profile cpu up train-cpu`).
- [ ] Commit an environment snapshot (`scripts/env_snapshot.sh`).
- [ ] Ask Prof. Sinčák / university IT which HPC cluster to use and request accounts.
- [ ] Learning: `docs/learning-path.md` steps 1–2 (neural networks, RL basics).

**Done when:** every member has run the debug config, snapshot committed,
supervisor approves the setup.

## Week 2 — M0 → M1: HPC and baseline launch

- [ ] Build the Apptainer image (`docs/hpc.md`).
- [ ] Interactive test: `slurm/00_interactive.sh`.
- [ ] Smoke test: `sbatch slurm/01_smoke.sbatch`. Note its wall-clock per 1k steps.
- [ ] Launch the **official Crafter baseline**: `sbatch slurm/02_train.sbatch 0`
      (seed 0, code unchanged, `--configs crafter`).
- [ ] Register the run in `EXPERIMENTS.md`.
- [ ] Plan early checkpoint copies (needed in week 6, see below).
- [ ] Learning: `docs/learning-path.md` step 3 (world models).

**Done when:** smoke test passes and the baseline job runs.

**Early checkpoints.** DreamerV3 only keeps the latest checkpoint in
`<logdir>/ckpt`. Copy it at an early step (e.g. ~50k env steps) to
`<logdir>/ckpt_early/` before it is overwritten. Task 6 needs it.

## Week 3 — M2: understand the architecture

- [ ] Learning: DreamerV3 paper (`docs/learning-path.md` step 4).
- [ ] Task 2: trace one batch through `dreamerv3/agent.py` and `dreamerv3/rssm.py`.
      Record name and shape of: observation, action, encoder tokens,
      deterministic and stochastic latent state, predicted reward,
      continuation, actor output, critic value. Fill `docs/architecture.md`.
- [ ] Draw the original data-flow diagram. Mark where real data stops and
      imagination begins.
- [ ] If GPU budget allows: launch seeds 1 and 2 of the baseline.

**Done when:** supervisor approves `docs/architecture.md` and the diagram.

## Week 4 — M1 / M3: baseline and world-model losses

- [ ] Baseline finishes (1.1M env steps).
- [ ] Task 3: map each loss in `metrics.jsonl` to its component
      (`train/loss/image`, `rew`, `con`, `dyn`, `rep`, `policy`, `value`, `repval`).
- [ ] Plot: episode return vs env steps; world-model losses vs train steps.
- [ ] Compare the return curve with the published reference in `scores/`.

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

No new experiments this week. It is the buffer for reruns.

## Deliverables (assignment §7)

- [ ] 1. Working installation and documented environment
- [ ] 2. Reproduced baseline run (official config)
- [ ] 3. Original architecture / data-flow diagram
- [ ] 4. Training logs and learning curves
- [ ] 5. Prediction / imagination experiment
- [ ] 6. Random-agent and early-checkpoint baselines
- [ ] 7. One controlled sensitivity experiment
- [ ] 8. SLURM scripts and GPU execution evidence
- [ ] 9. Error analysis with failure cases
- [ ] 10. Reproducible repository with README
- [ ] 11. Final technical report
- [ ] 12. 10–15 min demo

## Risks

- **HPC access late.** Fallback: start a non-official baseline locally with
  `CONFIGS="crafter size12m"` (fits a 4 GB GPU). Label it as non-official.
- **Baseline too slow.** `train_ratio: 512` is heavy. Measure speed in the
  smoke test, extrapolate, and reduce the number of seeds if needed.
- **Lost early checkpoint.** Copy it in week 2–3, see above.
