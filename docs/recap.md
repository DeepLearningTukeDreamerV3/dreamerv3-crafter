# Project recap

One page to understand the whole project: what we did, how, why, what
we found, and what is left. For the details, follow the links. Terms in
*italics* are in `docs/glossary.md`.

State on 2026-11-25: all experiments are done. What remains is writing
(report, demo, GPU evidence) and a fresh-clone check.

## 1. Goal

Train DreamerV3 on Crafter, check that it learns, and look inside its
*world model*: how well does it predict the game, and how does one
design choice (the *imagination horizon*) change what the agent learns.

- **Crafter**: a 2D Minecraft-like game. 22 *achievements* (collect
  wood, place a table, make a pickaxe, defeat a zombie, ...). Reward +1
  the first time each achievement is unlocked in an episode, ±0.1 per
  health point. An episode ends when the player dies (or at 10,000
  steps).
- **DreamerV3** (Hafner et al., 2023): a model-based RL agent. It does
  not learn the policy from the real game directly. It learns a world
  model of the game, then trains the policy inside that model.

## 2. How DreamerV3 works (short)

Details and measured tensor shapes: `docs/architecture.md`.

1. **Act and record.** The agent plays the real game. Every frame,
   action and reward goes into a *replay buffer*.
2. **Learn the world model** from batches of real replay (16 sequences
   of 64 steps). The model compresses each frame into a *latent state*
   and learns to predict the next latent state from the action. Heads
   decode the frame (image loss), the reward and whether the episode
   continues.
3. **Imagine.** From real latent states, the model rolls forward
   *imagined* trajectories of 15 steps (the imagination horizon), with
   no real game involved.
4. **Train actor and critic on imagination.** The *critic* learns the
   expected future reward of a state. The *actor* (the policy) learns
   to pick actions that lead to high imagined reward plus critic value.
5. Repeat. `train_ratio 512` means one training batch every 2 env steps.

Real experience trains only the world model. The actor and critic only
ever see imagined data. This split is the heart of the method and of
our analysis.

## 3. How we ran it

| Piece | What | Where |
|---|---|---|
| Code | Upstream DreamerV3 commit `e3f0224`, unchanged (`upstream/`) | `README.md` |
| Config | Official `crafter` config (model size XL equivalent, `train_ratio 512`) | `upstream/dreamerv3/configs.yaml` |
| Environment | Docker image (Python 3.11, JAX) on laptops; same image as Apptainer `.sif` on the cluster | `DOCKER.md`, `docs/hpc.md` |
| Hardware | PERUN cluster (TUKE), one NVIDIA H200 per run, SLURM jobs | `slurm/`, `docs/hpc.md` |
| Cluster access | `scripts/perun.sh`: fixed commands (status, log, fetch, submit, eval, ...) run from the laptop; nothing heavy on the login node | `docs/hpc.md` |
| Registry | Every run has an ID and a row before it starts, failed ones included | `EXPERIMENTS.md` |
| Our scripts | Evaluation, open-loop prediction, plots, batch tracing | `scripts/`, `analysis/` |

**Speed.** About 16 env steps/s per run (baseline), 18 with horizon 5.
The limit is training compute (one batch of 16×64 per 2 env steps), not
the game. The GPU was busy ~85 % of the time. So we ran seeds in
parallel instead of trying to make one run faster.

**Budget.** The paper trains for 1M env steps (10 seeds). We trained to
**500k env steps, 3 seeds per setting** (~9 h per run from scratch).
This was a time decision: 1M steps would take ~18 h per run.

## 4. Runs

| Run | Setting | Seed | Steps | Use |
|---|---|---|---|---|
| `20261123-baseline-s0` | official `crafter` | 0 | 500k (stopped at 213k, resumed) | main run |
| `20261124-baseline-s1`, `-s2` | official `crafter` | 1, 2 | 500k | spread across seeds |
| `20261124-imag5-s0` | `--agent.imag_length 5` | 0 | 500k (213k, resumed) | sensitivity (Task 7) |
| `20261124-imag5-s1`, `-s2` | `--agent.imag_length 5` | 1, 2 | 500k | sensitivity across seeds |
| `*-eval-*`, `*-eval500k-*` | `scripts/eval_agent.py` | – | – | 50 fixed eval worlds per checkpoint |
| `*-openloop-*`, `*-openloop500k-*` | `scripts/openloop.py` | – | – | world-model prediction test |

Checkpoints are saved every 15 min and not at the end of a run, so
"500k" checkpoints are at ~495k steps and "213k" ones at ~208–213k.

## 5. Results

All numbers: `docs/results.md`. Figures: `docs/figures/`.

### 5.1 The agent learns

Every agent plays the same 50 Crafter worlds (world seed 100) in eval
mode with `scripts/eval_agent.py`. *Crafter score* = geometric mean of
the 22 *success rates*; it rewards a wide spread of achievements.

| Agent | Env steps | Return | Crafter score |
|---|---|---|---|
| Random | 0 | 1.52 | 1.4 % |
| Early checkpoint | ~40–52k | 2.64 | 2.7 % |
| Baseline seed 0 | ~213k | 4.44 | 4.3 % |
| Baseline, 3 seeds (mean ± sd across seeds) | ~495k | 7.82 ± 1.65 | 11.3 ± 4.9 % |
| per seed 0 / 1 / 2 | ~495k | 8.78 / 8.76 / 5.92 | 13.7 / 14.4 / 5.6 % |

- Each stage beats the one before, on both numbers. The agent learns.
- What it learned, at ~495k: wood, table and wood pickaxe in almost
  every episode; stone, coal and furnace often; a first skeleton. Iron
  and diamond never.
- At ~495k the agents also survive longer than random (217 vs 175
  steps per episode). At 213k they did not yet.
- **Seed 2 is stuck one stage behind**: wood pickaxe in only 8 % of
  episodes, so no stone and nothing after it. One missed early step in
  the tech tree blocks all later ones. This one seed makes most of the
  spread.

### 5.2 Learning curves over 3 seeds

![Baseline vs horizon 5, mean over 3 seeds](figures/20261124-imag5-s0-vs-20261123-baseline-s0-3seeds.svg)

Training return, mean of the last 50 episodes before each step count
(3 seeds: mean ± standard deviation across seeds):

| Env steps | Horizon 15 (baseline) | Horizon 5 |
|---|---|---|
| 100k | 3.51 ± 0.08 | 3.71 ± 0.39 |
| 213k | 4.33 ± 0.17 | 4.71 ± 0.75 |
| 300k | 5.55 ± 0.92 | 5.28 ± 0.79 |
| 400k | 6.63 ± 1.31 | 6.17 ± 0.53 |
| 500k | 7.87 ± 1.74 | 7.24 ± 0.10 |

- Return still rises at 500k for every seed: training was not finished.
- Baseline seeds spread out after 300k (5.98 to 9.40 at 500k). Seeds of
  horizon 5 stay together (7.18 to 7.36).

### 5.3 Sensitivity: imagination horizon 15 vs 5 (Task 7)

One flag changed: `--agent.imag_length 5`. The actor and critic learn
from 5-step imagined rollouts instead of 15.

| Eval at ~495k (50 worlds) | Horizon 15 (baseline) | Horizon 5 |
|---|---|---|
| Return, mean ± sd across seeds | 7.82 ± 1.65 | 7.47 ± 0.31 |
| Crafter score, mean ± sd | 11.3 ± 4.9 % | 9.4 ± 2.3 % |
| Score per seed | 13.7 / 14.4 / 5.6 % | 7.8 / 12.0 / 8.4 % |
| Defeat zombie (success rate) | 35 % | 57 % |
| Place furnace | 25 % | 3 % |
| Wood pickaxe | 63 % (seed 2: 8 %) | 79 % |

**What we found.**

- Early (100k–213k) horizon 5 is ahead; from ~300k the baseline is.
- At ~495k the means are close and inside the spread: with 3 seeds we
  cannot call one setting better on average.
- The two best agents use horizon 15 (13.7 and 14.4 %); no horizon-5
  seed passes 12.0 %.
- Horizon 5 is more consistent across seeds and kills more zombies (a
  quick reward), but rarely builds a furnace (the longest chain the
  agents reach).
- At 213k horizon 5 never made a wood pickaxe; at ~495k all three seeds
  do (72–88 %). The short horizon delayed the long chains; it did not
  block them.

**Why.** An achievement chain like wood → table → pickaxe pays its +1
many steps after the first action. With a 5-step rollout the actor sees
that reward only through the critic's estimate at the end of the
rollout, not as an imagined reward. Quick rewards (a zombie kill) fit
inside 5 steps. So horizon 5 learns quick rewards early and is ahead
at first, while horizon 15 is better at the long chains that matter
more later. Horizon 5 is also ~13 % faster per env step (shorter
rollouts).

### 5.4 Inside the world model: open-loop prediction (Task 4, Task 8)

Test: the model sees 32 real frames, then predicts the next 32 from the
actions alone (*open loop*). We compare the predicted frames to the
real ones (pixel MSE) and to a trivial guess (repeat the last real
frame). Details: `docs/failures.md`.

| Checkpoint | MSE in context | 1 step ahead | 8 steps | 32 steps | Copy last frame, 8 / 32 | Reward per window, predicted / real |
|---|---|---|---|---|---|---|
| ~213k | 0.009 | 0.011 | 0.023 | 0.032 | 0.030 / 0.034 | 0.53 / 0.26 |
| ~495k | 0.021 | 0.020 | 0.029 | 0.048 | 0.029 / 0.038 | 0.50 / 0.70 |

- **At 213k**: error triples over 32 steps (mistakes add up, no real
  frame to correct them). The model beats the copy from step 7 on. The
  player and large terrain survive; trees, cows and sand fade. Reward is
  predicted twice too high (false alarms).
- **At 495k the error is higher, but the test is not the same.** The
  500k agent survives longer (episodes 258 vs 202 steps on average) and
  reaches **night** and **caves**. Crafter darkens night frames and adds
  random noise to them, which no model can predict pixel by pixel.
  `runs/20261125-openloop500k-s0/media/openloop.png` shows several
  dark windows. So the two rows measure different kinds of frames. A
  fair comparison would feed the same recorded episodes to both
  checkpoints (not done).
- Reward at 495k is now *under*-predicted (0.50 vs 0.70): the agent
  collects more rewards and the model does not see them all coming.
- Continue head: ≥ 0.96 at every step, no death inside any window, so
  death prediction is untested.

### 5.5 Comparison with the paper (issue #17)

Paper (arXiv 2301.04104 v1, Table M.1, 1M steps, 10 seeds): Crafter
score 14.5 ± 1.6 %, return 11.7 ± 1.9. Random agent 1.6 %.

| | Env steps | Seeds | Crafter score | Return |
|---|---|---|---|---|
| Paper | 1M | 10 | 14.5 ± 1.6 % | 11.7 ± 1.9 |
| Ours | ~495k | 3 | 11.3 ± 4.9 % | 7.82 ± 1.65 |
| Ours, seeds 0 and 1 | ~495k | 2 | 13.7 / 14.4 % | 8.78 / 8.76 |
| Ours, random agent | – | – | 1.4 % | 1.52 |

- **Random agent matches the paper** (1.4 vs 1.6 %): our evaluation and
  score formula behave like theirs.
- **Two of three seeds reach the paper's score with half its steps.**
  The mean is lower because of the stuck seed 2.
- Return stays lower (7.8 vs 11.7): the paper's agents unlock more per
  episode, our best ones a similar spread. Return still rises at 500k.
- Other differences: 3 vs 10 seeds, 50 eval episodes of one checkpoint
  instead of all training episodes, 2024 code vs the paper's 2023 code.

### 5.6 What failed (Task 8)

Full list: `docs/failures.md`, `EXPERIMENTS.md`.

- Smoke job killed by the time limit: upstream ignores
  `--run.duration`. Fixed with `--run.steps`.
- Open-loop job crashed on GPU only (JAX transfer guard). Fixed with an
  explicit `jax.device_get`.
- Baseline stopped by hand at 213k for time, then resumed to 500k.
- 500k evals first failed to submit: the resumed runs rewrote their
  committed `env.txt` on the cluster, so `git switch` refused.
  `perun.sh` now restores such files when they match the committed copy.

## 6. Limits

- 500k of the paper's 1M steps, 3 of its 10 seeds.
- Evaluation: 50 episodes on fixed worlds of one checkpoint, not the
  benchmark protocol (success rates over all training episodes).
- Open-loop: pixel MSE favours blur; 213k and 495k tests see different
  states (night); no deaths in the windows.
- Explanation of the horizon effect follows from how imagination
  training works; we did not test it directly (for example by
  measuring imagined reward per step for each horizon).

## 7. What is left

| # | Deliverable (assignment §7) | State | To do |
|---|---|---|---|
| 1 | Working installation, documented environment | done | tick in ROADMAP after the fresh-clone check |
| 2 | Reproduced baseline run (official config) | done (500k, 3 seeds) | – |
| 3 | Architecture / data-flow diagram | done (`docs/architecture.md`) | – |
| 4 | Training logs and learning curves | done (`runs/`, `docs/figures/`) | – |
| 5 | Prediction / imagination experiment | done (213k and 495k open loop) | optional: same episodes for both checkpoints |
| 6 | Random-agent and early-checkpoint baselines | done | – |
| 7 | One controlled sensitivity experiment | done (3 seeds) | – |
| 8 | SLURM scripts and GPU execution evidence | done (`docs/gpu-evidence.md`: job IDs, wall-clock, ~88 % GPU use) | – |
| 9 | Error analysis with failure cases | done (`docs/failures.md`, 213k and 495k) | turn into the report section |
| 10 | Reproducible repository with README | mostly | fresh clone on a clean machine, follow README only |
| 11 | Final technical report | not started | write in `docs/report/` from `docs/results.md`, `docs/failures.md`, `docs/architecture.md`; cite one general and one robotics world-model survey |
| 12 | 10–15 min demo | not started | game footage, open-loop GIF, results; rehearse |

Also open: old PRs #38 and #39, issue #1, and whether to commit
`analysis/progress_deck.py`.

## 8. Where things are

| Need | File |
|---|---|
| All numbers | `docs/results.md` |
| Every run, job ID, commit | `EXPERIMENTS.md` |
| Model, shapes, losses | `docs/architecture.md` |
| Failure cases | `docs/failures.md` |
| Cluster how-to | `docs/hpc.md` |
| Words | `docs/glossary.md` |
| Plan and checkboxes | `ROADMAP.md` |
