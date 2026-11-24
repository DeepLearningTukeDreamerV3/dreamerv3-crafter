# Failure cases (Task 8)

Where the world model and our runs fail, with the numbers that show it.
Draft for the error-analysis section of the report (deliverable 9).

Data: open-loop run `20261124-openloop-s0` (`scripts/openloop.py`,
baseline checkpoint ~213k env steps, 30 windows of 64 steps). In each
window the model sees 32 real frames (context), then predicts the next 32
from the actions alone (open loop). Numbers and figure:

```sh
python analysis/failures.py 20261124-openloop-s0
```

![Image error per step ahead, and predicted vs real reward per window](figures/20261124-openloop-s0-failures.svg)

## 1. Image error grows with the horizon

| Steps ahead | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| World model MSE | 0.011 | 0.012 | 0.018 | 0.023 | 0.027 | 0.032 |
| Copy last real frame | 0.006 | 0.009 | 0.015 | 0.030 | 0.037 | 0.034 |

- The error triples in 32 steps (0.011 to 0.032). Each imagined step
  starts from the previous imagined state, so mistakes add up. The model
  never sees a real frame again to correct itself.
- For the first 6 steps the model is **worse** than simply repeating the
  last real frame. The scene barely changes in a few steps, so copying is
  hard to beat. The decoder also blurs details even in context
  (MSE 0.008 at the last context frame, where copying scores 0).
- From step 7 on the model is better (26 of 32 steps). It knows how the
  view moves when the player walks; the copy does not.
- At step 32 both are close again (0.032 vs 0.034): the prediction has
  lost most of what is specific to this window.

What fails visually (`runs/20261124-openloop-s0/media/openloop.png`): the
player and the large terrain (grass, water) stay. Small and rare things
fade: trees, sand edges, cows. The model keeps the
average look of the world and drops the details that matter for
achievements (a cow to eat, a tree to collect wood).

## 2. Reward model: false alarms and missed sparse rewards

Crafter gives +1 the first time an achievement is unlocked and ±0.1 when
health changes. We sum the reward over the 32 open-loop steps of each
window and call a sum of at least 0.5 "an achievement happened".

| | Real: achievement | Real: none |
|---|---|---|
| **Predicted: achievement** | 6 (hit) | 9 (false alarm) |
| **Predicted: none** | 4 (missed) | 11 |

- **Over-estimation.** Mean predicted reward per window is 0.53, real
  0.26: twice too high. Most of the gap comes from the 9 false alarms,
  where the model predicts a full +1 (dots at x ≈ 1, y = 0 in the
  figure). Possible causes (not tested): the imagined world drifted
  from the real one, or the model does not know which achievements
  are already unlocked (+1 is paid only the first time).
- **Missed sparse rewards.** In 4 of the 10 windows with a real
  achievement, the model predicted none (dots at x ≈ 0, y ≥ 0.7), one window with +2. Achievements
  are rare in replay, so the reward head sees few examples of them.
- **Health penalties are missed.** 8 windows lose health (real sum
  −0.34 on average); the model predicts +0.24 there. It does not see the
  damage coming (hunger, thirst, zombies) without real frames.

Why it matters: the actor learns only from imagined rewards
(`docs/architecture.md`, Imagination). False alarms push it toward
actions that pay nothing in the real game. Missed rewards and missed
damage hide what it should learn or avoid.

## 3. Continuation: no deaths to test

The continue head predicts ≥ 0.997 at every step, and no window contains
an episode end. Episodes last 158–430 steps, so a 64-step window rarely
reaches a death. This is a gap in the test, not evidence that the head
predicts deaths well. To test it: cut windows that end at the last step
of an episode.

## 4. Crashed and failed runs

From `EXPERIMENTS.md` (rows are never deleted):

| Run / job | What failed | Cause | Fix |
|---|---|---|---|
| `20261109-smoke-j148884` | killed by the 30 min time limit | `--run.duration` is not read by upstream `e3f0224`; the run never stopped | smoke uses `--run.steps 100000` |
| `20261124-openloop-s0`, job 172547 | crashed after recording the episodes | upstream's JAX transfer guard refuses an implicit GPU-to-host copy; CPU tests did not show it | explicit `jax.device_get` (commit `7bf2bb4`), job 172548 |
| `20261123-baseline-s0` | stopped by hand at 213,620 of 1.1M steps | time budget of the project, not a crash | resume with the same run ID if needed |

Lesson for both crashes: a flag or code path that works on the laptop
CPU can fail on the cluster GPU. The staged workflow (`docs/hpc.md`:
local, interactive, smoke, production) caught both before a long run
was lost.

## Limits of this analysis

- One checkpoint, one seed, 30 windows: the counts in section 2 are
  small. Treat them as examples, not rates.
- MSE is a pixel measure. It rewards blur and ignores whether a small,
  important object (a cow) is right.
- The windows follow the trained policy, so they cover the states that
  policy visits, not all of Crafter.
