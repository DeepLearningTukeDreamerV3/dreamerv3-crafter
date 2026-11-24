# Architecture and data flow (Task 2, milestone M2)

Fill this in during week 3. Record real tensor shapes from a run with the
`crafter` config (not `debug`).

## Code map

| Component | File / function | Notes |
|---|---|---|
| Encoder | `upstream/dreamerv3/agent.py` | TODO |
| RSSM: observe (posterior, real data) | `upstream/dreamerv3/rssm.py:61` `observe` | TODO |
| RSSM: imagine (prior, imagination) | `upstream/dreamerv3/rssm.py:94` `imagine` | TODO |
| Decoder | `upstream/dreamerv3/agent.py` | TODO |
| Reward head | | TODO |
| Continuation head | | TODO |
| Actor (policy) | `upstream/dreamerv3/agent.py:115` `policy` | TODO |
| Critic (value) | | TODO |
| Losses | `upstream/dreamerv3/agent.py:137` `train` | TODO |
| Report / open-loop video | `upstream/dreamerv3/agent.py:247` `report` | TODO |

## Tensors for one batch

Batch size `B = 16`, sequence length `T = 64` (`configs.yaml` defaults).

| Name | Shape | dtype | Meaning |
|---|---|---|---|
| observation `image` | `(B, T, 64, 64, 3)` | uint8 | TODO |
| action | | | |
| encoder tokens | | | |
| deterministic state `deter` | | | |
| stochastic state `stoch` | | | |
| predicted reward | | | |
| continuation | | | |
| actor output | | | |
| critic value | | | |

## Losses

All losses are computed in `Agent.loss` (`upstream/dreamerv3/agent.py:156`).
They are multiplied by `loss_scales`, summed into **one** number
(`agent.py:240`) and minimised by **one** optimizer over all modules
(`agent.py:76`). What decides which network a loss trains is where the code
stops the gradient: `sg(x)` (`agent.py:17`, `jax.lax.stop_gradient`) makes
`x` a constant, so nothing before `x` learns from that loss.
`sg(x, skip=True)` does nothing.

| Loss key in `metrics.jsonl` | What it measures | Trains | Scale | Code |
|---|---|---|---|---|
| `train/loss/image` | how badly the decoder rebuilds the real image from the posterior latent state (pixel error) | decoder, RSSM, encoder | 1.0 (`rec`) | `agent.py:178-182` |
| `train/loss/rew` | how badly the reward head predicts the real reward | reward head, RSSM, encoder (`reward_grad: true`) | 1.0 | `agent.py:172-173` |
| `train/loss/con` | how badly the continuation head predicts "episode goes on" | continuation head, RSSM, encoder | 1.0 | `agent.py:174-177` |
| `train/loss/dyn` | KL from posterior (with image) to prior (without image); posterior frozen | **prior** (dynamics: predict the next state) | 1.0 | `rssm.py:125` |
| `train/loss/rep` | same KL; prior frozen | **posterior** and encoder (representation: stay predictable) | 0.1 | `rssm.py:126` |
| `train/loss/policy` | actor loss on imagined trajectories: raise the probability of actions with a high advantage, plus an entropy bonus | actor only | 1.0 | `agent.py:411-415` |
| `train/loss/value` | critic error on imagined trajectories vs the λ-return (and vs the slow critic) | critic only | 1.0 | `agent.py:417-422` |
| `train/loss/repval` | the same critic error, but on **real** replayed states | critic, and RSSM + encoder (`repval_grad: true`) | 0.3 | `agent.py:219-234`, `471` |

Notes:

- **World-model losses:** `image`, `rew`, `con`, `dyn`, `rep`. They
  use real data from the replay buffer only.
- **Imagination losses:** `policy`, `value`. Imagined states are wrapped
  in `sg` (`agent.py:196`, `ac_grads: false`), so these losses never
  change the world model. The actor learns with REINFORCE-style
  gradients (`logpi * advantage`), not by backpropagating through the
  model.
- **`dyn` and `rep` have the same value** in every log line: both are
  the KL between posterior and prior. They differ only in which side is
  frozen by `sg`, so `dyn` moves the prior towards the posterior and
  `rep` (10× weaker) moves the posterior towards the prior.
- **Free nats** (`free_nats: 1.0`, `rssm.py:127-129`): a KL below 1 nat
  is clipped to 1 and gives no gradient. A `dyn`/`rep` curve flat at 1.0
  means "prior and posterior agree well enough", not "nothing is
  learned". In the baseline the KL is still ~2 at 45k steps; in the
  debug smoke test it sat at the 1.0 floor.
- `image` is the largest number because it is a sum over 64×64×3 pixels.
  Compare its trend, not its size, with the other losses.

## Diagram

Original diagram (draw.io / Excalidraw, export as SVG to
`docs/figures/dataflow.svg`). It must show where real data stops and
imagination begins.

## Imagination: how the actor and critic learn (Task 5)

The actor and critic never train on real frames. They train on
**imagined rollouts**: sequences that the world model predicts from its
latent state, without the environment and without images. Code:
`Agent.loss` (`upstream/dreamerv3/agent.py:188-214`) and `imag_loss`
(`agent.py:382`). Settings from our run's `config.yaml`.

### Steps, for one training batch

1. **Start states.** Every posterior state of the replayed batch is a
   start: `B·T = 16·64 = 1024` starts (`imag_last: 0` means "all T
   steps"). `dyn.starts`, `agent.py:191`.
2. **Rollout.** 15 times (`imag_length`): the actor picks an action from
   the feature; the RSSM core updates `deter`; `stoch` is sampled from
   the **prior** (no image). `rssm.py:94` `imagine`. The real start state
   is put in front, so each rollout has 16 states (`agent.py:196`).
3. **Heads on every imagined state** (`agent.py:202-209`): reward head
   → reward `r`; continuation head → `c`, probability the episode goes
   on; critic → value `v`; slow critic → `v_slow`.
4. **λ-return** (`lambda_return`, `agent.py:482`), computed backwards
   from the last state:

   ```
   R_15 = v_15
   R_t  = r_t+1 + c_t+1 · ((1 − λ) · v_t+1 + λ · R_t+1)
   ```

   With λ = 0.95 the target trusts the imagined rewards most and the
   critic a little. The discount is inside `c`: with `contdisc: true` the
   continuation head is trained on `(1 − 1/horizon) · not terminal`
   (`agent.py:174-176`), so `c ≈ 0.997` on safe steps.
5. **Weight.** `weight = cumprod(c)` (`agent.py:402`): a step counts less
   when the model thinks the episode may already have ended.
6. **Critic loss `value`** (`agent.py:418-422`): the critic's 255-bin
   output is pulled toward `R_t` (cross-entropy), plus a second term
   that pulls it toward the slow critic (`slowreg: 1.0`). The slow critic
   follows the critic slowly (`slowvalue.rate: 0.02` per update). A second
   critic loss, `repval` (scale 0.3), trains the same critic on real
   replayed sequences.
7. **Actor loss `policy`** (`agent.py:407-415`): REINFORCE:
   `−log π(a_t) · advantage − 0.0003 · entropy`, with
   `advantage = (R_t − v_t) / S`. `S` is the spread of the returns, the
   95th minus 5th percentile and at least 1 (`retnorm: perc, limit 1`),
   so the same settings work for small and large rewards. The gradient
   only flows through `log π`, not through the world model.

**Nothing in steps 2–7 trains the world model**: the start states and
imagined features pass through a stop-gradient (`agent.py:196`,
`ac_grads: false`). The world model learns only from real data (losses
`image`, `rew`, `con`, `dyn`, `rep`).

### Settings

| Setting | Value | Meaning |
|---|---|---|
| `imag_length` | 15 | Imagined steps per rollout. Longer: the actor sees further ahead, but model errors pile up. Our sensitivity run uses 5 (`20261124-imag5-s0`) |
| `horizon` | 333 | Discount γ = 1 − 1/333 ≈ 0.997. A reward 333 steps ahead counts ≈ 0.997³³³ ≈ 0.37 of an immediate one. Rewards beyond 15 imagined steps enter only through the critic `v_15` |
| `imag_loss.lam` | 0.95 | λ of the λ-return: 1 = sum of imagined rewards + critic only at the end; 0 = one reward + critic at the next step |
| `imag_loss.actent` | 0.0003 | Entropy bonus: keeps the actor from becoming deterministic too early |
| `retnorm` | 5th–95th percentile, limit 1 | Scale of the advantage (step 7) |
| `slowvalue.rate` | 0.02 | How fast the slow critic follows the critic |

### Evidence from the baseline

Figure: `docs/figures/20261123-baseline-s0-imagination.svg`
(`analysis/plot_curves.py`). Top row: real side (episode return, three
world-model losses). Bottom row: imagination side. Values from
`runs/20261123-baseline-s0/metrics.jsonl`, mean of the first 3 vs the
last 10 log lines:

| Metric (`train/...`) | Start (~3k steps) | End (~213k) | Reading |
|---|---|---|---|
| `ent/action` actor entropy | 1.18 | 0.19 | Uniform over 17 actions would be ln 17 = 2.83 nats. The actor is almost deterministic in imagination at the end |
| `val` critic value | 1.33 | 1.66 | Expected return rises, like the real return (2.0 → 4.5 per episode) |
| `adv_mag` \|advantage\| | 0.081 | 0.034 | Imagined actions are rarely much better than the critic expects: the actor has settled |
| `weight` | 0.93 | 0.95 | Over 16 imagined steps the model rarely predicts the end of the episode |
| `rew` imagined reward per step | 0.036 | 0.035 | Real: 4.45 / 199 ≈ 0.022 per step (last 100 training episodes) |

The imagined reward per step is about 1.6 times the real one. This
suggests the world model is **optimistic** about reward, or the actor
finds rewards in imagination that it does not reach in the real game.
Task 4 (open-loop prediction) and Task 8 (reward-model errors) check this
on real trajectories.
