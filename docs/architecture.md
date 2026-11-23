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

## Imagination settings (Task 5)

From `upstream/dreamerv3/configs.yaml`: `imag_length: 15`, `horizon: 333`,
`imag_loss.lam: 0.95`. Explain each.
