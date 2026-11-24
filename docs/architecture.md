# Architecture and data flow (Task 2, milestone M2)

Shapes below are measured, not guessed: `analysis/trace_batch.py` rebuilds
the network from the config of our baseline run
(`runs/20261123-baseline-s0/config.yaml`, official `crafter` config, not
`debug`) and follows one batch through the same calls as `Agent.loss`.
It uses `jax.eval_shape`, so it runs on a laptop CPU in seconds and needs
no checkpoint:

```sh
docker compose --profile cpu run --rm train-cpu python analysis/trace_batch.py
```

## Code map

| Component | File / function | Notes |
|---|---|---|
| Encoder | `upstream/dreamerv3/rssm.py:179` `Encoder` (call: `:210`) | 4 conv layers (depth 64×(2,3,4,4), kernel 5), each followed by a 2×2 max-pool: 64×64 image → 4×4×256 → flat token of 4096 |
| RSSM: observe (posterior, real data) | `upstream/dreamerv3/rssm.py:61` `observe`, step `:75` `_observe` | Per time step: `_core` updates `deter` from previous `deter`, `stoch`, action; then `deter` + image token → posterior logits → sample `stoch` |
| RSSM: core (sequence model) | `upstream/dreamerv3/rssm.py:135` `_core` | Block GRU, 8 blocks, `deter` 8192 |
| RSSM: prior | `upstream/dreamerv3/rssm.py:161` `_prior` | 2 MLP layers on `deter` only → prior logits (no image) |
| RSSM: imagine (prior, imagination) | `upstream/dreamerv3/rssm.py:94` `imagine` | Same `_core`, but `stoch` sampled from the prior; the action comes from the actor |
| Decoder | `upstream/dreamerv3/rssm.py:253` `Decoder` (call: `:288`) | Feature → reconstructed image; used only for the `image` loss |
| Reward head | `upstream/dreamerv3/agent.py:57` `self.rew` (MLPHead) | Feature → reward, 255-bin `symexp_twohot` output, used at `:173` (real) and `:205` (imagined) |
| Continuation head | `upstream/dreamerv3/agent.py:58` `self.con` (MLPHead) | Feature → probability the episode continues, used at `:177` and `:206` |
| Actor (policy) | `upstream/dreamerv3/agent.py:62` `self.pol` (MLPHead) | Feature → logits over 17 actions. Acting: `policy` `:115`; training: imagined features `:192`, `:207` |
| Critic (value) | `upstream/dreamerv3/agent.py:65` `self.val`, slow copy `:66` | Feature → expected λ-return, 255-bin output, used at `:208` (imagined) and `:228` (replay) |
| Losses | `upstream/dreamerv3/agent.py:156` `loss`, called by `train` `:137` | See "Losses" below |
| Report / open-loop video | `upstream/dreamerv3/agent.py:247` `report` | Task 4 |

"Feature" = `feat2tensor` (`agent.py:51`): `deter` and the flattened
`stoch` side by side, 8192 + 32×64 = 10240 numbers per time step. Every
head (reward, continuation, actor, critic, decoder) reads it.

## Tensors for one batch

From `analysis/trace_batch.py` with the baseline config. `B = 16`
sequences of `T = 64` steps are sampled from the replay buffer
(`batch_size`, `batch_length`). Imagination starts from **every** real
step, so `B·T = 1024` start states, each rolled out `H = 15` steps
(`imag_length`). `bfloat16` is a 16-bit float used for speed; the heads
output `float32`.

Real data (world-model training, `agent.py:164-177`):

| Name | Shape | dtype | Meaning |
|---|---|---|---|
| observation `image` | `(B, T, 64, 64, 3)` | uint8 | RGB frame, values 0–255 |
| `reward`, `is_first`, `is_last`, `is_terminal` | `(B, T)` | float32 / bool | Reward of the step; episode start; episode end; end by death (not time limit) |
| action (`prevact`) | `(B, T)` | int32 | Index of the action taken *before* each frame (17 Crafter actions) |
| encoder tokens | `(B, T, 4096)` | bfloat16 | Image compressed by the CNN encoder |
| deterministic state `deter` | `(B, T, 8192)` | bfloat16 | Recurrent memory of the RSSM (block GRU output) |
| posterior logits | `(B, T, 32, 64)` | bfloat16 | 32 categorical variables × 64 classes, computed *with* the image |
| prior logits | `(B, T, 32, 64)` | bfloat16 | Same, predicted from `deter` *without* the image; `dyn`/`rep` compare the two |
| stochastic state `stoch` | `(B, T, 32, 64)` | bfloat16 | One-hot sample of the posterior: 32 choices out of 64 |
| feature | `(B, T, 10240)` | bfloat16 | `deter` + flattened `stoch`, input of every head |
| reconstructed image | `(B, T, 64, 64, 3)` | float32 | Decoder output (mean), compared with `image / 255` |
| predicted reward | `(B, T)` | float32 | Reward head |
| continuation | `(B, T)` | float32 | Probability the episode goes on |
| critic value | `(B, T)` | float32 | Used by the `repval` loss on real data |

Imagination (actor and critic training, `agent.py:189-214`). No image,
no encoder, no decoder from here on: this is where real data stops.

| Name | Shape | dtype | Meaning |
|---|---|---|---|
| start `deter` | `(B·T, 8192)` = `(1024, 8192)` | bfloat16 | Every posterior state of the batch becomes a start point |
| imagined `deter` | `(1024, 15, 8192)` | bfloat16 | `_core` rolled forward 15 steps |
| imagined `stoch` | `(1024, 15, 32, 64)` | bfloat16 | Sampled from the **prior** (no image) |
| imagined feature | `(1024, 15, 10240)` | bfloat16 | Input of the heads |
| actor output (logits) | `(1024, 15, 17)` | float32 | Scores for the 17 actions; the action is sampled from them |
| imagined action | `(1024, 15)` | int32 | Sampled action per imagined step |
| predicted reward | `(1024, 15)` | float32 | Reward head on imagined features |
| continuation | `(1024, 15)` | float32 | Continuation head on imagined features |
| critic value | `(1024, 15)` | float32 | Critic on imagined features |

`Agent.loss` then puts the real start state in front of the 15 imagined
steps (`agent.py:196`), so the imagination losses see `H + 1 = 16` steps
per start (`assert` at `:200`).

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
