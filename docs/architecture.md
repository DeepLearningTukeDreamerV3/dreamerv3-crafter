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

| Loss key in `metrics.jsonl` | Trains | Scale (`loss_scales`) |
|---|---|---|
| `train/loss/image` | encoder, RSSM, decoder | 1.0 |
| `train/loss/rew` | | 1.0 |
| `train/loss/con` | | 1.0 |
| `train/loss/dyn` | | 1.0 |
| `train/loss/rep` | | 0.1 |
| `train/loss/policy` | | 1.0 |
| `train/loss/value` | | 1.0 |
| `train/loss/repval` | | 0.3 |

## Diagram

Original diagram (draw.io / Excalidraw, export as SVG to
`docs/figures/dataflow.svg`). It must show where real data stops and
imagination begins.

## Imagination settings (Task 5)

From `upstream/dreamerv3/configs.yaml`: `imag_length: 15`, `horizon: 333`,
`imag_loss.lam: 0.95`. Explain each.
