# Glossary

Plain-language definitions. Add terms when you meet new ones.

## Deep learning

- **Loss** — a number that says how wrong the model is. Training lowers it.
- **Gradient descent** — change the weights a little in the direction that
  lowers the loss. Repeat many times.
- **CNN** (convolutional neural network) — network for images. DreamerV3
  uses one as the **encoder** (image → vector) and the **decoder**
  (vector → image).
- **MLP** — plain stack of fully connected layers.
- **KL divergence** — distance between two probability distributions.
- **JAX** — Python library for numerical computing on GPU. DreamerV3 is
  written in it.
- **`jax.jit`** — compiles a function for speed. The first call is slow.
- **`jax.eval_shape`** — runs a function on placeholders that only have a
  shape and a dtype. Returns the output shapes without computing anything
  (`analysis/trace_batch.py`).
- **Logits** — raw scores before softmax. Softmax turns them into
  probabilities.
- **One-hot** — a vector of zeros with a single 1 that marks the chosen
  class.
- **Two-hot** — how DreamerV3's reward head and critic output a number:
  a distribution over 255 fixed bins, where the target value is split
  between the two nearest bins. Works for small and large values alike.
- **bfloat16** — 16-bit floating-point number. Half the memory of
  float32, faster on GPU, less precise. DreamerV3 computes in it.
- **Batch** (`B`, `batch_size`) and **sequence length** (`T`,
  `batch_length`) — one training step uses `B` sequences of `T`
  consecutive steps from the replay buffer (16 × 64 here).

## Reinforcement learning

- **Agent** — the learner that chooses actions.
- **Environment** — the world the agent acts in. Here: Crafter, a 2D
  Minecraft-like game with 64×64 images and 17 discrete actions.
- **Observation** — what the agent sees at one step (an image).
- **Action** — what the agent does at one step.
- **Reward** — number the environment gives after an action. Crafter gives
  +1 the first time each achievement is unlocked, so rewards are rare
  (**sparse**), and ±0.1 when health goes up or down.
- **False alarm** / **missed reward** — errors of the reward head in
  [`failures.md`](failures.md): it predicts an achievement that does not
  happen, or does not predict one that does.
- **Achievement** — one of Crafter's 22 goals (collect wood, place a
  table, make a wood pickaxe, defeat a zombie, ...). Many depend on
  earlier ones, like a tech tree.
- **Seed** — the number that fixes all random choices of a run (network
  start weights, worlds, sampling). Runs with different seeds show how
  much a result depends on luck; we report mean ± standard deviation
  across seeds (`docs/results.md`).
- **Episode** — one game from start to end.
- **Return** / **episode score** — sum of rewards in an episode.
- **Success rate** — share of episodes in which an achievement was
  unlocked at least once.
- **Crafter score** — the benchmark number in the Crafter and DreamerV3
  papers: geometric mean of the 22 success rates (in %),
  `exp(mean(ln(1 + s_i))) - 1`. Rewards rare achievements more than the
  plain return does. Computed by `scripts/eval_agent.py`.
- **Policy** / **actor** — function: state → action.
- **Value function** / **critic** — predicts the future return from a state.
- **Model-free RL** — learns policy and value directly from real experience.
- **Model-based RL** — also learns a model of the environment and uses it.

## World models and DreamerV3

- **World model** — network that predicts how the environment changes
  after an action.
- **Latent state** — compact vector the model uses instead of the image.
- **Encoder tokens** — the image compressed by the CNN encoder into one
  vector per step (4096 numbers in our config).
- **Feature** (`feat2tensor`) — `deter` and the flattened `stoch` side by
  side (10240 numbers). Every head reads it.
- **RSSM** (recurrent state-space model) — DreamerV3's world model
  (`upstream/dreamerv3/rssm.py`). Its state has two parts:
  - **deterministic state** (`deter`) — memory, updated by a recurrent network
  - **stochastic state** (`stoch`) — sampled categorical variables, models
    uncertainty
- **Posterior** — latent state computed *with* the current image
  (`observe`). Used on real data.
- **Prior** — latent state predicted *without* the image, from the previous
  state and action. Used in imagination (`imagine`).
- **`dyn` / `rep` losses** — KL terms that pull prior and posterior together.
- **Stop-gradient** (`sg`) — marks a value as a constant during training,
  so a loss does not change the networks that produced it. Decides which
  network each DreamerV3 loss trains (`docs/architecture.md`).
- **Free nats** — a KL below this value (1 nat) is clipped and gives no
  gradient, so the model does not over-fit the latent state.
- **λ-return** — target for the critic: a mix of real or imagined rewards
  over the next steps and the critic's own estimate further ahead
  (`lam: 0.95`).
- **Discount** (γ) — weight for future rewards: a reward `k` steps ahead
  counts `γ^k`. DreamerV3 sets γ = 1 − 1/`horizon` = 0.997.
- **Entropy** (of the policy) — how spread out the action choice is.
  Maximum (ln 17 ≈ 2.83 nats for Crafter) = uniformly random; 0 = always
  the same action.
- **REINFORCE** — policy-gradient rule: raise `log π(a)` of actions with
  a positive advantage, lower it for a negative one.
- **Advantage** — how much better an action turned out than the critic
  expected (return minus value). The actor raises the probability of
  actions with a positive advantage (REINFORCE-style policy gradient).
- **Slow critic** — a slowly updated copy of the critic, used as a stable
  target.
- **Reward head** / **continuation head** (`rew`, `con`) — predict the reward
  and whether the episode continues.
- **Imagination** / **imagined rollout** — sequence of prior states produced
  by the world model and the actor, without touching the real environment.
  Length: `imag_length` (15), the **imagination horizon**. Our
  sensitivity run uses 5 (`docs/results.md`).
- **Horizon 15 (default)** / **horizon 5** — our two settings. Horizon
  15 is the official `crafter` config, unchanged. Horizon 5 is the same
  config with `--agent.imag_length 5`. Both learn from real play and
  from imagination; only the imagination length differs. The horizon 15
  runs were first called "baseline", and their run IDs keep that word
  (`20261123-baseline-s0`, `20261124-baseline-s1`, `-s2`).
- **Sensitivity experiment** — a run that changes one setting only, to
  see its effect against the default (here: horizon 5 vs horizon 15).
- **Open-loop prediction** — start from real observations, then predict the
  future using only actions. Shown as `openloop/*` videos in the logs, and
  made from a checkpoint by `scripts/openloop.py` (Task 4).
- **Context** (open loop) — the real frames the world model sees before it
  predicts alone. `scripts/openloop.py`: first half of each window.
- **Horizon** (open loop) — how many steps ahead a prediction is. The error
  usually grows with it.
- **Mean squared error (MSE)** — average of (predicted − real)² over all
  pixels, with pixels scaled to 0..1. 0 = identical frames.
- **Replay buffer** — storage of past real experience used for training.
- **`train_ratio`** — replayed steps trained per environment step. Crafter
  default: 512.

## HPC

- **SLURM** — job scheduler on a cluster. See [`hpc.md`](hpc.md).
- **Partition** — group of nodes (e.g. GPU nodes).
- **Apptainer** (formerly Singularity) — container runtime on clusters.
  Runs `.sif` images converted from our Docker image.
- **Scratch** — fast, large, temporary cluster storage for logdirs. On
  PERUN: `/mnt/scratch/$USER`, deleted 60 days after last access.
- **Login node** — the machine you `ssh` into. Shared by all users: only
  edit, `git` and submit jobs there, never compute.
- **Compute node** — machine with CPUs/GPUs where SLURM runs your job.
- **`.sif`** — single-file Apptainer image, built from our Docker image.
- **GPU utilisation** — share of time at least one GPU kernel is running
  (nvidia-smi "GPU Utilization"). High (~90 %) means the GPU is the
  bottleneck. See [`gpu-evidence.md`](gpu-evidence.md).
- **Memory-bandwidth utilisation** — share of time GPU memory is being
  read or written. Not the same as how much memory is used.
- **Wall-clock time** — real elapsed time of a job, as on a clock.
