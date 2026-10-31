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

## Reinforcement learning

- **Agent** — the learner that chooses actions.
- **Environment** — the world the agent acts in. Here: Crafter, a 2D
  Minecraft-like game with 64×64 images and 17 discrete actions.
- **Observation** — what the agent sees at one step (an image).
- **Action** — what the agent does at one step.
- **Reward** — number the environment gives after an action. Crafter gives
  +1 per new achievement, so rewards are rare (**sparse**).
- **Episode** — one game from start to end.
- **Return** / **episode score** — sum of rewards in an episode.
- **Policy** / **actor** — function: state → action.
- **Value function** / **critic** — predicts the future return from a state.
- **Model-free RL** — learns policy and value directly from real experience.
- **Model-based RL** — also learns a model of the environment and uses it.

## World models and DreamerV3

- **World model** — network that predicts how the environment changes
  after an action.
- **Latent state** — compact vector the model uses instead of the image.
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
- **Reward head** / **continuation head** (`rew`, `con`) — predict the reward
  and whether the episode continues.
- **Imagination** / **imagined rollout** — sequence of prior states produced
  by the world model and the actor, without touching the real environment.
  Length: `imag_length` (15).
- **Open-loop prediction** — start from real observations, then predict the
  future using only actions. Shown as `openloop/*` videos in the logs.
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
- **Virtual environment** (venv) — a folder with its own Python and
  packages, separate from the system's. Our temporary non-container
  setup on PERUN uses `~/dreamer-venv`.
- **uv** — fast tool that installs Python versions and packages. Used by
  `slurm/setup_native.sh`.
