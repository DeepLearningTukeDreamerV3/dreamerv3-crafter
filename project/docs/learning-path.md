# Learning path

For a team new to deep learning. Each step lists what to read or watch and
what you must be able to explain afterwards. Do the steps in order. Discuss
the checklists together in a short team meeting at the end of each step.

Terms in **bold** are explained in [`glossary.md`](glossary.md).

## Step 1 — Neural networks (week 1)

- 3Blue1Brown, *Neural Networks* video series (chapters 1–4):
  https://www.3blue1brown.com/topics/neural-networks
- JAX tutorial "JAX 101" / "Thinking in JAX": https://docs.jax.dev/
  (only to read code; we do not write JAX models from scratch)

You can explain:

- [ ] what a layer, weight, activation and **loss** are
- [ ] what **gradient descent** and backpropagation do
- [ ] what a **CNN** encoder does with an image
- [ ] what `jax.jit` and `jax.grad` do

## Step 2 — Reinforcement learning (week 1–2)

- OpenAI *Spinning Up*, "Part 1: Key Concepts in RL" and "Part 2: Kinds of
  RL Algorithms": https://spinningup.openai.com/
- Optional: Sutton & Barto, *Reinforcement Learning: An Introduction*,
  chapters 1, 3 and 13: http://incompleteideas.net/book/the-book.html

You can explain:

- [ ] **agent**, **environment**, **observation**, **action**, **reward**,
      **episode**, **return**
- [ ] **policy** (actor) and **value function** (critic)
- [ ] model-free vs **model-based** RL

## Step 3 — World models (week 2–3)

- Ha & Schmidhuber, *World Models* (2018), interactive article:
  https://worldmodels.github.io/
- Ding et al. (2025) survey, intro and taxonomy:
  https://doi.org/10.1145/3746449

You can explain:

- [ ] what a **world model** is and why it is useful
- [ ] **latent state** vs observation
- [ ] **imagination** / dreaming: training a policy inside the model

## Step 4 — DreamerV3 (week 3)

- DreamerV3 project page: https://danijar.com/project/dreamerv3/
- Paper: https://arxiv.org/abs/2301.04104 (sections 1–3, figures 2–3)
- Code: `dreamerv3/rssm.py` (world model), `dreamerv3/agent.py` (losses,
  actor, critic), `dreamerv3/configs.yaml`

You can explain (assignment §8 requires this):

- [ ] the difference between observation, latent state, predicted next
      state, reward prediction, imagined rollout and action
- [ ] **RSSM**: deterministic + stochastic state; **posterior** vs **prior**
- [ ] each world-model loss: `image` (reconstruction), `rew`, `con`,
      `dyn`, `rep`
- [ ] how actor and critic train on imagined trajectories of length
      `imag_length`
- [ ] what `train_ratio` means

## Step 5 — Surveys for the report (weeks 4–7)

At least one general and one robotics survey must be cited (assignment A.11):
Ding et al. (general) and Hou et al. (robot learning).

- Ding et al. (2025), general survey: <https://doi.org/10.1145/3746449>
- Hou et al. (2026), robot learning: <https://arxiv.org/abs/2605.00080>
- Li et al. (2025), embodied AI: <https://arxiv.org/abs/2510.16732>
- Kirchner et al. (2026), physical AI: <https://doi.org/10.1007/s44163-026-02122-1>
- Optional: Guan et al. (2024) <https://doi.org/10.1109/TIV.2024.3398357>,
  Long et al. (2025) <https://arxiv.org/abs/2507.00917>

Suggested order: Ding → Hou → Li → Kirchner.
