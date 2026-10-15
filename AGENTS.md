# Instructions for AI coding assistants

Rules for any AI assistant (Claude Code, Codex, Copilot, Cursor, ...)
working in this repo. `CLAUDE.md` imports this file.

## Project

TUKE MSc group project: train DreamerV3 on Crafter and analyse the world
model. Plan: `project/ROADMAP.md`. Issues and board:
https://github.com/orgs/DeepLearningTukeDreamerV3/projects/1. Start page
for humans: `README.md`.

The team is new to deep learning. Every member must be able to explain
any code we submit (assignment §6). So:

- Explain what you change and why, in plain language. Define terms or
  point to `project/docs/glossary.md`.
- Prefer small, readable changes over clever ones. No new dependencies
  or abstractions unless asked.
- Do not do a member's learning task for them (`learning` label issues).
  Explain and point to sources instead.

## Running code

- Everything runs in the Docker image (Python 3.11, JAX). Never
  `pip install` on the host; the pinned packages do not install on other
  Python versions.
- Shell with everything installed:
  `docker compose --profile cpu run --rm train-cpu bash`
- Local check: `docker compose --profile cpu run --rm train-cpu python project/scripts/check_local.py`
- No HPC access for now. Do not plan steps that need a cluster or the
  supervisor. `project/slurm/` and `project/docs/hpc.md` are for later.
- GPU baseline: use the exact command in `project/ROADMAP.md` week 2. Do
  not use `docker compose up train` for real runs: it forces
  `--run.train_ratio 32` (the `crafter` config uses 512) and a new
  `{timestamp}` logdir on every start, so runs cannot resume.
- Executable scripts need a shebang (`#!/usr/bin/env python3`).

## Code rules

- Do not edit `dreamerv3/` or `embodied/` (upstream, commit `e3f0224`)
  before the baseline is reproduced (milestone M1). After that, one PR
  per change, with the reason.
- Prefer config flags (`--agent.imag_length 5`) over code changes.
- New code goes in `project/` (`scripts/`, `analysis/`).

## Documentation

Docs are part of the change, in the same PR, never "later":

- Changed how something is installed, run, named or located: update
  every doc that describes it (`README.md`, `project/ROADMAP.md`,
  `project/CONTRIBUTING.md`, `project/docs/*`, `DOCKER.md`, this file).
  `grep -rn` for the old name or command before opening the PR.
- New script or folder: add it to the layout in `README.md` with one line
  on what it does.
- New technical term in docs or code comments: add it to
  `project/docs/glossary.md`.
- PR finishes a roadmap item: tick its checkbox in `project/ROADMAP.md`.
- A doc you read is wrong or outdated: fix it or say so; do not work
  around it silently.

## Experiments

- Every run gets an ID `<date>-<short-name>-s<seed>` and a row in
  `project/EXPERIMENTS.md` **before** it starts.
- Keep failed and crashed runs in the registry with a note.
- One sensitivity experiment changes one flag only.
- Commit only small logs (`metrics.jsonl`, `scores.jsonl`, `config.yaml`)
  to `runs/<run-id>/` and plots. Never checkpoints, replay buffers,
  logdirs, `.sif`, `.env`, `.venv`.

## Git and GitHub

Full guide: `project/docs/git-workflow.md`.

- Never push to `main`. Branch `<type>/<short-name>` (`infra/`, `exp/`,
  `analysis/`, `docs/`), then a PR. Squash merge only.
- One topic per PR. PR body contains `Closes #N` for its issue.
- Stage files by name. Never `git add -A` blindly; never force-push.
- Files over 5 MB fail the "Large file check" CI.
- `gh` targets `DeepLearningTukeDreamerV3/dreamerv3-crafter`. The
  `upstream` remote (danijar/dreamerv3) is read-only; never open PRs
  there.
- Do not add AI attribution to commits or PRs: no `Co-Authored-By` AI
  trailer, no "Generated with ..." line. The member who runs the
  assistant is the author.
- Ask before anything hard to undo or visible to others: merging PRs,
  closing issues, changing repo or org settings, deleting files or
  branches.

## Private material

The repo is public. The assignment (`project/docs/assignment.{md,pdf}`)
is local only and git-ignored. Never commit it, quote it at length, or
copy the supervisor's contact details into the repo.
