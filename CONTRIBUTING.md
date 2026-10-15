# How we work

## Git

Step-by-step guide, conflicts and rescue: [`docs/git-workflow.md`](docs/git-workflow.md).

- `main` is protected. Every change goes through a pull request with one
  review from a teammate. Squash merge only.
- Branch names: `<type>/<short-name>`, e.g. `infra/slurm-smoke`,
  `exp/imag-length-5`, `analysis/loss-curves`, `docs/architecture`.
- Small PRs. One topic per PR.
- Commit messages: short imperative, e.g. `Add smoke test sbatch script`.
- Never commit checkpoints, replay buffers, `.sif` images or `.env`.
  Commit only small logs (`metrics.jsonl`, `scores.jsonl`, `config.yaml`)
  and plots.

## Upstream code

- Do not edit `upstream/dreamerv3/` or `upstream/embodied/` before the official baseline is
  reproduced (M1). After that, every change is a separate PR that explains
  why.
- Prefer config flags (`--agent.imag_length 5`) over code changes.

## Issues and board

- One GitHub issue per roadmap item. Assign yourself before starting.
- Labels: `infra`, `experiment`, `analysis`, `report`, `learning`.
- Milestones: M0–M5. Move cards on the project board.

## Experiments

Every run gets an ID and a row in `EXPERIMENTS.md` **before** it starts.

ID format: `<date>-<short-name>-s<seed>`, e.g. `20261027-baseline-s0`.

For every run, keep:

- exact git commit (`git rev-parse HEAD`)
- full command line
- config blocks and changed flags
- seed
- hardware (`nvidia-smi`), machine owner; cluster and job ID if on HPC
- environment snapshot (`scripts/env_snapshot.sh <run-id>`)
- start/end time, status

Failed and crashed runs stay in the registry with a note on what went
wrong (assignment §8: failures must be analysed, not hidden).

Logdir: `LOGDIR` in `.env` (default `./logdir`, git-ignored). On a cluster:
`$SCRATCH/dreamer/logdir/<run-id>`.
Copy `metrics.jsonl`, `scores.jsonl`, `config.yaml` to `runs/<run-id>/`
in git after the run.

## AI assistants

Allowed. Every member must be able to explain any code we submit
(assignment §6).
Rules the assistant must follow: [`AGENTS.md`](AGENTS.md) (Claude Code
reads it through `CLAUDE.md`).
