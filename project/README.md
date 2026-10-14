# World Model Project: DreamerV3 on Crafter

TUKE MSc group project. Supervisor: prof. Ing. Peter Sinčák, CSc.

Research question: can a learned latent world model support useful
prediction and control from visual experience?

- Assignment: [`docs/assignment.md`](docs/assignment.md)
- Plan: [`ROADMAP.md`](ROADMAP.md)
- How we work: [`CONTRIBUTING.md`](CONTRIBUTING.md)
- Run registry: [`EXPERIMENTS.md`](EXPERIMENTS.md)
- New to deep learning? [`docs/learning-path.md`](docs/learning-path.md), [`docs/glossary.md`](docs/glossary.md)

## Upstream

Based on [danijar/dreamerv3](https://github.com/danijar/dreamerv3) at commit
`e3f02248693a79dc8b0ebd62c93683888ddaccfe` (MIT license). History was not
kept. Local changes to upstream files: `Dockerfile` (DMLab install
disabled, see `DOCKER.md`), `.gitignore`.

## Repository layout

```
dreamerv3/, embodied/   upstream DreamerV3 code (commit e3f0224), do not edit before M1
compose.yaml, DOCKER.md Docker setup (local)
project/
  ROADMAP.md            8-week plan with checkboxes
  CONTRIBUTING.md       branches, PRs, experiment naming
  EXPERIMENTS.md        every run, including failed ones
  docs/                 assignment, learning path, glossary, architecture, HPC, report
  slurm/                SLURM job scripts for the cluster
  scripts/              env snapshot, local check, random-agent baseline
  analysis/             plotting scripts and notebooks
```

## Quickstart (local)

Needs Docker. GPU optional (NVIDIA Container Toolkit, see `DOCKER.md`).

```sh
git clone git@github.com:<org>/dreamerv3-crafter.git
cd dreamerv3-crafter
cp .env.example .env
docker compose build                     # long the first time

# 1. Local check: imports, env, config, one step.
docker compose --profile cpu run --rm train-cpu python project/scripts/check_local.py

# 2. CPU debug training (tiny network, stops with Ctrl+C).
docker compose --profile cpu up train-cpu

# 3. Look at the logs.
docker compose up viewer                 # http://localhost:8000
```

On a small GPU (4 GB) use `CONFIGS="crafter size12m"` in `.env`, then
`docker compose --profile gpu up train`.

## HPC

See [`docs/hpc.md`](docs/hpc.md).

## GitHub setup (once)

1. One member creates a free GitHub organization (github.com → + → New
   organization) and invites the others as **Owners**.
2. Create the empty repo `dreamerv3-crafter` in the org.
3. Push this repo:
   ```sh
   git remote add origin git@github.com:<org>/dreamerv3-crafter.git
   git push -u origin main
   ```
   The upstream code stays reachable through the `upstream` remote.
4. Create an org Project (Projects → New → Roadmap template), milestones
   M0–M5 and one issue per roadmap item.
