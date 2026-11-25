# World Model Project: DreamerV3 on Crafter

TUKE MSc group project. Supervisor: prof. Ing. Peter Sinčák, CSc.

Research question: can a learned latent world model support useful
prediction and control from visual experience?

We train [DreamerV3](https://github.com/danijar/dreamerv3) on the
[Crafter](https://github.com/danijar/crafter) game, then look inside it:
what it predicts, how it imagines, where it fails.

**Results dashboard:** https://deeplearningtukedreamerv3.github.io/dreamerv3-crafter/
(rebuilt from `main` on every merge by `.github/workflows/pages.yml`; local copy:
`python analysis/dashboard.py`, then open `dashboard.html`).

**Project recap page:** https://deeplearningtukedreamerv3.github.io/dreamerv3-crafter/recap.html
(source `docs/recap.html`, copied to the site by the same workflow).

**Gameplay at each training stage** (synced videos on the same worlds):
`python analysis/gameplay.py`, then open `gameplay.html`. The same workflow
publishes it as `gameplay.html` on the site once recordings are merged.

## Results (state on 2026-11-25)

All experiments are done; the 12 deliverables are ticked in
[`ROADMAP.md`](ROADMAP.md).

- Six runs of the official `crafter` config to ~495k env steps on one
  H200 each: 3 seeds with the default imagination horizon (15), 3 with
  horizon 5.
- Crafter score: random 1.4 %, early checkpoint 2.7 %, ~213k 4.3 %,
  ~495k **11.3 ± 4.9 %** (seeds 13.7 / 14.4 / 5.6 %). Paper: 14.5 % at 1M
  steps.
- The world model predicts frames better than "copy the last frame" from
  7 steps ahead, loses small objects first, and predicts rewards
  unreliably over 32 steps.

| Read | For |
|---|---|
| [`docs/report/report.md`](docs/report/report.md) | **Final technical report**: model, real vs imagined data, experiments, results, failures, limits |
| [`docs/report/demo.md`](docs/report/demo.md) | 10–15 min demo script |
| [`docs/recap.md`](docs/recap.md) | Whole project on one page (web version: [`recap.html`](https://deeplearningtukedreamerv3.github.io/dreamerv3-crafter/recap.html)) |
| [`docs/results.md`](docs/results.md) | Every evaluation number, paper comparison |
| [`docs/failures.md`](docs/failures.md) | Error analysis |
| [`docs/gpu-evidence.md`](docs/gpu-evidence.md) | PERUN jobs, GPU use, wall-clock |
| [`EXPERIMENTS.md`](EXPERIMENTS.md) | Every run with commit, flags and job ID |

To reproduce: [Quickstart](#quickstart-local) for the local setup, then
[`docs/hpc.md`](docs/hpc.md) and `slurm/` for the cluster runs. Every
figure's command is in the report's appendix.

## Start here (new member, in this order)

1. **This page**, to the end.
2. **[`docs/git-workflow.md`](docs/git-workflow.md)**,
   section "One-time setup". Then clone the repo.
3. **[Quickstart](#quickstart-local)** below: run `./setup.sh`, then the
   CPU debug run.
4. **[`ROADMAP.md`](ROADMAP.md)**: the 8-week plan. Read
   the overview table and the current week.
5. **[Project board](https://github.com/orgs/DeepLearningTukeDreamerV3/projects/1)**:
   pick an issue of the current week, assign yourself.
6. **[`docs/learning-path.md`](docs/learning-path.md)**:
   start step 1. New to deep learning? This is for you.

Then, when you need them:

| When | Read |
|---|---|
| Before your first PR | [`CONTRIBUTING.md`](CONTRIBUTING.md) |
| Unknown word | [`docs/glossary.md`](docs/glossary.md) |
| Before starting a training run | [`EXPERIMENTS.md`](EXPERIMENTS.md) |
| Docker or GPU problem | [`DOCKER.md`](DOCKER.md) |
| Reading the DreamerV3 code (week 3) | [`docs/architecture.md`](docs/architecture.md) |
| Original DreamerV3 README | [`upstream/README.md`](upstream/README.md) |
| Cluster (PERUN, SLURM) | [`docs/hpc.md`](docs/hpc.md) |

The assignment is not in this public repo. Ask the team for the PDF; keep
a local copy at `docs/assignment.{md,pdf}` (git-ignored).

## Repository layout

```
README.md            you are here
setup.sh             one-time setup for new members (./setup.sh)
AGENTS.md            rules for AI assistants
CLAUDE.md            imports AGENTS.md for Claude Code
.claude/skills/      Claude Code skills (perun-run: submit and collect PERUN runs)
.claude/settings.json, .claude/hooks/   what Claude may run; blocks raw ssh to PERUN
ROADMAP.md           8-week plan with checkboxes
CONTRIBUTING.md      branches, PRs, experiment naming
EXPERIMENTS.md       every run, including failed ones
docs/                learning path, glossary, git workflow, architecture, HPC,
                     recap.md (whole project on one page: how, results, what is left),
                     recap.html (same recap as one web page, on GitHub Pages),
                     results (evaluation table, paper comparison),
                     failure cases (Task 8), gpu-evidence.md (PERUN jobs, GPU use)
docs/report/         report.md (final technical report), demo.md (10–15 min demo script)
scripts/             setup check, env snapshot, random agent, issue list,
                     perun.sh (fixed PERUN commands, run from the laptop),
                     eval_agent.py (checkpoint or random agent on fixed worlds, Task 6),
                     gameplay_recorder.py (videos for eval_agent.py --video),
                     openloop.py (world model predicts frames alone, Task 4)
analysis/            plotting scripts and notebooks;
                     plot_curves.py (return + loss SVGs of one run, Task 3),
                     trace_batch.py (tensor shapes of one batch, Task 2),
                     dashboard.py (one offline HTML page: runs, evals, findings, roadmap),
                     failures.py (open-loop error growth and reward errors, Task 8),
                     compare_runs.py (two or three runs, or seed means, on the same charts, Task 7),
                     gpu_usage.py (GPU use and speed of PERUN runs, deliverable 8),
                     gameplay.py + gameplay_template.html (synced gameplay videos per training stage)
docs/figures/        figures for the report, named after the run ID
runs/                small logs of finished runs (metrics, scores, config)
slurm/               SLURM job scripts for the PERUN cluster
Dockerfile, compose.yaml, DOCKER.md   Docker setup
.env.example         defaults for compose; setup.sh copies it to .env
.devcontainer/       dev container for VS Code, Codespaces, PyCharm Pro
.github/             CI (large file check, dashboard, recap and gameplay on GitHub Pages), PR template, CODEOWNERS
LICENSE              MIT, from upstream
upstream/            danijar/dreamerv3 @ e3f0224, do not edit before M1
  dreamerv3/         agent, world model (rssm.py), configs.yaml
  embodied/          environments, replay, training loop
  scores/            published reference scores
  README.md          original DreamerV3 README
```

## Quickstart (local)

Needs Docker. GPU optional (NVIDIA Container Toolkit, see `DOCKER.md`).

Do not `pip install -r requirements.txt` on your machine. The pinned
packages need Python 3.11 and only install inside the Docker image.
See [Editor and Python](#editor-and-python).

```sh
git clone https://github.com/DeepLearningTukeDreamerV3/dreamerv3-crafter.git
cd dreamerv3-crafter

# 1. One-time setup: checks prerequisites, creates .env, builds the image
#    (20-40 min the first time), runs the local check, tests the GPU.
#    Safe to rerun. ./setup.sh --check only checks, changes nothing.
./setup.sh

# 2. CPU debug training (tiny network, stops with Ctrl+C).
docker compose --profile cpu up train-cpu

# 3. Look at the logs.
docker compose up viewer                 # http://localhost:8000
```

GPU runs (smoke test, baseline): use the exact commands in
[`ROADMAP.md`](ROADMAP.md), week 2.

## Editor and Python

Edit files with any editor. Run Python inside the Docker image: the whole
repo is mounted at `/app`, so edits are live without rebuilding.

```sh
docker compose --profile cpu run --rm train-cpu bash   # shell with everything installed
python scripts/check_local.py                          # inside that shell
```

For code completion and a debugger, open the repo in the dev container
(`.devcontainer/devcontainer.json`), CPU only:

- **VS Code** (free): install the *Dev Containers* extension, then
  "Dev Containers: Reopen in Container". First start builds the image.
  VS Code installed as a Flatpak: its terminal cannot see `docker`. Use a
  normal terminal, or prefix commands with `flatpak-spawn --host`.
- **GitHub Codespaces**: repo page → Code → Codespaces. Nothing to install
  locally; free hours for students with GitHub Education.
- **PyCharm Pro** (free for students): Remote Development → Dev Containers,
  or Python Interpreter → On Docker Compose, service `train-cpu`,
  interpreter `/venv/bin/python`. PyCharm Community cannot use Docker
  interpreters; use the shell above.

Files created inside the container belong to `root` on Linux hosts. Fix
with `sudo chown -R $USER: .` if needed.

## Issues offline

`scripts/issues_md.sh` writes `ISSUES.md` (git-ignored): every
issue grouped by roadmap week, with board status and assignees. Rerun to
refresh. Needs `gh` with the `project` scope (`gh auth refresh -h github.com -s project`).

## HPC

Real training runs on the PERUN cluster (TUKE) with SLURM and Apptainer:
[`docs/hpc.md`](docs/hpc.md). Never work on the login node: everything
except `git`, editing and job commands runs in a SLURM job.

## Upstream

Based on [danijar/dreamerv3](https://github.com/danijar/dreamerv3) at commit
`e3f02248693a79dc8b0ebd62c93683888ddaccfe` (MIT license). History was not
kept. All upstream files live in `upstream/`, unchanged, except
`Dockerfile`, `.dockerignore` and `.gitignore`, which moved to the repo
root and were adapted (DMLab install disabled, see `DOCKER.md`; paths
point to `upstream/`). `LICENSE` is copied at the root.

## GitHub

Org: [DeepLearningTukeDreamerV3](https://github.com/DeepLearningTukeDreamerV3).
New members are added to the `members` team, which is requested for review
on every PR (`.github/CODEOWNERS`). Branch rules on `main` are listed in
[`docs/git-workflow.md`](docs/git-workflow.md#rules-enforced-by-github-on-main).
