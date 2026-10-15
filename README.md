# World Model Project: DreamerV3 on Crafter

TUKE MSc group project. Supervisor: prof. Ing. Peter Sinčák, CSc.

Research question: can a learned latent world model support useful
prediction and control from visual experience?

We train [DreamerV3](https://github.com/danijar/dreamerv3) on the
[Crafter](https://github.com/danijar/crafter) game, then look inside it:
what it predicts, how it imagines, where it fails.

## Start here (new member, in this order)

1. **This page**, to the end.
2. **[`project/docs/git-workflow.md`](project/docs/git-workflow.md)**,
   section "One-time setup". Then clone the repo.
3. **[Quickstart](#quickstart-local)** below: run `./setup.sh`, then the
   CPU debug run.
4. **[`project/ROADMAP.md`](project/ROADMAP.md)**: the 8-week plan. Read
   the overview table and the current week.
5. **[Project board](https://github.com/orgs/DeepLearningTukeDreamerV3/projects/1)**:
   pick an issue of the current week, assign yourself.
6. **[`project/docs/learning-path.md`](project/docs/learning-path.md)**:
   start step 1. New to deep learning? This is for you.

Then, when you need them:

| When | Read |
|---|---|
| Before your first PR | [`project/CONTRIBUTING.md`](project/CONTRIBUTING.md) |
| Unknown word | [`project/docs/glossary.md`](project/docs/glossary.md) |
| Before starting a training run | [`project/EXPERIMENTS.md`](project/EXPERIMENTS.md) |
| Docker or GPU problem | [`DOCKER.md`](DOCKER.md) |
| Reading the DreamerV3 code (week 3) | [`project/docs/architecture.md`](project/docs/architecture.md) |
| Original DreamerV3 README | [`UPSTREAM.md`](UPSTREAM.md) |
| Cluster (not used yet) | [`project/docs/hpc.md`](project/docs/hpc.md) |

The assignment is not in this public repo. Ask the team for the PDF; keep
a local copy at `project/docs/assignment.{md,pdf}` (git-ignored).

## Repository layout

```
README.md               you are here
setup.sh                one-time setup for new members (./setup.sh)
dreamerv3/, embodied/   upstream DreamerV3 code; do not edit before M1
compose.yaml, DOCKER.md Docker setup
UPSTREAM.md             original DreamerV3 README
project/
  ROADMAP.md            8-week plan with checkboxes
  CONTRIBUTING.md       branches, PRs, experiment naming
  EXPERIMENTS.md        every run, including failed ones
  docs/                 learning path, glossary, git workflow, architecture, HPC, report
  scripts/              local check, env snapshot, random agent, issue list
  analysis/             plotting scripts and notebooks
  slurm/                SLURM job scripts (unused until we get HPC access)
runs/                   small logs of finished runs (metrics, scores, config)
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
[`project/ROADMAP.md`](project/ROADMAP.md), week 2.

## Editor and Python

Edit files with any editor. Run Python inside the Docker image: the repo
folders `dreamerv3/`, `embodied/`, `project/`, `runs/` are mounted, so
edits are live without rebuilding.

```sh
docker compose --profile cpu run --rm train-cpu bash   # shell with everything installed
python project/scripts/check_local.py                  # inside that shell
```

For code completion and a debugger, open the repo in the dev container
(`.devcontainer/devcontainer.json`), CPU only:

- **VS Code** (free): install the *Dev Containers* extension, then
  "Dev Containers: Reopen in Container". First start builds the image.
- **GitHub Codespaces**: repo page → Code → Codespaces. Nothing to install
  locally; free hours for students with GitHub Education.
- **PyCharm Pro** (free for students): Remote Development → Dev Containers,
  or Python Interpreter → On Docker Compose, service `train-cpu`,
  interpreter `/venv/bin/python`. PyCharm Community cannot use Docker
  interpreters; use the shell above.

Files created inside the container belong to `root` on Linux hosts. Fix
with `sudo chown -R $USER: .` if needed.

## Issues offline

`project/scripts/issues_md.sh` writes `ISSUES.md` (git-ignored): every
issue grouped by roadmap week, with board status and assignees. Rerun to
refresh. Needs `gh` with the `project` scope (`gh auth refresh -h github.com -s project`).

## HPC

No cluster access for now; all training is local. Cluster instructions,
for later: [`project/docs/hpc.md`](project/docs/hpc.md).

## Upstream

Based on [danijar/dreamerv3](https://github.com/danijar/dreamerv3) at commit
`e3f02248693a79dc8b0ebd62c93683888ddaccfe` (MIT license). History was not
kept. Local changes to upstream files: `Dockerfile` (DMLab install
disabled, see `DOCKER.md`), `.gitignore`, `README.md` (moved to
`UPSTREAM.md`).

## GitHub

Org: [DeepLearningTukeDreamerV3](https://github.com/DeepLearningTukeDreamerV3).
New members are added to the `members` team, which is requested for review
on every PR (`.github/CODEOWNERS`). Branch rules on `main` are listed in
[`project/docs/git-workflow.md`](project/docs/git-workflow.md#rules-enforced-by-github-on-main).
