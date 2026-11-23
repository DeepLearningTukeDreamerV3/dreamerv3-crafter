# HPC and SLURM (PERUN)

Real training runs on **PERUN**, the TUKE cluster
([wiki](https://wiki.perun.tuke.sk/)). The laptop cannot hold the replay
buffer of the official baseline (~23 GB RAM for 1.1M steps, see
`EXPERIMENTS.md` run `20261017-smoke-s0`).

## Rule: never work on the login node

The supervisor was clear about this. The login node is shared by every
user; heavy work there slows everyone down.

| On the login node | In a SLURM job (`srun` / `sbatch`) |
|---|---|
| `ssh`, `git`, editing files | `python`, training, tests, analysis |
| `sbatch`, `srun`, `squeue`, `scancel`, `sacct`, `sinfo` | `apptainer exec`, `apptainer build` |
| reading logs (`tail`, `less`) | copying checkpoints or replay (GBs) |
| `rsync` / `scp` uploads | anything that runs longer than a few seconds |

Need a shell to try things? Ask for one on a compute node:
`srun -p cpu_short -A perun26012624 --qos=perun26012624 -c 4 --mem=8G -t 01:00:00 --pty bash`.

## What SLURM is

A cluster has a login node and many compute nodes with GPUs. You never
train on the login node. You ask SLURM, the job scheduler, for resources,
and it runs your job on a compute node when they are free.

| Command | Does |
|---|---|
| `srun --pty ... bash` | interactive job: a shell on a compute node, for testing |
| `sbatch file.sbatch` | queues a batch job; runs without you |
| `squeue -u $USER` | your jobs and their state (`PD` waiting, `R` running) |
| `scancel <jobid>` | stops a job |
| `sacct -j <jobid>` | info about a finished job (time, exit code) |
| `sinfo` | partitions and free nodes |

`#SBATCH` lines at the top of a `.sbatch` file request resources:
partition, number of GPUs (`--gres=gpu:1`), time limit, memory.

PERUN runs **Apptainer**, not Docker. We convert our Docker image to a
`.sif` file once and run everything inside it.

## PERUN facts

| What | Value |
|---|---|
| Login nodes | `login01.perun.tuke.sk`, `login02.perun.tuke.sk` (port 22) |
| Access | SSH key uploaded in the PERUN Portal ("SSH Keys"); VPN when off campus |
| GPU partitions | `gpu_short` (max 2 days), `gpu_long` (max 4 days); nodes with 8× NVIDIA H200 |
| CPU partitions | `cpu_short`, `cpu_long` |
| `/mnt/home/$USER` | 500 GB, not purged: repo clone and `dreamerv3.sif` |
| `/mnt/scratch/$USER` | fast Lustre storage, **deleted 60 days after last access**: logdirs (created by the admins; until it exists, `slurm/common.sh` uses home) |
| Project | `perun26012624` ("Learning and Testing a Simple World Model with DREAMERV3"). Every job sets `--account` and `--qos` to it |
| `/mnt/project/perun26012624` | 1000 GB shared team storage, not purged: results |
| Disk usage, project, job limits | `perunfsusage` |
| Choosing nodes | not allowed (`--nodelist` / `--exclude`) |
| Support | `hpc@helpdesk.tuke.sk` |

Our scripts use these values in `slurm/common.sh` (paths) and in the
`#SBATCH` lines (partition, project).

### SSH shortcut

Add to `~/.ssh/config` on your laptop (replace the username), then
`chmod 600 ~/.ssh/config`. After that `ssh perun`, `rsync ... perun:...`
work without the long names.

```
Host perun
    HostName login01.perun.tuke.sk
    User <your-perun-username>
    IdentityFile ~/.ssh/perun
    IdentitiesOnly yes
    ServerAliveInterval 60
```

Never share or upload the private key (`~/.ssh/perun`). Only the `.pub`
file goes to the portal.

## From the laptop: `scripts/perun.sh`

After the SSH shortcut works, most cluster tasks are one command on the
laptop. The script sends only commands allowed on the login node. The
checkpoint copy runs in a job through `srun`.

| Command | Does |
|---|---|
| `scripts/perun.sh status` | `squeue` + `sacct` of your jobs (last 7 days) |
| `scripts/perun.sh log <jobid> [lines]` | end of a job's `slurm-*-<jobid>.out` |
| `scripts/perun.sh fetch <run-id>` | small logs (or eval results) into `runs/<run-id>/`, warns over 5 MB |
| `scripts/perun.sh smoke <branch>` | pull the branch on the cluster, submit the smoke test |
| `scripts/perun.sh submit <branch> <run-id> <seed> <name> [flags]` | pull, then `sbatch slurm/02_train.sbatch` |
| `scripts/perun.sh early-ckpt <run-id> <step>` | copy `ckpt` to `ckpt_early_<step>`, in a job |
| `scripts/perun.sh cancel <jobid>` | `scancel` a job; a run resumes from its last checkpoint (saved every 15 min) |
| `scripts/perun.sh eval <branch> <eval-id> <run-id> <ckpt\|random> [episodes] [seed]` | pull, then `sbatch slurm/03_eval.sbatch` (Task 6, see below) |

`smoke`, `submit`, `early-ckpt`, `cancel` and `eval` change the cluster, so they ask y/N.
Without a terminal, put `-y` first: `scripts/perun.sh -y smoke main`.
The cluster clone is only updated with `git pull --ff-only`. Arguments
with spaces or shell characters are refused. The script gives up after 10 s
if PERUN cannot be reached (VPN off). If your key has a passphrase, run
`ssh-add ~/.ssh/perun` once per session: without a terminal the script
cannot ask for the passphrase.

**With Claude Code.** In Claude Code, type `! scripts/perun.sh status`.
The output then shows up in the chat, so you do not copy it by hand.
Claude may also run `status`, `log` and `fetch` without asking. It must ask you before
`smoke`, `submit`, `early-ckpt`, `cancel` and `eval`. `.claude/settings.json` and the hook
`.claude/hooks/block_perun.py` block every other way to reach PERUN
(`ssh`, `scp`, `rsync`, also inside `bash -c` or a new script). They also
stop Claude from changing these files without your approval. This
is a guard rail on the laptop, not a hard security boundary: review
changes to these files in PRs. After changing the hook, run its tests:
`python3 -I .claude/hooks/test_block_perun.py`.

## Build the image (on your laptop)

The laptop has Docker but no Apptainer, so we run Apptainer inside
Docker to convert the image. Do this outside the repo (the files are
large and must never be committed):

```sh
mkdir -p ~/hpc-build && cd ~/hpc-build
docker save dreamerv3:latest -o dreamerv3.tar
docker run --rm --privileged -v ~/hpc-build:/work --entrypoint apptainer \
  ghcr.io/apptainer/apptainer:1.3.6 \
  build /work/dreamerv3.sif docker-archive:///work/dreamerv3.tar
rsync -P dreamerv3.sif perun:~/        # -P: resumes if the VPN drops
```

Rebuild only when the `Dockerfile` or `upstream/requirements.txt`
changes. Code changes do not need a rebuild: the jobs mount the repo
clone over `/app`.

## Get the code on PERUN (login node)

```sh
cd ~ && git clone https://github.com/DeepLearningTukeDreamerV3/dreamerv3-crafter.git
cd dreamerv3-crafter
```

Submit jobs from this folder. `git pull` before each new run, and
register the run in `EXPERIMENTS.md` first.

## Staged workflow (assignment §5)

Do not skip stages. Production GPU time must not find basic errors.

1. **Local test** — `scripts/check_local.py` on your laptop.
2. **Interactive test** — ask for a GPU shell, then run the checks in it:
   ```sh
   srun -p gpu_short -A perun26012624 --qos=perun26012624 --gres=gpu:1 -c 8 --mem=32G -t 00:30:00 --pty bash   # login node
   bash slurm/00_interactive.sh                                            # GPU node
   exit
   ```
   Checks GPU visibility, JAX device, model init, a few train steps, logs.
3. **Smoke test** — `sbatch slurm/01_smoke.sbatch`. Debug config on
   GPU, 100k steps, ~11 min. Check that a checkpoint is written. Note steps per minute.
4. **Production** — `sbatch slurm/02_train.sbatch <seed> [run-name] [extra flags]`.

Register every job in `EXPERIMENTS.md` with its job ID. Watch a job
from the login node with `squeue -u $USER` and
`tail -f slurm-dreamer-train-<jobid>.out`.

## Long runs

`gpu_long` allows 4 days. If the baseline is not finished by then,
resubmit with the same run ID; DreamerV3 resumes from `<logdir>/ckpt`:

```sh
RUN_ID=<the-run-id> sbatch slurm/02_train.sbatch 0
```

Scratch is purged 60 days after last access. Copy the small logs to
`runs/<run-id>/` in git when the run ends.

## Early checkpoint for Task 6

Only the latest checkpoint is kept. Copy it early (~50k env steps), in a
job, not on the login node:

```sh
srun -p cpu_short -A perun26012624 --qos=perun26012624 -t 00:20:00 cp -r \
  /mnt/scratch/$USER/dreamer/logdir/<run-id>/ckpt \
  /mnt/scratch/$USER/dreamer/logdir/<run-id>/ckpt_early_<step>
```

## Evaluation (Task 6)

`slurm/03_eval.sbatch` runs `scripts/eval_agent.py` in a GPU job: N
episodes on the same Crafter worlds (fixed seed) for every agent, with
return, length and unlocked achievements per episode. Login node:

```sh
sbatch slurm/03_eval.sbatch <eval-id> <run-id> ckpt              # final checkpoint
sbatch slurm/03_eval.sbatch <eval-id> <run-id> ckpt_early_<step> # early checkpoint
sbatch slurm/03_eval.sbatch <eval-id> <run-id> random            # random agent
```

Optional 4th and 5th arguments: episodes (default 50), world seed
(default 100). Results go to `<logdir>/<eval-id>/eval.jsonl` and
`summary.json` (mean return, success rate per achievement, Crafter score).
