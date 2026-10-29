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
`srun -p cpu_short -c 4 --mem=8G -t 01:00:00 --pty bash`.

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
| `/mnt/scratch/$USER` | fast Lustre storage, **deleted 60 days after last access**: logdirs |
| `/mnt/project/...` | shared team storage for results |
| Disk usage | `perunfsusage` |

Our scripts use these values in `slurm/common.sh` (paths) and in the
`#SBATCH` lines (partitions).

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
   srun -p gpu_short --gres=gpu:1 -c 8 --mem=32G -t 00:30:00 --pty bash   # login node
   bash slurm/00_interactive.sh                                            # GPU node
   exit
   ```
   Checks GPU visibility, JAX device, model init, a few train steps, logs.
3. **Smoke test** — `sbatch slurm/01_smoke.sbatch`. Debug config on
   GPU, ~20 min. Check that a checkpoint is written. Note steps per minute.
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
srun -p cpu_short -t 00:20:00 cp -r \
  /mnt/scratch/$USER/dreamer/logdir/<run-id>/ckpt \
  /mnt/scratch/$USER/dreamer/logdir/<run-id>/ckpt_early_<step>
```
