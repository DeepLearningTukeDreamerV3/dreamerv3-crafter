# HPC and SLURM

> **Status: not used yet.** The team has no cluster access. All runs are
> local (see `ROADMAP.md`). This page and `project/slurm/` are kept for
> when access becomes available.

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
partition, account, number of GPUs (`--gres=gpu:1`), time limit, memory.

Clusters run **Apptainer**, not Docker. We convert our Docker image to a
`.sif` file once and run everything inside it.

## Fill in once we know the cluster

Edit `project/slurm/common.sh`:

| Variable | Ask IT / docs | Example |
|---|---|---|
| `PARTITION` | GPU partition name | `gpu` |
| `ACCOUNT` | project / billing account | `p123-26-t` |
| `SCRATCH_DIR` | fast storage path | `/scratch/$USER` |
| `SIF` | path to the `.sif` image | `$SCRATCH_DIR/dreamerv3.sif` |
| `MODULES` | module to load Apptainer, if any | `apptainer` |

## Build the image

On a machine with Docker (see `DOCKER.md`, section HPC):

```sh
docker compose build
docker save dreamerv3:latest -o dreamerv3.tar
apptainer build dreamerv3.sif docker-archive://dreamerv3.tar
scp dreamerv3.sif <user>@<cluster>:<scratch>/
```

## Staged workflow (assignment §5)

Do not skip stages. Production GPU time must not find basic errors.

1. **Local test** — `project/scripts/check_local.py` on your laptop.
2. **Interactive test** — on the cluster:
   ```sh
   srun -p <partition> -A <account> --gres=gpu:1 -t 00:30:00 --pty bash
   bash project/slurm/00_interactive.sh
   ```
   Checks GPU visibility, JAX device, model init, a few train steps, logs.
3. **Smoke test** — `sbatch project/slurm/01_smoke.sbatch`. Debug config on
   GPU, ~20 min. Check that a checkpoint is written. Note steps per minute.
4. **Production** — `sbatch project/slurm/02_train.sbatch <seed> [extra flags]`.

Register every job in `EXPERIMENTS.md` with its job ID.

## Long runs

Time limits are often 24–48 h. The Crafter baseline may need longer.
DreamerV3 resumes from `<logdir>/ckpt` when started with the same
`--logdir`, so resubmit the same command with the same run ID.

## Early checkpoint for Task 6

Only the latest checkpoint is kept. Copy it early:

```sh
cp -r $SCRATCH_DIR/dreamer/logdir/<run-id>/ckpt \
      $SCRATCH_DIR/dreamer/logdir/<run-id>/ckpt_early_<step>
```
