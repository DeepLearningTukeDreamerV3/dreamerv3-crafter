---
name: perun-run
description: Register, submit, watch, resume or collect a DreamerV3 training run on the PERUN cluster (SLURM). Use when the user wants to start a run on PERUN, check a job, resume a run after the time limit, copy an early checkpoint, or bring a finished run's small logs back into runs/<run-id>/.
---

# Run on PERUN

Full background: `docs/hpc.md`. Rules: `AGENTS.md`.

The assistant runs on the user's laptop. It cannot run commands on
PERUN. Give the user the commands to paste, and label each block
**login node** or **in a job**. Never put `python`, `apptainer`, or a
large `cp` on the login node.

## 1. Before submitting (laptop)

1. Pick the run ID: `<YYYYMMDD>-<short-name>-s<seed>` (date of
   submission). Sensitivity runs change one flag only.
2. Add a row to `EXPERIMENTS.md` with status `planned`: run ID, date,
   commit, configs + flags, seed, `PERUN H200, job <pending>`.
3. Commit and push it on a branch (`exp/<short-name>`). The cluster
   clone must have the code the run uses.

## 2. Submit (login node)

```sh
cd ~/dreamerv3-crafter && git fetch && git switch <branch> && git pull
RUN_ID=<run-id> sbatch slurm/02_train.sbatch <seed> <short-name> [flags...]
```

Ask the user for the job ID that `sbatch` prints. Update the row:
`job <id>`, status `running`.

Staged tests come first for a new image or new scripts:
`slurm/00_interactive.sh` in an `srun` GPU shell, then
`sbatch slurm/01_smoke.sbatch` (`docs/hpc.md`).

## 3. Watch (login node)

```sh
squeue -u $USER
tail -n 30 slurm-dreamer-train-<jobid>.out
sacct -j <jobid> --format=JobID,State,Elapsed,ExitCode
```

`PD` = waiting for a GPU, `R` = running. Ask the user to paste
errors verbatim, then diagnose before changing anything.

## 4. Resume after the 4-day limit or a crash (login node)

Same run ID, same command; DreamerV3 continues from `<logdir>/ckpt`:

```sh
RUN_ID=<run-id> sbatch slurm/02_train.sbatch <seed> <short-name> [same flags]
```

Add the new job ID to the same row (`job <id1>, <id2>`).

## 5. Early checkpoint for Task 6 (in a job)

At ~50k env steps (`step` in `metrics.jsonl`):

```sh
L=/mnt/scratch/$USER/dreamer/logdir/<run-id>
srun -p cpu_short -t 00:20:00 cp -r $L/ckpt $L/ckpt_early_<step>
```

## 6. Collect results (laptop)

Only small files come back. Never checkpoints, replay or `.sif`.

```sh
mkdir -p runs/<run-id>
rsync -av perun:/mnt/scratch/<user>/dreamer/logdir/<run-id>/{metrics.jsonl,scores.jsonl,config.yaml} runs/<run-id>/
rsync -av perun:dreamerv3-crafter/runs/<run-id>/env.txt runs/<run-id>/
```

Check each file is under 5 MB (CI limit). Update the `EXPERIMENTS.md`
row: status `done` / `failed` / `killed`, wall-clock, notes. Commit on
the run's branch and open a PR.
