---
name: perun-run
description: Register, submit, watch, resume or collect a DreamerV3 training run on the PERUN cluster (SLURM). Use when the user wants to start a run on PERUN, check a job, resume a run after the time limit, copy an early checkpoint, or bring a finished run's small logs back into runs/<run-id>/.
---

# Run on PERUN

Full background: `docs/hpc.md`. Rules: `AGENTS.md`.

The assistant runs on the user's laptop and reaches PERUN **only**
through `scripts/perun.sh` (table in `docs/hpc.md`). Never call `ssh`,
`scp` or `rsync` directly: `.claude/settings.json` and
`.claude/hooks/block_perun.py` block it.

- Run `scripts/perun.sh status`, `log <jobid>` and `fetch <run-id>`
  yourself whenever you need them (read-only, no prompt).
- `smoke`, `submit`, `early-ckpt`, `cancel`, `eval` and `openloop` change the cluster. Run them with
  `-y` (`scripts/perun.sh -y submit ...`): the user approves the
  permission prompt. Or ask the user to type `! scripts/perun.sh ...`.
- Anything the script cannot do: give the user the raw commands to
  paste (sections below), each block labelled **login node** or
  **in a job**. Never put `python`, `apptainer`, or a large `cp` on the
  login node. If a task comes back often, propose a new subcommand in a
  PR instead.

## 1. Before submitting (laptop)

1. Pick the run ID: `<YYYYMMDD>-<short-name>-s<seed>` (date of
   submission). Sensitivity runs change one flag only.
2. Add a row to `EXPERIMENTS.md` with status `planned`: run ID, date,
   commit, configs + flags, seed, `PERUN H200, job <pending>`.
3. Commit and push it on a branch (`exp/<short-name>`). The cluster
   clone must have the code the run uses.

## 2. Submit (login node)

`scripts/perun.sh -y submit <branch> <run-id> <seed> <short-name> [flags...]`
prints the job ID. Fallback to paste:

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

`scripts/perun.sh status` and `scripts/perun.sh log <jobid> [lines]`.
Fallback to paste:

```sh
squeue -u $USER
tail -n 30 slurm-dreamer-train-<jobid>.out
sacct -j <jobid> --format=JobID,State,Elapsed,ExitCode
```

`PD` = waiting for a GPU, `R` = running. Read errors in the `log`
output verbatim, then diagnose before changing anything.

## 4. Resume after the 4-day limit or a crash (login node)

Same run ID, same command; DreamerV3 continues from `<logdir>/ckpt`.
`scripts/perun.sh -y submit` with the same arguments, or paste:

```sh
RUN_ID=<run-id> sbatch slurm/02_train.sbatch <seed> <short-name> [same flags]
```

Add the new job ID to the same row (`job <id1>, <id2>`).

To stop a run early (step budget reached): wait for a `Saved checkpoint`
line in `log` after the target step, then
`scripts/perun.sh -y cancel <jobid>`. Note the last checkpoint step in
the row.

## 5. Early checkpoint for Task 6 (in a job)

`scripts/perun.sh -y early-ckpt <run-id> <step>`. Fallback:

At ~50k env steps (`step` in `metrics.jsonl`):

```sh
L=/mnt/scratch/$USER/dreamer/logdir/<run-id>
srun -p cpu_short -A perun26012624 --qos=perun26012624 -t 00:20:00 cp -r $L/ckpt $L/ckpt_early_<step>
```

## 6. Collect results (laptop)

`scripts/perun.sh fetch <run-id>` does the copy and the 5 MB check.
Fallback:

Only small files come back. Never checkpoints, replay or `.sif`.

```sh
mkdir -p runs/<run-id>
rsync -av perun:/mnt/scratch/<user>/dreamer/logdir/<run-id>/{metrics.jsonl,scores.jsonl,config.yaml} runs/<run-id>/
rsync -av perun:dreamerv3-crafter/runs/<run-id>/env.txt runs/<run-id>/
```

Check each file is under 5 MB (CI limit). Update the `EXPERIMENTS.md`
row: status `done` / `failed` / `killed`, wall-clock, notes. Commit on
the run's branch and open a PR.
