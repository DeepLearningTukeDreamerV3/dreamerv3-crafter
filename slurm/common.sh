# Cluster settings shared by all SLURM scripts. Source this file.
# Values for the PERUN cluster (TUKE), see docs/hpc.md. The partition is
# set in the #SBATCH lines of each script, not here.

# Repo checkout on the cluster. Bind-mounted over /app so the job runs
# the committed code, not the code baked into the image.
# Found from this file's location: compute nodes have no git.
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Our home folder is the parent of the repo clone (~/dreamerv3-crafter).
# We do not trust $HOME: on PERUN compute nodes it can be wrong
# (/var/lib/slurm) while an account is not fully set up.
USER_HOME="$(dirname "$REPO_DIR")"
PERUN_USER="$(basename "$USER_HOME")"
if [ ! -d "${HOME:-}" ]; then
  export HOME="$USER_HOME"
fi

# Fast storage for logdirs (replay, checkpoints). Deleted 60 days after
# last access, so copy results to /mnt/project or the repo. Admins create
# /mnt/scratch/<user>; until they do, fall back to home (500 GB, enough).
if [ -z "${SCRATCH_DIR:-}" ]; then
  if [ -d "/mnt/scratch/$PERUN_USER" ]; then
    SCRATCH_DIR="/mnt/scratch/$PERUN_USER"
  else
    SCRATCH_DIR="$USER_HOME"
  fi
fi
# The image lives in home: home is not purged.
SIF="${SIF:-$USER_HOME/dreamerv3.sif}"
MODULES="${MODULES:-}"
LOGROOT="$SCRATCH_DIR/dreamer/logdir"

# How jobs run Python. apptainer (default): inside dreamerv3.sif.
# native: temporary workaround while Apptainer fails on PERUN; uses the
# environment built by slurm/setup_native.sh. Choose with
# `RUNTIME=native sbatch ...`.
RUNTIME="${RUNTIME:-apptainer}"
VENV="${VENV:-$USER_HOME/dreamer-venv}"
# The logdir root as the job sees it: the container mounts it at /logdir.
if [ "$RUNTIME" = native ]; then
  RUN_LOGROOT="$LOGROOT"
else
  RUN_LOGROOT=/logdir
fi
# Exported for scripts/env_snapshot.sh, which runs as a separate process.
export SIF RUNTIME VENV
echo "common.sh: repo $REPO_DIR, runtime $RUNTIME, logs $LOGROOT"

if [ -n "$MODULES" ]; then
  module load $MODULES
fi

mkdir -p "$LOGROOT"

# Run a command inside the container with GPU, code and logdir mounted.
# --no-home: do not mount $HOME, as the PERUN wiki recommends.
# With RUNTIME=native: run it from the repo with the venv's python.
dreamer_exec() {
  if [ "$RUNTIME" = native ]; then
    (cd "$REPO_DIR" && PATH="$VENV/bin:$PATH" PYTHONUNBUFFERED=1 \
      PYTHONDONTWRITEBYTECODE=1 "$@")
    return
  fi
  apptainer exec --nv --no-home \
    -B "$REPO_DIR:/app" \
    -B "$LOGROOT:/logdir" \
    --pwd /app \
    "$SIF" \
    sh /app/upstream/entrypoint.sh "$@"
}
