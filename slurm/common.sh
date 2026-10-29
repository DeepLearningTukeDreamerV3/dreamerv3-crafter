# Cluster settings shared by all SLURM scripts. Source this file.
# Values for the PERUN cluster (TUKE), see docs/hpc.md. The partition is
# set in the #SBATCH lines of each script, not here.

# Repo checkout on the cluster. Bind-mounted over /app so the job runs
# the committed code, not the code baked into the image.
REPO_DIR="${REPO_DIR:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"

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
echo "common.sh: repo $REPO_DIR, image $SIF, logs $LOGROOT"

if [ -n "$MODULES" ]; then
  module load $MODULES
fi

mkdir -p "$LOGROOT"

# Run a command inside the container with GPU, code and logdir mounted.
# --no-home: do not mount $HOME, as the PERUN wiki recommends.
dreamer_exec() {
  apptainer exec --nv --no-home \
    -B "$REPO_DIR:/app" \
    -B "$LOGROOT:/logdir" \
    --pwd /app \
    "$SIF" \
    sh /app/upstream/entrypoint.sh "$@"
}
