# Cluster settings shared by all SLURM scripts. Source this file.
# TODO: fill in when we know the cluster (see docs/hpc.md).

PARTITION="${PARTITION:-CHANGE_ME_gpu_partition}"
ACCOUNT="${ACCOUNT:-CHANGE_ME_account}"
SCRATCH_DIR="${SCRATCH_DIR:-${SCRATCH:-$HOME}}"
SIF="${SIF:-$SCRATCH_DIR/dreamerv3.sif}"
MODULES="${MODULES:-}"

# Repo checkout on the cluster. Bind-mounted over /app so the job runs
# the committed code, not the code baked into the image.
REPO_DIR="${REPO_DIR:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
LOGROOT="$SCRATCH_DIR/dreamer/logdir"

if [ -n "$MODULES" ]; then
  module load $MODULES
fi

mkdir -p "$LOGROOT"

# Run a command inside the container with GPU, code and logdir mounted.
dreamer_exec() {
  apptainer exec --nv \
    -B "$REPO_DIR:/app" \
    -B "$LOGROOT:/logdir" \
    --pwd /app \
    "$SIF" \
    sh /app/upstream/entrypoint.sh "$@"
}
