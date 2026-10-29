# Cluster settings shared by all SLURM scripts. Source this file.
# Values for the PERUN cluster (TUKE), see docs/hpc.md. The partition is
# set in the #SBATCH lines of each script, not here.

# Fast storage for logdirs (replay, checkpoints). Deleted 60 days after
# last access, so copy results to /mnt/project or the repo.
SCRATCH_DIR="${SCRATCH_DIR:-/mnt/scratch/$USER}"
# The image lives in home: home is not purged.
SIF="${SIF:-$HOME/dreamerv3.sif}"
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
# --no-home: do not mount $HOME, as the PERUN wiki recommends.
dreamer_exec() {
  apptainer exec --nv --no-home \
    -B "$REPO_DIR:/app" \
    -B "$LOGROOT:/logdir" \
    --pwd /app \
    "$SIF" \
    sh /app/upstream/entrypoint.sh "$@"
}
