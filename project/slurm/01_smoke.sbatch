#!/bin/bash
# Stage 3: short GPU smoke test with the debug config.
# Usage: sbatch -p <partition> -A <account> project/slurm/01_smoke.sbatch
#SBATCH --job-name=dreamer-smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source project/slurm/common.sh

RUN_ID="$(date +%Y%m%d)-smoke-j${SLURM_JOB_ID:-local}"
bash project/scripts/env_snapshot.sh "$RUN_ID"

dreamer_exec python dreamerv3/main.py \
  --logdir "/logdir/$RUN_ID" \
  --configs crafter debug \
  --jax.platform cuda \
  --run.duration 1200

ls "$LOGROOT/$RUN_ID/ckpt"/*/done && echo "OK: checkpoint written"
