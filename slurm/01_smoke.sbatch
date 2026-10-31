#!/bin/bash
# Stage 3: short GPU smoke test with the debug config.
# Usage: sbatch slurm/01_smoke.sbatch
#SBATCH --job-name=dreamer-smoke
#SBATCH --partition=gpu_short
#SBATCH --account=perun26012624
#SBATCH --qos=perun26012624
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source slurm/common.sh

RUN_ID="$(date +%Y%m%d)-smoke-j${SLURM_JOB_ID:-local}"
bash scripts/env_snapshot.sh "$RUN_ID"

dreamer_exec python upstream/dreamerv3/main.py \
  --logdir "$RUN_LOGROOT/$RUN_ID" \
  --configs crafter debug \
  --jax.platform cuda \
  --run.duration 1200

ls "$LOGROOT/$RUN_ID/ckpt"/*/done && echo "OK: checkpoint written"
