#!/bin/bash
# Stage 4: production run. Official Crafter config unless extra flags given.
# Usage:
#   sbatch slurm/02_train.sbatch <seed> [run-name] [extra flags...]
# Examples:
#   sbatch slurm/02_train.sbatch 0                                   # official baseline
#   sbatch slurm/02_train.sbatch 0 imag5 --agent.imag_length 5       # sensitivity
# Resubmit the same command to resume (same run ID -> same logdir).
#SBATCH --job-name=dreamer-train
#SBATCH --partition=gpu_long
#SBATCH --account=perun26012624
#SBATCH --qos=perun26012624
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source slurm/common.sh

SEED="${1:?seed required}"
NAME="${2:-baseline}"
shift $(( $# >= 2 ? 2 : 1 ))

RUN_ID="${RUN_ID:-$(date +%Y%m%d)-$NAME-s$SEED}"
echo "Run ID: $RUN_ID (set RUN_ID=... to resume an older run)"
bash scripts/env_snapshot.sh "$RUN_ID"

dreamer_exec python upstream/dreamerv3/main.py \
  --logdir "$RUN_LOGROOT/$RUN_ID" \
  --configs crafter \
  --seed "$SEED" \
  "$@"
