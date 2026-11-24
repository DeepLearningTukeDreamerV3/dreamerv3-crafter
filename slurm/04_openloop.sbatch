#!/bin/bash
# Task 4: open-loop predictions of a trained world model.
# The world model sees 32 real frames, then predicts 32 more from the
# real actions alone. Writes images and the error per step.
# Usage:
#   sbatch slurm/04_openloop.sbatch <openloop-id> <run-id> [ckpt-folder] [episodes] [seed]
# Example:
#   sbatch slurm/04_openloop.sbatch 20261124-openloop-s0 20261123-baseline-s0 ckpt
# Results: <logdir>/<openloop-id>/openloop.json and media/openloop.{gif,png}.
#SBATCH --job-name=dreamer-openloop
#SBATCH --partition=gpu_short
#SBATCH --account=perun26012624
#SBATCH --qos=perun26012624
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source slurm/common.sh

OL_ID="${1:?openloop-id required}"
RUN_ID="${2:?run-id required}"
CKPT="${3:-ckpt}"
EPISODES="${4:-10}"
SEED="${5:-100}"
echo "Open loop $OL_ID: run $RUN_ID, checkpoint $CKPT, $EPISODES episodes, world seed $SEED"
bash scripts/env_snapshot.sh "$OL_ID"

dreamer_exec python scripts/openloop.py \
  --run-dir "/logdir/$RUN_ID" \
  --ckpt "$CKPT" \
  --episodes "$EPISODES" \
  --seed "$SEED" \
  --platform cuda \
  --out "/logdir/$OL_ID"
