#!/bin/bash
# Task 6: evaluate a checkpoint, or a random agent, on fixed Crafter worlds.
# Every agent plays the same worlds (same --seed), so results compare.
# Usage:
#   sbatch slurm/03_eval.sbatch <eval-id> <run-id> <ckpt-folder|random> [episodes] [seed]
# Examples:
#   sbatch slurm/03_eval.sbatch 20261124-eval-final-s0 20261123-baseline-s0 ckpt
#   sbatch slurm/03_eval.sbatch 20261124-eval-early-s0 20261123-baseline-s0 ckpt_early_50120
#   sbatch slurm/03_eval.sbatch 20261124-eval-random 20261123-baseline-s0 random
# <run-id> is the training run whose config.yaml defines the network
# (for random: only the env settings are used).
# Results: <logdir>/<eval-id>/eval.jsonl and summary.json.
#SBATCH --job-name=dreamer-eval
#SBATCH --partition=gpu_short
#SBATCH --account=perun26012624
#SBATCH --qos=perun26012624
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source slurm/common.sh

EVAL_ID="${1:?eval-id required}"
RUN_ID="${2:?run-id required}"
CKPT="${3:?ckpt folder or random required}"
EPISODES="${4:-50}"
SEED="${5:-100}"
echo "Eval $EVAL_ID: run $RUN_ID, agent $CKPT, $EPISODES episodes, world seed $SEED"
bash scripts/env_snapshot.sh "$EVAL_ID"

if [ "$CKPT" = random ]; then AGENT=(--random); else AGENT=(--ckpt "$CKPT"); fi

dreamer_exec python scripts/eval_agent.py \
  --run-dir "/logdir/$RUN_ID" \
  "${AGENT[@]}" \
  --episodes "$EPISODES" \
  --seed "$SEED" \
  --platform cuda \
  --out "/logdir/$EVAL_ID/eval.jsonl"
