#!/bin/bash
# Stage 2: interactive check on a GPU node. Run inside an srun shell:
#   srun -p <partition> -A <account> --gres=gpu:1 -t 00:30:00 --pty bash
#   bash project/slurm/00_interactive.sh
set -euo pipefail
source "$(dirname "$0")/common.sh"

echo "== GPU visible on host"
nvidia-smi

echo "== JAX devices inside container"
dreamer_exec python -c "import jax; print(jax.devices()); assert jax.devices()[0].platform == 'gpu'"

echo "== Local checks inside container"
dreamer_exec python project/scripts/check_local.py

echo "== Model init + a few train steps + log writing (debug config on GPU)"
RUN_ID="$(date +%Y%m%d)-interactive"
dreamer_exec python dreamerv3/main.py \
  --logdir "/logdir/$RUN_ID" \
  --configs crafter debug \
  --jax.platform cuda \
  --run.steps 500

test -s "$LOGROOT/$RUN_ID/metrics.jsonl" && echo "OK: metrics written to $LOGROOT/$RUN_ID"
