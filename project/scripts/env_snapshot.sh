#!/bin/bash
# Save everything needed to reproduce a run: commit, diff, versions, GPU.
# Usage: project/scripts/env_snapshot.sh <run-id>
# Output: runs/<run-id>/env.txt (commit this file).
set -uo pipefail

RUN_ID="${1:?run id required}"
ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
OUT="$ROOT/runs/$RUN_ID"
mkdir -p "$OUT"

PY=python
if [ -n "${SIF:-}" ] && [ -f "${SIF:-}" ]; then
  PY="apptainer exec $SIF python"
fi

{
  echo "# Run $RUN_ID"
  echo "date: $(date -Iseconds)"
  echo "host: $(hostname)"
  echo "slurm_job: ${SLURM_JOB_ID:-none}"
  echo
  echo "## Git"
  git -C "$ROOT" rev-parse HEAD
  git -C "$ROOT" status --short
  git -C "$ROOT" diff --stat
  echo
  echo "## Python / JAX"
  $PY --version
  $PY -c "import jax, jaxlib; print('jax', jax.__version__, 'jaxlib', jaxlib.__version__)" 2>&1
  echo
  echo "## pip freeze"
  $PY -m pip freeze 2>&1
  echo
  echo "## GPU"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>&1 || echo "no GPU"
} > "$OUT/env.txt"

echo "Snapshot: $OUT/env.txt"
