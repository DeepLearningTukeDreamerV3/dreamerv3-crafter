#!/bin/bash
# Save everything needed to reproduce a run: commit, diff, versions, GPU.
# Usage: scripts/env_snapshot.sh <run-id>
# Output: runs/<run-id>/env.txt (commit this file).
set -uo pipefail

RUN_ID="${1:?run id required}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/runs/$RUN_ID"
mkdir -p "$OUT"

PY=python
GIT=git
if [ "${RUNTIME:-}" = native ]; then
  PY="$VENV/bin/python"
elif [ -n "${SIF:-}" ] && [ -f "${SIF:-}" ]; then
  PY="apptainer exec $SIF python"
  # Cluster compute nodes have no git; the image has it.
  if ! command -v git >/dev/null; then
    GIT="apptainer exec -B $ROOT $SIF git"
  fi
fi

{
  echo "# Run $RUN_ID"
  echo "date: $(date -Iseconds)"
  echo "host: $(hostname)"
  echo "slurm_job: ${SLURM_JOB_ID:-none}"
  echo
  echo "## Git"
  if [ "${RUNTIME:-}" = native ] && ! command -v git >/dev/null; then
    # No git on the node: read the commit from .git directly.
    HEAD_REF="$(cat "$ROOT/.git/HEAD")"
    echo "$HEAD_REF"
    case "$HEAD_REF" in
      ref:*) cat "$ROOT/.git/${HEAD_REF#ref: }" 2>/dev/null \
               || grep " ${HEAD_REF#ref: }\$" "$ROOT/.git/packed-refs" ;;
    esac
    echo "(git not available: no status or diff)"
  else
    $GIT -C "$ROOT" rev-parse HEAD
    $GIT -C "$ROOT" status --short
    $GIT -C "$ROOT" diff --stat
  fi
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
