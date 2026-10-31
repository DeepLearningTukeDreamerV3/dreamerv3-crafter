#!/bin/bash
# Build a Python environment without Apptainer, for RUNTIME=native.
# Temporary workaround: while compute nodes cannot look up our user name,
# every `apptainer` command fails. Remove once Apptainer works.
# Run in a job, never on the login node (downloads and installs ~5 GB):
#   srun -p cpu_short -A perun26012624 --qos=perun26012624 -c 4 --mem=8G -t 01:00:00 bash slurm/setup_native.sh
set -euo pipefail
source "$(dirname "$0")/common.sh"

# uv: one-file installer that also downloads Python 3.11 (the nodes only
# have 3.9; our pinned packages need 3.11). Pinned version.
UV_VERSION=0.8.0
UV_DIR="$USER_HOME/.local/uv-$UV_VERSION"
if [ ! -x "$UV_DIR/uv" ]; then
  mkdir -p "$UV_DIR"
  curl -LsSf "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-x86_64-unknown-linux-gnu.tar.gz" \
    | tar -xz -C "$UV_DIR" --strip-components=1
fi
UV="$UV_DIR/uv"
export UV_CACHE_DIR="$USER_HOME/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$USER_HOME/.local/uv-python"

# --seed adds pip, so scripts/env_snapshot.sh can run `pip freeze`.
"$UV" venv --seed --python 3.11 "$VENV"
"$UV" pip install --python "$VENV/bin/python" -r "$REPO_DIR/slurm/requirements-native.txt"

"$VENV/bin/python" --version
"$VENV/bin/python" -c "import jax, crafter; print('jax', jax.__version__)"
echo "OK: environment ready in $VENV"
