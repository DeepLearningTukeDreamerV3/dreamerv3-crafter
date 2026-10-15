#!/usr/bin/env bash
# One-time setup for a new team member. Run from the repo root after cloning:
#
#   ./setup.sh            full setup (builds the Docker image, 20-40 min first time)
#   ./setup.sh --check    only check prerequisites, change nothing
#
# Safe to run again: every step skips what is already done.
# Linux and macOS. On Windows, run it inside WSL2 with Docker Desktop.
set -uo pipefail

cd "$(dirname "$0")"

CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

ok()   { printf '  \033[32mok\033[0m    %s\n' "$*"; }
warn() { printf '  \033[33mwarn\033[0m  %s\n' "$*"; WARNINGS=$((WARNINGS + 1)); }
fail() { printf '  \033[31mfail\033[0m  %s\n' "$*"; FAILURES=$((FAILURES + 1)); }
step() { printf '\n\033[1m%s\033[0m\n' "$*"; }
WARNINGS=0
FAILURES=0

step "1. Prerequisites"
if [ -f /.flatpak-info ]; then
  warn "running inside a Flatpak sandbox (e.g. the VS Code Flatpak terminal):
          docker, gh and nvidia-smi are hidden here. Run ./setup.sh from a normal
          terminal, or here with: flatpak-spawn --host ./setup.sh"
fi
if command -v git >/dev/null; then ok "git $(git --version | cut -d' ' -f3)"; else fail "git missing"; fi
if command -v docker >/dev/null; then
  ok "docker $(docker version --format '{{.Client.Version}}' 2>/dev/null)"
  if docker info >/dev/null 2>&1; then
    ok "docker daemon reachable"
  else
    fail "cannot talk to the docker daemon. Start Docker, or on Linux run:
          sudo usermod -aG docker \$USER   (then log out and in again)"
  fi
  if docker compose version >/dev/null 2>&1; then
    ok "docker compose $(docker compose version --short)"
  else
    fail "docker compose v2 missing (the 'docker compose' subcommand)"
  fi
else
  fail "docker missing: https://docs.docker.com/get-docker/"
fi
if command -v gh >/dev/null; then
  if gh auth status >/dev/null 2>&1; then ok "gh logged in"
  else warn "gh not logged in. Run: gh auth login && gh auth setup-git"; fi
else
  warn "gh (GitHub CLI) missing. Optional, used in docs/git-workflow.md"
fi
GPU=0
if ! command -v nvidia-smi >/dev/null; then
  warn "no NVIDIA GPU found. Fine: CPU debug runs work; training needs a GPU machine"
elif SMI=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1); then
  GPU=1
  ok "NVIDIA GPU: $(echo "$SMI" | head -1)"
else
  warn "nvidia-smi fails: $(echo "$SMI" | head -1)
          'Driver/library version mismatch' after a driver update: reboot."
fi

step "2. Git identity (this repo)"
if [ -n "$(git config user.name)" ] && [ -n "$(git config user.email)" ]; then
  ok "$(git config user.name) <$(git config user.email)>"
else
  warn "git user.name / user.email not set. Run:
          git config user.name \"Your Name\"
          git config user.email \"you@example.com\""
fi

if [ "$CHECK_ONLY" = 1 ] || [ "$FAILURES" -gt 0 ]; then
  step "Summary"
  echo "  $FAILURES failure(s), $WARNINGS warning(s)."
  [ "$FAILURES" -gt 0 ] && echo "  Fix the failures above, then run ./setup.sh again."
  exit $((FAILURES > 0))
fi

step "3. Local config"
if [ -f .env ]; then ok ".env exists (kept)"; else cp .env.example .env && ok "created .env from .env.example"; fi
mkdir -p logdir && ok "logdir/ ready"

step "4. Docker image (long the first time)"
if docker compose build; then ok "image dreamerv3 built"; else fail "docker compose build failed"; fi

if [ "$FAILURES" -eq 0 ]; then
  step "5. Local check inside the image"
  if docker compose --profile cpu run --rm -T train-cpu python scripts/check_local.py; then
    ok "check_local.py passed"
  else
    fail "check_local.py failed (output above)"
  fi
fi

if [ "$FAILURES" -eq 0 ] && [ "$GPU" = 1 ]; then
  step "6. GPU inside Docker"
  if docker compose --profile gpu run --rm -T train python -c \
      "import jax; d = jax.devices(); print(d); assert d[0].platform == 'gpu'" 2>&1 | tail -3; then
    ok "JAX sees the GPU"
  else
    warn "JAX cannot use the GPU in Docker. Install the NVIDIA Container Toolkit, see DOCKER.md"
  fi
fi

step "Summary"
echo "  $FAILURES failure(s), $WARNINGS warning(s)."
if [ "$FAILURES" -eq 0 ]; then
  cat <<'EOF'

  Next:
    1. CPU debug run:  docker compose --profile cpu up train-cpu   (Ctrl+C to stop)
    2. Logs viewer:    docker compose up viewer   then http://localhost:8000
    3. Read README.md, section "Start here", from step 4.
EOF
fi
exit $((FAILURES > 0))
