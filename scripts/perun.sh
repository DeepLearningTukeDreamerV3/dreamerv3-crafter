#!/usr/bin/env bash
# Run a fixed set of PERUN commands from the laptop, over `ssh perun`.
#
# Why: so nobody has to copy commands into an ssh terminal and copy the
# output back. Each subcommand sends only commands that are allowed on
# the login node (squeue, sacct, tail, git, sbatch, srun). Heavy work
# (the checkpoint copy) runs inside a SLURM job through srun.
# There is no "run anything" subcommand on purpose.
#
# Needs the `perun` SSH alias from docs/hpc.md.
#
# Usage:
#   scripts/perun.sh status                       # your jobs, last 7 days
#   scripts/perun.sh log <jobid> [lines]          # end of a job's output
#   scripts/perun.sh fetch <run-id>               # small logs -> runs/<run-id>/
#   scripts/perun.sh smoke <branch>               # submit the smoke test
#   scripts/perun.sh submit <branch> <run-id> <seed> <name> [flags...]
#   scripts/perun.sh early-ckpt <run-id> <step>   # copy ckpt, in a job
#   scripts/perun.sh cancel <jobid>               # stop a job (scancel)
#   scripts/perun.sh eval <branch> <eval-id> <run-id> <ckpt|random> [episodes] [seed]
#   scripts/perun.sh openloop <branch> <openloop-id> <run-id> [ckpt] [episodes]
#
# smoke, submit, early-ckpt, cancel, eval and openloop change the cluster: they ask y/N first.
# Without a terminal (run from Claude Code) they refuse unless -y comes
# first, e.g. `scripts/perun.sh -y smoke main`.
set -euo pipefail

HOST="${PERUN_HOST:-perun}"
REPO="dreamerv3-crafter"    # the clone in the cluster home folder
PROJECT="perun26012624"     # SLURM account and QOS
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

die() { echo "perun.sh: $*" >&2; exit 1; }

# Accept only safe characters, so a value can never become a second
# shell command on the server (no spaces, ;, |, $, quotes, ...).
check() {  # check <what> <value> <regex>
  [[ "$2" =~ $3 ]] || die "bad $1: '$2'"
}
NAME_RE='^[A-Za-z0-9._/-]+$'
NUM_RE='^[0-9]+$'
FLAG_RE='^[A-Za-z0-9._=:/,+-]+$'

confirm() {
  echo "Will run on $HOST (login node):"
  echo "  $1"
  [ "$YES" = 1 ] && return 0
  [ -t 0 ] || die "no terminal to ask y/N: add -y before the subcommand"
  read -r -p "Continue? [y/N] " answer
  [[ "$answer" == "y" || "$answer" == "Y" ]] || die "cancelled"
}

# Fail after 10 s instead of hanging (VPN off, host down). Without a
# terminal, never wait for a password or passphrase prompt nobody sees:
# load the key first with `ssh-add ~/.ssh/perun`.
SSH_OPTS=(-o ConnectTimeout=10)
[ -t 0 ] || SSH_OPTS+=(-o BatchMode=yes)
export RSYNC_RSH="ssh ${SSH_OPTS[*]}"   # rsync uses the same options

# shellcheck disable=SC2029  # values are built here on purpose, then checked
remote() { ssh "${SSH_OPTS[@]}" "$HOST" "$1"; }

# Shell code that sets $L to the run's logdir on the cluster. Same rule
# as slurm/common.sh: scratch if it exists, else the home folder.
logdir_of() {
  echo "if [ -d /mnt/scratch/\$USER ]; then L=/mnt/scratch/\$USER/dreamer/logdir/$1; else L=\$HOME/dreamer/logdir/$1; fi; test -d \$L || { echo \"no logdir \$L\" >&2; exit 1; }"
}

# Update the cluster clone to a branch. --ff-only: never rewrite history.
pull() {
  echo "cd ~/$REPO && git fetch --quiet && git switch --quiet $1 && git pull --quiet --ff-only && git log --oneline -1"
}

YES=0
if [ "${1:-}" = "-y" ]; then YES=1; shift; fi
cmd="${1:-}"
[ $# -gt 0 ] && shift

case "$cmd" in
  status)
    [ $# -eq 0 ] || die "usage: status"
    # sacct needs the cluster's accounting database, which is sometimes
    # down. Then still show squeue and exit 0.
    remote "squeue -u \$USER; echo; sacct -u \$USER -S now-7days -X --format=JobID,JobName%20,State,Elapsed,ExitCode 2>/dev/null || echo '(sacct unavailable: job history not shown)'"
    ;;

  log)
    [ $# -ge 1 ] && [ $# -le 2 ] || die "usage: log <jobid> [lines]"
    check jobid "$1" "$NUM_RE"
    lines="${2:-30}"
    check lines "$lines" "$NUM_RE"
    remote "tail -n $lines ~/$REPO/slurm-*-$1.out"
    ;;

  fetch)
    [ $# -eq 1 ] || die "usage: fetch <run-id>"
    check run-id "$1" "$NAME_RE"
    dest="$LOCAL_ROOT/runs/$1"
    mkdir -p "$dest"
    # Find the logdir on the cluster, then copy only the small files.
    L="$(remote "$(logdir_of "$1"); echo \$L")"
    # Training runs have metrics/scores/config, evaluations eval/summary,
    # open-loop runs openloop.json and media/ (a few images).
    rsync -av --ignore-missing-args "$HOST:$L/metrics.jsonl" "$HOST:$L/scores.jsonl" \
      "$HOST:$L/config.yaml" "$HOST:$L/eval.jsonl" "$HOST:$L/summary.json" \
      "$HOST:$L/openloop.json" "$HOST:$L/media" "$dest/"
    rsync -av "$HOST:$REPO/runs/$1/env.txt" "$dest/" || echo "no env.txt for $1"
    # The repo CI rejects files over 5 MB.
    big="$(find "$dest" -type f -size +5M)"
    [ -z "$big" ] || echo "WARNING: over 5 MB, do not commit as is: $big"
    ;;

  smoke)
    [ $# -eq 1 ] || die "usage: smoke <branch>"
    check branch "$1" "$NAME_RE"
    c="$(pull "$1") && sbatch slurm/01_smoke.sbatch"
    confirm "$c"
    remote "$c"
    ;;

  submit)
    [ $# -ge 4 ] || die "usage: submit <branch> <run-id> <seed> <name> [flags...]"
    check branch "$1" "$NAME_RE"
    check run-id "$2" "$NAME_RE"
    check seed "$3" "$NUM_RE"
    check name "$4" "$NAME_RE"
    branch="$1" run_id="$2" seed="$3" name="$4"
    shift 4
    for f in "$@"; do check flag "$f" "$FLAG_RE"; done
    c="$(pull "$branch") && RUN_ID=$run_id sbatch slurm/02_train.sbatch $seed $name $*"
    confirm "$c"
    remote "$c"
    echo "Add the job ID above to the run's row in EXPERIMENTS.md."
    ;;

  early-ckpt)
    [ $# -eq 2 ] || die "usage: early-ckpt <run-id> <step>"
    check run-id "$1" "$NAME_RE"
    check step "$2" "$NUM_RE"
    # The copy is GBs: it runs on a CPU node through srun, not on the login node.
    c="$(logdir_of "$1"); srun -p cpu_short -A $PROJECT --qos=$PROJECT -t 00:20:00 cp -r \$L/ckpt \$L/ckpt_early_$2"
    confirm "$c"
    remote "$c"
    ;;

  cancel)
    # Stops a job. A training run keeps its last checkpoint (saved every
    # 15 min) and resumes from it when resubmitted with the same run ID.
    [ $# -eq 1 ] || die "usage: cancel <jobid>"
    check jobid "$1" "$NUM_RE"
    c="scancel $1"
    confirm "$c"
    remote "$c"
    ;;

  eval)
    # Task 6: evaluate a checkpoint folder of <run-id> (ckpt,
    # ckpt_early_<step>) or a random agent, in a GPU job (slurm/03_eval.sbatch).
    [ $# -ge 4 ] && [ $# -le 6 ] || die "usage: eval <branch> <eval-id> <run-id> <ckpt|random> [episodes] [seed]"
    check branch "$1" "$NAME_RE"
    check eval-id "$2" "$NAME_RE"
    check run-id "$3" "$NAME_RE"
    check ckpt "$4" "$NAME_RE"
    episodes="${5:-50}" seed="${6:-100}"
    check episodes "$episodes" "$NUM_RE"
    check seed "$seed" "$NUM_RE"
    c="$(pull "$1") && sbatch slurm/03_eval.sbatch $2 $3 $4 $episodes $seed"
    confirm "$c"
    remote "$c"
    echo "Add the job ID above to the eval's row in EXPERIMENTS.md."
    ;;

  openloop)
    # Task 4: open-loop predictions of a checkpoint folder of <run-id>,
    # in a GPU job (slurm/04_openloop.sbatch).
    [ $# -ge 3 ] && [ $# -le 5 ] || die "usage: openloop <branch> <openloop-id> <run-id> [ckpt] [episodes]"
    check branch "$1" "$NAME_RE"
    check openloop-id "$2" "$NAME_RE"
    check run-id "$3" "$NAME_RE"
    ckpt="${4:-ckpt}" episodes="${5:-10}"
    check ckpt "$ckpt" "$NAME_RE"
    check episodes "$episodes" "$NUM_RE"
    c="$(pull "$1") && sbatch slurm/04_openloop.sbatch $2 $3 $ckpt $episodes"
    confirm "$c"
    remote "$c"
    echo "Add the job ID above to the run's row in EXPERIMENTS.md."
    ;;

  *)
    sed -n '12,21p' "$0" >&2   # the usage lines at the top of this file
    exit 1
    ;;
esac
