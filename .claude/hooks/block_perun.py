#!/usr/bin/env python3
"""Claude Code hook: keep the AI assistant away from raw SSH to PERUN.

Runs before every Bash command the assistant wants to run (PreToolUse).
The only allowed way to reach the cluster is scripts/perun.sh, which
sends a fixed set of login-node-safe commands. This hook blocks:

- any other command that uses ssh, scp, sftp or rsync, or names the
  cluster host (also when hidden inside `bash -c "..."` or Python);
- a scripts/perun.sh call chained with other commands (; & | $( `);
- shell commands that change scripts/perun.sh or this guard
  (sed -i, >, tee, mv, cp, rm, chmod, ...);
- writing a script file (.sh, .py, .sbatch, ...) that uses ssh, so the
  assistant cannot hide an ssh call in a file and then run that file.

Exit code 2 tells Claude Code to block the command and show the reason
to the assistant. The user's own `!` commands do not go through hooks.
This is a guard rail on the laptop, not a hard security boundary.
"""
import json
import re
import sys

SAFE_CALL = re.compile(
    r"^\s*(\./)?scripts/perun\.sh( -y)? "
    r"(status|log|fetch|smoke|submit|early-ckpt)( [A-Za-z0-9._=:/,+-]+)*\s*$"
)
REMOTE = re.compile(r"\b(ssh|scp|sftp|rsync)\b|perun\.tuke\.sk|\bperun:")
GUARDED = re.compile(r"perun\.sh|\.claude/(settings|hooks)")
WRITES = re.compile(r"sed\s+-i|>|\btee\b|\bmv\b|\bcp\b|\brm\b|\bchmod\b|\bln\b|python|perl|truncate")


SCRIPT = re.compile(r"\.(sh|bash|py|sbatch|pl|rb|js)$")
# These files may mention ssh. The settings ask the user before they change.
EXEMPT = re.compile(r"(scripts/perun\.sh|\.claude/hooks/[^/]+)$")


def check_file_write(tool_input):
    path = tool_input.get("file_path", "")
    text = tool_input.get("content", "") + tool_input.get("new_string", "")
    if EXEMPT.search(path):
        return 0
    if SCRIPT.search(path) and REMOTE.search(text):
        print("Blocked: scripts must not call ssh, scp, sftp or rsync to "
              "PERUN. Add a subcommand to scripts/perun.sh instead.",
              file=sys.stderr)
        return 2
    return 0


def main():
    event = json.load(sys.stdin)
    if event.get("tool_name") in ("Write", "Edit"):
        return check_file_write(event.get("tool_input", {}))
    if event.get("tool_name") != "Bash":
        return 0
    command = event.get("tool_input", {}).get("command", "")
    if SAFE_CALL.match(command):
        return 0
    if REMOTE.search(command):
        print("Blocked: reach PERUN only through scripts/perun.sh "
              "(see docs/hpc.md). Ask the user to run other cluster "
              "commands themselves.", file=sys.stderr)
        return 2
    if GUARDED.search(command) and WRITES.search(command):
        print("Blocked: scripts/perun.sh and .claude/ settings and hooks "
              "change only through the Edit tool, with the user's approval.",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
