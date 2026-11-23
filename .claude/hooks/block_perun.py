#!/usr/bin/env python3
"""Claude Code hook: keep the AI assistant away from raw SSH to PERUN.

Runs before every Bash, Write and Edit call of the assistant (PreToolUse).
The only allowed way to reach the cluster is scripts/perun.sh, which
sends a fixed set of login-node-safe commands. This hook blocks:

- running ssh, scp, sftp or rsync, also inside `bash -c "..."`, `xargs`,
  `env`, `timeout`, or as a quoted program name (Python subprocess);
- connecting to the cluster host with a network tool (nc, curl, ...);
- a scripts/perun.sh call chained with other commands (; & | $( `);
- shell commands that change scripts/perun.sh or the .claude settings
  and hooks (sed -i, >, tee, mv, cp, rm, chmod, ...);
- writing a script file (.sh, .py, .sbatch, ...) that runs ssh, so the
  assistant cannot hide an ssh call in a file and then run that file.

Text that only mentions these words is allowed: a commit message, a PR
title, `git add scripts/perun.sh`, `cmd 2>&1`, a DNS lookup.

Exit code 2 tells Claude Code to block the call and show the reason to
the assistant. The user's own `!` commands do not go through hooks.
This is a guard rail on the laptop, not a hard security boundary.
Tests: python3 -I .claude/hooks/test_block_perun.py
"""
import json
import re
import shlex
import sys

REMOTE_TOOLS = {"ssh", "scp", "sftp", "rsync"}
NET_TOOLS = {"nc", "ncat", "netcat", "telnet", "socat", "curl", "wget", "sshpass"}
# Words that run the next word as a command: `timeout 30 ssh`, `env X=1 ssh`.
WRAPPERS = {"sudo", "env", "exec", "nohup", "time", "timeout", "xargs",
            "command", "builtin", "nice", "setsid", "stdbuf", "watch"}
SHELLS = {"bash", "sh", "zsh", "dash"}
WRITE_TOOLS = {"tee", "mv", "cp", "rm", "chmod", "chown", "ln", "truncate",
               "install", "dd", "shred", "unlink"}

SAFE_CALL = re.compile(
    r"^\s*(\./)?scripts/perun\.sh( -y)? "
    r"(status|log|fetch|smoke|submit|early-ckpt)( [A-Za-z0-9._=:/,+-]+)*\s*$"
)
HOST = re.compile(r"perun\.tuke\.sk|^perun:")
GUARDED = re.compile(r"(^|/)scripts/perun\.sh$|(^|/)\.claude/(settings[^/]*\.json|hooks/)")
GUARDED_ANY = re.compile(r"scripts/perun\.sh|\.claude/(settings|hooks)")
# A program name in quotes, as in subprocess.run(["ssh", ...]) or os.system('ssh ...').
QUOTED_REMOTE = re.compile(r"""["'](ssh|scp|sftp|rsync)(["'\s])""")
SCRIPT = re.compile(r"\.(sh|bash|py|sbatch|pl|rb|js)$")
# Files that may mention ssh. The settings ask the user before they change.
EXEMPT = re.compile(r"(scripts/perun\.sh|\.claude/hooks/[^/]+)$")
SEPARATORS = {";", "&&", "||", "|", "&", "\n", "(", ")", "$(", "`", "{", "}"}


def tokens(command):
    """Split a shell command into words and separators (; && | ...)."""
    lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|()`<>")
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        return list(lexer)
    except ValueError:  # unbalanced quotes: fall back to a rough split
        return command.split()


def segments(command):
    """Yield the word lists of each simple command in a command line."""
    current = []
    for tok in tokens(command.replace("$(", " ( ").replace("\n", " ; ")):
        if tok in SEPARATORS or set(tok) <= set(";&|()`"):
            if current:
                yield current
            current = []
        else:
            current.append(tok)
    if current:
        yield current


def program(words):
    """Return (name, rest) of the program a segment runs, past wrappers."""
    i = 0
    while i < len(words):
        w = words[i]
        name = w.rsplit("/", 1)[-1]
        if "=" in w and not w.startswith("-") and name == w:  # VAR=value
            i += 1
        elif name in WRAPPERS:
            i += 1
            while i < len(words) and (words[i].startswith("-") or words[i].isdigit()):
                i += 1
        else:
            return name, words[i + 1:]
    return "", []


def remote_reason(command, depth=0):
    if QUOTED_REMOTE.search(command):
        return "runs ssh/scp/rsync as a quoted program name"
    for words in segments(command):
        name, rest = program(words)
        if name in REMOTE_TOOLS:
            return f"runs {name}"
        if name in NET_TOOLS and any(HOST.search(w) for w in rest):
            return f"connects to PERUN with {name}"
        if name in SHELLS and depth < 3:
            for i, w in enumerate(rest):
                if w == "-c" and i + 1 < len(rest):
                    reason = remote_reason(rest[i + 1], depth + 1)
                    if reason:
                        return reason
    return None


def guard_write_reason(command):
    words_all = tokens(command)
    # Redirect into a guarded file: `... > scripts/perun.sh`.
    for i, w in enumerate(words_all[:-1]):
        if w in (">", ">>") and GUARDED.search(words_all[i + 1]):
            return f"redirects into {words_all[i + 1]}"
    for words in segments(command):
        name, rest = program(words)
        targets = [w for w in rest if GUARDED.search(w)]
        if not targets:
            continue
        # A program's own name (scripts/perun.sh, a hook script) is not a write target.
        if name in WRITE_TOOLS:
            return f"{name} changes {targets[0]}"
        if name == "sed" and any(w.startswith("-i") or w == "--in-place" for w in rest):
            return f"sed -i changes {targets[0]}"
    return None


def inline_code_reason(command):
    """Inline interpreter code (python -c, perl -e, heredoc) that names a guard file."""
    for words in segments(command):
        name, rest = program(words)
        if not name.startswith(("python", "perl", "ruby", "node")):
            continue
        inline = any(w in ("-c", "-e", "-") for w in rest) or "<<" in command
        if inline and GUARDED_ANY.search(command):
            return f"{name} code names a guard file"
    return None


def check_bash(command):
    if SAFE_CALL.match(command):
        return 0
    reason = remote_reason(command)
    if reason:
        print(f"Blocked ({reason}): reach PERUN only through scripts/perun.sh "
              "(see docs/hpc.md). Ask the user to run other cluster commands "
              "themselves.", file=sys.stderr)
        return 2
    reason = guard_write_reason(command) or inline_code_reason(command)
    if reason:
        print(f"Blocked ({reason}): scripts/perun.sh and .claude/ settings "
              "and hooks change only through the Edit tool, with the user's "
              "approval.", file=sys.stderr)
        return 2
    return 0


def check_file_write(tool_input):
    path = tool_input.get("file_path", "")
    text = tool_input.get("content", "") + tool_input.get("new_string", "")
    if EXEMPT.search(path) or not SCRIPT.search(path):
        return 0
    if path.endswith(".py"):
        reason = "runs ssh" if QUOTED_REMOTE.search(text) else None
    else:
        reason = remote_reason(text)
    if reason:
        print(f"Blocked ({reason}): scripts must not call ssh, scp, sftp or "
              "rsync. Add a subcommand to scripts/perun.sh instead.",
              file=sys.stderr)
        return 2
    return 0


def main():
    event = json.load(sys.stdin)
    tool = event.get("tool_name")
    tool_input = event.get("tool_input", {})
    if tool in ("Write", "Edit"):
        return check_file_write(tool_input)
    if tool == "Bash":
        return check_bash(tool_input.get("command", ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
