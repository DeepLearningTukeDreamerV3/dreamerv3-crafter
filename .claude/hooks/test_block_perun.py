#!/usr/bin/env python3
"""Tests for block_perun.py. Run: python3 -I .claude/hooks/test_block_perun.py

Each case feeds one tool call to the hook and checks the exit code:
0 = allowed, 2 = blocked.
"""
import json
import pathlib
import subprocess
import sys

HOOK = pathlib.Path(__file__).with_name("block_perun.py")

BASH = [
    # Allowed: the script itself, and text that only mentions the words.
    (0, "scripts/perun.sh status"),
    (0, "scripts/perun.sh -y submit exp/baseline 20261123-baseline-s0 0 baseline --agent.imag_length 5"),
    (0, "scripts/perun.sh status 2>&1"),
    (0, "git add scripts/perun.sh .claude/settings.json .claude/hooks/block_perun.py"),
    (0, "git log --format='%an <%ae>' -1"),
    (0, 'gh pr create --title "Block raw ssh from Claude Code" --body-file pr.md'),
    (0, 'git commit -m "docs: explain ssh alias"'),
    (0, "getent hosts login01.perun.tuke.sk"),
    (0, "cat scripts/perun.sh"),
    (0, "shellcheck scripts/perun.sh && echo ok"),
    (0, "git switch infra/perun"),
    (0, "ls -la > /tmp/out.txt"),
    # Blocked: real ssh, also hidden.
    (2, "ssh perun hostname"),
    (2, "/usr/bin/ssh perun hostname"),
    (2, "scripts/perun.sh status; ssh perun ls"),
    (2, "echo hi && scp file perun:"),
    (2, "rsync -av perun:x ."),
    (2, 'bash -c "ssh perun hostname"'),
    (2, "timeout 30 ssh perun hostname"),
    (2, "env FOO=1 sftp perun"),
    (2, "echo perun | xargs ssh"),
    (2, "x=$(ssh perun cat f)"),
    (2, """python3 -c 'import subprocess; subprocess.run(["ssh","perun"])'"""),
    (2, "nc login01.perun.tuke.sk 22"),
    # Blocked: shell writes to the guard files.
    (2, "sed -i s/a/b/ scripts/perun.sh"),
    (2, "echo {} > .claude/settings.json"),
    (2, "echo x >> .claude/hooks/block_perun.py"),
    (2, "rm .claude/hooks/block_perun.py"),
    (2, "chmod -x .claude/hooks/block_perun.py"),
    (2, "cp /tmp/x .claude/settings.json"),
    (2, "python3 -c 'open(\".claude/settings.json\",\"w\")'"),
    (0, "python3 -I .claude/hooks/test_block_perun.py"),
]

FILES = [
    (0, "/repo/docs/hpc.md", "Use ssh perun to log in."),
    (0, "/repo/scripts/perun.sh", "ssh perun hostname"),
    (0, "/repo/scripts/plot.py", "# results come from rsync, see docs\nprint(1)"),
    (0, "/repo/scripts/x.sh", "# copy with rsync later\necho hi"),
    (2, "/repo/scripts/x.sh", "#!/usr/bin/env bash\nssh perun hostname\n"),
    (2, "/repo/scripts/x.py", 'import subprocess\nsubprocess.run(["ssh", "perun"])\n'),
]


def run(event):
    return subprocess.run([sys.executable, "-I", str(HOOK)],
                          input=json.dumps(event), text=True,
                          capture_output=True).returncode


def main():
    failed = 0
    for want, command in BASH:
        got = run({"tool_name": "Bash", "tool_input": {"command": command}})
        if got != want:
            failed += 1
            print(f"FAIL want {want} got {got}: {command}")
    for want, path, content in FILES:
        got = run({"tool_name": "Write",
                   "tool_input": {"file_path": path, "content": content}})
        if got != want:
            failed += 1
            print(f"FAIL want {want} got {got}: Write {path}")
    total = len(BASH) + len(FILES)
    print(f"{total - failed}/{total} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
