# Experiment registry

Add a row **before** a run starts. Update status when it ends.
Never delete a row. Failed runs are evidence (Task 8).

Status: `planned`, `running`, `done`, `failed`, `killed`.

| Run ID | Date | Commit | Configs + flags | Seed | Hardware / job | Wall-clock | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| 20261001-debug-cpu | 2026-10-01 | e3f0224 | `crafter debug`, `--jax.platform cpu` | 0 | local CPU, Docker | ~3 h | done | First local smoke run, `logdir/20261001T153148` |
