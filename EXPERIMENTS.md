# Experiment registry

Add a row **before** a run starts. Update status when it ends.
Never delete a row. Failed runs are evidence (Task 8).

Status: `planned`, `running`, `done`, `failed`, `killed`.

| Run ID | Date | Commit | Configs + flags | Seed | Hardware / job | Wall-clock | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| 20261001-debug-cpu | 2026-10-01 | e3f0224 | `crafter debug`, `--jax.platform cpu` | 0 | local CPU, Docker | ~3 h | done | First local smoke run, `logdir/20261001T153148` |
| 20261017-smoke-s0 | 2026-10-17 | 35364df | `crafter size12m` | 0 | local RTX 3050 Laptop 4 GB, 13.5 GB RAM, Docker | 17 min | killed | GPU smoke test (#6), stopped with Ctrl+C on purpose at 6,350 env steps. Steady speed ~5.5 env steps/s (`fps/policy`), so ~3 min per 1k env steps and ~56 h (~2.3 days) for 1.1M. GPU compute ~95%, peak GPU memory 56%. Replay RAM grows ~21 KB per step (`replay/ram_gb` 0.15 at 6,286 items); `replay.size` 5e6 keeps every step, so 1.1M steps need ~23 GB RAM: more than this laptop has. `logdir/20261017-smoke-s0`, snapshot `runs/20261017-smoke-s0/env.txt` |
