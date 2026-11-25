# Analysis

Plotting scripts and notebooks. Read `runs/<run-id>/metrics.jsonl` and
`scores.jsonl` (one JSON object per line). `upstream/plot.py` compares runs
with the published curves in `upstream/scores/`, but that folder has no
Crafter file: take the Crafter reference from the DreamerV3 paper.
Our runs log episode return, not per-achievement success
(`env.crafter.logs` is false), so the paper's Crafter score must come from
an evaluation run.

Save figures to `docs/figures/` with the run ID in the file name.

Three horizons on the same axes, at an equal budget (mean over seeds;
`--max-step 500000` ignores the steps trained after a resume):

```sh
python analysis/compare_runs.py --max-step 500000 \
    20261123-baseline-s0,20261124-baseline-s1,20261124-baseline-s2 \
    20261124-imag5-s0,20261124-imag5-s1,20261124-imag5-s2 \
    20261125-imag30-s0,20261125-imag30-s1,20261125-imag30-s2
```

Use `--max-step 1000000` for the 1M figure. The 2-setting figure without
`--max-step` in `docs/results.md` was made when the runs ended at 500k:
after the 1M logs are fetched, remake it only with `--max-step 500000`.

`dashboard.py` reads the checkpoint step and the horizon of an evaluation
from `EXPERIMENTS.md`. Keep the text `(~495k steps` in the evaluation row
and `--agent.imag_length N` in the training row.

## Gameplay at each training stage

`gameplay.py` builds `gameplay.html` from `runs/*/media/gameplay.json`
and the videos next to it: one panel per agent, same worlds, played in
step. All agents start in exactly the same world. After that the lives
differ, because the agents act differently and because Crafter itself is
not fully repeatable (it loops over sets of objects whose order depends
on memory addresses). Recording does not change the game: with that order
fixed, the same actions gave the same game with and without the recorder.
The panel label comes from the registry row, as for `dashboard.py`.
