# Analysis

Plotting scripts and notebooks. Read `runs/<run-id>/metrics.jsonl` and
`scores.jsonl` (one JSON object per line). `upstream/plot.py` compares runs
with the published curves in `upstream/scores/`, but that folder has no
Crafter file: take the Crafter reference from the DreamerV3 paper.
Our runs log episode return, not per-achievement success
(`env.crafter.logs` is false), so the paper's Crafter score must come from
an evaluation run.

Save figures to `docs/figures/` with the run ID in the file name.
