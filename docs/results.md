# Results

Numbers for the report. Every number comes from a file under `runs/`;
the run IDs are in `EXPERIMENTS.md`.

## Evaluation table (assignment A.7, Task 6)

All agents play the **same 50 Crafter worlds** (world seed 100, one env,
50 episodes in a row) with `scripts/eval_agent.py`. Trained agents act in
eval mode (no training during the evaluation). Return = +1 per new
achievement, ±0.1 per health point; mean ± standard deviation over the
50 episodes. Crafter score = geometric mean of the 22 success rates
(glossary).

| System stage | Env steps | Episode return | Crafter score | Prediction evidence | Comments | Run ID |
|---|---|---|---|---|---|---|
| Random agent | 0 | 1.52 ± 1.08 | 1.4 % | none (no model) | Reference lower baseline | `20261124-eval-random` |
| Early DreamerV3 checkpoint | ~40–52k | 2.64 ± 1.37 | 2.7 % | Task 4 (todo) | World model still immature | `20261124-eval-early-s0` |
| Trained DreamerV3 | ~213k | 4.44 ± 1.61 | 4.3 % | Task 4 (todo) | Main result, stopped at 213,620 steps (ROADMAP week 4) | `20261124-eval-final-s0` |
| Sensitivity variant | 213,620 | todo | todo | todo | `--agent.imag_length 5` | `20261124-imag5-s0` |

The order random < early < trained holds for both return and score.
Mean episode length is the same for all three (174–176 steps): at
213k steps the agent collects more achievements per episode, but it does
not yet survive longer than a random agent.

### Success rate per achievement (%)

Share of the 50 episodes in which the achievement was unlocked at least
once. Achievements at 0 % for all three agents are left out:
`collect_coal`, `collect_diamond`, `collect_iron`, `defeat_skeleton`,
`eat_plant`, `make_iron_pickaxe`, `make_iron_sword`,
`make_stone_pickaxe`, `make_stone_sword`, `place_furnace`.

| Achievement | Random | Early | Trained |
|---|---|---|---|
| wake_up | 96 | 92 | 92 |
| collect_wood | 26 | 70 | 86 |
| collect_sapling | 60 | 50 | 88 |
| place_plant | 54 | 48 | 82 |
| place_table | 2 | 36 | 68 |
| collect_drink | 6 | 56 | 64 |
| make_wood_pickaxe | 0 | 6 | 32 |
| defeat_zombie | 0 | 4 | 12 |
| collect_stone | 0 | 0 | 6 |
| make_wood_sword | 0 | 0 | 2 |
| place_stone | 0 | 0 | 2 |
| eat_cow | 0 | 2 | 0 |

Training made the first steps of the tech tree reliable (wood → table →
wood pickaxe) and started the next one (stone). Nothing beyond stone
yet: no coal, iron, furnace or stone tools.

## Comparison with the published result (issue #17)

Reference: DreamerV3 paper, arXiv 2301.04104 **v1** (Jan 2023),
appendix M, Table M.1. Setting from its appendix A: 1M env steps, train
ratio 512, model size XL, 1 env. The 2024 revision (v2) shows Crafter
only in a scaling figure (Fig. 6c–d), with no table.

| | Env steps | Seeds | Crafter score | Return |
|---|---|---|---|---|
| Paper: DreamerV3 | 1M | 10 | 14.5 ± 1.6 % | 11.7 ± 1.9 |
| Paper: PPO | 1M | 10 | 4.6 ± 0.3 % | 4.2 ± 1.2 |
| Paper: Random | — | 10 | 1.6 ± 0.0 % | 2.1 ± 1.3 |
| Ours: DreamerV3 | 213k | 1 | 4.3 % | 4.44 ± 1.61 |
| Ours: Random | — | 1 (50 episodes) | 1.4 % | 1.52 ± 1.08 |

**Random agent: matches.** 1.4 % vs 1.6 %, return 1.5 vs 2.1 (inside
the paper's spread of ±1.3). This checks that our evaluation script and
the score formula behave like the paper's.

**Trained agent: about 30 % of the paper's score, after about 21 % of
its steps.** Reasons for the gap, largest first:

1. **Budget.** We stopped at 213,620 of 1M env steps (decision
   2026-11-23, ROADMAP week 4). The paper's reward curve (Fig. M.1b) is
   still rising at 1M, so a run stopped at 21 % is expected to score far
   lower. The paper gives the 200k value only in that figure, so we do
   not quote a number for it.
2. **Protocol.** The Crafter benchmark computes success rates over all
   episodes collected during training. Ours come from 50 eval-mode
   episodes of one checkpoint. As a cross-check, our mean training return
   over the last 100 training episodes (4.45, `metrics.jsonl`) equals the
   eval return (4.44), so the protocol difference is small at this stage.
3. **Seeds.** One seed (0) against the paper's 10. We cannot give a
   spread across seeds.
4. **Code version.** We run upstream commit `e3f0224` (2024 code,
   `crafter` config in `configs.yaml`). The v1 numbers come from the 2023
   code. The network and defaults changed between the two, so even at
   1M steps the numbers need not match exactly.

What would close the gap: resume `20261123-baseline-s0` with the same
run ID up to 1M steps (~15 h more on one H200) and add seeds 1 and 2
(issue #14).
