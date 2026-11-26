# Demo script (10–15 min)

Deliverable 12. Shows the environment's behaviour and the world model's
internal workflow. Every slide or screen below already exists in the
repository; nothing needs a GPU during the demo. Rehearse once with a
timer; record it if the assessment asks for a video.

Total: about 13 minutes, with 2 minutes of buffer for questions.

## Before the demo (laptop)

- Open in browser tabs: `docs/report/report.md` on GitHub (the mermaid
  diagram renders there), the open-loop GIFs and the figures below.
- Build the gameplay page (`python analysis/gameplay.py`) and open
  `gameplay.html`. It works offline. Play the world you will show once,
  so you know where to pause.
- Optional live part: start the CPU debug run
  (`docker compose --profile cpu up train-cpu`) 5 minutes before and show
  its logs in `docker compose up viewer` (http://localhost:8000). It
  shows the training loop running; the tiny debug model learns nothing
  useful in 5 minutes.
  If anything fails, use the GIFs and figures instead; do not debug live.
- Decide who presents each part. Every member must be able to answer
  questions on any part (assignment rule), so rehearse swapping.

## Timeline

| Time | Part | Show | Say (key points) |
|---|---|---|---|
| 0:00–1:30 | 1. Goal | Title, `README.md` | Train DreamerV3 on Crafter, then measure and explain imagination-based control. The goal is understanding, not the highest score. |
| 1:30–3:00 | 2. The environment | Crafter frames: top row of `runs/20261125-openloop500k-s0/media/openloop.gif` (real frames; rows are real / predicted / error, green border = model sees the frame, red = open loop) | 64×64 image, 17 actions, 22 achievements, +1 the first time each is unlocked, health ±0.1. Long chains: wood, table, pickaxe, stone, furnace. Day and night. |
| 3:00–6:00 | 3. Inside the model | Mermaid diagram, `docs/report/report.md` section 3.3; table of six terms (3.1) | Encoder turns a frame into a token. RSSM: `deter` (memory, 8192) + `stoch` (32×64 categories). Posterior uses the frame, prior does not. Reward, continue, actor and critic heads read the 10240-number feature. Point at `sg`: real data stops, imagination begins. |
| 6:00–7:30 | 4. Real vs imagined | `docs/figures/20261123-baseline-s0-imagination.svg`; report section 4 table | World model learns only from replayed real frames. Actor and critic learn only from 15-step imagined rollouts, 1024 per training step. Imagined reward per step was 1.6× too high at 213k, matches the real one at 495k. |
| 7:30–9:30 | 5. Can it predict? | `runs/20261124-openloop-s0/media/openloop.gif`, then `docs/figures/20261124-openloop-s0-failures.svg` | 32 real frames, then 32 predicted from actions only. Layout and player stay; trees and cows fade. Beats "copy last frame" from step 7. Reward: false alarms at 213k, misses at 495k, health losses never foreseen. |
| 9:30–11:30 | 6. Results | `gameplay.html` for under a minute (one world at 4×: same start, one panel per training stage); then `docs/figures/20261125-imag30-s0-vs-20261124-imag5-s0-vs-20261123-baseline-s0-3seeds.svg`; evaluation table (report 7.2) | Random 1.4 %, early 2.7 %, 213k 4.3 %, 495k 11.3 ± 4.9 %, ~960k 23.0 ± 1.0 %. Return 11.56 matches the paper's 11.7; our score is higher because we score the final agent only. Seed 2 stuck at 495k, caught up by 960k. Horizon 15 ends best; horizon 30 fastest until ~600k, then passed; horizon 5 last, rarely builds a furnace. |
| 11:30–12:30 | 7. How we ran it | `docs/figures/gpu-usage.svg`; staged-workflow table (report 5.2) | Local test, interactive, smoke, production. Smoke and first open-loop job each found a GPU-only bug. One H200 per run, GPU busy 87–89 %, ~9 h per 500k run. The 1M runs died at ~960k from an out-of-memory kill while saving a checkpoint. |
| 12:30–13:00 | 8. Conclusion | Report section 10 | Learns; real and imagined data are separate in code and logs; model is good short-term, weak on rewards long-term; default horizon 15 beats 5 and 30. Next: same-episode comparison, deaths in open loop, imagined-reward error per rollout step, more seeds. |

## Likely questions

Prepare one answer each; the report section is in brackets.

- Why does the actor not see real rewards? (4)
- What is the difference between the prior and the posterior? (3.1, 3.4)
- Why is the 495k open-loop error higher than at 213k? (8.1: night frames, different episodes)
- Why did one seed score 5.6 % at 495k? (7.2: stuck before the wood pickaxe; caught up by 960k)
- Is horizon 5 or 30 better? (7.5: no; 15 ends best, 30 leads only until ~600k)
- Why is our score above the paper's when the return is the same? (7.6: protocol, final agent vs all training episodes)
- Why ~960k and not 1M? (8.4: out-of-memory kill while saving a checkpoint)
- What does `train_ratio 512` mean and why is the GPU the bottleneck? (5.3)
- What do the surveys say about world models in robotics? (2)
