#!/usr/bin/env python3
"""Evaluate a DreamerV3 checkpoint (or a random agent) on Crafter (Task 6).

Same protocol for every agent, so the numbers can be compared:
- one Crafter env with a fixed world seed (--seed), N episodes in a row,
  so every agent plays the same sequence of worlds;
- the policy in 'eval' mode (no training, nothing written to the run);
- for each episode: return, length, and which of the 22 achievements were
  unlocked at least once.

Writes one JSON line per episode to --out and prints a summary: mean
return, success rate per achievement, and the Crafter score (geometric
mean of the success rates, as in the Crafter and DreamerV3 papers):
  score = exp(mean(ln(1 + s_i))) - 1, with s_i in percent.

Examples (in Docker or in a SLURM job, never on the PERUN login node):
  python scripts/eval_agent.py --run-dir <logdir>/<run-id> --ckpt ckpt \\
      --episodes 50 --seed 100 --out runs/<eval-id>/eval.jsonl
  python scripts/eval_agent.py --run-dir <logdir>/<run-id> --random \\
      --episodes 50 --seed 100 --out runs/<eval-id>/eval.jsonl

--run-dir is a training logdir: its config.yaml defines the network.
--ckpt is a folder inside it (`ckpt`, or `ckpt_early_<step>`); the
checkpoint that its `latest` file names is loaded.
"""

import argparse
import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'upstream'))

import elements  # noqa: E402
import embodied  # noqa: E402
import numpy as np  # noqa: E402
import ruamel.yaml as yaml  # noqa: E402

from dreamerv3 import main as dmain  # noqa: E402
from embodied.envs import crafter  # noqa: E402


def crafter_score(rates):
  """Geometric mean of success rates in percent (Crafter paper)."""
  return math.exp(np.mean([math.log(1 + 100 * r) for r in rates])) - 1


def main():
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument('--run-dir', type=pathlib.Path, required=True)
  parser.add_argument('--ckpt', default='ckpt')
  parser.add_argument('--random', action='store_true',
                      help='random actions instead of the checkpoint')
  parser.add_argument('--episodes', type=int, default=50)
  parser.add_argument('--seed', type=int, default=100,
                      help='world seed; use the same one for every agent')
  parser.add_argument('--platform', default=None,
                      help='override jax.platform (cpu or cuda)')
  parser.add_argument('--out', type=pathlib.Path, required=True)
  args = parser.parse_args()

  # The network is rebuilt from the training config, then the weights are
  # loaded into it. logdir points to the output folder so that nothing is
  # written into the training run.
  configs = yaml.YAML(typ='safe').load((args.run_dir / 'config.yaml').read_text())
  config = elements.Config(configs)
  config = config.update(logdir=str(args.out.parent), random_agent=args.random)
  if args.platform:
    config = config.update({'jax.platform': args.platform})

  agent = dmain.make_agent(config)
  if not args.random:
    ckdir = args.run_dir / args.ckpt
    path = ckdir / (ckdir / 'latest').read_text().strip()
    cp = elements.Checkpoint()
    cp.agent = agent
    cp.load(elements.Path(path), keys=['agent'])

  # logs=True adds log/achievement_<name> counters to each observation.
  def make_env():
    env = crafter.Crafter(
        'reward', size=tuple(config.env.crafter.size), logs=True,
        seed=args.seed)
    return dmain.wrap_env(env, config)

  lines = []
  current = {'score': 0.0, 'length': 0}

  def on_step(tran, worker):
    if tran['is_first']:
      current.update(score=0.0, length=0)
    current['score'] += float(tran['reward'])
    current['length'] += 1
    if tran['is_last']:
      unlocked = {
          k[len('log/achievement_'):]: int(tran[k] > 0)
          for k in tran if k.startswith('log/achievement_')}
      lines.append({
          'episode': len(lines), 'episode/score': current['score'],
          'episode/length': current['length'], 'achievements': unlocked})
      print(f"episode {len(lines) - 1}: return {current['score']:.1f}, "
            f"length {current['length']}, achievements {sum(unlocked.values())}",
            flush=True)

  driver = embodied.Driver([make_env], parallel=False)
  driver.on_step(on_step)
  driver.reset(agent.init_policy)
  policy = lambda *a: agent.policy(*a, mode='eval')
  while len(lines) < args.episodes:
    driver(policy, episodes=1)
  driver.close()
  lines = lines[:args.episodes]

  names = sorted(lines[0]['achievements'])
  rates = {k: float(np.mean([x['achievements'][k] for x in lines])) for k in names}
  returns = np.array([x['episode/score'] for x in lines])
  summary = {
      'agent': 'random' if args.random else str(args.run_dir / args.ckpt),
      'episodes': len(lines), 'seed': args.seed,
      'return_mean': float(returns.mean()), 'return_std': float(returns.std()),
      'crafter_score': crafter_score(list(rates.values())),
      'success_rates': rates,
  }

  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(''.join(json.dumps(x) + '\n' for x in lines))
  (args.out.parent / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
  print(f"return {summary['return_mean']:.2f} +- {summary['return_std']:.2f}, "
        f"Crafter score {summary['crafter_score']:.1f}% over {len(lines)} episodes")
  print('Wrote', args.out, 'and summary.json')


if __name__ == '__main__':
  main()
