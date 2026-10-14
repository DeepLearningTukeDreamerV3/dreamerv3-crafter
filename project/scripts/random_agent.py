"""Random-action baseline for Crafter (Task 6).

Plays N episodes with uniformly random actions and writes one JSON line per
episode, in the same format as DreamerV3's scores.jsonl.

  python project/scripts/random_agent.py --episodes 100 --seed 0 \
      --out runs/<run-id>/scores.jsonl
"""

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from embodied.envs import crafter  # noqa: E402


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--episodes', type=int, default=100)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--out', type=pathlib.Path, default=None)
  args = parser.parse_args()

  rng = np.random.default_rng(args.seed)
  env = crafter.Crafter('reward', size=(64, 64), seed=args.seed)
  num_actions = int(env.act_space['action'].high)

  lines = []
  step = 0
  for episode in range(args.episodes):
    obs = env.step({'action': 0, 'reset': True})
    score, length = 0.0, 0
    while not obs['is_last']:
      action = int(rng.integers(num_actions))
      obs = env.step({'action': action, 'reset': False})
      score += float(obs['reward'])
      length += 1
    step += length
    lines.append({'step': step, 'episode': episode,
                  'episode/score': score, 'episode/length': length})
    print(f'episode {episode}: score {score:.1f}, length {length}')
  env.close()

  scores = np.array([x['episode/score'] for x in lines])
  print(f'mean score {scores.mean():.2f} +- {scores.std():.2f} '
        f'over {len(scores)} episodes')

  if args.out:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(''.join(json.dumps(x) + '\n' for x in lines))
    print('Wrote', args.out)


if __name__ == '__main__':
  main()
