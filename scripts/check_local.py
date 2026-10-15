#!/usr/bin/env python3
"""Stage 1 local check (assignment section 5).

Checks imports, config parsing, Crafter environment creation and one step.
Run from the repo root, inside the Docker image:

  docker compose --profile cpu run --rm train-cpu python scripts/check_local.py
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'upstream'
sys.path.insert(0, str(UPSTREAM))


def main():
  print('1. Imports')
  import elements
  import jax
  import ruamel.yaml as yaml
  import embodied
  from embodied.envs import crafter
  print('   jax', jax.__version__, 'devices:', jax.devices())

  print('2. Config parsing (defaults + crafter)')
  configs = yaml.YAML(typ='safe').load(
      (UPSTREAM / 'dreamerv3' / 'configs.yaml').read_text())
  config = elements.Config(configs['defaults']).update(configs['crafter'])
  print('   task:', config.task, '| steps:', config.run.steps,
        '| train_ratio:', config.run.train_ratio,
        '| imag_length:', config.agent.imag_length)
  assert config.task == 'crafter_reward'

  print('3. Environment creation')
  env = crafter.Crafter('reward', size=(64, 64), seed=0)
  print('   obs space:', {k: v.shape for k, v in env.obs_space.items()})
  print('   act space:', env.act_space['action'])

  print('4. One environment step')
  obs = env.step({'action': 0, 'reset': True})
  assert obs['is_first'] and obs['image'].shape == (64, 64, 3)
  obs = env.step({'action': 1, 'reset': False})
  print('   image', obs['image'].shape, obs['image'].dtype,
        '| reward', obs['reward'])
  env.close()

  print('OK: all local checks passed')


if __name__ == '__main__':
  main()
