#!/usr/bin/env python3
"""Task 2: follow one training batch through the DreamerV3 networks.

Prints the name, shape and dtype of each tensor that `Agent.loss`
(upstream/dreamerv3/agent.py) builds for one batch: observation, action,
encoder tokens, deterministic and stochastic latent state, predicted
reward, continuation, actor output and critic value, for the real data
and for the imagined rollouts.

How: the network is rebuilt from a run's config.yaml (default: our
baseline, so the sizes are the official `crafter` ones). The same calls
as `Agent.loss` lines 164-202 run under `jax.eval_shape`. That traces the
code and reports shapes without computing anything or allocating the
weights, so it runs on a laptop CPU in seconds, even for the 200M model.
No checkpoint is needed: shapes do not depend on the weight values.

Run in Docker (never on the PERUN login node):
  docker compose --profile cpu run --rm train-cpu \\
      python analysis/trace_batch.py [runs/<run-id>/config.yaml]
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'upstream'))

import elements  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import ninjax as nj  # noqa: E402
import numpy as np  # noqa: E402
import ruamel.yaml as yaml  # noqa: E402

from dreamerv3 import agent as agent_mod  # noqa: E402
from dreamerv3 import main as dmain  # noqa: E402
from embodied.jax import internal  # noqa: E402
from embodied.jax.agent import Options  # noqa: E402

DEFAULT = ROOT / 'runs' / '20261123-baseline-s0' / 'config.yaml'
ORDER = []  # tensor names in the order trace() creates them


def build_model(config):
  """The inner Agent (the networks), without the outer training wrapper.

  dmain.make_agent() also creates the weights and compiles the training
  step, which needs a GPU-sized machine for the official config. Here we
  only need the network definitions, so we repeat the first part of
  embodied.jax.Agent.__new__ by hand.
  """
  env = dmain.make_env(config, 0)
  obs_space = {k: v for k, v in env.obs_space.items() if not k.startswith('log/')}
  act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}
  env.close()
  setup = {k: v for k, v in config.jax.items()
           if k not in Options.__dataclass_fields__}
  setup['platform'] = 'cpu'
  internal.setup(**setup)  # sets the compute dtype (bfloat16)
  agent_config = elements.Config(
      **config.agent, logdir='/tmp/trace', seed=0, jax=config.jax,
      batch_size=config.batch_size, batch_length=config.batch_length,
      replay_context=config.replay_context, report_length=config.report_length,
      replica=0, replicas=1)
  model = object.__new__(agent_mod.Agent)
  model.__init__(obs_space, act_space, agent_config)
  return model, obs_space, act_space


def trace(model, obs, prevact, B, T, H):
  """Same steps as Agent.loss, returning the tensors instead of losses."""
  out = {}
  enc_carry, dyn_carry, dec_carry, _ = model.init_train(B)
  reset = obs['is_first']

  # World model on real data (agent.py:164-177).
  _, _, tokens = model.enc(enc_carry, obs, reset, True)
  dyn_carry, entries, _, repfeat, _ = model.dyn.loss(
      dyn_carry, tokens, prevact, reset, True)
  _, _, recons = model.dec(dec_carry, repfeat, reset, True)
  feat = model.feat2tensor(repfeat)
  out['real/tokens'] = tokens
  out['real/deter'] = repfeat['deter']
  out['real/stoch'] = repfeat['stoch']
  out['real/post_logit'] = repfeat['logit']
  out['real/prior_logit'] = model.dyn._prior(repfeat['deter'])
  out['real/feat'] = feat
  out['real/recon_image'] = recons['image'].pred()
  out['real/reward_pred'] = model.rew(feat, 2).pred()
  out['real/cont_prob'] = model.con(feat, 2).prob(1)
  out['real/value'] = model.val(feat, 2).pred()

  # Imagination (agent.py:189-202): every real step is a start state.
  starts = model.dyn.starts(entries, dyn_carry, T)
  policyfn = lambda f: agent_mod.sample(model.pol(model.feat2tensor(f), 1))
  _, imgfeat, imgact = model.dyn.imagine(starts, policyfn, H, True)
  imgin = model.feat2tensor(imgfeat)
  pol = model.pol(imgin, 2)
  out['imag/start_deter'] = starts['deter']
  out['imag/deter'] = imgfeat['deter']
  out['imag/stoch'] = imgfeat['stoch']
  out['imag/feat'] = imgin
  out['imag/action'] = imgact['action']
  out['imag/actor_logits'] = pol['action'].logits
  out['imag/reward_pred'] = model.rew(imgin, 2).pred()
  out['imag/cont_prob'] = model.con(imgin, 2).prob(1)
  out['imag/value'] = model.val(imgin, 2).pred()
  ORDER[:] = list(out)
  return out


def main():
  path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
  config = elements.Config(yaml.YAML(typ='safe').load(path.read_text()))
  B, T, H = config.batch_size, config.batch_length, config.agent.imag_length
  model, obs_space, act_space = build_model(config)

  # Abstract inputs: shape and dtype only, no values.
  spec = lambda s: jax.ShapeDtypeStruct((B, T, *s.shape), s.dtype)
  obs = {k: spec(s) for k, s in obs_space.items()}
  prevact = {k: spec(s) for k, s in act_space.items()}
  fn = nj.pure(lambda o, a: trace(model, o, a, B, T, H))
  _, out = jax.eval_shape(
      lambda o, a: fn({}, o, a, seed=np.array([0, 0], np.uint32), create=True),
      obs, prevact)

  print(f'Config: {path}')
  print(f'B = batch_size = {B}, T = batch_length = {T}, '
        f'H = imag_length = {H}, B*T = {B * T} imagination starts\n')
  rows = [(f'obs/{k}', obs[k]) for k in sorted(obs)]
  rows += [(f'prevact/{k}', prevact[k]) for k in prevact]
  rows += [(k, out[k]) for k in ORDER]  # JAX sorts dict keys; use code order
  for name, x in rows:
    print(f'{name:22} {str(tuple(x.shape)):24} {jnp.dtype(x.dtype).name}')


if __name__ == '__main__':
  main()
