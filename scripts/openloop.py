#!/usr/bin/env python3
"""Open-loop predictions of a trained world model on Crafter (Task 4).

"Open loop" means: the world model sees the real frames for a while (the
context), then must predict the next frames alone, from the real actions
only, without seeing the real frames. The predictions drift away from
reality as the horizon grows. How fast they drift tells how good the
world model is.

Steps:
1. Load the checkpoint as scripts/eval_agent.py does.
2. Play --episodes episodes with the trained policy (eval mode) on a
   fixed world seed, and record every step (frame, reward, action).
3. Cut windows of --length steps out of the episodes. For each window
   call the upstream `agent.report` (upstream/dreamerv3/agent.py), which
   observes the first half and imagines the second half with the real
   actions, then decodes both halves to images. Same code as the
   `openloop/image` video that training writes.
4. Compare predicted and real frames, step by step.
5. Same windows again through the reward and continuation heads
   (heads_fn below): does the world model foresee rewards and the end of
   the episode without seeing the frames?

Writes to --out (a folder):
  openloop.json      mean squared error per step (context and open loop),
                     plus a "copy last frame" baseline for scale;
                     reward error per step, predicted vs real reward sums,
                     predicted continuation probability per step
  media/openloop.gif upstream-style video: rows real / predicted / error,
                     green border = context, red border = open loop
  media/openloop.png the same as a still: a few time steps side by side

Example (in Docker or in a SLURM job, never on the PERUN login node):
  python scripts/openloop.py --run-dir <logdir>/<run-id> --ckpt ckpt \\
      --out runs/<openloop-id>
"""

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'upstream'))

import elements  # noqa: E402
import embodied  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import ninjax as nj  # noqa: E402
import numpy as np  # noqa: E402
import ruamel.yaml as yaml  # noqa: E402
from PIL import Image  # noqa: E402

from dreamerv3 import main as dmain  # noqa: E402
from embodied.envs import crafter  # noqa: E402
from embodied.jax import internal  # noqa: E402

RB = 6  # agent.report shows at most 6 windows per call (RB in agent.py)


def record(agent, config, episodes, seed):
  """Play episodes with the eval policy; return one list of steps each."""
  def make_env():
    env = crafter.Crafter(
        'reward', size=tuple(config.env.crafter.size), seed=seed)
    return dmain.wrap_env(env, config)

  done, current = [], []

  def on_step(tran, worker):
    if tran['is_first']:
      current.clear()
    current.append({k: np.asarray(v) for k, v in tran.items()})
    if tran['is_last']:
      done.append(list(current))
      print(f'episode {len(done) - 1}: {len(current)} steps', flush=True)

  driver = embodied.Driver([make_env], parallel=False)
  driver.on_step(on_step)
  driver.reset(agent.init_policy)
  policy = lambda *a: agent.policy(*a, mode='eval')
  while len(done) < episodes:
    driver(policy, episodes=1)
  driver.close()
  return done[:episodes]


def windows(eps, length, per_episode, rng):
  """Start indices (episode, step) of windows that fit in one episode."""
  out = []
  for e, ep in enumerate(eps):
    last = len(ep) - length
    if last < 0:
      continue
    starts = sorted(rng.choice(last + 1, min(per_episode, last + 1), replace=False))
    out += [(e, int(s)) for s in starts]
  return out


def batch(agent, eps, starts, length):
  """The data dict agent.report expects, for the given windows.

  agent.report drops the first `replay_context` steps of the window (they
  only carry the replay state, which we do not use: consec=1 below makes
  it start from a fresh state). So we pass context + length steps and
  start each window one step early when possible.
  """
  K = agent.config.replay_context
  data = agent._zeros(agent.spaces, (len(starts), K + length))
  for b, (e, s) in enumerate(starts):
    steps = eps[e][max(0, s - K): s + length]
    steps = [steps[0]] * (K + length - len(steps)) + steps  # pad the front
    for key in data:
      if key in steps[0]:
        data[key][b] = np.stack([x[key] for x in steps]).astype(data[key].dtype)
    data['consec'][b] = 1   # not the first chunk: do not use replay entries
  return data


def heads_fn(model, ctx):
  """Reward and continuation predicted for every step of the windows.

  Same steps as the open loop in agent.report: the encoder and the
  posterior see the first ctx frames, then the prior imagines the rest
  from the real actions. The reward and continuation heads read the
  model state at each step (agent.loss uses the same calls).
  """
  def fn(obs, prevact):
    B, T = obs['is_first'].shape
    enc_carry, dyn_carry, _, _ = model.init_policy(B)
    _, _, tokens = model.enc(enc_carry, obs, obs['is_first'], training=False)
    first = lambda xs: jax.tree.map(lambda x: x[:, :ctx], xs)
    second = lambda xs: jax.tree.map(lambda x: x[:, ctx:], xs)
    dyn_carry, _, obsfeat = model.dyn.observe(
        dyn_carry, first(tokens), first(prevact), first(obs['is_first']),
        training=False)
    _, imgfeat, _ = model.dyn.imagine(
        dyn_carry, second(prevact), length=T - ctx, training=False)
    feat = jax.tree.map(lambda a, b: jnp.concatenate([a, b], 1), obsfeat, imgfeat)
    inp = model.feat2tensor(feat)
    return model.rew(inp, 2).pred(), model.con(inp, 2).prob(1)
  pure = nj.pure(fn)
  # The weights (agent.params) are only read; the new state is dropped.
  # seed: the model samples its latent state, so it needs random numbers.
  return jax.jit(lambda params, obs, prevact, seed: pure(
      params, obs, prevact, seed=seed)[1])


def split_grid(grid, n, length):
  """Undo the layout of agent.report's openloop/image grid.

  grid: (length + 10, 3H + 4, n (W + 4), 3) uint8. Returns real and
  predicted frames, each (n, length, H, W, 3).
  """
  H = (grid.shape[1] - 4) // 3
  W = grid.shape[2] // n - 4
  real, pred = [], []
  for b in range(n):
    x = b * (W + 4) + 2
    cell = grid[:length, 2:2 + 3 * H, x:x + W]
    real.append(cell[:, :H])
    pred.append(cell[:, H:2 * H])
  return np.stack(real), np.stack(pred)


def mse(a, b):
  """Mean squared error per window and step, pixels scaled to 0..1."""
  d = (a.astype(np.float32) - b.astype(np.float32)) / 255
  return (d ** 2).mean(axis=(2, 3, 4))


def rounded(xs):
  return [round(float(x), 6) for x in xs]


def save_media(grid, real, pred, ctx, out):
  """GIF of the upstream grid, PNG strip of one window at a few steps."""
  media = out / 'media'
  media.mkdir(parents=True, exist_ok=True)
  big = lambda a: Image.fromarray(a).resize(
      (a.shape[1] * 2, a.shape[0] * 2), Image.NEAREST)
  # Native size: the repo CI rejects files over 5 MB; browsers can zoom.
  frames = [Image.fromarray(f) for f in grid]
  frames[0].save(media / 'openloop.gif', save_all=True,
                 append_images=frames[1:], duration=150, loop=0)

  T = real.shape[1]
  steps = sorted({0, ctx - 1, ctx, ctx + 2, ctx + 5, ctx + 10, ctx + 20, T - 1})
  steps = [t for t in steps if 0 <= t < T]
  rows = min(3, len(real))
  H, W = real.shape[2:4]
  strip = np.full((rows * (2 * H + 6), len(steps) * (W + 4), 3), 255, np.uint8)
  for r in range(rows):
    for c, t in enumerate(steps):
      y, x = r * (2 * H + 6), c * (W + 4)
      strip[y:y + H, x:x + W] = real[r, t]
      strip[y + H + 2:y + 2 * H + 2, x:x + W] = pred[r, t]
  big(strip).save(media / 'openloop.png')
  return steps


def main():
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument('--run-dir', type=pathlib.Path, required=True)
  parser.add_argument('--ckpt', default='ckpt')
  parser.add_argument('--episodes', type=int, default=10)
  parser.add_argument('--windows-per-episode', type=int, default=3)
  parser.add_argument('--length', type=int, default=64,
                      help='steps per window: first half context, second half open loop')
  parser.add_argument('--seed', type=int, default=100, help='world seed')
  parser.add_argument('--random-weights', action='store_true',
                      help='skip loading the checkpoint (pipeline test only)')
  parser.add_argument('--platform', default=None,
                      help='override jax.platform (cpu or cuda)')
  parser.add_argument('--out', type=pathlib.Path, required=True)
  args = parser.parse_args()
  args.out.mkdir(parents=True, exist_ok=True)

  configs = yaml.YAML(typ='safe').load((args.run_dir / 'config.yaml').read_text())
  # precompile off: otherwise report is compiled for the training shapes
  # (batch_size x report_length) and refuses our windows. The key is not
  # in configs.yaml (default of Options in upstream/embodied/jax/agent.py),
  # so it is added to the dict before the Config is built.
  configs['jax']['precompile'] = False
  config = elements.Config(configs).update(logdir=str(args.out))
  if args.platform:
    config = config.update({'jax.platform': args.platform})
  agent = dmain.make_agent(config)
  if not args.random_weights:
    ckdir = args.run_dir / args.ckpt
    path = ckdir / (ckdir / 'latest').read_text().strip()
    cp = elements.Checkpoint()
    cp.agent = agent
    cp.load(elements.Path(path), keys=['agent'])

  eps = record(agent, config, args.episodes, args.seed)
  rng = np.random.default_rng(args.seed)
  starts = windows(eps, args.length, args.windows_per_episode, rng)
  if not starts:
    sys.exit(f'no episode has {args.length} steps; lower --length')
  ctx = args.length // 2   # agent.report observes T // 2 steps

  heads = heads_fn(agent.model, ctx)
  K = agent.config.replay_context
  errors, copy_errors, first = [], [], None
  rew_pred, rew_real, con_pred = [], [], []
  for i in range(0, len(starts), RB):
    chunk = starts[i:i + RB]
    data = batch(agent, eps, chunk, args.length)
    # Reward and continuation: same windows, without the replay step K.
    obs = {k: data[k][:, K:] for k in agent.model.obs_space}
    prevact = {k: data[k][:, K - 1:-1] for k in agent.model.act_space}
    # Upstream forbids implicit copies to the GPU: put the inputs there.
    obs, prevact = internal.device_put((obs, prevact), agent.train_sharded)
    seed = internal.device_put(np.array([args.seed, i], np.uint32), agent.train_mirrored)
    rp, cp = jax.device_get(heads(agent.params, obs, prevact, seed))  # explicit copy back
    rew_pred.append(np.asarray(rp, np.float32))
    rew_real.append(data['reward'][:, K:].astype(np.float32))
    con_pred.append(np.asarray(cp, np.float32))
    data = internal.device_put(data, agent.train_sharded)  # as in stream()
    data['seed'] = agent._seeds(i, agent.train_mirrored)
    carry = agent.init_report(len(chunk))
    _, mets = agent.report(carry, data)
    grid = np.asarray(mets['openloop/image'])
    real, pred = split_grid(grid, len(chunk), args.length)
    errors.append(mse(pred, real))
    # Baseline: keep showing the last real context frame.
    copy = np.repeat(real[:, ctx - 1:ctx], args.length, axis=1)
    copy_errors.append(mse(copy, real))
    if first is None:
      first = (grid, real, pred)
  errors = np.concatenate(errors)
  copy_errors = np.concatenate(copy_errors)
  rew_pred, rew_real = np.concatenate(rew_pred), np.concatenate(rew_real)
  con_pred = np.concatenate(con_pred)

  shown = save_media(*first, ctx, args.out)
  result = {
      'run_dir': str(args.run_dir), 'ckpt': None if args.random_weights else args.ckpt,
      'seed': args.seed, 'episodes': len(eps), 'windows': len(errors),
      'length': args.length, 'context': ctx,
      'episode_lengths': [len(e) for e in eps],
      'mse_by_step': rounded(errors.mean(0)),
      'mse_by_step_std': rounded(errors.std(0)),
      'copy_last_mse_by_step': rounded(copy_errors.mean(0)),
      'png_steps': shown,
      'reward_abs_error_by_step': rounded(np.abs(rew_pred - rew_real).mean(0)),
      'reward_pred_by_step': rounded(rew_pred.mean(0)),
      'reward_real_by_step': rounded(rew_real.mean(0)),
      # One pair per window: total reward over the open-loop half.
      'reward_sum_openloop': [
          [round(float(p), 4), round(float(r), 4)] for p, r in
          zip(rew_pred[:, ctx:].sum(1), rew_real[:, ctx:].sum(1))],
      'continue_prob_by_step': rounded(con_pred.mean(0)),
      'windows_with_episode_end': int(sum(
          eps[e][s + args.length - 1]['is_last'] for e, s in starts)),
  }
  (args.out / 'openloop.json').write_text(json.dumps(result, indent=1) + '\n')
  ctx_mse = errors[:, :ctx].mean()
  ol = errors[:, ctx:].mean(0)
  print(f'windows: {len(errors)}, context mse {ctx_mse:.5f}, '
        f'open loop mse step 1 {ol[0]:.5f}, step {len(ol)} {ol[-1]:.5f}')
  print(f'reward |error| context {np.abs(rew_pred - rew_real)[:, :ctx].mean():.4f}, '
        f'open loop {np.abs(rew_pred - rew_real)[:, ctx:].mean():.4f}')
  print('wrote', args.out / 'openloop.json', 'and', args.out / 'media')


if __name__ == '__main__':
  main()
