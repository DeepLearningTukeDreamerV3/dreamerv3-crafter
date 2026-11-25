"""Record the agent's gameplay as videos (used by eval_agent.py --video).

For each recorded episode:
- world<N>.mp4: one video frame per env step, 10 frames per second,
  drawn at 360x360 so people can watch it (the agent still sees its own
  64x64 image);
- an entry in gameplay.json: what happened at each step (action, reward,
  health, food, drink, energy) and when each achievement was unlocked.

Watching must not change the game. At night, Crafter adds random noise to
the picture and takes it from the same random generator that moves the
zombies. An extra drawing would therefore change what happens next. So
frame() saves the state of that generator before drawing and puts it back
after.
"""

import json
import subprocess

import crafter
import numpy as np

FPS = 10
SIZE = (360, 360)  # 9x9 map cells of 40 pixels
STATS = ('health', 'food', 'drink', 'energy')


class Recorder:

  def __init__(self, folder, episodes):
    self.folder = folder
    self.episodes = episodes  # how many episodes to record
    self.done = []            # finished episodes
    self.frames = []
    self.log = None
    folder.mkdir(parents=True, exist_ok=True)

  def frame(self, env):
    """Draw the current state of the crafter.Env, without side effects."""
    state = env._world.random.get_state()
    image = env.render(SIZE)
    env._world.random.set_state(state)
    return image

  def step(self, env, tran):
    """Call once per env step, with the crafter.Env and the transition."""
    if len(self.done) >= self.episodes:
      return
    if tran['is_first']:
      self.frames = []
      self.log = {'actions': [], 'reward': [], 'unlocked': [],
                  **{k: [] for k in STATS}}
    self.frames.append(self.frame(env))
    log, t = self.log, len(self.frames) - 1
    # The action chosen after seeing this frame.
    log['actions'].append(int(tran['action']))
    log['reward'].append(round(float(tran['reward']), 2))
    for k in STATS:
      log[k].append(int(env._player.inventory[k]))
    seen = {name for _, name in log['unlocked']}
    for name, count in env._player.achievements.items():
      if count > 0 and name not in seen:
        log['unlocked'].append([t, name])
    if tran['is_last']:
      log['died'] = bool(tran['is_terminal'])
      self.save()

  def save(self):
    world = len(self.done) + 1
    name = f'world{world}.mp4'
    # yuv420p and faststart: plays in every browser. A key frame every
    # 10 frames (-g 10) makes jumping to a step fast.
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
           '-pix_fmt', 'rgb24', '-s', f'{SIZE[0]}x{SIZE[1]}', '-r', str(FPS),
           '-i', '-', '-c:v', 'libx264', '-crf', '26', '-preset', 'slow',
           '-g', '10', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
           str(self.folder / name)]
    subprocess.run(cmd, input=np.stack(self.frames).tobytes(), check=True)
    self.done.append({'world': world, 'video': name, **self.log})
    # Written after each episode, so a crash keeps the finished ones.
    (self.folder / 'gameplay.json').write_text(json.dumps({
        'fps': FPS, 'action_names': list(crafter.constants.actions),
        'achievements': sorted(crafter.constants.achievements),
        'episodes': self.done}, separators=(',', ':')) + '\n')
    print(f'recorded {name}: {len(self.frames)} frames', flush=True)
