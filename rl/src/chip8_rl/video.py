"""Records evaluation games as MP4 clips."""

from pathlib import Path

import imageio
import numpy as np
from torch import nn

from chip8_rl.dqn import EVAL_EPSILON, select_action
from chip8_rl.envs import make_env

CLIP_FPS = 15  # Real game speed: 60 frames/s with frame skip 4.


def record_clip(net: nn.Module, game: str, path: Path, seed: int, max_steps: int = 900) -> dict:
    """Plays one game with `net`, writing up to `max_steps` decisions to `path` as MP4.

    Returns the final step's info, with both scores.
    """
    rng = np.random.default_rng(seed)
    env = make_env(game, render_mode="rgb_array")
    obs, info = env.reset(seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=CLIP_FPS, codec="libx264", macro_block_size=16) as writer:
        writer.append_data(env.render())
        for _ in range(max_steps):
            action, _ = select_action(net, obs, EVAL_EPSILON, rng)
            obs, _, terminated, truncated, info = env.step(action)
            writer.append_data(env.render())
            if terminated or truncated:
                break
    env.close()
    return info
