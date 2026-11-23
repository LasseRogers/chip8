"""Environment factories for training and evaluation."""

import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation

from chip8_rl import PONG_ID

# One screen can't show the ball's direction or speed; four can.
STACK_SIZE = 4


def make_pong_env(render_mode: str | None = None) -> gym.Env:
    """Pong with the last `STACK_SIZE` screens stacked, oldest first."""
    return FrameStackObservation(gym.make(PONG_ID, render_mode=render_mode), STACK_SIZE)
