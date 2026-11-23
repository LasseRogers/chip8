"""Environment factories for training and evaluation."""

import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation

from chip8_rl import GAMES

# One screen can't show the ball's direction or speed; four can.
STACK_SIZE = 4


def make_env(game: str, render_mode: str | None = None) -> gym.Env:
    """The named game with the last `STACK_SIZE` screens stacked, oldest first."""
    if game not in GAMES:
        raise ValueError(f"unknown game {game!r}; choose from: {', '.join(GAMES)}")
    return FrameStackObservation(gym.make(GAMES[game], render_mode=render_mode), STACK_SIZE)
