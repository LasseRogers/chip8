"""Tests for environment factories."""

import gymnasium as gym
import numpy as np
import pytest

from chip8_rl import PONG_ID
from chip8_rl.envs import STACK_SIZE, make_env


def test_stacks_last_screens_oldest_first():
    stacked, plain = make_env("pong"), gym.make(PONG_ID)
    obs, _ = stacked.reset(seed=0)
    history = [plain.reset(seed=0)[0]]
    assert obs.shape == (STACK_SIZE, 32, 64)

    for action in [1, 1, 2, 2, 1, 0]:
        obs, *_ = stacked.step(action)
        history.append(plain.step(action)[0])

    assert (obs == np.stack(history[-STACK_SIZE:])).all()


def test_unknown_game_lists_valid_names():
    with pytest.raises(ValueError, match="pong, invaders"):
        make_env("tetris")
