"""Tests for PongEnv."""

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from chip8_rl import PongEnv
from chip8_rl.pong import RENDER_SCALE, WIN_SCORE


@pytest.fixture
def env() -> PongEnv:
    return PongEnv()


def play(env: PongEnv, seed: int) -> list[tuple[float, dict]]:
    """Plays one seeded random episode; returns (reward, info) per step."""
    env.reset(seed=seed)
    env.action_space.seed(seed)
    trace = []
    while True:
        _, reward, terminated, truncated, info = env.step(env.action_space.sample())
        trace.append((reward, info))
        if terminated or truncated:
            return trace


def test_passes_gymnasium_env_checker():
    check_env(PongEnv(render_mode="rgb_array"))


def test_same_seed_reproduces_episode(env):
    assert play(env, seed=0) == play(env, seed=0)


def test_episode_ends_at_win_score_with_unit_rewards(env):
    trace = play(env, seed=1)
    final = trace[-1][1]
    assert WIN_SCORE in (final["agent_score"], final["opponent_score"])
    assert {reward for reward, _ in trace} <= {-1.0, 0.0, 1.0}
    assert sum(reward for reward, _ in trace) == final["agent_score"] - final["opponent_score"]


def test_rgb_array_render_is_upscaled_screen():
    env = PongEnv(render_mode="rgb_array")
    env.reset(seed=0)
    for _ in range(10):
        obs, *_ = env.step(0)
    frame = env.render()
    assert frame.shape == (32 * RENDER_SCALE, 64 * RENDER_SCALE, 3)
    assert frame.dtype == np.uint8
    assert (frame[::RENDER_SCALE, ::RENDER_SCALE, 0] == obs * 255).all()
