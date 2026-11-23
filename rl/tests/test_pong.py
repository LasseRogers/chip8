"""Tests for PongEnv."""

import os

# Exercise the human render path without opening real windows.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from chip8_rl import PONG_ID
from chip8_rl.pong import RENDER_SCALE, WIN_SCORE


@pytest.fixture
def env() -> gym.Env:
    return gym.make(PONG_ID)


def play(env: gym.Env, seed: int) -> list[tuple[float, dict]]:
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
    # The checker wants the raw env; it reads the spec that make() attached.
    check_env(gym.make(PONG_ID, render_mode="rgb_array").unwrapped)


def test_same_seed_reproduces_episode(env):
    assert play(env, seed=0) == play(env, seed=0)


def test_episode_ends_at_win_score_with_unit_rewards(env):
    trace = play(env, seed=1)
    final = trace[-1][1]
    assert WIN_SCORE in (final["agent_score"], final["opponent_score"])
    assert {reward for reward, _ in trace} <= {-1.0, 0.0, 1.0}
    assert sum(reward for reward, _ in trace) == final["agent_score"] - final["opponent_score"]


def test_rgb_array_render_is_upscaled_screen():
    env = gym.make(PONG_ID, render_mode="rgb_array")
    env.reset(seed=0)
    for _ in range(10):
        obs, *_ = env.step(0)
    frame = env.render()
    assert frame.shape == (32 * RENDER_SCALE, 64 * RENDER_SCALE, 3)
    assert frame.dtype == np.uint8
    assert (frame[::RENDER_SCALE, ::RENDER_SCALE, 0] == obs * 255).all()


def test_human_render_runs_and_closes():
    env = gym.make(PONG_ID, render_mode="human")
    env.reset(seed=0)
    for _ in range(3):
        env.step(0)
    env.close()


def test_closing_human_window_interrupts():
    import pygame

    env = gym.make(PONG_ID, render_mode="human")
    env.reset(seed=0)
    pygame.event.post(pygame.event.Event(pygame.QUIT))
    with pytest.raises(KeyboardInterrupt):
        env.step(0)
