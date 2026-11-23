"""Tests for InvadersEnv."""

import gymnasium as gym
import pytest
from gymnasium.utils.env_checker import check_env

from chip8_rl import INVADERS_ID, InvadersEnv
from chip8_rl.invaders import FIRST_ROW, InvadersState


@pytest.fixture
def env() -> gym.Env:
    return gym.make(INVADERS_ID)


def play(env: gym.Env, action: int | None, seed: int = 0) -> list[tuple[float, bool]]:
    """Plays one episode with a fixed action, or random ones if `action` is None."""
    env.reset(seed=seed)
    env.action_space.seed(seed)
    trace = []
    while True:
        chosen = env.action_space.sample() if action is None else action
        _, reward, terminated, truncated, _ = env.step(chosen)
        trace.append((reward, terminated))
        if terminated or truncated:
            return trace


def test_passes_gymnasium_env_checker():
    check_env(gym.make(INVADERS_ID, render_mode="rgb_array").unwrapped)


def test_reset_starts_in_a_full_first_wave(env):
    _, info = env.reset(seed=0)
    assert info == {"invaders_alive": 4, "invader_row": FIRST_ROW}


def test_idle_player_loses_without_reward(env):
    trace = play(env, action=0)
    assert trace[-1][1], "episode should end by game over, not time limit"
    assert sum(reward for reward, _ in trace) == 0


def test_firing_kills_invaders(env):
    fire = env.unwrapped.action_names.index("fire")
    assert sum(reward for reward, _ in play(env, action=fire)) >= 1


def test_same_seed_reproduces_episode(env):
    assert play(env, action=None, seed=3) == play(env, action=None, seed=3)


@pytest.mark.parametrize(
    ("before", "after", "kills"),
    [
        (InvadersState(0, 0b1111, 4), InvadersState(0, 0b1011, 4), 1),  # one hit
        (InvadersState(0, 0b1111, 4), InvadersState(0, 0b1111, 6), 0),  # moved down
        (InvadersState(0, 0b0001, 8), InvadersState(6, 0b1111, 4), 1),  # last one: wave cleared
        (InvadersState(0, 0b0000, 8), InvadersState(6, 0b1111, 4), 0),  # clear seen earlier
    ],
)
def test_reward_counts_kills_including_wave_clears(before, after, kills):
    assert InvadersEnv()._reward(before, after) == kills
