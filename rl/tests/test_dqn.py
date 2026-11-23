"""Tests for DQN components."""

import numpy as np
import torch

from chip8_rl.device import best_device
from chip8_rl.dqn import QNetwork, ReplayBuffer
from chip8_rl.envs import STACK_SIZE


def test_outputs_one_q_value_per_action():
    net = QNetwork(n_actions=3)
    screens = torch.randint(0, 2, (5, STACK_SIZE, 32, 64), dtype=torch.uint8)
    assert net(screens).shape == (5, 3)


def test_runs_on_best_device():
    device = best_device()
    net = QNetwork(n_actions=3).to(device)
    screens = torch.zeros((1, STACK_SIZE, 32, 64), dtype=torch.uint8, device=device)
    assert net(screens).device.type == device.type


OBS_SHAPE = (STACK_SIZE, 32, 64)


def random_obs(rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 2, OBS_SHAPE, dtype=np.uint8)


def test_replay_buffer_round_trips_packed_screens():
    rng = np.random.default_rng(0)
    obs, next_obs = random_obs(rng), random_obs(rng)
    buffer = ReplayBuffer(capacity=1, obs_shape=OBS_SHAPE)
    buffer.add(obs, action=2, reward=-1.0, next_obs=next_obs, done=True)

    batch = buffer.sample(1, rng)
    assert (batch.obs[0] == obs).all()
    assert (batch.next_obs[0] == next_obs).all()
    assert (batch.actions[0], batch.rewards[0], batch.dones[0]) == (2, -1.0, True)


def test_replay_buffer_overwrites_oldest_when_full():
    rng = np.random.default_rng(0)
    buffer = ReplayBuffer(capacity=3, obs_shape=OBS_SHAPE)
    for action in range(5):
        buffer.add(random_obs(rng), action, 0.0, random_obs(rng), False)

    assert len(buffer) == 3
    assert set(buffer.sample(100, rng).actions) == {2, 3, 4}


def test_replay_buffer_sample_shapes():
    rng = np.random.default_rng(0)
    buffer = ReplayBuffer(capacity=10, obs_shape=OBS_SHAPE)
    for _ in range(10):
        buffer.add(random_obs(rng), 0, 0.0, random_obs(rng), False)

    batch = buffer.sample(4, rng)
    assert batch.obs.shape == batch.next_obs.shape == (4, *OBS_SHAPE)
    assert batch.actions.shape == batch.rewards.shape == batch.dones.shape == (4,)
