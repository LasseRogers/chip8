"""Tests for DQN components."""

import numpy as np
import pytest
import torch

from chip8_rl.device import best_device
from chip8_rl.dqn import (
    QNetwork,
    ReplayBuffer,
    dqn_update,
    linear_epsilon,
    next_state_values,
    select_action,
    td_targets,
)
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


def test_linear_epsilon_anneals_then_holds():
    assert linear_epsilon(0, 1.0, 0.05, 100) == 1.0
    assert linear_epsilon(50, 1.0, 0.05, 100) == pytest.approx(0.525)
    assert linear_epsilon(100, 1.0, 0.05, 100) == pytest.approx(0.05)
    assert linear_epsilon(1_000, 1.0, 0.05, 100) == pytest.approx(0.05)


class FixedQ(torch.nn.Module):
    """Stand-in network that always returns the same Q-values."""

    def __init__(self, q_values: list[float]) -> None:
        super().__init__()
        self.q = torch.nn.Parameter(torch.tensor(q_values), requires_grad=False)

    def forward(self, screens: torch.Tensor) -> torch.Tensor:
        return self.q.expand(len(screens), -1)


def test_greedy_picks_highest_q_value():
    rng = np.random.default_rng(0)
    net = FixedQ([0.1, 0.9, -0.5])
    for _ in range(20):
        action, q_values = select_action(net, random_obs(rng), epsilon=0.0, rng=rng)
        assert action == 1
    assert q_values.tolist() == pytest.approx([0.1, 0.9, -0.5])


def test_full_exploration_covers_all_actions():
    rng = np.random.default_rng(0)
    net = FixedQ([0.1, 0.9, -0.5])
    actions = {select_action(net, random_obs(rng), epsilon=1.0, rng=rng)[0] for _ in range(100)}
    assert actions == {0, 1, 2}


def test_td_targets_masks_future_after_termination():
    targets = td_targets(
        rewards=torch.tensor([1.0, 0.0, -1.0]),
        next_q_max=torch.tensor([2.0, 2.0, 2.0]),
        dones=torch.tensor([False, False, True]),
        gamma=0.5,
    )
    assert targets.tolist() == pytest.approx([2.0, 1.0, -1.0])


def test_dqn_update_reduces_loss_on_fixed_batch():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    buffer = ReplayBuffer(capacity=32, obs_shape=OBS_SHAPE)
    for i in range(32):
        buffer.add(random_obs(rng), i % 3, float(i % 3 - 1), random_obs(rng), True)
    batch = buffer.sample(32, rng)

    online, target = QNetwork(n_actions=3), QNetwork(n_actions=3)
    optimizer = torch.optim.Adam(online.parameters(), lr=1e-3)
    first = dqn_update(online, target, optimizer, batch, gamma=0.99)
    for _ in range(50):
        last = dqn_update(online, target, optimizer, batch, gamma=0.99)
    assert last < first / 10


def test_double_dqn_values_the_online_choice_with_the_target_net():
    online = FixedQ([1.0, 0.0])  # Prefers action 0...
    target = FixedQ([0.2, 0.9])  # ...which the target network rates low.
    next_obs = torch.zeros((2, STACK_SIZE, 32, 64), dtype=torch.uint8)

    plain = next_state_values(online, target, next_obs, double=False)
    double = next_state_values(online, target, next_obs, double=True)
    assert plain.tolist() == pytest.approx([0.9, 0.9])
    assert double.tolist() == pytest.approx([0.2, 0.2])
