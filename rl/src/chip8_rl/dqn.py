"""Deep Q-Network components."""

import math
from typing import NamedTuple

import numpy as np
import torch
from torch import nn

from chip8_rl.envs import STACK_SIZE


class QNetwork(nn.Module):
    """Maps stacked CHIP-8 screens to one Q-value per action.

    DeepMind's Atari DQN architecture, resized for 32x64 binary screens.
    """

    def __init__(self, n_actions: int, in_frames: int = STACK_SIZE) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_frames, 32, kernel_size=4, stride=2),  # -> 32 x 15 x 31
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # -> 64 x 6 x 14
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # -> 64 x 4 x 12
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 4 * 12, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, screens: torch.Tensor) -> torch.Tensor:
        """`screens`: uint8 (batch, frames, 32, 64) of 0/1 pixels."""
        # Pixels are already 0/1, so no scaling: just make them floats.
        return self.layers(screens.float())


class Batch(NamedTuple):
    obs: np.ndarray  # uint8 (batch, *obs_shape)
    actions: np.ndarray  # int64 (batch,)
    rewards: np.ndarray  # float32 (batch,)
    next_obs: np.ndarray  # uint8 (batch, *obs_shape)
    dones: np.ndarray  # bool (batch,): episode terminated, not truncated


class ReplayBuffer:
    """Fixed-size ring buffer of transitions for experience replay.

    Screens hold only 0/1 pixels, so they are bit-packed (8 per byte):
    100k transitions take ~200 MB instead of ~1.6 GB.
    """

    def __init__(self, capacity: int, obs_shape: tuple[int, ...]) -> None:
        self.capacity = capacity
        self._obs_shape = obs_shape
        self._pixels = math.prod(obs_shape)
        packed_size = (self._pixels + 7) // 8
        self._obs = np.zeros((capacity, packed_size), dtype=np.uint8)
        self._next_obs = np.zeros((capacity, packed_size), dtype=np.uint8)
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.bool_)
        self._next = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(
        self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool
    ) -> None:
        """Stores a transition, overwriting the oldest once full.

        `done` must be True only when the episode terminated; a truncated
        episode's next state still has future value.
        """
        i = self._next
        self._obs[i] = np.packbits(obs)
        self._next_obs[i] = np.packbits(next_obs)
        self._actions[i] = action
        self._rewards[i] = reward
        self._dones[i] = done
        self._next = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> Batch:
        """Uniformly samples `batch_size` stored transitions, with replacement."""
        idx = rng.integers(0, self._size, size=batch_size)
        return Batch(
            obs=self._unpack(self._obs[idx]),
            actions=self._actions[idx],
            rewards=self._rewards[idx],
            next_obs=self._unpack(self._next_obs[idx]),
            dones=self._dones[idx],
        )

    def _unpack(self, packed: np.ndarray) -> np.ndarray:
        pixels = np.unpackbits(packed, axis=1, count=self._pixels)
        return pixels.reshape(-1, *self._obs_shape)


def linear_epsilon(step: int, start: float, end: float, decay_steps: int) -> float:
    """Anneals linearly from `start` to `end` over `decay_steps`, then holds."""
    fraction = min(step / decay_steps, 1.0)
    return start + fraction * (end - start)


@torch.no_grad()
def select_action(
    net: nn.Module, obs: np.ndarray, epsilon: float, rng: np.random.Generator
) -> tuple[int, np.ndarray]:
    """Epsilon-greedy action for one observation.

    Also returns the network's Q-values, computed even when acting randomly,
    so callers can always display what the network thinks.
    """
    device = next(net.parameters()).device
    q_values = net(torch.as_tensor(obs, device=device).unsqueeze(0))[0].cpu().numpy()
    if rng.random() < epsilon:
        return int(rng.integers(len(q_values))), q_values
    return int(q_values.argmax()), q_values


def td_targets(
    rewards: torch.Tensor, next_q_max: torch.Tensor, dones: torch.Tensor, gamma: float
) -> torch.Tensor:
    """Bellman targets: reward plus discounted best next value; no future after termination."""
    return rewards + gamma * next_q_max * (1.0 - dones.float())


def dqn_update(
    online: nn.Module,
    target: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Batch,
    gamma: float,
    max_grad_norm: float = 10.0,
) -> float:
    """One gradient step of `online` towards targets from the frozen `target`; returns the loss."""
    device = next(online.parameters()).device
    obs, actions, rewards, next_obs, dones = (torch.as_tensor(x, device=device) for x in batch)

    with torch.no_grad():
        targets = td_targets(rewards, target(next_obs).max(dim=1).values, dones, gamma)
    predicted = online(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
    # Huber loss: quadratic near zero, linear for large errors, so outliers don't explode.
    loss = nn.functional.smooth_l1_loss(predicted, targets)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(), max_grad_norm)
    optimizer.step()
    return loss.item()
