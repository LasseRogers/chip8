"""Tests for DQN components."""

import torch

from chip8_rl.device import best_device
from chip8_rl.dqn import QNetwork
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
