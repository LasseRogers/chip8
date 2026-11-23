"""Deep Q-Network components."""

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
