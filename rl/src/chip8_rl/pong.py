"""One-player CHIP-8 Pong (pong1p.ch8) as a Gymnasium environment.

The agent controls the left paddle against the ROM's built-in opponent.
"""

from pathlib import Path

import chip8
import gymnasium as gym
import numpy as np
from gymnasium import spaces

ROM_PATH = Path(__file__).resolve().parents[3] / "roms" / "pong1p.ch8"
SCREEN_SHAPE = (32, 64)
# Score digits written by the ROM's FX33: agent (tens of VE), opponent (ones).
SCORE_ADDR = 0x2F3
# Both scores share VE, so a 10th opponent point would carry into ours.
WIN_SCORE = 9
# Frames per agent decision, as in DeepMind's Atari setup.
FRAME_SKIP = 4


class PongEnv(gym.Env):
    metadata = {"render_modes": []}

    # Action index -> CHIP-8 key held during the step; None holds nothing.
    ACTION_KEYS = (None, 0x1, 0x4)  # noop, up, down

    def __init__(self, rom_path: Path = ROM_PATH) -> None:
        self._rom = Path(rom_path).read_bytes()
        self._chip8: chip8.Chip8 | None = None
        self.action_space = spaces.Discrete(len(self.ACTION_KEYS))
        self.observation_space = spaces.Box(0, 1, SCREEN_SHAPE, dtype=np.uint8)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        # Seed the emulator from Gymnasium's RNG so reset(seed=...) is reproducible.
        self._chip8 = chip8.Chip8(seed=int(self.np_random.integers(2**63)))
        self._chip8.load_rom(self._rom)
        return self._observation(), self._info()

    def step(self, action: int):
        key = self.ACTION_KEYS[action]
        agent_before, opponent_before = self._scores()

        if key is not None:
            self._chip8.set_key(key, True)
        for _ in range(FRAME_SKIP):
            self._chip8.run_frame()
        if key is not None:
            self._chip8.set_key(key, False)

        agent, opponent = self._scores()
        reward = float((agent - agent_before) - (opponent - opponent_before))
        terminated = WIN_SCORE in (agent, opponent)
        return self._observation(), reward, terminated, False, self._info()

    def _scores(self) -> tuple[int, int]:
        agent, opponent = self._chip8.read_memory(SCORE_ADDR, 2)
        return agent, opponent

    def _info(self) -> dict:
        agent, opponent = self._scores()
        return {"agent_score": agent, "opponent_score": opponent}

    def _observation(self) -> np.ndarray:
        return np.frombuffer(self._chip8.screen(), dtype=np.uint8).reshape(SCREEN_SHAPE)
