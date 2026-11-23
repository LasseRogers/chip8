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
# Screen pixels per CHIP-8 pixel in rendered frames.
RENDER_SCALE = 8


class PongEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60 // FRAME_SKIP}

    # Action index -> CHIP-8 key held during the step; None holds nothing.
    ACTION_KEYS = (None, 0x1, 0x4)  # noop, up, down
    ACTION_NAMES = ("nothing", "up", "down")

    def __init__(self, render_mode: str | None = None, rom_path: Path = ROM_PATH) -> None:
        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError(f"unsupported render_mode: {render_mode!r}")
        self.render_mode = render_mode
        self._window = None
        self._clock = None
        rom_path = Path(rom_path)
        if not rom_path.is_file():
            raise FileNotFoundError(
                f"ROM not found: {rom_path}. Download pong1p.ch8 into the repo's roms/ folder."
            )
        self._rom = rom_path.read_bytes()
        self._chip8: chip8.Chip8 | None = None
        self.action_space = spaces.Discrete(len(self.ACTION_KEYS))
        self.observation_space = spaces.Box(0, 1, SCREEN_SHAPE, dtype=np.uint8)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        # Seed the emulator from Gymnasium's RNG so reset(seed=...) is reproducible.
        self._chip8 = chip8.Chip8(seed=int(self.np_random.integers(2**63)))
        self._chip8.load_rom(self._rom)
        if self.render_mode == "human":
            self._draw_window()
        return self._observation(), self._info(*self._scores())

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
        if self.render_mode == "human":
            self._draw_window()
        return self._observation(), reward, terminated, False, self._info(agent, opponent)

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            return self._frame()
        if self.render_mode == "human":
            self._draw_window()
        return None

    def close(self) -> None:
        if self._window is not None:
            import pygame

            pygame.display.quit()
            self._window = None

    def _draw_window(self) -> None:
        """Shows the current frame and paces the loop to real time."""
        # Imported lazily: only human mode needs a display.
        import pygame

        from chip8_rl.display import blit_frame

        if self._window is None:
            pygame.display.init()  # Display only; full init also opens audio.
            self._window = pygame.display.set_mode((64 * RENDER_SCALE, 32 * RENDER_SCALE))
            pygame.display.set_caption("chip8/Pong-v0")
            self._clock = pygame.time.Clock()
        # Draining events keeps the window responsive. Closing it stops the
        # program, the same as Ctrl+C.
        if any(event.type == pygame.QUIT for event in pygame.event.get()):
            self.close()
            raise KeyboardInterrupt("window closed")
        blit_frame(self._window, self._frame())
        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])

    def _frame(self) -> np.ndarray:
        """The screen as an upscaled RGB image, white on black."""
        pixels = (self._observation() * 255).repeat(RENDER_SCALE, 0).repeat(RENDER_SCALE, 1)
        return np.stack([pixels] * 3, axis=-1)

    def _scores(self) -> tuple[int, int]:
        agent, opponent = self._chip8.read_memory(SCORE_ADDR, 2)
        return agent, opponent

    @staticmethod
    def _info(agent: int, opponent: int) -> dict:
        return {"agent_score": agent, "opponent_score": opponent}

    def _observation(self) -> np.ndarray:
        return np.frombuffer(self._chip8.screen(), dtype=np.uint8).reshape(SCREEN_SHAPE)
