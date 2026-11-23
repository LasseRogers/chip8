"""Shared Gymnasium plumbing for CHIP-8 games."""

from pathlib import Path
from typing import Any

import chip8
import gymnasium as gym
import numpy as np
from gymnasium import spaces

ROMS_DIR = Path(__file__).resolve().parents[3] / "roms"
SCREEN_SHAPE = (32, 64)
# Frames per agent decision, as in DeepMind's Atari setup.
FRAME_SKIP = 4
# Screen pixels per CHIP-8 pixel in rendered frames.
RENDER_SCALE = 8


class Chip8GameEnv(gym.Env):
    """A CHIP-8 ROM as a Gymnasium environment; each subclass defines one game.

    Subclasses set `ROM`, `ACTIONS` and optionally `PROFILE`, and implement
    `_progress`, `_reward` and `_terminated`. `_start` and `_info` are optional hooks.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60 // FRAME_SKIP}

    ROM: str  # File name in ROMS_DIR.
    PROFILE = "cosmac"  # Interpreter behaviour the ROM expects.
    ACTIONS: dict[str, tuple[int, ...]]  # Action name -> CHIP-8 keys held during the step.

    def __init__(self, render_mode: str | None = None, rom_path: Path | None = None) -> None:
        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError(f"unsupported render_mode: {render_mode!r}")
        self.render_mode = render_mode
        self._window = None
        self._clock = None
        rom_path = Path(rom_path) if rom_path is not None else ROMS_DIR / self.ROM
        if not rom_path.is_file():
            raise FileNotFoundError(
                f"ROM not found: {rom_path}. Download {self.ROM} into the repo's roms/ folder."
            )
        self._rom = rom_path.read_bytes()
        self._chip8: chip8.Chip8 | None = None
        self._state: Any = None
        self._action_keys = tuple(self.ACTIONS.values())
        self.action_space = spaces.Discrete(len(self.ACTIONS))
        self.observation_space = spaces.Box(0, 1, SCREEN_SHAPE, dtype=np.uint8)

    @property
    def action_names(self) -> tuple[str, ...]:
        return tuple(self.ACTIONS)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        # Seed the emulator from Gymnasium's RNG so reset(seed=...) is reproducible.
        self._chip8 = chip8.Chip8(seed=int(self.np_random.integers(2**63)), profile=self.PROFILE)
        self._chip8.load_rom(self._rom)
        self._start()
        self._state = self._progress()
        if self.render_mode == "human":
            self._draw_window()
        return self._observation(), self._info(self._state)

    def step(self, action: int):
        before = self._state
        self._run_frames(FRAME_SKIP, self._action_keys[action])
        self._state = self._progress()
        reward = float(self._reward(before, self._state))
        terminated = self._terminated(self._state)
        if self.render_mode == "human":
            self._draw_window()
        return self._observation(), reward, terminated, False, self._info(self._state)

    # --- Game-specific hooks ---

    def _start(self) -> None:
        """Runs after the ROM loads, e.g. to press a start key."""

    def _progress(self) -> Any:
        """Reads the game state that rewards and termination depend on."""
        raise NotImplementedError

    def _reward(self, before: Any, after: Any) -> float:
        raise NotImplementedError

    def _terminated(self, state: Any) -> bool:
        raise NotImplementedError

    def _info(self, state: Any) -> dict:
        return {}

    # --- Emulation and rendering ---

    def _run_frames(self, frames: int, keys: tuple[int, ...] = ()) -> None:
        """Runs `frames` frames with `keys` held, then releases them."""
        for key in keys:
            self._chip8.set_key(key, True)
        for _ in range(frames):
            self._chip8.run_frame()
        for key in keys:
            self._chip8.set_key(key, False)

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
            pygame.display.set_caption(self.spec.id if self.spec else type(self).__name__)
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

    def _observation(self) -> np.ndarray:
        return np.frombuffer(self._chip8.screen(), dtype=np.uint8).reshape(SCREEN_SHAPE)
