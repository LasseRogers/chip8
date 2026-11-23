"""Gymnasium environments backed by the chip8 emulator."""

import os

# Set before any module imports pygame: silences its startup banner.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from gymnasium.envs.registration import register

from chip8_rl.invaders import InvadersEnv
from chip8_rl.pong import PongEnv

PONG_ID = "chip8/Pong-v0"
INVADERS_ID = "chip8/Invaders-v0"
# Short names for scripts, e.g. `--game invaders`.
GAMES = {"pong": PONG_ID, "invaders": INVADERS_ID}

# Episodes end on their own; the cap stops endless rallies or waves.
register(id=PONG_ID, entry_point=PongEnv, max_episode_steps=20_000)
register(id=INVADERS_ID, entry_point=InvadersEnv, max_episode_steps=20_000)

__all__ = ["GAMES", "INVADERS_ID", "PONG_ID", "InvadersEnv", "PongEnv"]
