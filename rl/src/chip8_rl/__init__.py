"""Gymnasium environments backed by the chip8 emulator."""

import os

# Set before any module imports pygame: silences its startup banner.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from gymnasium.envs.registration import register

from chip8_rl.pong import PongEnv

PONG_ID = "chip8/Pong-v0"

# Natural episodes end at 9 points (1k-5.5k steps); the cap stops endless rallies.
register(id=PONG_ID, entry_point=PongEnv, max_episode_steps=20_000)

__all__ = ["PONG_ID", "PongEnv"]
