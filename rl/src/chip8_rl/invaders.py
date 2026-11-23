"""CHIP-8 Space Invaders (David Winter, SUPER-CHIP behaviour).

Shoot each wave of four invaders before they reach the bottom. The ROM keeps
no score and no lives in memory: its state lives in registers, read directly.
The game has no randomness, so every episode starts identically.
"""

from typing import NamedTuple

from chip8_rl.base import Chip8GameEnv

# Register indices, from the ROM's disassembly.
WAVE = 0x9  # Steps by 6 per cleared wave (an offset into the sprite table).
INVADER_ROW = 0xC  # Grows from FIRST_ROW as the invaders descend.
ALIVE_MASK = 0xE  # One bit per invader in the current wave.

FIRST_ROW = 0x04
GAME_OVER_ROW = 0x18
FULL_WAVE = 0x0F
START_KEY = 0x5
MAX_START_FRAMES = 300


class InvadersState(NamedTuple):
    wave: int
    alive: int
    row: int


class InvadersEnv(Chip8GameEnv):
    ROM = "invaders.ch8"
    PROFILE = "superchip"
    # DeepMind's Atari action set: move and fire can combine in one decision.
    ACTIONS = {
        "nothing": (),
        "left": (0x4,),
        "right": (0x6,),
        "fire": (0x5,),
        "left+fire": (0x4, 0x5),
        "right+fire": (0x6, 0x5),
    }

    def _start(self) -> None:
        """Holds fire until the title screen gives way to a full first wave."""
        self._chip8.set_key(START_KEY, True)
        try:
            for _ in range(MAX_START_FRAMES):
                self._chip8.run_frame()
                # Both registers: the ROM sets the row a few instructions before the mask.
                if self._progress()[1:] == (FULL_WAVE, FIRST_ROW):
                    return
            raise RuntimeError("Space Invaders did not leave its title screen")
        finally:
            self._chip8.set_key(START_KEY, False)

    def _progress(self) -> InvadersState:
        registers = self._chip8.registers()
        return InvadersState(registers[WAVE], registers[ALIVE_MASK], registers[INVADER_ROW])

    def _reward(self, before: InvadersState, after: InvadersState) -> float:
        """+1 per invader killed."""
        if after.wave != before.wave:
            # Wave cleared: the mask was refilled, so every invader still alive died.
            return before.alive.bit_count()
        return (before.alive & ~after.alive).bit_count()

    def _terminated(self, state: InvadersState) -> bool:
        return state.row >= GAME_OVER_ROW

    def _info(self, state: InvadersState) -> dict:
        return {"invaders_alive": state.alive.bit_count(), "invader_row": state.row}
