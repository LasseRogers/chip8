"""One-player CHIP-8 Pong (pong1p.ch8).

The agent controls the left paddle against the ROM's built-in opponent.
"""

from chip8_rl.base import Chip8GameEnv

# Score digits written by the ROM's FX33: agent (tens of VE), opponent (ones).
SCORE_ADDR = 0x2F3
# Both scores share VE, so a 10th opponent point would carry into ours.
WIN_SCORE = 9


class PongEnv(Chip8GameEnv):
    ROM = "pong1p.ch8"
    ACTIONS = {"nothing": (), "up": (0x1,), "down": (0x4,)}

    def _progress(self) -> tuple[int, int]:
        agent, opponent = self._chip8.read_memory(SCORE_ADDR, 2)
        return agent, opponent

    def _reward(self, before: tuple[int, int], after: tuple[int, int]) -> float:
        return (after[0] - before[0]) - (after[1] - before[1])

    def _terminated(self, state: tuple[int, int]) -> bool:
        return WIN_SCORE in state

    def _info(self, state: tuple[int, int]) -> dict:
        agent, opponent = state
        return {"agent_score": agent, "opponent_score": opponent}
