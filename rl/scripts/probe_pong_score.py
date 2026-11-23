"""Confirm where pong1p keeps the score.

pong1p stores both scores in VE as `left * 10 + right` and writes its decimal
digits to 0x2F2..0x2F4 (via FX33) whenever it redraws the score. With the
agent's (left) paddle idle, the opponent's digit should climb to 9.
"""

from pathlib import Path

import chip8

ROM = Path(__file__).resolve().parents[2] / "roms" / "pong1p.ch8"
SCORE_ADDR = 0x2F3  # agent (tens digit), then opponent (ones digit)
MAX_FRAMES = 60 * 60 * 30  # 30 minutes of game time at 60 fps


def main() -> None:
    c = chip8.Chip8(seed=0)
    c.load_rom(ROM.read_bytes())
    last = None
    for frame in range(MAX_FRAMES):
        c.run_frame()
        agent, opponent = c.read_memory(SCORE_ADDR, 2)
        if (agent, opponent) != last:
            print(f"frame {frame:>6} ({frame / 60:6.1f}s)  agent {agent}  opponent {opponent}")
            last = (agent, opponent)
        if 9 in last:
            return
    print("no side reached 9 within the frame limit")


if __name__ == "__main__":
    main()
