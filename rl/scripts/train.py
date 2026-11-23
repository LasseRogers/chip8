"""Train a DQN agent on chip8/Pong-v0.

Usage: uv run python scripts/train.py [--steps N] [--seed N] [--run-dir DIR]
"""

import argparse
import time
from pathlib import Path

from chip8_rl.train import Config, train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a DQN agent on chip8/Pong-v0.")
    parser.add_argument("--steps", type=int, default=Config.total_steps)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument(
        "--run-dir", type=Path, default=Path("runs") / time.strftime("%Y%m%d-%H%M%S")
    )
    args = parser.parse_args()
    try:
        train(Config(total_steps=args.steps, seed=args.seed), args.run_dir)
    except KeyboardInterrupt:
        print("stopped; checkpoint saved")


if __name__ == "__main__":
    main()
