"""Compare training runs: final and best avg20 reward, and how fast they reached a target.

Usage: uv run python scripts/compare.py RUN_DIR... [--target REWARD]
"""

import argparse
from collections import deque
from pathlib import Path

import numpy as np
import torch

from chip8_rl.metrics import MetricsReader
from chip8_rl.train import CHECKPOINT_NAME, METRICS_NAME

WINDOW = 20  # Matches the trainer's avg20.


def summarize(run_dir: Path, target: float) -> dict:
    config = torch.load(run_dir / CHECKPOINT_NAME, map_location="cpu", weights_only=True)["config"]
    recent = deque(maxlen=WINDOW)
    averages, reached = [], None
    for record in MetricsReader(run_dir / METRICS_NAME).read_new():
        recent.append(record["reward"])
        averages.append(float(np.mean(recent)))
        if reached is None and len(recent) == WINDOW and averages[-1] >= target:
            reached = record["step"]
    return {
        "run": run_dir.name,
        # Runs from before these options existed lack the keys.
        "game": config.get("game", "pong"),
        "dqn": "double" if config.get("double_dqn", False) else "plain",
        "final": averages[-1],
        "best": max(averages),
        "reached": f"{reached:,}" if reached is not None else "never",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare training runs.")
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--target", type=float, default=0.0, help="avg20 reward to reach")
    args = parser.parse_args()

    print(
        f"{'run':<14} {'game':<10} {'dqn':<7} {'final':>7} {'best':>7}  "
        f"avg20 >= {args.target:g} at step"
    )
    for run_dir in args.run_dirs:
        s = summarize(run_dir, args.target)
        print(
            f"{s['run']:<14} {s['game']:<10} {s['dqn']:<7} "
            f"{s['final']:>+7.1f} {s['best']:>+7.1f}  {s['reached']}"
        )


if __name__ == "__main__":
    main()
