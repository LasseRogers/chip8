"""Watch a training run's latest checkpoint play Pong, live.

Usage: uv run python scripts/view.py RUN_DIR

Between games, reloads the checkpoint if training has saved a newer one,
so the window follows a run while it trains. Beside the game, a panel shows
the network's Q-value for each action at every decision; below, the run's
training curves update as episodes finish.
"""

import argparse
import os
import time
from collections import deque
from pathlib import Path

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame

from chip8_rl.device import best_device
from chip8_rl.display import (
    HIGHLIGHT_COLOR,
    MUTED_COLOR,
    TEXT_COLOR,
    blit_frame,
    draw_line_chart,
    draw_q_values,
)
from chip8_rl.dqn import EVAL_EPSILON, select_action
from chip8_rl.envs import make_pong_env
from chip8_rl.metrics import MetricsReader
from chip8_rl.train import CHECKPOINT_NAME, METRICS_NAME, load_checkpoint

FPS = 15  # Decisions per second: real game speed with frame skip 4.
METRICS_POLL_SECONDS = 2.0
AVERAGE_WINDOW = 20  # Matches the trainer's avg20.
STATUS_HEIGHT = 28
PANEL_WIDTH = 280
CHART_HEIGHT = 160
PADDING = 16


def quit_requested() -> bool:
    return any(event.type == pygame.QUIT for event in pygame.event.get())


class Viewer:
    """Game top-left, Q-values top-right, status below the game, curves along the bottom."""

    def __init__(self, game_size: tuple[int, int], labels: tuple[str, ...], title: str) -> None:
        pygame.display.init()
        pygame.font.init()
        self.game_width, self.game_height = game_size
        self.labels = labels
        self.window = pygame.display.set_mode(
            (self.game_width + PANEL_WIDTH, self.game_height + STATUS_HEIGHT + CHART_HEIGHT)
        )
        pygame.display.set_caption(title)
        self.font = pygame.font.Font(None, 22)
        self.rewards: list[float] = []
        self.average_rewards: list[float] = []
        self.losses: list[float] = []
        self.epsilons: list[float] = []
        self._recent = deque(maxlen=AVERAGE_WINDOW)

    def add_records(self, records: list[dict]) -> None:
        for record in records:
            self._recent.append(record["reward"])
            self.rewards.append(record["reward"])
            self.average_rewards.append(float(np.mean(self._recent)))
            self.epsilons.append(record["epsilon"])
            if record["loss"] is not None:  # No loss before learning starts.
                self.losses.append(record["loss"])

    def draw(
        self,
        status: str,
        frame: np.ndarray | None = None,
        q_values: np.ndarray | None = None,
        action: int = 0,
    ) -> None:
        self.window.fill((0, 0, 0))
        if frame is not None:
            blit_frame(self.window, frame)
        if q_values is not None:
            left = self.game_width + PADDING
            self.window.blit(self.font.render("Q-values", True, TEXT_COLOR), (left, PADDING))
            panel = pygame.Rect(
                left, PADDING + 28, PANEL_WIDTH - 2 * PADDING, 36 * len(self.labels)
            )
            draw_q_values(self.window, panel, self.labels, q_values, action, self.font)
        self.window.blit(self.font.render(status, True, TEXT_COLOR), (8, self.game_height + 6))
        self._draw_charts()
        pygame.display.flip()

    def _draw_charts(self) -> None:
        top = self.game_height + STATUS_HEIGHT
        width = (self.window.get_width() - 4 * PADDING) // 3
        charts = [
            (
                "reward (avg20)",
                [(self.rewards, MUTED_COLOR), (self.average_rewards, HIGHLIGHT_COLOR)],
            ),
            ("loss", [(self.losses, HIGHLIGHT_COLOR)]),
            ("epsilon", [(self.epsilons, HIGHLIGHT_COLOR)]),
        ]
        for i, (title, series) in enumerate(charts):
            rect = pygame.Rect(PADDING + i * (width + PADDING), top, width, CHART_HEIGHT - PADDING)
            draw_line_chart(self.window, rect, title, series, self.font)


def view(run_dir: Path) -> None:
    checkpoint = run_dir / CHECKPOINT_NAME
    metrics = MetricsReader(run_dir / METRICS_NAME)
    device = best_device()
    rng = np.random.default_rng()
    env = make_pong_env(render_mode="rgb_array")
    env.reset()
    height, width, _ = env.render().shape
    viewer = Viewer((width, height), env.unwrapped.action_names, f"chip8 viewer: {run_dir}")
    clock = pygame.time.Clock()
    net, step, loaded_mtime = None, 0, None
    last_poll = 0.0

    def poll_metrics() -> None:
        nonlocal last_poll
        if time.monotonic() - last_poll >= METRICS_POLL_SECONDS:
            viewer.add_records(metrics.read_new())
            last_poll = time.monotonic()

    while True:
        # Wait for the first checkpoint; training saves one every 10k steps.
        while not checkpoint.exists():
            poll_metrics()
            viewer.draw(f"waiting for {checkpoint} ...")
            if quit_requested():
                return
            time.sleep(0.5)

        mtime = checkpoint.stat().st_mtime
        if mtime != loaded_mtime:
            net, data = load_checkpoint(checkpoint, device)
            step, loaded_mtime = data["step"], mtime

        obs, info = env.reset()
        done = False
        while not done:
            if quit_requested():
                return
            poll_metrics()
            action, q_values = select_action(net, obs, EVAL_EPSILON, rng)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            score = f"{info['agent_score']} - {info['opponent_score']}"
            status = f"checkpoint step {step:,}   |   score {score}"
            viewer.draw(status, env.render(), q_values, action)
            clock.tick(FPS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch a run's latest checkpoint play Pong.")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    try:
        view(args.run_dir)
    except KeyboardInterrupt:
        pass
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
