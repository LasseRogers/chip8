"""Watch a training run's latest checkpoint play Pong, live.

Usage: uv run python scripts/view.py RUN_DIR

Between games, reloads the checkpoint if training has saved a newer one,
so the window follows a run while it trains.
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame

from chip8_rl.device import best_device
from chip8_rl.display import blit_frame
from chip8_rl.dqn import select_action
from chip8_rl.envs import make_pong_env
from chip8_rl.train import CHECKPOINT_NAME, load_checkpoint

EPSILON = 0.02  # A little randomness, so an untrained net can't freeze forever.
FPS = 15  # Decisions per second: real game speed with frame skip 4.
STATUS_HEIGHT = 28
TEXT_COLOR = (200, 200, 200)


def quit_requested() -> bool:
    return any(event.type == pygame.QUIT for event in pygame.event.get())


def draw(
    window: pygame.Surface, font: pygame.font.Font, frame: np.ndarray | None, text: str
) -> None:
    window.fill((0, 0, 0))
    if frame is not None:
        blit_frame(window, frame)
    window.blit(font.render(text, True, TEXT_COLOR), (8, window.get_height() - STATUS_HEIGHT + 6))
    pygame.display.flip()


def view(run_dir: Path) -> None:
    checkpoint = run_dir / CHECKPOINT_NAME
    device = best_device()
    rng = np.random.default_rng()
    env = make_pong_env(render_mode="rgb_array")
    env.reset()
    height, width, _ = env.render().shape

    pygame.display.init()
    pygame.font.init()
    window = pygame.display.set_mode((width, height + STATUS_HEIGHT))
    pygame.display.set_caption(f"chip8 viewer: {run_dir}")
    font = pygame.font.Font(None, 22)
    clock = pygame.time.Clock()
    net, step, loaded_mtime = None, 0, None

    while True:
        # Wait for the first checkpoint; training saves one every 10k steps.
        while not checkpoint.exists():
            draw(window, font, None, f"waiting for {checkpoint} ...")
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
            action, _ = select_action(net, obs, EPSILON, rng)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            score = f"{info['agent_score']} - {info['opponent_score']}"
            status = f"checkpoint step {step:,}   |   score {score}"
            draw(window, font, env.render(), status)
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
