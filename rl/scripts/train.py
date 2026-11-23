"""Train a DQN agent on chip8/Pong-v0.

Usage: uv run python scripts/train.py [--steps N] [--seed N]
"""

import argparse
import copy
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from chip8_rl.device import best_device
from chip8_rl.dqn import QNetwork, ReplayBuffer, dqn_update, linear_epsilon, select_action
from chip8_rl.envs import make_pong_env


@dataclass(frozen=True)
class Config:
    """DeepMind's Atari DQN settings, scaled down for CHIP-8 Pong."""

    total_steps: int = 500_000  # agent decisions; x4 frames with frame skip
    buffer_size: int = 100_000
    learning_starts: int = 10_000  # act randomly first to fill the buffer
    batch_size: int = 32
    train_every: int = 4
    target_sync_every: int = 2_000
    gamma: float = 0.99
    lr: float = 1e-4
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 100_000
    seed: int = 0


def train(cfg: Config) -> None:
    device = best_device()
    print(f"training on {device} for {cfg.total_steps:,} steps (seed {cfg.seed})")
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    env = make_pong_env()
    online = QNetwork(env.action_space.n).to(device)
    target = copy.deepcopy(online)
    optimizer = torch.optim.Adam(online.parameters(), lr=cfg.lr)
    buffer = ReplayBuffer(cfg.buffer_size, env.observation_space.shape)

    obs, _ = env.reset(seed=cfg.seed)
    episode, episode_reward, losses = 0, 0.0, []
    recent_rewards = deque(maxlen=20)
    start = time.perf_counter()

    for step in range(cfg.total_steps):
        epsilon = linear_epsilon(step, cfg.epsilon_start, cfg.epsilon_end, cfg.epsilon_decay_steps)
        action, _ = select_action(online, obs, epsilon, rng)
        next_obs, reward, terminated, truncated, info = env.step(action)
        buffer.add(obs, action, reward, next_obs, terminated)
        obs, episode_reward = next_obs, episode_reward + reward

        if step >= cfg.learning_starts and step % cfg.train_every == 0:
            batch = buffer.sample(cfg.batch_size, rng)
            losses.append(dqn_update(online, target, optimizer, batch, cfg.gamma))
        if step % cfg.target_sync_every == 0:
            target.load_state_dict(online.state_dict())

        if terminated or truncated:
            episode += 1
            recent_rewards.append(episode_reward)
            loss = f"{np.mean(losses):.4f}" if losses else "  -   "
            steps_per_sec = (step + 1) / (time.perf_counter() - start)
            print(
                f"episode {episode:>4}  step {step + 1:>7,}  eps {epsilon:.2f}  "
                f"score {info['agent_score']}-{info['opponent_score']}  "
                f"reward {episode_reward:+3.0f}  avg20 {np.mean(recent_rewards):+5.1f}  "
                f"loss {loss}  {steps_per_sec:,.0f} steps/s",
                flush=True,
            )
            obs, _ = env.reset()
            episode_reward, losses = 0.0, []


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a DQN agent on chip8/Pong-v0.")
    parser.add_argument("--steps", type=int, default=Config.total_steps)
    parser.add_argument("--seed", type=int, default=Config.seed)
    args = parser.parse_args()
    try:
        train(Config(total_steps=args.steps, seed=args.seed))
    except KeyboardInterrupt:
        print("stopped")


if __name__ == "__main__":
    main()
