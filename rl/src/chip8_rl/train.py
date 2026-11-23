"""DQN training loop with checkpoints and per-episode metrics."""

import copy
import json
import os
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from chip8_rl.device import best_device
from chip8_rl.dqn import QNetwork, ReplayBuffer, dqn_update, linear_epsilon, select_action
from chip8_rl.envs import make_pong_env

CHECKPOINT_NAME = "latest.pt"
METRICS_NAME = "metrics.jsonl"


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
    checkpoint_every: int = 10_000
    seed: int = 0


def save_checkpoint(path: Path, net: QNetwork, cfg: Config, step: int) -> None:
    """Saves atomically via rename, so concurrent readers never see a partial file."""
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "model": net.state_dict(),
            # Plain int: the safe loader rejects NumPy scalars.
            "n_actions": int(net.layers[-1].out_features),
            "config": asdict(cfg),
            "step": step,
        },
        tmp,
    )
    os.replace(tmp, path)


def load_checkpoint(path: Path, device: torch.device) -> tuple[QNetwork, dict]:
    """Returns the checkpoint's network, in eval mode, and the raw checkpoint dict."""
    data = torch.load(path, map_location=device, weights_only=True)
    net = QNetwork(data["n_actions"]).to(device)
    net.load_state_dict(data["model"])
    return net.eval(), data


def train(cfg: Config, run_dir: Path) -> None:
    """Trains on Pong, writing `latest.pt` and `metrics.jsonl` to `run_dir`."""
    run_dir.mkdir(parents=True, exist_ok=True)
    device = best_device()
    print(f"training on {device} for {cfg.total_steps:,} steps (seed {cfg.seed}) -> {run_dir}")
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    env = make_pong_env()
    # Gymnasium's `n` is a NumPy integer; keep NumPy types out of the model.
    online = QNetwork(int(env.action_space.n)).to(device)
    target = copy.deepcopy(online)
    optimizer = torch.optim.Adam(online.parameters(), lr=cfg.lr)
    buffer = ReplayBuffer(cfg.buffer_size, env.observation_space.shape)
    checkpoint = run_dir / CHECKPOINT_NAME

    obs, _ = env.reset(seed=cfg.seed)
    episode, episode_reward, losses, steps_done = 0, 0.0, [], 0
    recent_rewards = deque(maxlen=20)
    start = time.perf_counter()

    with open(run_dir / METRICS_NAME, "a") as metrics:
        try:
            for step in range(cfg.total_steps):
                epsilon = linear_epsilon(
                    step, cfg.epsilon_start, cfg.epsilon_end, cfg.epsilon_decay_steps
                )
                action, _ = select_action(online, obs, epsilon, rng)
                next_obs, reward, terminated, truncated, info = env.step(action)
                buffer.add(obs, action, reward, next_obs, terminated)
                obs, episode_reward = next_obs, episode_reward + reward

                if step >= cfg.learning_starts and step % cfg.train_every == 0:
                    batch = buffer.sample(cfg.batch_size, rng)
                    losses.append(dqn_update(online, target, optimizer, batch, cfg.gamma))
                if step % cfg.target_sync_every == 0:
                    target.load_state_dict(online.state_dict())

                steps_done = step + 1
                if steps_done % cfg.checkpoint_every == 0:
                    save_checkpoint(checkpoint, online, cfg, steps_done)

                if terminated or truncated:
                    episode += 1
                    recent_rewards.append(episode_reward)
                    record = {
                        "episode": episode,
                        "step": steps_done,
                        "epsilon": round(epsilon, 4),
                        "agent_score": info["agent_score"],
                        "opponent_score": info["opponent_score"],
                        "reward": episode_reward,
                        "loss": float(np.mean(losses)) if losses else None,
                        "elapsed": round(time.perf_counter() - start, 1),
                    }
                    metrics.write(json.dumps(record) + "\n")
                    metrics.flush()
                    print(_format(record, np.mean(recent_rewards)), flush=True)
                    obs, _ = env.reset()
                    episode_reward, losses = 0.0, []
        finally:
            # Also runs on Ctrl+C, so an interrupted run keeps its progress.
            save_checkpoint(checkpoint, online, cfg, steps_done)


def _format(record: dict, avg_reward: float) -> str:
    loss = f"{record['loss']:.4f}" if record["loss"] is not None else "  -   "
    return (
        f"episode {record['episode']:>4}  step {record['step']:>7,}  eps {record['epsilon']:.2f}  "
        f"score {record['agent_score']}-{record['opponent_score']}  "
        f"reward {record['reward']:+3.0f}  avg20 {avg_reward:+5.1f}  "
        f"loss {loss}  {record['step'] / record['elapsed']:,.0f} steps/s"
    )
