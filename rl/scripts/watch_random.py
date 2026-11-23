"""Watch a random agent play one game of Pong at real speed."""

import gymnasium as gym

from chip8_rl import PONG_ID


def main() -> None:
    env = gym.make(PONG_ID, render_mode="human")
    env.reset(seed=0)
    env.action_space.seed(0)
    terminated = truncated = False
    while not (terminated or truncated):
        _, _, terminated, truncated, info = env.step(env.action_space.sample())
    env.close()
    print(f"final score: agent {info['agent_score']}, opponent {info['opponent_score']}")


if __name__ == "__main__":
    main()
