"""Tests for clip recording."""

import imageio

from chip8_rl.dqn import QNetwork
from chip8_rl.video import record_clip


def test_record_clip_writes_playable_mp4(tmp_path):
    path = tmp_path / "clips" / "clip.mp4"
    info = record_clip(QNetwork(n_actions=3), "pong", path, seed=0, max_steps=20)

    assert {"agent_score", "opponent_score"} <= info.keys()
    reader = imageio.get_reader(path)
    assert reader.count_frames() == 21  # Reset frame plus one per step.
    assert reader.get_meta_data()["size"] == (512, 256)
