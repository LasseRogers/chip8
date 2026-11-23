"""Tests for the training loop's outputs."""

import json

import torch

from chip8_rl.train import CHECKPOINT_NAME, METRICS_NAME, Config, load_checkpoint, train


def test_short_run_writes_metrics_and_loadable_checkpoint(tmp_path):
    cfg = Config(total_steps=3_000, buffer_size=3_000, learning_starts=500, checkpoint_every=1_000)
    train(cfg, tmp_path)

    lines = (tmp_path / METRICS_NAME).read_text().splitlines()
    records = [json.loads(line) for line in lines]
    assert records
    assert {"episode", "step", "epsilon", "reward", "loss"} <= records[0].keys()

    net, data = load_checkpoint(tmp_path / CHECKPOINT_NAME, torch.device("cpu"))
    assert data["step"] == 3_000
    assert data["config"]["total_steps"] == 3_000
    assert net(torch.zeros((1, 4, 32, 64), dtype=torch.uint8)).shape == (1, 3)
