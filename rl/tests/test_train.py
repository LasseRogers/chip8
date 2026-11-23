"""Tests for the training loop's outputs."""

import json

import pytest
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


def test_refuses_to_reuse_a_run_dir(tmp_path):
    (tmp_path / METRICS_NAME).write_text("")
    with pytest.raises(FileExistsError, match="already holds a run"):
        train(Config(total_steps=1), tmp_path)


def test_trains_on_any_registered_game(tmp_path):
    cfg = Config(game="invaders", total_steps=1_500, buffer_size=1_500, learning_starts=500)
    train(cfg, tmp_path)

    record = json.loads((tmp_path / METRICS_NAME).read_text().splitlines()[0])
    assert "invaders_alive" in record["info"]
    _, data = load_checkpoint(tmp_path / CHECKPOINT_NAME, torch.device("cpu"))
    assert (data["config"]["game"], data["n_actions"]) == ("invaders", 6)
