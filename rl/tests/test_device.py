"""Tests for device selection; hardware availability is faked."""

import torch

from chip8_rl.device import best_device


def test_prefers_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert best_device().type == "cuda"


def test_falls_back_to_mps_then_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert best_device().type == "mps"
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert best_device().type == "cpu"
