"""Compute device selection."""

import torch


def best_device() -> torch.device:
    """CUDA if available, then Apple's MPS, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
