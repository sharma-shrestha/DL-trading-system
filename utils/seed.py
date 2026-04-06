"""Global random seed setter for full reproducibility."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set the random seed across all stochastic components.

    Applies the seed to Python's built-in ``random`` module, NumPy,
    and PyTorch (both CPU and CUDA).  Pass the same seed to
    Stable-Baselines3 agents at construction time to achieve full
    pipeline reproducibility.

    Args:
        seed: Integer seed value (e.g. 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
