"""
DiffusionInference: thin wrapper around DiffusionModel for checkpoint loading
and scenario generation. Used by TradingEnv to generate future price scenarios.
"""

import numpy as np
import torch

from diffusion.model import DiffusionModel


class DiffusionInference:
    """Loads a trained DiffusionModel from disk and exposes a generate() method.

    Config keys used:
        config['diffusion']['checkpoint_path']  — path to saved .pt checkpoint
        config['diffusion']['future_steps']     — default number of future steps
    """

    def __init__(self, config: dict):
        self._config = config
        diff_cfg = config["diffusion"]
        checkpoint_path: str = diff_cfg["checkpoint_path"]

        self._future_steps: int = diff_cfg["future_steps"]
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build model architecture then load weights
        self._model = DiffusionModel(config)
        checkpoint = torch.load(checkpoint_path, map_location=self._device)

        # Support both raw state_dict and wrapped checkpoint dicts
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        self._model.load_state_dict(state_dict)
        self._model.to(self._device)
        self._model.eval()

    def generate(self, context: np.ndarray, n_steps: int = None) -> np.ndarray:
        """Generate future price scenarios using DDIM inference.

        Args:
            context: np.ndarray of shape (B, 100, 4) or (100, 4) for a single sample
            n_steps: number of future steps to generate (default: config future_steps)

        Returns:
            np.ndarray of shape (B, n_steps, 4)
        """
        if n_steps is None:
            n_steps = self._future_steps

        # Handle single-sample input: (100, 4) → (1, 100, 4)
        squeeze = False
        if context.ndim == 2:
            context = context[np.newaxis]  # (1, 100, 4)
            squeeze = True

        ctx_tensor = torch.from_numpy(context.astype(np.float32)).to(self._device)

        with torch.no_grad():
            output = self._model.generate(ctx_tensor, n_steps)  # (B, n_steps, 4)

        result = output.cpu().numpy()

        if squeeze:
            result = result[0]  # (n_steps, 4)

        return result
