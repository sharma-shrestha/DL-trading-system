"""
Diffusion model training loop (DDPM).

Trains DiffusionModel on sliding windows using standard DDPM loss:
MSE between predicted noise and actual noise.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from diffusion.model import DiffusionModel


def train_diffusion(model: DiffusionModel, windows: np.ndarray, config: dict) -> DiffusionModel:
    """
    Train the diffusion model on sliding windows.

    Args:
        model: DiffusionModel instance
        windows: np.ndarray of shape (N, window_size, 4) — training windows
                 window_size must be >= context_len + future_steps (100 + 20 = 120)
        config: full config dict

    Returns:
        Trained DiffusionModel
    """
    diff_cfg = config["diffusion"]
    context_len: int = diff_cfg["context_len"]       # 100
    future_steps: int = diff_cfg["future_steps"]     # 20
    batch_size: int = diff_cfg["batch_size"]         # 256
    epochs: int = diff_cfg["epochs"]                 # 50
    checkpoint_path: str = diff_cfg["checkpoint_path"]  # outputs/models/diffusion.pt
    seed: int = config["seed"]                       # 42

    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Adapt context/target split to actual window size.
    # If window_size >= context_len + future_steps, use the designed split.
    # Otherwise split the window in half: first half = context, second half = target.
    actual_window = windows.shape[1]
    if actual_window >= context_len + future_steps:
        context_np = windows[:, :context_len, :]
        target_np = windows[:, context_len:context_len + future_steps, :]
    else:
        # window too small — use first half as context, last future_steps rows as target
        ctx_len = max(1, actual_window - future_steps)
        context_np = windows[:, :ctx_len, :]
        target_np = windows[:, ctx_len:ctx_len + future_steps, :]
        # Clamp future_steps if still not enough rows
        if target_np.shape[1] == 0:
            future_steps = max(1, actual_window // 2)
            ctx_len = actual_window - future_steps
            context_np = windows[:, :ctx_len, :]
            target_np = windows[:, ctx_len:, :]

    context_t = torch.tensor(context_np, dtype=torch.float32)
    target_t = torch.tensor(target_np, dtype=torch.float32)

    dataset = TensorDataset(context_t, target_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    mse_loss = nn.MSELoss()

    T = model.train_timesteps  # 1000

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (context_batch, target_batch) in enumerate(loader, 1):
            context_batch = context_batch.to(device)
            target_batch = target_batch.to(device)
            B = target_batch.shape[0]

            t = torch.randint(0, T, (B,), device=device, dtype=torch.long)
            eps = torch.randn_like(target_batch)

            sqrt_alpha = model.sqrt_alphas_cumprod[t].view(B, 1, 1)
            sqrt_one_minus_alpha = model.sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1)
            x_noisy = sqrt_alpha * target_batch + sqrt_one_minus_alpha * eps

            eps_pred = model(x_noisy, t, context_batch)
            loss = mse_loss(eps_pred, eps)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            # Print batch progress every 50 batches
            if batch_idx % 50 == 0:
                print(f"  Epoch [{epoch}/{epochs}] Batch [{batch_idx}/{len(loader)}] loss: {loss.item():.6f}", flush=True)

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch [{epoch}/{epochs}] avg loss: {avg_loss:.6f}", flush=True)

    # Save checkpoint
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

    return model
