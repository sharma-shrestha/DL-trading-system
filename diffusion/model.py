"""
DiffusionModel: 1D UNet-based DDPM conditioned on past 100 time steps.
Supports DDIM inference for fast scenario generation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sinusoidal_embedding(t: Tensor, dim: int) -> Tensor:
    """Sinusoidal positional embedding for diffusion timestep t.

    Args:
        t:   (batch,) integer timestep indices
        dim: embedding dimension (must be even)

    Returns:
        (batch, dim) float tensor
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / (half - 1)
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)


class ResBlock1d(nn.Module):
    """Conv1d residual block with GroupNorm + SiLU and time/context conditioning."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        """
        x:     (B, in_ch, L)
        t_emb: (B, time_dim)
        """
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.time_proj(t_emb).unsqueeze(-1)  # broadcast over L
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.skip(x)


# ---------------------------------------------------------------------------
# DiffusionModel
# ---------------------------------------------------------------------------

class DiffusionModel(nn.Module):
    """1D UNet denoising diffusion model conditioned on a past price window.

    Architecture:
        - Sinusoidal time embedding
        - Context encoder: flatten(100×4) → linear projection → time_dim
        - Encoder: 3 ResBlock1d stages with 2× downsampling
        - Bottleneck: 2 ResBlocks
        - Decoder: 3 ResBlock1d stages with 2× upsampling + skip connections
        - Output projection: → 4 channels

    Config keys used:
        config['diffusion']['train_timesteps']  (default 1000)
        config['diffusion']['inference_steps']  (default 50)
        config['diffusion']['future_steps']     (default 20)
    """

    def __init__(self, config: dict):
        super().__init__()
        diff_cfg = config["diffusion"]
        self.train_timesteps: int = diff_cfg["train_timesteps"]
        self.inference_steps: int = diff_cfg["inference_steps"]
        self.future_steps: int = diff_cfg["future_steps"]

        in_channels = 4          # OHLCV-derived features
        # context_len is the actual number of timesteps fed as context.
        # When window_size < context_len, the actual context is window_size.
        window_size = config.get("windows", {}).get("window_size", 50)
        context_len = min(diff_cfg["context_len"], window_size)
        time_dim = 128
        base_ch = 64

        # ---- Time embedding ------------------------------------------------
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        self._time_dim = time_dim

        # ---- Context encoder -----------------------------------------------
        context_input_dim = context_len * in_channels
        self.context_encoder = nn.Sequential(
            nn.Linear(context_input_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        self._context_input_dim = context_input_dim

        # ---- Input projection ----------------------------------------------
        self.input_proj = nn.Conv1d(in_channels, base_ch, kernel_size=1)

        # ---- Encoder -------------------------------------------------------
        ch = base_ch
        self.enc1 = ResBlock1d(ch, ch, time_dim)
        self.down1 = nn.Conv1d(ch, ch * 2, kernel_size=3, stride=2, padding=1)

        ch2 = ch * 2
        self.enc2 = ResBlock1d(ch2, ch2, time_dim)
        self.down2 = nn.Conv1d(ch2, ch2 * 2, kernel_size=3, stride=2, padding=1)

        ch4 = ch2 * 2
        self.enc3 = ResBlock1d(ch4, ch4, time_dim)
        self.down3 = nn.Conv1d(ch4, ch4 * 2, kernel_size=3, stride=2, padding=1)

        ch8 = ch4 * 2

        # ---- Bottleneck ----------------------------------------------------
        self.mid1 = ResBlock1d(ch8, ch8, time_dim)
        self.mid2 = ResBlock1d(ch8, ch8, time_dim)

        # ---- Decoder -------------------------------------------------------
        self.up3 = nn.ConvTranspose1d(ch8, ch4, kernel_size=2, stride=2)
        self.dec3 = ResBlock1d(ch4 + ch4, ch4, time_dim)

        self.up2 = nn.ConvTranspose1d(ch4, ch2, kernel_size=2, stride=2)
        self.dec2 = ResBlock1d(ch2 + ch2, ch2, time_dim)

        self.up1 = nn.ConvTranspose1d(ch2, ch, kernel_size=2, stride=2)
        self.dec1 = ResBlock1d(ch + ch, ch, time_dim)

        # ---- Output projection ---------------------------------------------
        self.out_proj = nn.Conv1d(ch, in_channels, kernel_size=1)

        # ---- DDPM noise schedule -------------------------------------------
        self._register_noise_schedule()

    # ------------------------------------------------------------------
    # Noise schedule
    # ------------------------------------------------------------------

    def _register_noise_schedule(self):
        T = self.train_timesteps
        betas = torch.linspace(1e-4, 0.02, T)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt())

    # ------------------------------------------------------------------
    # Forward pass (noise prediction)
    # ------------------------------------------------------------------

    def forward(self, x_noisy: Tensor, t: Tensor, context: Tensor) -> Tensor:
        """Predict noise given noisy future sequence, timestep, and past context.

        Args:
            x_noisy: (B, n_steps, 4)  — noisy future price sequence
            t:       (B,)             — diffusion timestep indices (int)
            context: (B, 100, 4)      — past price window

        Returns:
            (B, n_steps, 4) predicted noise
        """
        B = x_noisy.shape[0]

        # Time embedding
        t_emb = sinusoidal_embedding(t, self._time_dim)   # (B, time_dim)
        t_emb = self.time_mlp(t_emb)                       # (B, time_dim)

        # Context embedding — add to time embedding
        ctx = context.reshape(B, -1)                       # (B, context_len*4)
        # Pad or truncate to match the expected context_input_dim
        if ctx.shape[1] != self._context_input_dim:
            if ctx.shape[1] < self._context_input_dim:
                pad = torch.zeros(B, self._context_input_dim - ctx.shape[1], device=ctx.device)
                ctx = torch.cat([ctx, pad], dim=1)
            else:
                ctx = ctx[:, :self._context_input_dim]
        ctx_emb = self.context_encoder(ctx)                # (B, time_dim)
        t_emb = t_emb + ctx_emb                            # fuse

        # x_noisy: (B, n_steps, 4) → (B, 4, n_steps) for Conv1d
        x = x_noisy.permute(0, 2, 1)
        x = self.input_proj(x)                             # (B, base_ch, L)

        # Encoder
        e1 = self.enc1(x, t_emb)                          # (B, 64, L)
        x = self.down1(e1)                                 # (B, 128, L/2)

        e2 = self.enc2(x, t_emb)                          # (B, 128, L/2)
        x = self.down2(e2)                                 # (B, 256, L/4)

        e3 = self.enc3(x, t_emb)                          # (B, 256, L/4)
        x = self.down3(e3)                                 # (B, 512, L/8)

        # Bottleneck
        x = self.mid1(x, t_emb)
        x = self.mid2(x, t_emb)

        # Decoder with skip connections
        x = self.up3(x)                                    # (B, 256, L/4)
        x = self._match_and_cat(x, e3)
        x = self.dec3(x, t_emb)

        x = self.up2(x)                                    # (B, 128, L/2)
        x = self._match_and_cat(x, e2)
        x = self.dec2(x, t_emb)

        x = self.up1(x)                                    # (B, 64, L)
        x = self._match_and_cat(x, e1)
        x = self.dec1(x, t_emb)

        x = self.out_proj(x)                               # (B, 4, L)
        return x.permute(0, 2, 1)                          # (B, L, 4)

    @staticmethod
    def _match_and_cat(x: Tensor, skip: Tensor) -> Tensor:
        """Trim or pad x to match skip's spatial dim, then concatenate."""
        if x.shape[-1] != skip.shape[-1]:
            x = x[..., : skip.shape[-1]]
        return torch.cat([x, skip], dim=1)

    # ------------------------------------------------------------------
    # DDIM inference
    # ------------------------------------------------------------------

    def generate(self, context: Tensor, n_steps: int) -> Tensor:
        """Generate future price scenarios via DDIM inference.

        Args:
            context: (B, 100, 4) — past price window (conditioning)
            n_steps: number of future steps to generate, must be in [10, 50]

        Returns:
            (B, n_steps, 4) generated future sequence
        """
        if not (10 <= n_steps <= 50):
            raise ValueError(f"n_steps must be in [10, 50], got {n_steps}")

        device = context.device
        B = context.shape[0]
        T = self.train_timesteps
        S = self.inference_steps

        # DDIM timestep schedule: evenly spaced from T-1 down to 0
        ddim_timesteps = torch.linspace(T - 1, 0, S, dtype=torch.long, device=device)

        # Start from pure noise
        x = torch.randn(B, n_steps, 4, device=device)

        self.eval()
        with torch.no_grad():
            for i, t_val in enumerate(ddim_timesteps):
                t_batch = t_val.expand(B)                  # (B,)

                # Predicted noise
                eps = self.forward(x, t_batch, context)    # (B, n_steps, 4)

                # DDIM update
                alpha_t = self.alphas_cumprod[t_val]
                if i + 1 < S:
                    t_prev = ddim_timesteps[i + 1]
                    alpha_prev = self.alphas_cumprod[t_prev]
                else:
                    alpha_prev = torch.ones(1, device=device)

                # Predicted x0
                x0_pred = (x - (1 - alpha_t).sqrt() * eps) / alpha_t.sqrt()
                x0_pred = x0_pred.clamp(-10.0, 10.0)

                # Direction pointing to x_t
                dir_xt = (1 - alpha_prev).sqrt() * eps

                x = alpha_prev.sqrt() * x0_pred + dir_xt

        return x
