"""
TradingEnv: Gymnasium-compatible trading environment for the BTC RL Trading System.

Observation space (flat vector):
    - Price window:      50 × 4 = 200 values (flattened)
    - Volatility scalar: 1 value
    - Diffusion output:  N × 4 values (flattened, N = config['diffusion']['future_steps'])
    Total: 200 + 1 + N*4

Action space: Discrete(3)
    0 = Hold
    1 = Buy  (opens long when flat; penalty when already long)
    2 = Sell (closes long when long; penalty when flat)

Reward:
    reward = profit - transaction_cost - risk_penalty - holding_penalty
"""

from __future__ import annotations

import numpy as np
import gymnasium
from gymnasium import spaces


class TradingEnv(gymnasium.Env):
    """Gymnasium trading environment backed by pre-built price windows.

    Args:
        windows:   np.ndarray of shape (N, window_size, 4) — price windows.
                   Column index 0 is 'close' (normalized).
        diffusion: DiffusionInference instance, or None for RL-only mode
                   (diffusion portion of observation is zeroed out).
        config:    Full config dict (see config.yaml).
    """

    metadata = {"render_modes": []}

    def __init__(self, windows: np.ndarray, diffusion, config: dict):
        super().__init__()

        self._windows = windows          # (num_windows, window_size, 4)
        self._diffusion = diffusion
        self._config = config

        # ---- Config shortcuts ------------------------------------------------
        env_cfg = config["env"]
        self._transaction_cost: float = env_cfg["transaction_cost"]
        self._risk_penalty_weight: float = env_cfg["risk_penalty_weight"]
        self._holding_penalty: float = env_cfg["holding_penalty"]
        self._min_portfolio_value: float = env_cfg["min_portfolio_value"]
        self._initial_balance: float = env_cfg["initial_balance"]

        window_size: int = config["windows"]["window_size"]
        future_steps: int = config["diffusion"]["future_steps"]

        # ---- Observation / action spaces ------------------------------------
        obs_dim = window_size * 4 + 1 + future_steps * 4
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)

        # ---- Pre-cache diffusion outputs for all windows -------------------
        # This avoids running the expensive UNet at every RL step.
        self._diff_cache: np.ndarray | None = None
        if self._diffusion is not None:
            self._diff_cache = self._precompute_diffusion(windows, future_steps)

        # ---- Internal state (initialised in reset) --------------------------
        self._step_idx: int = 0
        self._position: int = 0          # 0 = flat, 1 = long
        self._entry_price: float = 0.0
        self._portfolio_value: float = self._initial_balance
        self._num_trades: int = 0
        self._invalid_penalty: float = self._transaction_cost * 1.0

    def _precompute_diffusion(self, windows: np.ndarray, future_steps: int) -> np.ndarray:
        """Pre-generate diffusion outputs for all windows in one batched pass.

        Returns array of shape (N, future_steps * 4).
        """
        context_len: int = self._config["diffusion"]["context_len"]
        N = len(windows)
        batch_size = 256  # process in chunks to avoid OOM
        results = []

        print(f"[TradingEnv] Pre-caching diffusion outputs for {N} windows...", flush=True)
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = windows[start:end]  # (B, window_size, 4)

            # Build context for each window
            B, W, _ = batch.shape
            if W >= context_len:
                ctx = batch[:, -context_len:, :].astype(np.float32)
            else:
                pad = np.zeros((B, context_len - W, 4), dtype=np.float32)
                ctx = np.concatenate([pad, batch.astype(np.float32)], axis=1)

            out = self._diffusion.generate(ctx, n_steps=future_steps)  # (B, future_steps, 4)
            results.append(out.reshape(end - start, -1).astype(np.float32))

            if (start // batch_size) % 10 == 0:
                print(f"  Cached {end}/{N} windows", flush=True)

        print("[TradingEnv] Diffusion cache ready.", flush=True)
        return np.concatenate(results, axis=0)  # (N, future_steps*4)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._step_idx = 0
        self._position = 0
        self._entry_price = 0.0
        self._portfolio_value = self._initial_balance
        self._num_trades = 0

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int):
        window = self._windows[self._step_idx]   # (window_size, 4)
        close_price: float = float(window[-1, 0])  # last row, close column

        # Volatility: std of the close column across the window
        volatility: float = float(np.std(window[:, 0]))

        profit: float = 0.0
        transaction_cost: float = 0.0
        invalid_penalty: float = 0.0

        # ---- Position state machine ----------------------------------------
        if action == 1:  # Buy
            if self._position == 0:
                # Open long
                self._position = 1
                self._entry_price = close_price
                transaction_cost = self._transaction_cost * abs(close_price)
                self._num_trades += 1
            else:
                # Already long — invalid
                invalid_penalty = self._invalid_penalty

        elif action == 2:  # Sell
            if self._position == 1:
                # Close long — realize P&L
                profit = close_price - self._entry_price
                transaction_cost = self._transaction_cost * abs(close_price)
                self._portfolio_value += profit
                self._position = 0
                self._entry_price = 0.0
                self._num_trades += 1
            else:
                # No position — invalid
                invalid_penalty = self._invalid_penalty

        # action == 0 (Hold): always valid, no trade

        # ---- Reward components ---------------------------------------------
        risk_penalty: float = self._risk_penalty_weight * volatility
        holding_penalty: float = self._holding_penalty if self._position == 1 else 0.0

        # Diffusion alignment bonus: reward agent when diffusion predicted direction correctly
        diffusion_bonus: float = 0.0
        if self._diff_cache is not None and self._step_idx > 0 and profit != 0.0:
            prev_idx = self._step_idx - 1
            if prev_idx < len(self._diff_cache):
                diff_out = self._diff_cache[prev_idx]  # (future_steps*4,)
                # First predicted close value (index 0 = close feature)
                predicted_next = float(diff_out[0])
                actual_next = close_price
                entry = self._entry_price if self._entry_price != 0.0 else actual_next
                # Bonus if diffusion predicted the right direction
                predicted_up = predicted_next > entry
                actual_up = actual_next > entry
                if predicted_up == actual_up:
                    diffusion_bonus = abs(profit) * 0.1  # 10% bonus on correct prediction

        reward: float = (
            profit
            - transaction_cost
            - risk_penalty
            - holding_penalty
            - invalid_penalty
            + diffusion_bonus
        )

        # ---- Advance step --------------------------------------------------
        self._step_idx += 1

        # ---- Termination conditions ----------------------------------------
        min_value = self._min_portfolio_value * self._initial_balance
        terminated: bool = (
            self._step_idx >= len(self._windows)
            or self._portfolio_value < min_value
        )
        truncated: bool = False

        obs = self._get_obs() if not terminated else self._zero_obs()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def render(self):
        """Print a brief summary of the current state."""
        print(
            f"Step {self._step_idx} | "
            f"Position: {'long' if self._position else 'flat'} | "
            f"Portfolio: {self._portfolio_value:.2f} | "
            f"Trades: {self._num_trades}"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Build the flat observation vector for the current step."""
        if self._step_idx >= len(self._windows):
            return self._zero_obs()

        window = self._windows[self._step_idx]   # (window_size, 4)

        # Price window — flattened
        price_part = window.flatten().astype(np.float32)

        # Volatility scalar (std of close column)
        volatility = np.array([np.std(window[:, 0])], dtype=np.float32)

        # Diffusion output — use pre-cached value (fast lookup)
        future_steps: int = self._config["diffusion"]["future_steps"]
        if self._diff_cache is not None:
            diff_part = self._diff_cache[self._step_idx]  # already (future_steps*4,)
        else:
            diff_part = np.zeros(future_steps * 4, dtype=np.float32)

        obs = np.concatenate([price_part, volatility, diff_part])
        return obs

    def _zero_obs(self) -> np.ndarray:
        """Return a zero observation of the correct shape (used after termination)."""
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def _get_info(self) -> dict:
        return {
            "portfolio_value": float(self._portfolio_value),
            "num_trades": int(self._num_trades),
            "position": int(self._position),
        }
