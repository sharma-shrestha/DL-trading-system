"""
TradeExecutor: Level-2 PPO/SAC agent for the BTC RL Trading System.

Wraps Stable-Baselines3 PPO or SAC. Receives the base TradingEnv observation
concatenated with the strategy index (scalar) from the StrategySelector.

The AugmentedEnv wrapper appends the strategy index to every observation and
exposes a set_strategy() method so the caller can inject the current strategy
before each step.

Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 10.1, 10.2, 10.3
"""

from __future__ import annotations

import os
import numpy as np
import gymnasium
from gymnasium import spaces
from stable_baselines3 import PPO, SAC


class AugmentedEnv(gymnasium.Env):
    """Wraps TradingEnv and appends the current strategy index to every observation.

    The strategy index is a scalar in {0, 1, 2} appended as the last element
    of the observation vector, making the augmented observation length
    ``base_obs_dim + 1``.

    Use ``set_strategy(strategy_idx)`` to update the strategy before calling
    ``step()``.
    """

    metadata = {"render_modes": []}

    def __init__(self, env: gymnasium.Env):
        super().__init__()
        self._env = env
        self._strategy: int = 0  # default strategy index

        base_dim = int(np.prod(env.observation_space.shape))
        aug_dim = base_dim + 1  # append scalar strategy index

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(aug_dim,),
            dtype=np.float32,
        )
        self.action_space = env.action_space

    # ------------------------------------------------------------------
    # Strategy injection
    # ------------------------------------------------------------------

    def set_strategy(self, strategy_idx: int) -> None:
        """Set the current strategy index (0=Scalping, 1=Momentum, 2=Mean-Reversion)."""
        self._strategy = int(strategy_idx)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        obs, info = self._env.reset(seed=seed, options=options)
        return self._augment(obs), info

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self._env.step(action)
        return self._augment(obs), reward, terminated, truncated, info

    def render(self):
        return self._env.render()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _augment(self, obs: np.ndarray) -> np.ndarray:
        """Append the strategy index scalar to the observation."""
        strategy_scalar = np.array([float(self._strategy)], dtype=np.float32)
        return np.concatenate([obs.astype(np.float32), strategy_scalar])


class TradeExecutor:
    """Level-2 hierarchical RL agent that executes precise Buy/Sell/Hold decisions.

    The agent receives the base TradingEnv observation concatenated with the
    strategy index provided by the StrategySelector.

    Actions:
        0 = Hold
        1 = Buy
        2 = Sell

    Args:
        env:    A Gymnasium-compatible environment (typically TradingEnv).
        config: Full config dict (see config.yaml).
    """

    def __init__(self, env: gymnasium.Env, config: dict):
        self._config = config
        agent_cfg = config["agents"]["trade_executor"]
        self._total_timesteps: int = agent_cfg["total_timesteps"]
        self._save_path: str = agent_cfg["save_path"]
        algorithm: str = agent_cfg.get("algorithm", "PPO").upper()
        seed: int = config.get("seed", 42)

        # Wrap the environment to augment observations with strategy index
        self._aug_env = AugmentedEnv(env)

        # Select algorithm: PPO (on-policy) or SAC (off-policy)
        if algorithm == "SAC":
            self._model = SAC(
                policy="MlpPolicy",
                env=self._aug_env,
                seed=seed,
                verbose=0,
            )
        else:
            # Default to PPO per requirements 9.1
            self._model = PPO(
                policy="MlpPolicy",
                env=self._aug_env,
                seed=seed,
                verbose=0,
            )

        self._algorithm = algorithm

    def train(self, total_timesteps: int = None) -> None:
        """Train the agent.

        Args:
            total_timesteps: Override the config value if provided.
        """
        steps = total_timesteps if total_timesteps is not None else self._total_timesteps
        self._model.learn(total_timesteps=steps)

    def predict(self, obs: np.ndarray, strategy: int) -> int:
        """Predict a trade action for the given observation and strategy.

        Args:
            obs:      Base observation from TradingEnv.
            strategy: Strategy index in {0, 1, 2} from StrategySelector.

        Returns:
            Action in {0, 1, 2}.
        """
        # Augment the observation with the strategy index
        strategy_scalar = np.array([float(strategy)], dtype=np.float32)
        aug_obs = np.concatenate([obs.astype(np.float32), strategy_scalar])
        action, _ = self._model.predict(aug_obs, deterministic=True)
        return int(action)

    def save(self, path: str = None) -> None:
        """Save the trained model using SB3's built-in serialization.

        Args:
            path: Override the config save path if provided.
        """
        save_path = path if path is not None else self._save_path
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        self._model.save(save_path)

    def load(self, path: str = None) -> None:
        """Load a previously saved model.

        Args:
            path: Override the config save path if provided.
        """
        load_path = path if path is not None else self._save_path
        if self._algorithm == "SAC":
            self._model = SAC.load(load_path, env=self._aug_env)
        else:
            self._model = PPO.load(load_path, env=self._aug_env)
