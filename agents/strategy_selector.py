"""
StrategySelector: Level-1 PPO agent for the BTC RL Trading System.

Wraps Stable-Baselines3 PPO and operates over a custom Gymnasium environment
that exposes the same observation space as TradingEnv but uses a 3-action
strategy space (0=Scalping, 1=Momentum, 2=Mean-Reversion).

Requirements: 8.1, 8.2, 8.3, 8.5
"""

from __future__ import annotations

import os
import numpy as np
import gymnasium
from gymnasium import spaces
from stable_baselines3 import PPO


class StrategySelectorEnv(gymnasium.Env):
    """Thin wrapper around TradingEnv that keeps the same observation space
    but replaces the action space with Discrete(3) for strategy selection.

    The underlying TradingEnv already has Discrete(3), so this wrapper is
    mainly a semantic layer that makes the strategy intent explicit and
    allows the StrategySelector to be trained independently.
    """

    metadata = {"render_modes": []}

    def __init__(self, env: gymnasium.Env):
        super().__init__()
        self._env = env
        # Inherit observation space from the wrapped env
        self.observation_space = env.observation_space
        # Strategy space: 0=Scalping, 1=Momentum, 2=Mean-Reversion
        self.action_space = spaces.Discrete(3)

    def reset(self, seed=None, options=None):
        return self._env.reset(seed=seed, options=options)

    def step(self, action: int):
        # Map strategy index to a trading action for the underlying env.
        # For training purposes we pass the strategy directly as a trade action
        # so the selector receives reward signal from the environment.
        return self._env.step(action)

    def render(self):
        return self._env.render()


class StrategySelector:
    """Level-1 hierarchical RL agent that selects a high-level trading strategy.

    Strategies:
        0 = Scalping
        1 = Momentum
        2 = Mean-Reversion

    Args:
        env:    A Gymnasium-compatible environment (typically TradingEnv).
        config: Full config dict (see config.yaml).
    """

    def __init__(self, env: gymnasium.Env, config: dict):
        self._config = config
        agent_cfg = config["agents"]["strategy_selector"]
        self._total_timesteps: int = agent_cfg["total_timesteps"]
        self._save_path: str = agent_cfg["save_path"]
        seed: int = config.get("seed", 42)

        # Wrap the environment
        self._env = StrategySelectorEnv(env)

        # Algorithm is always PPO per requirements 8.1
        self._model = PPO(
            policy="MlpPolicy",
            env=self._env,
            seed=seed,
            verbose=0,
        )

    def train(self, total_timesteps: int = None) -> None:
        """Train the PPO agent.

        Args:
            total_timesteps: Override the config value if provided.
        """
        steps = total_timesteps if total_timesteps is not None else self._total_timesteps
        self._model.learn(total_timesteps=steps)

    def predict(self, obs: np.ndarray) -> int:
        """Predict a strategy index for the given observation.

        Args:
            obs: Observation array matching the environment's observation space.

        Returns:
            Strategy index in {0, 1, 2}.
        """
        action, _ = self._model.predict(obs, deterministic=True)
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
        self._model = PPO.load(load_path, env=self._env)
