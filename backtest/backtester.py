"""
Backtester for the BTC RL Trading System.

Runs three strategies on the same test TradingEnv:
    1. HRL agent  (StrategySelector + TradeExecutor)
    2. Buy-and-Hold baseline
    3. Random-Action baseline

Returns a structured results dict keyed by strategy name.

Requirements: 11.1, 11.3, 11.4, 11.5
"""

from __future__ import annotations

import warnings
import numpy as np

from backtest.metrics import sharpe_ratio, max_drawdown, win_rate, profit_factor


# Type alias for a single strategy's backtest result
BacktestResult = dict  # keys: sharpe_ratio, max_drawdown, win_rate, profit_factor, portfolio_values


def _collect_episode(env, step_fn) -> BacktestResult:
    """Run one full episode using step_fn(obs) -> action and collect metrics.

    Args:
        env:     A reset-ready TradingEnv instance.
        step_fn: Callable(obs) -> int action.

    Returns:
        BacktestResult dict.
    """
    obs, _ = env.reset()
    portfolio_values: list[float] = [env._portfolio_value]
    trades: list[float] = []

    prev_portfolio = env._portfolio_value
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = step_fn(obs)
        obs, _reward, terminated, truncated, info = env.step(action)

        current_value = info["portfolio_value"]
        portfolio_values.append(current_value)

        # Detect a closed trade: portfolio value changed and position just went to 0
        # We track P&L as the delta in portfolio value when a sell occurs.
        # Since TradingEnv adds profit directly to portfolio_value on Sell,
        # we capture the delta whenever portfolio_value changes.
        delta = current_value - prev_portfolio
        if delta != 0.0 and info["position"] == 0:
            trades.append(delta)

        prev_portfolio = current_value

    # Compute step-level returns from portfolio value series
    pv = portfolio_values
    returns = [
        (pv[i] - pv[i - 1]) / pv[i - 1] if pv[i - 1] != 0.0 else 0.0
        for i in range(1, len(pv))
    ]

    return {
        "sharpe_ratio": sharpe_ratio(returns),
        "max_drawdown": max_drawdown(portfolio_values),
        "win_rate": win_rate(trades),
        "profit_factor": profit_factor(trades),
        "portfolio_values": portfolio_values,
    }


class Backtester:
    """Runs the HRL agent and two baselines on a held-out test TradingEnv.

    Args:
        config: Full config dict (see config.yaml).
    """

    def __init__(self, config: dict):
        self._config = config
        self._min_test_steps: int = config.get("backtest", {}).get("min_test_steps", 1000)

    def run(
        self,
        strategy_selector,  # StrategySelector instance
        trade_executor,     # TradeExecutor instance
        env,                # TradingEnv — test environment
    ) -> dict:
        """Run all three strategies and return structured results.

        Args:
            strategy_selector: Trained StrategySelector (Level-1 agent).
            trade_executor:    Trained TradeExecutor (Level-2 agent).
            env:               TradingEnv initialised with the test split.

        Returns:
            {
                'hrl_diffusion':  BacktestResult,
                'buy_and_hold':   BacktestResult,
                'random_action':  BacktestResult,
            }
        """
        num_steps = len(env._windows)
        if num_steps < self._min_test_steps:
            warnings.warn(
                f"Test split has only {num_steps} steps, which is fewer than the "
                f"recommended minimum of {self._min_test_steps}. "
                "Backtest results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )

        # ------------------------------------------------------------------ #
        # 1. HRL agent (StrategySelector → TradeExecutor)
        # ------------------------------------------------------------------ #
        def hrl_step(obs: np.ndarray) -> int:
            strategy = strategy_selector.predict(obs)
            action = trade_executor.predict(obs, strategy)
            return action

        hrl_result = _collect_episode(env, hrl_step)

        # ------------------------------------------------------------------ #
        # 2. Buy-and-Hold baseline
        #    Buy at step 0, hold until the last step, sell at the last step.
        # ------------------------------------------------------------------ #
        total_steps = len(env._windows)

        def bah_step(obs: np.ndarray) -> int:
            # Buy on the very first step (step_idx == 0 after reset means
            # we haven't advanced yet; after reset step_idx is 0).
            # We track state via a closure counter.
            step = bah_step._counter
            bah_step._counter += 1

            if step == 0:
                return 1  # Buy
            if step == total_steps - 1:
                return 2  # Sell on last step
            return 0  # Hold

        bah_step._counter = 0
        bah_result = _collect_episode(env, bah_step)

        # ------------------------------------------------------------------ #
        # 3. Random-Action baseline
        # ------------------------------------------------------------------ #
        rng = np.random.default_rng(self._config.get("seed", 42))

        def random_step(obs: np.ndarray) -> int:
            return int(rng.integers(0, 3))

        random_result = _collect_episode(env, random_step)

        return {
            "hrl_diffusion": hrl_result,
            "buy_and_hold": bah_result,
            "random_action": random_result,
        }
