"""
Backtest metric functions for the BTC RL Trading System.

Functions:
    sharpe_ratio(returns)       — annualized Sharpe ratio
    max_drawdown(portfolio_values) — largest peak-to-trough decline
    win_rate(trades)            — fraction of profitable trades
    profit_factor(trades)       — gross profit / gross loss

Requirements: 11.2
"""

from __future__ import annotations

import math


def sharpe_ratio(returns: list[float]) -> float:
    """Compute the annualized Sharpe ratio.

    Annualized Sharpe = (mean(returns) / std(returns)) * sqrt(252).
    Returns 0.0 if the standard deviation is zero or the list is empty.

    Args:
        returns: Sequence of per-step returns (e.g. daily P&L fractions).

    Returns:
        Annualized Sharpe ratio as a float.
    """
    if len(returns) < 2:
        return 0.0

    n = len(returns)
    mean = sum(returns) / n
    variance = sum((r - mean) ** 2 for r in returns) / (n - 1)
    std = math.sqrt(variance)

    if std == 0.0:
        return 0.0

    return (mean / std) * math.sqrt(252)


def max_drawdown(portfolio_values: list[float]) -> float:
    """Compute the maximum drawdown as a fraction of the peak value.

    Max drawdown = max over all t of (peak_t - value_t) / peak_t.
    Returns 0.0 if the list is empty or has only one element.

    Args:
        portfolio_values: Sequence of portfolio values over time.

    Returns:
        Maximum drawdown in [0, 1] (e.g. 0.25 means a 25% decline).
    """
    if len(portfolio_values) < 2:
        return 0.0

    peak = portfolio_values[0]
    max_dd = 0.0

    for value in portfolio_values:
        if value > peak:
            peak = value
        if peak > 0.0:
            drawdown = (peak - value) / peak
            if drawdown > max_dd:
                max_dd = drawdown

    return max_dd


def win_rate(trades: list[float]) -> float:
    """Compute the fraction of profitable trades.

    A trade is profitable if its P&L > 0.
    Returns 0.0 if the trades list is empty.

    Args:
        trades: Sequence of per-trade P&L values.

    Returns:
        Win rate in [0, 1].
    """
    if not trades:
        return 0.0

    wins = sum(1 for pnl in trades if pnl > 0)
    return wins / len(trades)


def profit_factor(trades: list[float]) -> float:
    """Compute the profit factor (gross profit / gross loss).

    Returns 0.0 if there are no winning trades (all losses or no trades).
    Returns math.inf if there are winning trades but no losing trades.

    Args:
        trades: Sequence of per-trade P&L values.

    Returns:
        Profit factor as a float (>= 0, possibly inf).
    """
    if not trades:
        return 0.0

    gross_profit = sum(pnl for pnl in trades if pnl > 0)
    gross_loss = sum(abs(pnl) for pnl in trades if pnl < 0)

    if gross_profit == 0.0:
        return 0.0

    if gross_loss == 0.0:
        return math.inf

    return gross_profit / gross_loss
