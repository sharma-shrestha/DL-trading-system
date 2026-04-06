"""
Evaluator for the BTC RL Trading System.

Compares three configurations:
    1. RL+Diffusion  (hrl_diffusion)
    2. RL-Only       (rl_only — same HRL agent with diffusion zeroed out)
    3. Buy-and-Hold  (buy_and_hold)

Generates comparison tables, PNG graphs, and saves a structured report.

Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6
"""

from __future__ import annotations

import os
import json
import math

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for headless environments
import matplotlib.pyplot as plt


# Row labels used in the comparison DataFrame
_ROW_LABELS = ["RL+Diffusion", "RL-Only", "Buy-and-Hold"]

# Keys in the results dict that map to each row
_RESULT_KEYS = ["hrl_diffusion", "rl_only", "buy_and_hold"]

# Metric columns
_METRIC_COLS = ["sharpe_ratio", "max_drawdown", "win_rate", "profit_factor"]


def _zero_result() -> dict:
    """Return a BacktestResult-shaped dict filled with zeros."""
    return {
        "sharpe_ratio": 0.0,
        "max_drawdown": 0.0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "portfolio_values": [1.0],
    }


def _resolve_results(results: dict) -> dict:
    """Ensure all three required keys are present.

    Falls back:
      - 'rl_only'    → 'random_action' if present, else zeros
      - 'buy_and_hold' → zeros if missing
    """
    resolved = dict(results)

    if "rl_only" not in resolved:
        if "random_action" in resolved:
            resolved["rl_only"] = resolved["random_action"]
        else:
            resolved["rl_only"] = _zero_result()

    if "buy_and_hold" not in resolved:
        resolved["buy_and_hold"] = _zero_result()

    if "hrl_diffusion" not in resolved:
        resolved["hrl_diffusion"] = _zero_result()

    return resolved


def _drawdown_series(portfolio_values: list[float]) -> list[float]:
    """Compute per-step drawdown from peak as a fraction."""
    if not portfolio_values:
        return []
    peak = portfolio_values[0]
    drawdowns = []
    for v in portfolio_values:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0.0 else 0.0
        drawdowns.append(dd)
    return drawdowns


class Evaluator:
    """Compares RL+Diffusion, RL-Only, and Buy-and-Hold configurations.

    Args:
        config: Full config dict (see config.yaml).
                Uses config['evaluation']['output_dir'] as the default output directory.
    """

    def __init__(self, config: dict):
        self._config = config
        self._default_output_dir: str = (
            config.get("evaluation", {}).get("output_dir", "outputs/reports/")
        )

    # ---------------------------------------------------------------------- #
    # Public API
    # ---------------------------------------------------------------------- #

    def compare(self, results: dict) -> pd.DataFrame:
        """Build a 3-row × 4-column comparison DataFrame.

        Rows:    'RL+Diffusion', 'RL-Only', 'Buy-and-Hold'
        Columns: 'sharpe_ratio', 'max_drawdown', 'win_rate', 'profit_factor'

        Args:
            results: Dict with keys 'hrl_diffusion', 'buy_and_hold', and optionally
                     'rl_only' (falls back to 'random_action' or zeros if absent).

        Returns:
            pd.DataFrame with shape (3, 4).
        """
        resolved = _resolve_results(results)

        rows = []
        for key in _RESULT_KEYS:
            r = resolved[key]
            rows.append({col: r.get(col, 0.0) for col in _METRIC_COLS})

        df = pd.DataFrame(rows, index=_ROW_LABELS, columns=_METRIC_COLS)
        return df

    def plot(self, results: dict, output_dir: str = None) -> None:
        """Generate 3 PNG files in output_dir.

        Files produced:
            1. portfolio_value.png  — portfolio value over time (line chart)
            2. drawdown.png         — drawdown curve over time
            3. metrics_comparison.png — bar chart of Sharpe, Win Rate, Profit Factor

        Args:
            results:    Dict with backtest results (same format as compare()).
            output_dir: Directory to write PNGs. Defaults to config output_dir.
        """
        out_dir = output_dir or self._default_output_dir
        os.makedirs(out_dir, exist_ok=True)

        resolved = _resolve_results(results)

        colors = {"RL+Diffusion": "#1f77b4", "RL-Only": "#ff7f0e", "Buy-and-Hold": "#2ca02c"}
        label_key_pairs = list(zip(_ROW_LABELS, _RESULT_KEYS))

        # ------------------------------------------------------------------ #
        # 1. Portfolio value over time
        # ------------------------------------------------------------------ #
        fig, ax = plt.subplots(figsize=(10, 5))
        for label, key in label_key_pairs:
            pv = resolved[key].get("portfolio_values", [1.0])
            ax.plot(pv, label=label, color=colors[label])
        ax.set_title("Portfolio Value Over Time")
        ax.set_xlabel("Step")
        ax.set_ylabel("Portfolio Value")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "portfolio_value.png"), format="png")
        plt.close(fig)

        # ------------------------------------------------------------------ #
        # 2. Drawdown curve over time
        # ------------------------------------------------------------------ #
        fig, ax = plt.subplots(figsize=(10, 5))
        for label, key in label_key_pairs:
            pv = resolved[key].get("portfolio_values", [1.0])
            dd = _drawdown_series(pv)
            ax.plot(dd, label=label, color=colors[label])
        ax.set_title("Drawdown Over Time")
        ax.set_xlabel("Step")
        ax.set_ylabel("Drawdown (fraction)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "drawdown.png"), format="png")
        plt.close(fig)

        # ------------------------------------------------------------------ #
        # 3. Bar chart — Sharpe, Win Rate, Profit Factor per config
        #    (max_drawdown excluded from bar chart as lower is better)
        # ------------------------------------------------------------------ #
        bar_metrics = ["sharpe_ratio", "win_rate", "profit_factor"]
        bar_labels = ["Sharpe Ratio", "Win Rate", "Profit Factor"]

        x = np.arange(len(bar_metrics))
        width = 0.25

        fig, ax = plt.subplots(figsize=(9, 5))
        for i, (label, key) in enumerate(label_key_pairs):
            r = resolved[key]
            values = []
            for m in bar_metrics:
                v = r.get(m, 0.0)
                # Cap math.inf for display purposes
                if not math.isfinite(v):
                    v = 0.0
                values.append(v)
            ax.bar(x + i * width, values, width, label=label, color=colors[label])

        ax.set_title("Metrics Comparison")
        ax.set_xticks(x + width)
        ax.set_xticklabels(bar_labels)
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "metrics_comparison.png"), format="png")
        plt.close(fig)

    def save_report(self, df: pd.DataFrame, output_dir: str = None) -> None:
        """Save the comparison DataFrame as CSV and JSON.

        Adds a 'highlight' column: True when RL+Diffusion Sharpe > RL-Only Sharpe.

        Files produced:
            - performance_report.csv
            - performance_report.json

        Args:
            df:         DataFrame returned by compare().
            output_dir: Directory to write files. Defaults to config output_dir.
        """
        out_dir = output_dir or self._default_output_dir
        os.makedirs(out_dir, exist_ok=True)

        df_out = df.copy()

        # Highlight condition: RL+Diffusion Sharpe > RL-Only Sharpe
        rl_diffusion_sharpe = df_out.loc["RL+Diffusion", "sharpe_ratio"] if "RL+Diffusion" in df_out.index else 0.0
        rl_only_sharpe = df_out.loc["RL-Only", "sharpe_ratio"] if "RL-Only" in df_out.index else 0.0
        highlight = rl_diffusion_sharpe > rl_only_sharpe

        df_out["highlight"] = highlight

        # CSV
        csv_path = os.path.join(out_dir, "performance_report.csv")
        df_out.to_csv(csv_path)

        # JSON — orient='index' gives {row_label: {col: value, ...}}
        json_path = os.path.join(out_dir, "performance_report.json")
        with open(json_path, "w") as f:
            json.dump(df_out.to_dict(orient="index"), f, indent=2, default=str)
