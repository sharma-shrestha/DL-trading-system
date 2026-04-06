"""BTC RL Trading System — pipeline entry point.

Six-phase pipeline:
  Phase 0: Load config + set seed
  Phase 1: DataLoader → FeatureEngineer → split_and_build (sliding windows)
  Phase 2: DiffusionModel training (train_diffusion) — skip if checkpoint exists
  Phase 3: Load DiffusionInference, build TradingEnv (train + test)
  Phase 4: Train StrategySelector + TradeExecutor — skip if saved models exist
  Phase 5: Backtester.run() — HRL+Diffusion, Buy-and-Hold, Random-Action
           RL-only mode: rebuild TradingEnv with diffusion=None, run HRL again
  Phase 6: Evaluator.compare() + Evaluator.plot() + Evaluator.save_report()
"""

import argparse
import os

from utils.config import load_config
from utils.seed import set_seed
from data.loader import DataLoader
from data.features import FeatureEngineer
from data.windows import WindowBuilder
from diffusion.model import DiffusionModel
from diffusion.train import train_diffusion
from diffusion.inference import DiffusionInference
from env.trading_env import TradingEnv
from agents.strategy_selector import StrategySelector
from agents.trade_executor import TradeExecutor
from backtest.backtester import Backtester
from evaluation.evaluator import Evaluator


def main(config_path: str = "config.yaml") -> None:
    # ------------------------------------------------------------------ #
    # Phase 0: Configuration & reproducibility                            #
    # ------------------------------------------------------------------ #
    print("[Phase 0] Loading config and setting seed...")
    config = load_config(config_path)
    set_seed(config["seed"])
    print(f"[Phase 0] Config loaded from '{config_path}' | seed={config['seed']}")

    # ------------------------------------------------------------------ #
    # Phase 1: Data loading & feature engineering                         #
    # ------------------------------------------------------------------ #
    print("\n[Phase 1] Loading data and engineering features...")

    loader = DataLoader(config)
    df = loader.load()
    print(f"[Phase 1] Data loaded: {len(df)} rows")

    engineer = FeatureEngineer(config)
    train_ratio: float = config["data"]["train_ratio"]
    split_idx = int(len(df) * train_ratio)

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    train_features = engineer.fit_transform(train_df)
    test_features = engineer.transform(test_df)
    print(f"[Phase 1] Features: train={train_features.shape}, test={test_features.shape}")

    window_size: int = config["windows"]["window_size"]
    builder = WindowBuilder(window_size=window_size)
    train_windows = builder.build(train_features)
    test_windows = builder.build(test_features)

    # Cap training samples to avoid excessive training time
    max_train = config["data"].get("max_train_samples")
    if max_train and len(train_windows) > max_train:
        train_windows = train_windows[:max_train]
        print(f"[Phase 1] Capped train windows to {max_train} samples")

    # Cap test windows too (use 20% of max_train for test)
    max_test = int(max_train * 0.2) if max_train else None
    if max_test and len(test_windows) > max_test:
        test_windows = test_windows[:max_test]
        print(f"[Phase 1] Capped test windows to {max_test} samples")

    print(f"[Phase 1] Windows: train={train_windows.shape}, test={test_windows.shape}")

    # ------------------------------------------------------------------ #
    # Phase 2: Diffusion model training                                   #
    # ------------------------------------------------------------------ #
    checkpoint_path: str = config["diffusion"]["checkpoint_path"]
    if os.path.exists(checkpoint_path):
        print(f"\n[Phase 2] Diffusion checkpoint found at '{checkpoint_path}' — skipping training.")
    else:
        print("\n[Phase 2] Training diffusion model...")
        diffusion_model = DiffusionModel(config)
        train_diffusion(diffusion_model, train_windows, config)
        print(f"[Phase 2] Diffusion model trained and saved to '{checkpoint_path}'")

    # ------------------------------------------------------------------ #
    # Phase 3: Load DiffusionInference, build TradingEnv                 #
    # ------------------------------------------------------------------ #
    print("\n[Phase 3] Loading diffusion inference and building trading environments...")
    diffusion_inference = DiffusionInference(config)
    train_env = TradingEnv(windows=train_windows, diffusion=diffusion_inference, config=config)
    test_env = TradingEnv(windows=test_windows, diffusion=diffusion_inference, config=config)

    # Also build RL-only envs (no diffusion) for fair comparison
    print("[Phase 3] Building RL-only environments (diffusion=None)...")
    train_env_rl_only = TradingEnv(windows=train_windows, diffusion=None, config=config)
    test_env_rl_only = TradingEnv(windows=test_windows, diffusion=None, config=config)
    print("[Phase 3] All environments ready.")

    # ------------------------------------------------------------------ #
    # Phase 4: Train StrategySelector + TradeExecutor (two separate pairs)#
    # ------------------------------------------------------------------ #
    ss_save_path: str = config["agents"]["strategy_selector"]["save_path"]
    te_save_path: str = config["agents"]["trade_executor"]["save_path"]
    ss_rl_path = ss_save_path + "_rl_only"
    te_rl_path = te_save_path + "_rl_only"

    # --- RL+Diffusion agents ---
    strategy_selector = StrategySelector(env=train_env, config=config)
    trade_executor = TradeExecutor(env=train_env, config=config)

    if os.path.exists(ss_save_path + ".zip") and os.path.exists(te_save_path + ".zip"):
        print("\n[Phase 4] Loading saved RL+Diffusion agents...")
        strategy_selector.load(ss_save_path)
        trade_executor.load(te_save_path)
    else:
        print("\n[Phase 4] Training RL+Diffusion agents...")
        strategy_selector.train()
        strategy_selector.save(ss_save_path)
        trade_executor.train()
        trade_executor.save(te_save_path)
        print("[Phase 4] RL+Diffusion agents saved.")

    # --- RL-Only agents (trained WITHOUT diffusion) ---
    ss_rl_only = StrategySelector(env=train_env_rl_only, config=config)
    te_rl_only = TradeExecutor(env=train_env_rl_only, config=config)

    if os.path.exists(ss_rl_path + ".zip") and os.path.exists(te_rl_path + ".zip"):
        print("[Phase 4] Loading saved RL-Only agents...")
        ss_rl_only.load(ss_rl_path)
        te_rl_only.load(te_rl_path)
    else:
        print("[Phase 4] Training RL-Only agents (no diffusion)...")
        rl_multiplier = config["agents"].get("rl_only_timesteps_multiplier", 1.0)
        rl_only_steps = int(config["agents"]["strategy_selector"]["total_timesteps"] * rl_multiplier)
        ss_rl_only.train(total_timesteps=rl_only_steps)
        ss_rl_only.save(ss_rl_path)
        te_rl_only.train(total_timesteps=rl_only_steps)
        te_rl_only.save(te_rl_path)
        print("[Phase 4] RL-Only agents saved.")

    # ------------------------------------------------------------------ #
    # Phase 5: Backtesting                                                #
    # ------------------------------------------------------------------ #
    print("\n[Phase 5] Running backtests...")
    backtester = Backtester(config)

    # RL+Diffusion on diffusion test env
    results = backtester.run(
        strategy_selector=strategy_selector,
        trade_executor=trade_executor,
        env=test_env,
    )
    print("[Phase 5] HRL+Diffusion backtest complete.")

    # RL-Only agents on rl-only test env (fair comparison — same obs space)
    print("[Phase 5] Running RL-only backtest...")
    rl_only_results = backtester.run(
        strategy_selector=ss_rl_only,
        trade_executor=te_rl_only,
        env=test_env_rl_only,
    )
    results["rl_only"] = rl_only_results["hrl_diffusion"]
    print("[Phase 5] RL-only backtest complete.")

    print(
        f"[Phase 5] Results summary:\n"
        f"  HRL+Diffusion — Sharpe: {results['hrl_diffusion']['sharpe_ratio']:.4f}, "
        f"MaxDD: {results['hrl_diffusion']['max_drawdown']:.4f}\n"
        f"  RL-Only       — Sharpe: {results['rl_only']['sharpe_ratio']:.4f}, "
        f"MaxDD: {results['rl_only']['max_drawdown']:.4f}\n"
        f"  Buy-and-Hold  — Sharpe: {results['buy_and_hold']['sharpe_ratio']:.4f}, "
        f"MaxDD: {results['buy_and_hold']['max_drawdown']:.4f}"
    )

    # ------------------------------------------------------------------ #
    # Phase 6: Evaluation & reporting                                     #
    # ------------------------------------------------------------------ #
    print("\n[Phase 6] Generating evaluation report...")
    evaluator = Evaluator(config)
    comparison_df = evaluator.compare(results)
    print("[Phase 6] Comparison table:")
    print(comparison_df.to_string())

    output_dir: str = config["evaluation"]["output_dir"]
    evaluator.plot(results, output_dir=output_dir)
    print(f"[Phase 6] Plots saved to '{output_dir}'")

    evaluator.save_report(comparison_df, output_dir=output_dir)
    print(f"[Phase 6] Report saved to '{output_dir}'")

    print("\n[main] Pipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BTC RL Trading System pipeline")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the YAML configuration file (default: config.yaml)",
    )
    args = parser.parse_args()
    main(config_path=args.config)
