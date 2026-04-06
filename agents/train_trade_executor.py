"""
Standalone script to train the TradeExecutor (Level-2 PPO/SAC agent).

Usage:
    python agents/train_trade_executor.py [--config config.yaml]

Pipeline:
    1. Load config from config.yaml
    2. Set global seed
    3. Load and preprocess data (DataLoader → FeatureEngineer → WindowBuilder → split_and_build)
    4. Build TradingEnv with train windows
    5. Instantiate and train TradeExecutor
    6. Save the trained model

Requirements: 9.5
"""

import argparse
import os
import sys

# Ensure project root is on the path when running as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config
from utils.seed import set_seed
from data.loader import DataLoader
from data.features import FeatureEngineer
from data.windows import split_and_build
from env.trading_env import TradingEnv
from agents.trade_executor import TradeExecutor


def main():
    parser = argparse.ArgumentParser(
        description="Train the TradeExecutor (Level-2 PPO/SAC agent)."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the YAML configuration file (default: config.yaml)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load configuration
    # ------------------------------------------------------------------
    print(f"[train_trade_executor] Loading config from '{args.config}' ...")
    config = load_config(args.config)

    # ------------------------------------------------------------------
    # 2. Set global seed
    # ------------------------------------------------------------------
    seed = config["seed"]
    print(f"[train_trade_executor] Setting global seed: {seed}")
    set_seed(seed)

    # ------------------------------------------------------------------
    # 3. Load and preprocess data
    # ------------------------------------------------------------------
    print("[train_trade_executor] Loading raw data ...")
    loader = DataLoader(config)
    df = loader.load()
    print(f"[train_trade_executor] Loaded {len(df)} rows.")

    print("[train_trade_executor] Engineering features ...")
    engineer = FeatureEngineer(config)
    train_ratio = config["data"]["train_ratio"]
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    train_features = engineer.fit_transform(train_df)
    test_features = engineer.transform(test_df)
    print(
        f"[train_trade_executor] Features — train: {train_features.shape}, "
        f"test: {test_features.shape}"
    )

    print("[train_trade_executor] Building sliding windows ...")
    import numpy as np
    all_features = np.concatenate([train_features, test_features], axis=0)
    window_size = config["windows"]["window_size"]
    train_windows, test_windows = split_and_build(
        all_features,
        train_ratio=train_ratio,
        window_size=window_size,
    )
    print(
        f"[train_trade_executor] Windows — train: {train_windows.shape}, "
        f"test: {test_windows.shape}"
    )

    # ------------------------------------------------------------------
    # 4. Build TradingEnv with train windows
    # ------------------------------------------------------------------
    print("[train_trade_executor] Building TradingEnv ...")

    # Optionally load diffusion model if checkpoint exists
    diffusion = None
    checkpoint_path = config["diffusion"]["checkpoint_path"]
    if os.path.exists(checkpoint_path):
        print(f"[train_trade_executor] Loading diffusion model from '{checkpoint_path}' ...")
        try:
            from diffusion.inference import DiffusionInference
            diffusion = DiffusionInference(config)
            print("[train_trade_executor] Diffusion model loaded.")
        except Exception as exc:
            print(f"[train_trade_executor] Warning: could not load diffusion model: {exc}")
            diffusion = None
    else:
        print(
            f"[train_trade_executor] Diffusion checkpoint not found at '{checkpoint_path}'. "
            "Running in RL-only mode (diffusion output zeroed)."
        )

    env = TradingEnv(train_windows, diffusion, config)
    algorithm = config["agents"]["trade_executor"].get("algorithm", "PPO").upper()
    print(
        f"[train_trade_executor] TradingEnv ready — "
        f"obs_dim={env.observation_space.shape[0]}, n_windows={len(train_windows)}, "
        f"algorithm={algorithm}"
    )

    # ------------------------------------------------------------------
    # 5. Instantiate and train TradeExecutor
    # ------------------------------------------------------------------
    total_timesteps = config["agents"]["trade_executor"]["total_timesteps"]
    print(
        f"[train_trade_executor] Training TradeExecutor ({algorithm}) for "
        f"{total_timesteps:,} timesteps ..."
    )
    executor = TradeExecutor(env, config)
    executor.train(total_timesteps=total_timesteps)
    print("[train_trade_executor] Training complete.")

    # ------------------------------------------------------------------
    # 6. Save the trained model
    # ------------------------------------------------------------------
    save_path = config["agents"]["trade_executor"]["save_path"]
    print(f"[train_trade_executor] Saving model to '{save_path}' ...")
    executor.save(save_path)
    print(f"[train_trade_executor] Model saved to '{save_path}'.")


if __name__ == "__main__":
    main()
