\# BTC RL Trading System — Full Project Documentation



This document explains the entire project in simple words: what each file does, what formulas are used, how training works, and how everything connects.



\---



\## What is this project?



We built an AI trading system for Bitcoin (BTC/USDT). The system:

1\. Reads historical BTC price data

2\. Learns patterns from the data

3\. Uses a \*\*Diffusion Model\*\* to imagine possible future price scenarios

4\. Uses a \*\*Reinforcement Learning (RL) agent\*\* to decide when to Buy, Sell, or Hold

5\. Evaluates performance and compares it against simple baselines



The core research question: \*\*Does giving the RL agent "imagined futures" from a diffusion model make it trade better?\*\*



\---



\## The Full Pipeline (Step by Step)



```

BTCUSDT.csv.7z (raw data)

&#x20;       ↓

&#x20; DataLoader  →  extracts and reads the CSV

&#x20;       ↓

&#x20; FeatureEngineer  →  computes close, returns, volatility, volume

&#x20;       ↓

&#x20; WindowBuilder  →  creates sliding windows of shape (N, 50, 4)

&#x20;       ↓

&#x20; DiffusionModel  →  trained to generate future price scenarios

&#x20;       ↓

&#x20; TradingEnv  →  simulated market where agent trades

&#x20;       ↓

&#x20; StrategySelector (PPO)  →  picks a strategy: Scalping / Momentum / Mean-Reversion

&#x20;       ↓

&#x20; TradeExecutor (PPO/SAC)  →  decides: Buy / Sell / Hold

&#x20;       ↓

&#x20; Backtester  →  runs agent on unseen test data

&#x20;       ↓

&#x20; Evaluator  →  compares RL+Diffusion vs RL-Only vs Buy-and-Hold

&#x20;       ↓

&#x20; outputs/reports/  →  graphs + CSV/JSON performance report

```



\---



Bottleneck (2x ResBlock)

&#x20;   ↓

Decoder Block 3 (upsample + skip connection from Encoder 3)

Decoder Block 2 (upsample + skip connection from Encoder 2)

Decoder Block 1 (upsample + skip connection from Encoder 1)

&#x20;   ↓

Output Projection → predicted noise (same shape as input)

```

This project builds an AI-based Bitcoin trading system using a combination of Reinforcement Learning (RL) and a Diffusion Model.

The pipeline:
- Processes historical BTC price data
- Extracts key features (returns, volatility, volume)
- Uses a Diffusion Model to generate possible future price scenarios
- Trains RL agents to make Buy / Sell / Hold decisions
- Evaluates performance against baselines like Buy-and-Hold

Goal:
Test whether giving RL agents "imagined futures" improves trading performance compared to standard RL.











