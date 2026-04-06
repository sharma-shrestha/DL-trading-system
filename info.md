# BTC RL Trading System — Full Project Documentation

This document explains the entire project in simple words: what each file does, what formulas are used, how training works, and how everything connects.

---

## What is this project?

We built an AI trading system for Bitcoin (BTC/USDT). The system:
1. Reads historical BTC price data
2. Learns patterns from the data
3. Uses a **Diffusion Model** to imagine possible future price scenarios
4. Uses a **Reinforcement Learning (RL) agent** to decide when to Buy, Sell, or Hold
5. Evaluates performance and compares it against simple baselines

The core research question: **Does giving the RL agent "imagined futures" from a diffusion model make it trade better?**

---

## The Full Pipeline (Step by Step)

```
BTCUSDT.csv.7z (raw data)
        ↓
  DataLoader  →  extracts and reads the CSV
        ↓
  FeatureEngineer  →  computes close, returns, volatility, volume
        ↓
  WindowBuilder  →  creates sliding windows of shape (N, 50, 4)
        ↓
  DiffusionModel  →  trained to generate future price scenarios
        ↓
  TradingEnv  →  simulated market where agent trades
        ↓
  StrategySelector (PPO)  →  picks a strategy: Scalping / Momentum / Mean-Reversion
        ↓
  TradeExecutor (PPO/SAC)  →  decides: Buy / Sell / Hold
        ↓
  Backtester  →  runs agent on unseen test data
        ↓
  Evaluator  →  compares RL+Diffusion vs RL-Only vs Buy-and-Hold
        ↓
  outputs/reports/  →  graphs + CSV/JSON performance report
```

---

## File-by-File Explanation

---

### `config.yaml` — Central Configuration

All settings live here. No magic numbers in code.

Key settings:
- `data.archive_path` — where your `.7z` file is
- `windows.window_size: 50` — each input is 50 time steps
- `diffusion.future_steps: 10` — model predicts 10 steps ahead
- `diffusion.train_timesteps: 1000` — DDPM noise schedule length
- `diffusion.epochs: 50` — how many times to train the diffusion model
- `agents.trade_executor.algorithm: PPO` — which RL algorithm to use
- `agents.strategy_selector.total_timesteps: 500000` — how long to train RL
- `data.max_train_samples: 50000` — cap on training data to keep it fast
- `seed: 42` — for reproducibility

---

### `utils/config.py` — Config Loader

Reads `config.yaml` using PyYAML. Raises clear errors if the file is missing or a required key is absent.

### `utils/seed.py` — Seed Setter

Sets the random seed on Python's `random`, NumPy, and PyTorch so results are reproducible.

```python
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
```

---

### `data/loader.py` — DataLoader

**What it does:** Extracts the compressed archive and loads the CSV.

**Steps:**
1. Opens `BTCUSDT.csv.7z` using `py7zr`
2. Finds the nested `.zip` inside and extracts it
3. Finds the `.gzip` inside that and decompresses it
4. Reads the final `BTCUSDT.csv` into a Pandas DataFrame
5. Validates that columns `timestamp, open, high, low, close, volume` exist
6. Logs row count and date range

**Why triple-compressed?** The original Binance dataset is distributed as `.7z → .zip → .gzip → .csv`. Our loader handles all three layers automatically.

---

### `data/features.py` — FeatureEngineer

**What it does:** Turns raw OHLCV data into 4 meaningful features.

**Features computed:**

| Feature | Formula | Why |
|---|---|---|
| `close` | raw close price | price reference |
| `returns` | `(close[t] - close[t-1]) / close[t-1]` | % change, removes price scale |
| `volatility` | rolling std of returns over 20 steps | measures risk/uncertainty |
| `volume` | raw volume | order pressure signal |

**Normalization:** Uses `StandardScaler` — subtracts mean, divides by std. Fitted ONLY on training data to prevent data leakage.

```
normalized = (x - mean_train) / std_train
```

Output shape: `(T', 4)` where T' = T minus NaN rows from rolling calculations.

---

### `data/windows.py` — WindowBuilder

**What it does:** Converts the feature matrix into sliding windows for sequential model input.

**How it works:**
- Given features of shape `(T, 4)`
- Produces windows of shape `(T - window_size, window_size, 4)`
- Window `i` = `features[i : i + window_size]`

**Example with window_size=50:**
- Window 0 = rows 0–49
- Window 1 = rows 1–50
- Window 2 = rows 2–51
- ...

**Why sliding windows?** Models need context — seeing the last 50 candles to predict what comes next.

**No data leakage:** Train and test windows are built independently after the chronological split.

---

### `diffusion/model.py` — DiffusionModel

**What it does:** A neural network that learns to generate realistic future price sequences.

**Theory — DDPM (Denoising Diffusion Probabilistic Model):**

The idea is simple:
- **Forward process:** Gradually add Gaussian noise to data over 1000 steps until it's pure noise
- **Reverse process:** Train a neural network to remove noise step by step, reconstructing the original data

**Forward process formula:**
```
x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε
```
Where:
- `x_0` = original clean data (future prices)
- `x_t` = noisy version at timestep t
- `ᾱ_t` = cumulative product of (1 - β_t), the noise schedule
- `ε` = random Gaussian noise

**Training objective:** Predict the noise `ε` that was added:
```
Loss = MSE(ε_predicted, ε_actual)
```

**Architecture — 1D UNet:**
```
Input (noisy future prices)
    ↓
Input Projection (Conv1d)
    ↓
Encoder Block 1 (ResBlock + downsample)
Encoder Block 2 (ResBlock + downsample)
Encoder Block 3 (ResBlock + downsample)
    ↓
Bottleneck (2x ResBlock)
    ↓
Decoder Block 3 (upsample + skip connection from Encoder 3)
Decoder Block 2 (upsample + skip connection from Encoder 2)
Decoder Block 1 (upsample + skip connection from Encoder 1)
    ↓
Output Projection → predicted noise (same shape as input)
```

**Conditioning:** The model also receives the past price window (context) as input. This is encoded via a linear layer and added to the time embedding, so the model knows "given this past, what noise was added to this future?"

**Inference — DDIM (50 steps instead of 1000):**
- Start from pure Gaussian noise
- Iteratively denoise using the trained model
- 50 steps instead of 1000 = much faster, similar quality
- Output: `(batch, future_steps, 4)` — plausible future price sequences

---

### `diffusion/train.py` — Diffusion Training Loop

**What it does:** Trains the DiffusionModel on historical windows.

**Training loop per batch:**
1. Take a window of shape `(50, 4)` — split into context (first 40 steps) and target (last 10 steps)
2. Sample random timestep `t ~ Uniform(0, 999)`
3. Sample noise `ε ~ N(0, I)`
4. Compute noisy target: `x_t = sqrt(ᾱ_t) * target + sqrt(1-ᾱ_t) * ε`
5. Feed `(x_t, t, context)` to model → get predicted noise `ε_pred`
6. Compute loss: `MSE(ε_pred, ε)`
7. Backpropagate, update weights with Adam optimizer

**Progress printed:** Every 50 batches and at end of each epoch.

**Saves:** `outputs/models/diffusion.pt` after training.

---

### `diffusion/inference.py` — DiffusionInference

**What it does:** Loads the trained diffusion model and exposes a simple `generate()` method.

```python
inference = DiffusionInference(config)
future = inference.generate(context)  # context: (50, 4) → future: (10, 4)
```

Used by `TradingEnv` at every step to give the RL agent a "vision of the future."

---

### `env/trading_env.py` — TradingEnv

**What it does:** A simulated trading market. The RL agent interacts with this environment.

**Follows the OpenAI Gymnasium interface:**
- `reset()` → start a new episode
- `step(action)` → take an action, get reward + next state
- `render()` → print current state

**Observation (what the agent sees):**
```
[price_window (50×4=200 values)] + [volatility (1 value)] + [diffusion_output (10×4=40 values)]
= 241-dimensional vector
```

**Actions:**
- `0` = Hold (do nothing)
- `1` = Buy (open a long position)
- `2` = Sell (close the position)

**Reward formula:**
```
reward = profit - transaction_cost - risk_penalty - holding_penalty
```
- `profit` = price_sell - price_buy (only when selling)
- `transaction_cost` = 0.1% of trade value (realistic exchange fee)
- `risk_penalty` = 0.1 × current_volatility (penalizes trading in volatile markets)
- `holding_penalty` = 0.0001 per step while holding (encourages decisive trading)

**Episode ends when:**
- All data is consumed, OR
- Portfolio value drops below 50% of starting balance

**RL-only mode:** When `diffusion=None`, the diffusion part of the observation is all zeros. This lets us compare RL+Diffusion vs RL-Only fairly.

---

### `agents/strategy_selector.py` — StrategySelector (Level 1)

**What it does:** The "big picture" agent. Decides which trading style to use.

**Strategies:**
- `0` = Scalping (many small quick trades)
- `1` = Momentum (follow the trend)
- `2` = Mean-Reversion (bet on price returning to average)

**Algorithm: PPO (Proximal Policy Optimization)**

PPO is an on-policy RL algorithm. It:
1. Collects experience by running the agent in the environment
2. Computes advantage estimates (how much better was this action than average?)
3. Updates the policy using a clipped objective to prevent too-large updates:

```
L_CLIP = E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]
```
Where `r_t = π_new(a|s) / π_old(a|s)` is the probability ratio and `A_t` is the advantage.

**Implementation:** Uses Stable-Baselines3's `PPO` with `MlpPolicy` (multi-layer perceptron).

---

### `agents/trade_executor.py` — TradeExecutor (Level 2)

**What it does:** The "execution" agent. Given the strategy chosen by Level 1, decides the exact Buy/Sell/Hold action.

**Observation:** Base observation (241 values) + strategy index (1 value) = 242 values total.

**Algorithm: PPO or SAC** (configurable in `config.yaml`)

**SAC (Soft Actor-Critic)** is an off-policy algorithm that:
- Maximizes both reward AND entropy (randomness) — encourages exploration
- More sample-efficient than PPO for continuous action spaces
- Uses a replay buffer to reuse past experience

**AugmentedEnv:** A wrapper around `TradingEnv` that appends the strategy index to every observation, so the executor knows which strategy it's supposed to follow.

---

### `backtest/metrics.py` — Metric Functions

Four standard trading performance metrics:

**Sharpe Ratio** — risk-adjusted return:
```
Sharpe = (mean(returns) / std(returns)) × sqrt(252)
```
Higher is better. sqrt(252) annualizes daily returns. Returns 0.0 if std=0.

**Max Drawdown** — worst loss from peak:
```
MaxDD = max over all t of (peak_t - value_t) / peak_t
```
Lower is better. 0.25 means the portfolio dropped 25% from its highest point.

**Win Rate** — fraction of profitable trades:
```
WinRate = (number of trades with P&L > 0) / (total trades)
```

**Profit Factor** — gross profit vs gross loss:
```
ProfitFactor = sum(winning trades) / sum(losing trades)
```
> 1.0 means you make more than you lose. Returns `inf` if no losing trades.

---

### `backtest/backtester.py` — Backtester

**What it does:** Runs three strategies on the held-out test data and collects metrics.

**Three strategies evaluated:**
1. **HRL+Diffusion** — our full system (StrategySelector + TradeExecutor + diffusion)
2. **Buy-and-Hold** — buy at step 0, hold until the end, sell at last step
3. **Random-Action** — random Buy/Sell/Hold at each step (worst case baseline)

**Why baselines?** To prove our system actually learned something useful.

---

### `evaluation/evaluator.py` — Evaluator

**What it does:** Compares all three strategies and produces the final report.

**Outputs:**
1. `portfolio_value.png` — line chart of portfolio value over time for all 3 strategies
2. `drawdown.png` — drawdown curve over time
3. `metrics_comparison.png` — bar chart comparing Sharpe, Win Rate, Profit Factor
4. `performance_report.csv` — table of all metrics
5. `performance_report.json` — same data in JSON format

**Highlight:** If RL+Diffusion Sharpe > RL-Only Sharpe, the report marks it as highlighted — this is the key research result.

---

### `main.py` — Pipeline Entry Point

Runs everything end-to-end:

```
python main.py --config config.yaml
```

**Smart skipping:** If a checkpoint already exists (diffusion model or RL agents), it skips retraining and loads from disk. This means you can re-run without waiting hours.

**RL-Only comparison:** After the main backtest, it rebuilds `TradingEnv` with `diffusion=None` and runs the same HRL agents again. This gives us the "RL without diffusion" result for comparison.

---

## Libraries Used

| Library | Purpose |
|---|---|
| `PyTorch` | Neural network framework (diffusion model) |
| `Stable-Baselines3` | RL algorithms (PPO, SAC) |
| `Gymnasium` | RL environment interface |
| `Pandas` | Data loading and manipulation |
| `NumPy` | Numerical operations, array math |
| `scikit-learn` | StandardScaler for normalization |
| `py7zr` | Extract `.7z` archives |
| `PyYAML` | Read `config.yaml` |
| `Matplotlib` | Generate performance graphs |

---

## Key Numbers (Default Config)

| Parameter | Value | Meaning |
|---|---|---|
| Window size | 50 | Agent sees last 50 candles |
| Future steps | 10 | Diffusion predicts 10 steps ahead |
| Train samples | 50,000 | Capped for speed |
| Diffusion epochs | 50 | Training passes over data |
| Diffusion timesteps | 1000 | Noise schedule length |
| DDIM inference steps | 50 | Fast generation |
| RL training steps | 500,000 | Per agent |
| Train/test split | 80/20 | Chronological |
| Initial balance | $10,000 | Starting portfolio |
| Transaction cost | 0.1% | Per trade |

---

## How to Run

```bash
# Install dependencies
python -m pip install -r requirements.txt

# Run the full pipeline
python main.py --config config.yaml

# Results saved to:
# outputs/models/   ← trained model weights
# outputs/reports/  ← graphs + performance report
```

---

## What "RL+Diffusion > RL-Only" Means

The diffusion model gives the RL agent extra information — a glimpse of possible futures. Instead of just seeing the past 50 candles, the agent also sees "here are 10 plausible future price paths." This richer observation should help the agent make better decisions, especially in uncertain market conditions.

If `Sharpe(RL+Diffusion) > Sharpe(RL-Only)`, we've proven the hypothesis.
