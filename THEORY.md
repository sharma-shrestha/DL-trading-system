# BTC RL Trading System — Complete Theory & Mathematics

## 1. System Overview

This project answers one research question:

> **Does giving a Reinforcement Learning agent "imagined futures" from a Diffusion Model improve trading performance compared to RL alone?**

The full pipeline is:

```
Raw OHLCV Data (BTCUSDT.csv)
        │
        ▼
  DataLoader          ← extracts .7z → .zip → .gzip → .csv
        │
        ▼
  FeatureEngineer     ← computes returns, volatility, normalizes
        │
        ▼
  WindowBuilder       ← sliding windows of shape (N, 50, 4)
        │
        ├──────────────────────────────────────────────┐
        ▼                                              ▼
  DiffusionModel                               TradingEnv (RL-Only)
  (trained on windows)                         (diffusion = None)
        │                                              │
        ▼                                              ▼
  DiffusionInference                         StrategySelector (PPO)
        │                                    TradeExecutor (PPO)
        ▼
  TradingEnv (RL+Diffusion)
        │
        ▼
  StrategySelector (PPO)   ← Level 1: picks strategy
  TradeExecutor (PPO/SAC)  ← Level 2: picks Buy/Sell/Hold
        │
        ▼
  Backtester               ← HRL+Diffusion vs Buy-and-Hold vs Random
        │
        ▼
  Evaluator                ← Sharpe, MaxDD, WinRate, ProfitFactor
        │
        ▼
  outputs/reports/         ← PNG charts + CSV/JSON report
```

---

## 2. Data Pipeline

### 2.1 DataLoader

The raw dataset is triple-compressed: `.7z → .zip → .gzip → .csv`.

The loader handles all three layers automatically:
1. Opens `.7z` with `py7zr`
2. Finds and extracts nested `.zip` with `zipfile`
3. Finds and decompresses nested `.gzip` with `gzip`
4. Reads the final CSV into a Pandas DataFrame

Required columns validated: `timestamp, open, high, low, close, volume`.

### 2.2 FeatureEngineer — The Math

From raw OHLCV data, four features are computed:

**Returns** (percentage change, removes price scale):
```
r_t = (close_t - close_{t-1}) / close_{t-1}
```

**Volatility** (rolling standard deviation of returns over 20 steps):
```
σ_t = std(r_{t-19}, r_{t-18}, ..., r_t)
```
This is the sample standard deviation:
```
σ_t = sqrt( (1/(w-1)) * Σ_{i=t-w+1}^{t} (r_i - r̄)² )
```
where `w = 20` and `r̄` is the mean of those 20 returns.

**Normalization** (StandardScaler — fitted ONLY on training data):
```
x_normalized = (x - μ_train) / σ_train
```
This prevents data leakage: the test set is scaled using training statistics, not its own.

Output shape: `(T', 4)` where `T' = T - 20` (NaN rows from rolling window dropped).

### 2.3 WindowBuilder — Sliding Windows

Given feature matrix of shape `(T, 4)`, produces windows of shape `(T - W, W, 4)` where `W = 50`.

Window `i` = `features[i : i + W]`

```
Window 0: rows [0, 49]
Window 1: rows [1, 50]
Window 2: rows [2, 51]
...
Window N: rows [N, N+49]
```

Implemented with `np.lib.stride_tricks.sliding_window_view` — zero-copy view, then `.copy()` to own memory.

**No look-ahead bias:** Train and test windows are built independently after the chronological 80/20 split. No window ever spans the split boundary.

---

## 3. Diffusion Model — Theory and Mathematics

### 3.1 What is a Diffusion Model?

A Denoising Diffusion Probabilistic Model (DDPM) learns to generate realistic data by learning to reverse a noise-adding process.

**Intuition:** If you gradually add Gaussian noise to a price sequence until it becomes pure noise, a neural network can learn to run that process backwards — starting from noise and recovering a realistic price sequence.

### 3.2 Forward Process (Adding Noise)

Given clean data `x_0` (a future price sequence), the forward process adds noise over `T = 1000` steps:

```
q(x_t | x_{t-1}) = N(x_t; sqrt(1 - β_t) * x_{t-1}, β_t * I)
```

Where `β_t` is the noise schedule — a linear ramp from `β_1 = 0.0001` to `β_{1000} = 0.02`.

Using the reparameterization trick, we can jump directly to any timestep `t`:

```
x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε,    ε ~ N(0, I)
```

Where:
- `α_t = 1 - β_t`
- `ᾱ_t = Π_{s=1}^{t} α_s`  (cumulative product)
- `sqrt(ᾱ_t)` = signal coefficient (how much of the original remains)
- `sqrt(1 - ᾱ_t)` = noise coefficient (how much noise is mixed in)

As `t → 1000`, `ᾱ_t → 0`, so `x_t → pure noise`.

In code:
```python
betas = torch.linspace(1e-4, 0.02, T)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)  # ᾱ_t for all t
sqrt_alphas_cumprod = alphas_cumprod.sqrt()
sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod).sqrt()
```

### 3.3 Training Objective

The model learns to predict the noise `ε` that was added:

```
L = E_{x_0, t, ε} [ || ε - ε_θ(x_t, t, context) ||² ]
```

This is a simple MSE loss between the actual noise and the model's prediction.

Training loop per batch:
1. Sample `t ~ Uniform(0, 999)`
2. Sample `ε ~ N(0, I)`
3. Compute `x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε`
4. Forward pass: `ε_pred = model(x_t, t, context)`
5. Loss: `MSE(ε_pred, ε)`
6. Backprop + Adam update

### 3.4 Architecture — 1D UNet

The model is a 1D UNet that processes sequences (not images).

```
Input: x_noisy (B, future_steps, 4)
       t       (B,)  — timestep index
       context (B, 50, 4)  — past price window

Time Embedding:
  sinusoidal_embedding(t, 128) → MLP → t_emb (B, 128)

Context Encoding:
  flatten(context) → Linear(200, 256) → SiLU → Linear(256, 128) → ctx_emb (B, 128)
  t_emb = t_emb + ctx_emb   ← fuse time and context

Input Projection:
  Conv1d(4, 64, kernel=1) → (B, 64, L)

Encoder:
  enc1: ResBlock1d(64→64)   → e1 (B, 64, L)
  down1: Conv1d stride=2    → (B, 128, L/2)
  enc2: ResBlock1d(128→128) → e2 (B, 128, L/2)
  down2: Conv1d stride=2    → (B, 256, L/4)
  enc3: ResBlock1d(256→256) → e3 (B, 256, L/4)
  down3: Conv1d stride=2    → (B, 512, L/8)

Bottleneck:
  mid1: ResBlock1d(512→512)
  mid2: ResBlock1d(512→512)

Decoder (with skip connections):
  up3: ConvTranspose1d → (B, 256, L/4)
  cat(up3, e3) → dec3: ResBlock1d(512→256)
  up2: ConvTranspose1d → (B, 128, L/2)
  cat(up2, e2) → dec2: ResBlock1d(256→128)
  up1: ConvTranspose1d → (B, 64, L)
  cat(up1, e1) → dec1: ResBlock1d(128→64)

Output Projection:
  Conv1d(64, 4, kernel=1) → (B, 4, L) → permute → (B, L, 4)
```

**ResBlock1d** (residual block with time conditioning):
```
h = SiLU(GroupNorm(Conv1d(x)))
h = h + Linear(t_emb).unsqueeze(-1)   ← inject time info
h = SiLU(GroupNorm(Conv1d(h)))
output = h + skip(x)                   ← residual connection
```

**Sinusoidal Time Embedding:**
```
freqs_k = exp(-log(10000) * k / (D/2 - 1)),  k = 0, ..., D/2-1
emb = [sin(t * freqs), cos(t * freqs)]        shape: (B, D)
```
This encodes the timestep `t` as a fixed-frequency sinusoidal vector, similar to positional encodings in Transformers.

### 3.5 DDIM Inference (Fast Generation)

Training uses 1000 steps. Inference uses DDIM (Denoising Diffusion Implicit Models) with only 50 steps — 20× faster.

DDIM update rule at each step:
```
x̂_0 = (x_t - sqrt(1 - ᾱ_t) * ε_θ(x_t, t)) / sqrt(ᾱ_t)
x̂_0 = clamp(x̂_0, -10, 10)
x_{t-1} = sqrt(ᾱ_{t-1}) * x̂_0 + sqrt(1 - ᾱ_{t-1}) * ε_θ(x_t, t)
```

Starting from `x_T ~ N(0, I)`, this iteratively denoises to produce a plausible future price sequence `x_0`.

**Why DDIM?** Standard DDPM sampling requires running the model 1000 times per generation. DDIM uses a non-Markovian process that achieves similar quality in 50 steps.

### 3.6 Conditioning on Past Context

The model is conditioned on the past 50 price steps (the current window). This context is:
1. Flattened: `(50, 4) → (200,)`
2. Encoded: `Linear(200, 256) → SiLU → Linear(256, 128)`
3. Added to the time embedding: `t_emb = t_emb + ctx_emb`

This means the model generates futures that are consistent with the observed past — not random noise, but plausible continuations.

### 3.7 Pre-caching Diffusion Outputs

Running the UNet at every RL step would be prohibitively slow. Instead, `TradingEnv` pre-computes diffusion outputs for all windows in batches of 256 before training starts:

```python
for start in range(0, N, 256):
    ctx = windows[start:end, -50:, :]   # (B, 50, 4)
    out = diffusion.generate(ctx, n_steps=10)  # (B, 10, 4)
    cache[start:end] = out.reshape(B, -1)
```

At each RL step, the agent gets `cache[step_idx]` — a fast array lookup.

---

## 4. Trading Environment

### 4.1 Observation Space

The agent sees a flat vector of 241 values:

```
obs = [price_window (50×4=200)] + [volatility (1)] + [diffusion_output (10×4=40)]
    = 241-dimensional float32 vector
```

In RL-only mode, the diffusion part is all zeros (40 zeros), giving the same 241-dim vector but without future information.

### 4.2 Action Space

`Discrete(3)`:
- `0` = Hold
- `1` = Buy (open long position)
- `2` = Sell (close long position)

Invalid actions (Buy when already long, Sell when flat) incur a small penalty equal to `transaction_cost`.

### 4.3 Reward Function

```
reward = profit - transaction_cost - risk_penalty - holding_penalty - invalid_penalty + diffusion_bonus
```

**Profit** (only realized on Sell):
```
profit = close_price_at_sell - close_price_at_buy
```

**Transaction cost** (applied on Buy and Sell):
```
transaction_cost = 0.001 × |close_price|
```
This models the 0.1% exchange fee on Binance.

**Risk penalty** (discourages trading in volatile markets):
```
risk_penalty = 0.1 × σ_window
```
where `σ_window = std(close prices in current window)`.

**Holding penalty** (encourages decisive trading):
```
holding_penalty = 0.0001  (per step while holding a position)
```

**Diffusion alignment bonus** (rewards the agent when diffusion predicted the right direction):
```
if predicted_direction == actual_direction:
    diffusion_bonus = |profit| × 0.1
```

**Episode termination:**
- All windows consumed, OR
- `portfolio_value < 0.5 × initial_balance` (50% loss limit)

### 4.4 Position State Machine

```
State: FLAT (position=0)
  action=Buy  → LONG (position=1), record entry_price, pay transaction_cost
  action=Sell → FLAT (invalid), pay invalid_penalty
  action=Hold → FLAT (valid, no cost)

State: LONG (position=1)
  action=Buy  → LONG (invalid), pay invalid_penalty
  action=Sell → FLAT (position=0), realize profit, pay transaction_cost
  action=Hold → LONG (valid), pay holding_penalty
```

---

## 5. Hierarchical Reinforcement Learning (HRL)

### 5.1 Why Hierarchical?

A single flat RL agent must simultaneously decide "what style of trading to use" and "when exactly to trade." These operate at different timescales and abstractions. HRL decomposes this into two levels:

- **Level 1 (StrategySelector):** High-level, slow decisions — which trading regime are we in?
- **Level 2 (TradeExecutor):** Low-level, fast decisions — given the regime, what exact action?

### 5.2 StrategySelector — Level 1

**Algorithm:** PPO (Proximal Policy Optimization)

**Strategies:**
- `0` = Scalping (many small quick trades)
- `1` = Momentum (follow the trend)
- `2` = Mean-Reversion (bet on price returning to average)

**Observation:** Same 241-dim vector as TradingEnv.

**PPO — The Math:**

PPO is an on-policy policy gradient algorithm. It collects trajectories, then updates the policy using a clipped surrogate objective:

```
L_CLIP(θ) = E_t [ min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t) ]
```

Where:
- `r_t(θ) = π_θ(a_t | s_t) / π_θ_old(a_t | s_t)` — probability ratio (new vs old policy)
- `A_t` — advantage estimate (how much better was this action than average?)
- `ε = 0.2` — clipping range (prevents too-large policy updates)

**Advantage estimation (GAE):**
```
δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
A_t = Σ_{l=0}^{∞} (γλ)^l * δ_{t+l}
```
Where `γ` is the discount factor and `λ` is the GAE smoothing parameter.

**Value function loss:**
```
L_VF = E_t [ (V_θ(s_t) - V_t^target)² ]
```

**Entropy bonus** (encourages exploration):
```
L_ENT = E_t [ H(π_θ(· | s_t)) ]
```

**Total PPO loss:**
```
L = L_CLIP - c1 * L_VF + c2 * L_ENT
```

Implemented via Stable-Baselines3 `PPO(policy="MlpPolicy", env=env)`.

### 5.3 TradeExecutor — Level 2

**Algorithm:** PPO or SAC (configurable)

**Observation:** Base 241-dim vector + strategy index scalar = **242-dim vector**

The strategy index is appended by `AugmentedEnv`:
```python
aug_obs = concat([base_obs, [float(strategy_idx)]])  # shape: (242,)
```

This tells the executor "you are currently in Scalping mode" (or Momentum, or Mean-Reversion), so it can adapt its behavior accordingly.

**SAC — The Math (if configured):**

SAC (Soft Actor-Critic) is an off-policy algorithm that maximizes a modified objective:

```
J(π) = E_τ~π [ Σ_t (r_t + α * H(π(· | s_t))) ]
```

Where `α` is the temperature parameter controlling the entropy-reward tradeoff.

SAC maintains:
- Actor: `π_φ(a | s)` — policy network
- Two critics: `Q_θ1(s, a)`, `Q_θ2(s, a)` — value networks (reduces overestimation)
- Target networks: soft-updated copies of critics

**Critic loss:**
```
y = r + γ * (min(Q_θ1', Q_θ2')(s', ã') - α * log π_φ(ã' | s'))
L_Q = E [ (Q_θ(s, a) - y)² ]
```

**Actor loss:**
```
L_π = E_s [ α * log π_φ(ã | s) - min(Q_θ1, Q_θ2)(s, ã) ]
```

SAC uses a **replay buffer** to reuse past experience, making it more sample-efficient than PPO.

---

## 6. Backtesting

### 6.1 Three Strategies Compared

| Strategy | Description |
|---|---|
| HRL+Diffusion | StrategySelector → TradeExecutor, with diffusion observations |
| Buy-and-Hold | Buy at step 0, hold until last step, sell |
| Random-Action | Random Buy/Sell/Hold at each step |

The RL-Only comparison (same HRL agents but `diffusion=None`) is added in `main.py` after the main backtest.

### 6.2 Metrics — The Math

**Sharpe Ratio** (risk-adjusted return, annualized):
```
Sharpe = (μ_r / σ_r) × sqrt(252)
```
Where `μ_r = mean(returns)`, `σ_r = std(returns)`, and `sqrt(252)` annualizes assuming 252 trading days.

Returns 0.0 if `σ_r = 0`.

**Maximum Drawdown** (worst peak-to-trough loss):
```
MaxDD = max_{t} [ (peak_t - value_t) / peak_t ]
```
Where `peak_t = max_{s ≤ t} value_s`.

A MaxDD of 0.25 means the portfolio dropped 25% from its highest point.

**Win Rate** (fraction of profitable trades):
```
WinRate = |{trades : PnL > 0}| / |{all trades}|
```

**Profit Factor** (gross profit vs gross loss):
```
ProfitFactor = Σ_{winning trades} PnL / Σ_{losing trades} |PnL|
```
Returns `inf` if no losing trades, `0.0` if no winning trades.

---

## 7. Evaluation

The `Evaluator` produces:

1. `portfolio_value.png` — line chart of portfolio value over time for all strategies
2. `drawdown.png` — per-step drawdown `(peak - value) / peak` over time
3. `metrics_comparison.png` — bar chart of Sharpe, Win Rate, Profit Factor
4. `performance_report.csv` — table of all metrics
5. `performance_report.json` — same data in JSON

**Highlight condition:** If `Sharpe(RL+Diffusion) > Sharpe(RL-Only)`, the report sets `highlight=True` — this is the key research result confirming the diffusion model adds value.

---

## 8. How Everything Links Together

```
config.yaml
    │
    ├── DataLoader ──────────────────────────────────────────────────────┐
    │       │                                                            │
    │   FeatureEngineer                                                  │
    │       │  returns = (close_t - close_{t-1}) / close_{t-1}          │
    │       │  volatility = rolling_std(returns, 20)                     │
    │       │  normalize: (x - μ_train) / σ_train                       │
    │       │                                                            │
    │   WindowBuilder                                                    │
    │       │  windows[i] = features[i : i+50]                          │
    │       │  shape: (N, 50, 4)                                         │
    │       │                                                            │
    │       ├──────────────────────────────────────────────────────────┐ │
    │       ▼                                                          ▼ │
    │   DiffusionModel (train)                                  TradingEnv
    │       │  Forward: x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε              │
    │       │  Loss: MSE(ε_pred, ε)                                     │
    │       │  Architecture: 1D UNet with skip connections              │
    │       │                                                            │
    │   DiffusionInference (DDIM, 50 steps)                             │
    │       │  x̂_0 = (x_t - √(1-ᾱ_t)*ε_θ) / √ᾱ_t                    │
    │       │  x_{t-1} = √ᾱ_{t-1}*x̂_0 + √(1-ᾱ_{t-1})*ε_θ            │
    │       │                                                            │
    │       └──────────────────────────────────────────────────────────┘
    │                           │
    │                    TradingEnv
    │                           │  obs = [window(200)] + [vol(1)] + [diff(40)]
    │                           │  reward = profit - cost - risk - hold + bonus
    │                           │
    │               ┌───────────┴───────────┐
    │               ▼                       ▼
    │       StrategySelector          TradeExecutor
    │       (PPO, Level 1)            (PPO/SAC, Level 2)
    │       picks: 0/1/2              obs = base(241) + strategy(1) = 242
    │       (Scalp/Mom/MeanRev)       picks: 0/1/2 (Hold/Buy/Sell)
    │               │                       │
    │               └───────────┬───────────┘
    │                           ▼
    │                      Backtester
    │                           │  HRL+Diffusion vs Buy-and-Hold vs Random
    │                           │
    │                      Evaluator
    │                           │  Sharpe, MaxDD, WinRate, ProfitFactor
    │                           ▼
    │                   outputs/reports/
    └────────────────────────────────────────────────────────────────────┘
```

---

## 9. Key Numbers and Their Meaning

| Parameter | Value | Why |
|---|---|---|
| `window_size` | 50 | Agent sees last 50 candles (~50 minutes on 1m data) |
| `future_steps` | 10 | Diffusion predicts 10 steps ahead |
| `volatility_window` | 20 | Standard 20-period volatility |
| `train_timesteps` | 1000 | DDPM noise schedule length (standard) |
| `inference_steps` | 50 | DDIM steps — 20× faster than full DDPM |
| `context_len` | 100 | Past steps fed to diffusion (capped to window_size=50) |
| `batch_size` | 256 | Diffusion training batch |
| `epochs` | 100 | Diffusion training epochs |
| `train_ratio` | 0.8 | 80% train, 20% test (chronological) |
| `max_train_samples` | 50000 | Cap to keep training time reasonable |
| `transaction_cost` | 0.001 | 0.1% per trade (realistic Binance fee) |
| `risk_penalty_weight` | 0.1 | Scales volatility penalty in reward |
| `holding_penalty` | 0.0001 | Per-step cost of holding a position |
| `min_portfolio_value` | 0.5 | Episode ends if portfolio drops 50% |
| `initial_balance` | $10,000 | Starting portfolio value |
| `RL total_timesteps` | 200,000 | Per agent (StrategySelector + TradeExecutor) |
| `seed` | 42 | Global reproducibility seed |

---

## 10. Reproducibility

`utils/seed.py` sets the seed on all random number generators:

```python
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

This ensures that given the same data and config, every run produces identical results.

---

## 11. Libraries and Their Roles

| Library | Role |
|---|---|
| `PyTorch` | DiffusionModel (1D UNet), training loop, DDIM inference |
| `Stable-Baselines3` | PPO and SAC implementations for RL agents |
| `Gymnasium` | Standard RL environment interface (TradingEnv) |
| `Pandas` | CSV loading, feature computation, report generation |
| `NumPy` | Array math, sliding windows, metric computation |
| `scikit-learn` | StandardScaler for feature normalization |
| `py7zr` | Extract `.7z` archives |
| `PyYAML` | Parse `config.yaml` |
| `Matplotlib` | Generate PNG performance charts |

---

## 12. The Core Hypothesis — Explained Simply

A standard RL trading agent sees only the past: the last 50 candles. It must decide whether to Buy, Sell, or Hold based solely on historical patterns.

The diffusion model adds a "crystal ball" — not a perfect prediction, but a plausible sample from the distribution of possible futures given the past. The agent's observation becomes:

```
[what happened in the last 50 steps] + [one plausible future for the next 10 steps]
```

If the diffusion model has learned the statistical structure of BTC price movements, this extra information should help the RL agent:
- Avoid buying just before a predicted drop
- Hold through predicted temporary dips
- Sell before predicted reversals

The experiment is validated by comparing `Sharpe(RL+Diffusion)` vs `Sharpe(RL-Only)`. If the former is higher, the hypothesis is confirmed.
