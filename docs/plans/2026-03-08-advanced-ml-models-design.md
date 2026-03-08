# Advanced ML Models for FPL — Design Document

## Overview

Build three tiers of increasingly sophisticated models for FPL team management:
1. Feature-engineered predictors (XGBoost + LSTM) paired with optimizers
2. Transfer-capable optimizers (greedy + LP/IP solver)
3. End-to-end RL agent (PPO)

All models conform to the existing `ActionModel` interface and are evaluated via `SeasonSimulator` backtesting.

## Dependencies

Add to `pyproject.toml` under `[project.optional-dependencies]`:
```
ml = ["scikit-learn>=1.4", "xgboost>=2.0", "pulp>=2.7", "torch>=2.0"]
```

## 1. Predictors

### 1a. XGBoost Predictor (`predictors/xgboost_predictor.py`)

Gradient-boosted tree model predicting next-GW `total_points` per player.

**Features** (engineered from `SeasonData`):
- **Form**: rolling mean/std of `total_points` over last 3, 5, 10 GWs
- **Expected stats**: rolling `xG`, `xA`, `xGI`, `xGC` (per 90 min)
- **ICT**: `influence`, `creativity`, `threat` rolling averages
- **Fixture context**: opponent `strength_defence_*` / `strength_attack_*` (home/away aware), fixture difficulty rating
- **Minutes trend**: rolling mean minutes (proxy for rotation risk)
- **BPS trend**: rolling `bps` average (bonus point likelihood)
- **Market momentum**: `transfers_balance` direction, `selected` ownership level
- **Position encoding**: one-hot `element_type`
- **Home/away**: binary flag from upcoming fixture
- **Rest days**: gap since last fixture (from fixture `kickoff_time`)

**Training**: `train()` builds features from all GWs across historical seasons, fits XGBoost regressor. Target is actual `total_points`.

**Prediction**: `predict()` builds features for the current GW, returns predicted points per player.

### 1b. Sequence Predictor (`predictors/sequence_predictor.py`)

LSTM that takes a player's recent GW history as a sequence and predicts next-GW points.

**Input**: Fixed-length sequence (last 10 GWs) of per-GW feature vectors (points, xG, xA, minutes, BPS, fixture difficulty, home/away). Padded with zeros for fewer GWs.

**Architecture**: `Embedding per position -> LSTM(hidden=64, layers=2) -> Linear -> scalar output`

**Training**: Each (player, GW) pair is a training sample. MSE loss on actual points.

Both predictors implement `Predictor` ABC and are interchangeable.

## 2. Optimizers

### 2a. Transfer-Capable Greedy Optimizer (extend `optimizers/greedy.py`)

Extend existing `GreedyOptimizer` with transfer evaluation before lineup selection:
- For each free transfer, evaluate all (player_out, player_in) swaps
- Score each swap as: `predicted_points_in - predicted_points_out` for next GW
- Only transfer if best gain exceeds threshold (default: 2.0 points)
- Respect budget and 3-per-team constraint
- For paid transfers (-4 hit), require gain > 4 + threshold
- After transfers, run existing lineup/captain selection

### 2b. LP Optimizer (`optimizers/lp_optimizer.py`)

Integer Linear Program using PuLP:

**Decision variables** (binary):
- `x[p]` = 1 if player p is in squad (15 total)
- `s[p]` = 1 if player p starts (11 total)
- `c[p]` = 1 if player p is captain
- `t_out[p]` = 1 if player p transferred out
- `t_in[p]` = 1 if player p transferred in

**Objective**: maximize `sum(predicted_points[p] * (s[p] + c[p]))` minus transfer cost penalties.

**Constraints**:
- Squad = 15, XI = 11
- Formation: 1 GK, >=3 DEF, >=2 MID, >=1 FWD in XI
- Squad composition: 2 GK, 5 DEF, 5 MID, 3 FWD total
- Max 3 players per team
- Budget: total cost <= current budget + sell value of outgoing
- Transfer cost: `max(0, num_transfers - free_transfers) * 4` deducted from objective
- Exactly 1 captain, must be a starter

**Multi-GW lookahead** (configurable `horizon`):
- Uses `PlayerPredictions.multi_gw` when available
- Sums predicted points over horizon, discounted by 0.9 per GW

**Chip heuristics**:
- **Bench Boost**: trigger when bench predicted points > 15 and double GW
- **Triple Captain**: trigger when captain predicted points > 12
- **Free Hit**: trigger on blank GWs (>3 squad players missing)
- **Wildcard**: trigger when optimal squad differs from current by >5 players

## 3. RL Agent

### 3a. Environment (`rl/environment.py`)

Wraps `SeasonSimulator` as an RL environment.

**State** (~310 floats, normalized to [0,1]):
- Per squad player (15 x 20): element_type, buy/sell price, rolling form (3/5/10 GW), rolling xG/xA, fixture difficulty next 3 opponents, home/away, minutes trend, BPS trend
- Global (~10): budget, free_transfers, current_gameweek, chips_remaining (4), season progress

**Action space** (multi-headed):
1. **Transfer head**: score per squad slot (out) + score per top-K=50 candidates (in). Argmax pair if above threshold, else no transfer. 0-2 transfers per step.
2. **Captain head**: softmax over 15 squad slots
3. **Lineup head**: sigmoid per slot, top 11 with formation masking
4. **Chip head**: softmax over 5 options (none, WC, FH, BB, TC)

**Reward**: raw GW points (after transfer costs). No shaping.

**Episode**: one full season (38 steps).

### 3b. PPO Agent (`rl/ppo.py`)

Implements `ActionModel` directly.

**Policy network**:
- Shared backbone: `Linear(310, 256) -> ReLU -> Linear(256, 128) -> ReLU`
- Transfer head: `Linear(128, 15)` (out) + `Linear(128, 50)` (in)
- Captain head: `Linear(128, 15)` -> softmax
- Lineup head: `Linear(128, 15)` -> sigmoid
- Chip head: `Linear(128, 5)` -> softmax

**Value network**:
- `Linear(310, 256) -> ReLU -> Linear(256, 128) -> ReLU -> Linear(128, 1)`

**Training**:
- Collect trajectories over N episodes
- GAE advantages (lambda=0.95, gamma=0.99)
- PPO clipped objective (epsilon=0.2), K epochs per batch
- Data augmentation for limited data (~10 seasons):
  - Random starting GW (partial seasons)
  - Gaussian noise on points (+/-10%)
  - Random initial squad selection

**Candidate selection**: pre-filter transfer-in to top 50 by form per GW.

## 4. Shared Feature Engineering (`features.py`)

Used by XGBoost, sequence predictor, and RL state encoder:
- `build_player_features(data, player_code, gw) -> dict` — single player/GW
- `build_feature_matrix(data) -> pd.DataFrame` — all players, current GW
- `build_sequence_features(data, player_code, gw, seq_len) -> np.ndarray` — time series

## 5. File Structure

```
src/fpl_model/models/
  predictors/
    form.py               # existing baseline
    xgboost_predictor.py   # new
    sequence_predictor.py  # new
  optimizers/
    greedy.py              # extend with transfers
    lp_optimizer.py        # new
  rl/
    environment.py         # SeasonSimulator wrapper
    ppo.py                 # PPO ActionModel
  features.py              # shared feature engineering
  base.py                  # existing
  registry.py              # existing
  defaults.py              # extend with new models
```

## 6. Model Registry

```
"form-greedy"    — FormPredictor + GreedyOptimizer (existing)
"xgb-greedy"     — XGBoostPredictor + GreedyOptimizer (with transfers)
"xgb-lp"         — XGBoostPredictor + LPOptimizer
"sequence-lp"    — SequencePredictor + LPOptimizer
"ppo-agent"      — PPOAgent (end-to-end)
```

## 7. Evaluation

All models evaluated via existing `SeasonSimulator` + `compute_metrics()` on held-out seasons. Compare: total points, transfer costs, points per GW, best/worst GW.
