# Mid-Season Retraining with Recency Weighting — Design

**Goal:** Allow models to retrain periodically during season simulation, with exponential decay weighting so recent data is prioritized.

## Architecture

The `SeasonSimulator` gains an optional `retrain_every_n_gws: int | None` parameter (default `None` = no retraining). When set, every N gameweeks during `run()`, it builds a `HistoricalData` object from all data available up to (but not including) the current GW and calls `model.train(historical)`. This is a no-op for models that don't implement training (like form-greedy).

For recency weighting, both `XGBoostPredictor` and `SequencePredictor` gain a `recency_decay: float` parameter (default `1.0` = no decay). When < 1.0, training samples from gameweek `gw` get weight `decay^(max_gw - gw)`, so recent data is weighted exponentially higher. XGBoost uses `sample_weight` natively; LSTM uses weighted MSE loss.

The `Predictor.train()` signature stays unchanged — `train(historical_data: HistoricalData)`. The decay is purely internal to each predictor.

## Components & Changes

### SeasonSimulator (`simulation/engine.py`)

- New param: `retrain_every_n_gws: int | None = None`
- New param: `train_seasons: list[str] | None = None` — if provided, loads those seasons fully plus the eval season up to current GW for retraining context. If `None`, just uses eval season data up to current GW.
- In the GW loop, before calling `model.recommend()`, check if it's a retraining GW. If so, build `HistoricalData` and call `model.train(historical)`.

### XGBoostPredictor (`predictors/xgboost_predictor.py`)

- New param: `recency_decay: float = 1.0`
- In `train()`, compute `weight = decay^(max_gw - gw)` per sample.
- Pass weights to `self.model.fit(X, y, sample_weight=weights)`.

### SequencePredictor (`predictors/sequence_predictor.py`)

- New param: `recency_decay: float = 1.0`
- In `train()`, compute per-sample weights the same way.
- Replace `MSELoss()` with manual weighted MSE: `(weights * (pred - y)^2).mean()`.

### No changes to

`ActionModel`, `Predictor` ABC, `Optimizer`, `PPOAgent`, `FormPredictor`, `HistoricalData`.

## Testing

- XGBoost trains with `recency_decay=0.9` without error and produces different predictions than `decay=1.0`.
- LSTM trains with `recency_decay=0.9` without error.
- `SeasonSimulator` with `retrain_every_n_gws=2` calls `model.train()` multiple times (mock/spy).
- `retrain_every_n_gws=None` (default) does not retrain mid-season.

## Notebook Integration

- Add retrained model variants (e.g. `xgb-greedy-retrain` with `retrain_every_n_gws=5`, `recency_decay=0.95`).
- Compare retrained vs non-retrained in the same charts.
