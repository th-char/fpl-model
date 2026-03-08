# fpl-model

ML pipeline for Fantasy Premier League (FPL) team recommendations. Trains predictive models on historical data, simulates full seasons, and compares strategies.

## Quick Start

```bash
# Install dependencies (requires uv)
uv sync

# Ingest historical data into SQLite
uv run fpl-model ingest --seasons 2023-24 2024-25

# Train and simulate a model
uv run fpl-model train --model xgb-greedy --seasons 2023-24
uv run fpl-model simulate --model xgb-greedy --season 2024-25

# Compare all models on a season
uv run fpl-model compare --season 2024-25

# Get transfer recommendations for next gameweek
uv run fpl-model recommend --model xgb-greedy --season 2024-25
```

## Models

| Model | Predictor | Optimizer | Transfers | Mid-Season Retrain |
|-------|-----------|-----------|-----------|-------------------|
| `form-greedy` | Recent form average | Greedy | No | No |
| `xgb-greedy` | XGBoost (25+ features) | Greedy | Yes (free only) | No |
| `xgb-lp` | XGBoost | Integer Linear Program (PuLP) | Yes | No |
| `sequence-lp` | LSTM sequence model | Integer Linear Program | Yes | No |
| `ppo-agent` | PPO (policy + value networks) | Policy network | No (lineup/captain) | No |

Models support **recency decay weighting** (`recency_decay` param) and **mid-season retraining** via `SeasonSimulator(retrain_every_n_gws=N)`.

## Architecture

```
src/fpl_model/
  data/         # ETL pipeline: Sources -> Transformers -> Unifier -> SQLite
  models/       # Predictors (XGBoost, LSTM, PPO), Optimizers (Greedy, LP)
  simulation/   # SeasonSimulator replays historical seasons per-GW
  evaluation/   # Comparison reports and metrics
  cli/          # Click-based CLI entry points
```

- **`ActionModel`** is the universal interface; **`PredictOptimizeModel`** composes a `Predictor` + `Optimizer`
- Shared feature engineering in `models/features.py` (used by XGBoost, LSTM, RL)
- Cross-season player linking via `code` field (stable across seasons, unlike `id`)
- Data: [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) GitHub repo for historical seasons

## Notebook

`notebooks/model_comparison.ipynb` trains all models, simulates the 2024-25 season, and produces:
- Comparison tables (total points, avg/GW, transfer hits)
- GW-by-GW and cumulative points charts
- Transfer activity and budget trajectory plots
- Interactive squad viewer (starting XI, transfers, chips per GW per model)

To run: open in VS Code, select the `.venv/bin/python` kernel, and run all cells.

## Development

```bash
# Run tests (150 tests)
uv run pytest -x -q

# Lint
uv run ruff check src/ tests/ --fix
```

## Data

- **Source:** vaastav/Fantasy-Premier-League GitHub repo (historical), FPL API (live)
- **Storage:** SQLite at `data/fpl.db` (gitignored)
- **Cache:** `data/cache/` (gitignored)
- Seasons available: 2016-17 through 2025-26
