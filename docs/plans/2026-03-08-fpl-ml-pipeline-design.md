# FPL ML Pipeline — Design Document

**Date**: 2026-03-08
**Status**: Approved

## Goal

Build an ML pipeline that recommends FPL team actions to maximize season points. It consumes historical and current-season data, trains models, and outputs weekly action recommendations (transfers, captaincy, lineup, chips). The system supports multiple model families, faithful backtesting, and interactive research.

## Project Structure

```
fpl-model/
├── pyproject.toml              # UV-managed, single package
├── uv.lock
├── src/
│   └── fpl_model/
│       ├── __init__.py
│       ├── data/               # Data layer
│       │   ├── sources/        # DataSource implementations
│       │   │   ├── base.py     # Abstract DataSource interface
│       │   │   ├── fpl_api.py  # Current season from FPL API
│       │   │   └── vaastav.py  # Historical seasons from GitHub raw URLs
│       │   ├── etl/            # Schema unification
│       │   │   ├── schemas.py  # Canonical unified schemas
│       │   │   ├── transformers.py  # Per-source transformers
│       │   │   └── unifier.py  # Column mapping, type coercion, null handling
│       │   ├── db.py           # SQLite connection, table creation, read/write
│       │   └── cache.py        # Local file cache for raw downloads
│       ├── models/             # Model layer
│       │   ├── base.py         # ActionModel, PredictOptimizeModel, Predictor, Optimizer
│       │   ├── registry.py     # Model registration and discovery
│       │   ├── predictors/     # Predictor implementations
│       │   └── optimizers/     # Optimizer implementations
│       ├── simulation/         # Simulation engine
│       │   ├── engine.py       # SeasonSimulator
│       │   ├── state.py        # SquadState
│       │   ├── rules.py        # Auto-subs, chips, scoring, transfer costs
│       │   └── actions.py      # Action dataclasses, validation
│       ├── evaluation/         # Evaluation framework
│       │   ├── metrics.py      # Scoring metrics
│       │   ├── comparison.py   # Multi-model comparison
│       │   └── reports.py      # Summary tables, export
│       └── cli/                # CLI entry points
│           └── main.py         # Click CLI group
├── tests/
├── notebooks/                  # Research & exploration
├── data/                       # Local data cache (gitignored)
│   ├── raw/                    # Downloaded CSVs and API JSON
│   └── fpl.db                  # SQLite database
└── docs/
    └── plans/
```

## Tooling

- **Package manager**: UV
- **Python**: 3.12+
- **Core dependencies**: httpx (async HTTP), pandas (data manipulation), click (CLI)
- **Persistence**: SQLite (stdlib sqlite3)
- **ML dependencies**: Optional extras per model (scikit-learn, torch, pulp/or-tools, etc.)
- **Testing**: pytest
- **Linting**: ruff

## Data Layer

### Data Sources

**Abstract interface**: Each `DataSource` implements `fetch(seasons, ...) → raw DataFrames`. New sources (betting odds, analyst sentiment) plug in by implementing this interface.

**FPL API** (`fpl_api.py`):
- Fetches current season from `https://fantasy.premierleague.com/api/bootstrap-static/`
- Additional endpoints for per-GW live data and per-player history
- Returns JSON, parsed into DataFrames

**vaastav GitHub** (`vaastav.py`):
- Downloads specific CSV files via GitHub raw URLs for requested seasons
- Files: `players_raw.csv`, `gws/merged_gw.csv`, `fixtures.csv`, `teams.csv`, `player_idlist.csv` per season
- Also: cross-season `master_team_list.csv`
- Supports all 10 seasons: 2016-17 through 2025-26

**Cache** (`cache.py`): Raw downloaded files cached in `data/raw/`, keyed by source + season + file type. Re-download only on explicit refresh.

### Canonical Schemas

A unified set of tables that all sources transform into:

- **`players`** — One row per player per season. Identity, position, team, `code` for cross-season linking.
- **`gameweek_performances`** — One row per player per gameweek. Points, goals, assists, minutes, xG, xA, bonus, value, etc.
- **`fixtures`** — One row per match. Teams, scores, difficulty ratings, date, GW.
- **`teams`** — One row per team per season. Strength ratings, name, short name.
- **`gameweeks`** — One row per GW per season. Deadline, average score, chip plays.

### Schema Evolution Handling

The unifier defines a superset of columns across all seasons. Missing columns in older seasons (e.g. `expected_goals` before 2019-20) become NULL. Removed columns (e.g. `loaned_in`) are dropped. Cross-season player linking uses the `code` field (stable across seasons, unlike `id`).

### SQLite Persistence

Tables mirror canonical schemas. `db.py` provides read/write helpers returning DataFrames. The DB is the single source of truth after ingestion — models and simulation read from it, never from raw files.

## Model Layer

### Core Abstractions

```python
class ActionModel(ABC):
    """Universal interface — every model implements this."""

    @abstractmethod
    def recommend(self, state: SquadState, data: SeasonData) -> list[Action]:
        """Given current squad state and available data, return actions."""
        ...

    def train(self, historical_data: HistoricalData) -> None:
        """Optional: train on historical data. No-op for rule-based models."""
        pass


class PredictOptimizeModel(ActionModel):
    """Subclass that composes a Predictor + Optimizer."""

    def __init__(self, predictor: Predictor, optimizer: Optimizer): ...

    def recommend(self, state: SquadState, data: SeasonData) -> list[Action]:
        predictions = self.predictor.predict(state, data)
        return self.optimizer.optimize(predictions, state)


class Predictor(ABC):
    """Predicts future player performance."""
    @abstractmethod
    def predict(self, state: SquadState, data: SeasonData) -> PlayerPredictions: ...

class Optimizer(ABC):
    """Given predictions + constraints, outputs optimal actions."""
    @abstractmethod
    def optimize(self, predictions: PlayerPredictions, state: SquadState) -> list[Action]: ...
```

### SeasonData

Provided to all models, includes:
- Historical gameweek performances (from DB)
- **Upcoming fixtures** — remaining matches with teams, difficulty, dates, double/blank GW info
- Player metadata (position, team, price, availability/news)
- Team strength ratings

### Action Types

Dataclasses:
- `Transfer(player_out, player_in)`
- `SetCaptain(player_id)`
- `SetViceCaptain(player_id)`
- `SetLineup(starting_xi, bench_order)`
- `PlayChip(chip_type)` — Wildcard, FreeHit, BenchBoost, TripleCaptain

### SquadState

Captures: current 15 players, budget, free transfers available, chips remaining, current GW, player sell prices.

### Registry

Dict-based registration so CLI/notebooks can look up models by name.

## Simulation Engine

### SeasonSimulator

Replays a historical season by calling `model.recommend()` each gameweek and tracking full FPL state.

```
For each gameweek:
  1. Model recommends actions
  2. Validate & apply actions (enforce transfer limits, budget, chip rules)
  3. Score the GW using actual historical results
  4. Apply auto-subs for non-playing starters
  5. Apply captain/vice-captain logic
  6. Apply chip effects (bench boost, triple captain)
  7. Update state: price changes, free transfer accrual
```

### FPL Rules Faithfully Implemented

- **Transfers**: -4 per extra transfer beyond free allowance, max 5 banked free transfers
- **Budget**: Player price changes affect squad value. Sell price = buy price + floor(rise / 0.2) * 0.1
- **Chips**: One chip per GW. Two of each per season (one per half-season, before/after GW19). Free Hit can't be GW1. Wildcard can't be GW1. Free Hit and Wildcard preserve banked transfers.
- **Auto-subs**: Ordered bench replacement respecting formation minimums (3 DEF, 2 MID, 1 FWD)
- **Captaincy**: Captain doubles points. If captain doesn't play, vice-captain activates. If neither plays, no double points.
- **Bench Boost**: All bench players score
- **Triple Captain**: Captain triples instead of doubles
- **Free Hit**: Unlimited transfers for one GW, squad reverts next GW

### SimulationResult

Total points, per-GW breakdown, all actions taken, budget history — enough for evaluation and debugging.

## Evaluation Framework

- **Single model evaluation**: Total points, per-GW points, rank vs historical averages, points left on table
- **Model comparison**: Run N models on same season(s), comparison table with total points, variance, best/worst GWs, transfer hit costs, chip timing
- **Predictor evaluation**: MAE/RMSE of predicted vs actual points per player per GW, calibration — independent of optimizer
- **Cross-season evaluation**: Run model across multiple seasons to test robustness

Reports are plain dataclass/dict structures consumable by CLI (printed), JSON (serialized), or notebooks (visualization).

## CLI

```
fpl-model ingest --seasons 2021-22,2022-23,...    # Download & ETL into SQLite
fpl-model ingest --current                        # Fetch current season from FPL API
fpl-model train <model-name> --seasons ...        # Train a model on historical data
fpl-model recommend <model-name> --gw 25          # Get action recommendations for a GW
fpl-model simulate <model-name> --season 2024-25  # Replay a full season
fpl-model evaluate <model-name> --seasons ...     # Evaluate across seasons
fpl-model compare <model1> <model2> --seasons ... # Compare models side by side
```

Models referenced by registry name. CLI is a thin wrapper — all logic lives in the layers, so notebooks call the same functions directly.

## Key Design Decisions

1. **Universal `ActionModel` interface** — every model conforms to one method. Predict+optimize is a subclass, not a separate tier.
2. **SeasonData includes upcoming fixtures** — all models can reason about schedule, difficulty, blanks/doubles.
3. **Full FPL rules simulation** — faithful backtesting including chips, auto-subs, budget, captaincy.
4. **All 10 seasons supported** — schema evolution handled via column superset + nulls.
5. **SQLite persistence** — single source of truth after ingestion. Simple, portable, SQL-queryable.
6. **Data downloaded on demand** — raw GitHub URLs, cached locally. No repo clone or submodule.
7. **Scripts + notebooks** — scripts for automated evaluation, notebooks for interactive research.
