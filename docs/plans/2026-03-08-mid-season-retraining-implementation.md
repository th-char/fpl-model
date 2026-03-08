# Mid-Season Retraining with Recency Weighting — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Allow predictors to retrain periodically during simulation with exponential decay weighting for recent data.

**Architecture:** `SeasonSimulator` checks every N GWs and calls `model.train()` with data up to current GW. Predictors accept a `recency_decay` float and weight training samples by `decay^(max_gw - gw)`. XGBoost uses native `sample_weight`; LSTM uses weighted MSE.

**Tech Stack:** Python 3.12, pytest, xgboost, PyTorch, UV (`/home/node/.local/bin/uv run`)

**Design doc:** `docs/plans/2026-03-08-mid-season-retraining-design.md`

---

### Task 1: XGBoost recency decay weighting

**Files:**
- Modify: `src/fpl_model/models/predictors/xgboost_predictor.py`
- Test: `tests/models/test_xgboost_predictor.py`

**Step 1: Write the failing tests**

Add to `tests/models/test_xgboost_predictor.py`:

```python
class TestXGBoostRecencyDecay:
    def test_train_with_decay_runs(self, tmp_path):
        """XGBoostPredictor trains successfully with recency_decay < 1."""
        # Reuse _setup_historical() from existing tests or inline it
        from tests.models.test_xgboost_predictor import _setup_historical
        historical = _setup_historical()
        predictor = XGBoostPredictor(recency_decay=0.9)
        predictor.train(historical)
        assert predictor.model is not None

    def test_decay_changes_predictions(self, tmp_path):
        """Different decay values produce different predictions."""
        from tests.models.test_xgboost_predictor import _setup_historical, _make_state_and_data
        historical = _setup_historical()

        p1 = XGBoostPredictor(recency_decay=1.0)
        p1.train(historical)

        p2 = XGBoostPredictor(recency_decay=0.5)
        p2.train(historical)

        state, data = _make_state_and_data()
        preds1 = p1.predict(state, data)
        preds2 = p2.predict(state, data)
        # With aggressive decay, predictions should differ
        assert preds1.predictions != preds2.predictions

    def test_default_decay_is_no_decay(self):
        """Default recency_decay=1.0 means uniform weighting."""
        predictor = XGBoostPredictor()
        assert predictor.recency_decay == 1.0
```

**Step 2: Run tests to verify they fail**

Run: `/home/node/.local/bin/uv run pytest tests/models/test_xgboost_predictor.py -v`
Expected: FAIL — `XGBoostPredictor() got an unexpected keyword argument 'recency_decay'`

**Step 3: Implement recency decay in XGBoostPredictor**

In `src/fpl_model/models/predictors/xgboost_predictor.py`:

1. Add `recency_decay: float = 1.0` parameter to `__init__`:
```python
def __init__(self, lookback_start: int = 5, recency_decay: float = 1.0) -> None:
    self.lookback_start = lookback_start
    self.recency_decay = recency_decay
    self.model = None
    self._feature_cols: list[str] | None = None
```

2. In `train()`, track the GW number for each sample. After building `rows_X` and `rows_y`, also build `rows_gw`:
```python
rows_gw = []  # add alongside rows_X and rows_y
# Inside the loop, after rows_y.append(...):
rows_gw.append(gw)
```

3. Before `self.model.fit(X, y)`, compute weights:
```python
import numpy as np
gw_array = np.array(rows_gw, dtype=np.float64)
max_gw = gw_array.max()
weights = np.power(self.recency_decay, max_gw - gw_array)
self.model.fit(X, y, sample_weight=weights)
```

**Step 4: Run tests to verify they pass**

Run: `/home/node/.local/bin/uv run pytest tests/models/test_xgboost_predictor.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/fpl_model/models/predictors/xgboost_predictor.py tests/models/test_xgboost_predictor.py
git commit -m "feat: add recency_decay weighting to XGBoostPredictor"
```

---

### Task 2: LSTM recency decay weighting

**Files:**
- Modify: `src/fpl_model/models/predictors/sequence_predictor.py`
- Test: `tests/models/test_sequence_predictor.py`

**Step 1: Write the failing tests**

Add to `tests/models/test_sequence_predictor.py`:

```python
class TestSequenceRecencyDecay:
    def test_train_with_decay_runs(self):
        """SequencePredictor trains successfully with recency_decay < 1."""
        from tests.models.test_sequence_predictor import _setup_historical
        historical = _setup_historical()
        predictor = SequencePredictor(recency_decay=0.9, epochs=2)
        predictor.train(historical)
        assert predictor.model is not None

    def test_default_decay_is_no_decay(self):
        """Default recency_decay=1.0 means uniform weighting."""
        predictor = SequencePredictor()
        assert predictor.recency_decay == 1.0
```

**Step 2: Run tests to verify they fail**

Run: `/home/node/.local/bin/uv run pytest tests/models/test_sequence_predictor.py -v`
Expected: FAIL — `SequencePredictor() got an unexpected keyword argument 'recency_decay'`

**Step 3: Implement recency decay in SequencePredictor**

In `src/fpl_model/models/predictors/sequence_predictor.py`:

1. Add `recency_decay: float = 1.0` parameter to `__init__`:
```python
def __init__(
    self,
    seq_len: int = 10,
    hidden_size: int = 64,
    num_layers: int = 2,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 64,
    recency_decay: float = 1.0,
) -> None:
    # ... existing assignments ...
    self.recency_decay = recency_decay
```

2. In `train()`, track GW numbers alongside samples. Add `gw_list = []` and append `gw` in the inner loop.

3. Compute weights tensor:
```python
gw_arr = torch.tensor(gw_list, dtype=torch.float32)
max_gw = gw_arr.max()
weights = torch.pow(torch.tensor(self.recency_decay), max_gw - gw_arr)
```

4. Replace the loss computation in the training loop:
```python
# Replace: loss = loss_fn(pred, y[idx])
# With:
diff_sq = (pred - y[idx]) ** 2
loss = (weights[idx] * diff_sq).mean()
```

**Step 4: Run tests to verify they pass**

Run: `/home/node/.local/bin/uv run pytest tests/models/test_sequence_predictor.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/fpl_model/models/predictors/sequence_predictor.py tests/models/test_sequence_predictor.py
git commit -m "feat: add recency_decay weighting to SequencePredictor"
```

---

### Task 3: Mid-season retraining in SeasonSimulator

**Files:**
- Modify: `src/fpl_model/simulation/engine.py:44-164`
- Test: `tests/simulation/test_engine.py`

**Step 1: Write the failing tests**

Add to `tests/simulation/test_engine.py`:

```python
from unittest.mock import MagicMock, patch

class TestMidSeasonRetraining:
    def test_retrain_called_periodically(self, tmp_path):
        """With retrain_every_n_gws=2, model.train() is called during simulation."""
        db = _setup_db(tmp_path, num_gws=5)  # reuse existing helper
        model = MagicMock(spec=ActionModel)
        model.recommend.return_value = []
        model.train.return_value = None

        sim = SeasonSimulator(model=model, season="2024-25", db=db, retrain_every_n_gws=2)
        sim.run()

        # train() should have been called at least once during the 5 GWs
        assert model.train.call_count >= 1

    def test_no_retrain_by_default(self, tmp_path):
        """With default retrain_every_n_gws=None, model.train() is never called."""
        db = _setup_db(tmp_path, num_gws=3)
        model = MagicMock(spec=ActionModel)
        model.recommend.return_value = []
        model.train.return_value = None

        sim = SeasonSimulator(model=model, season="2024-25", db=db)
        sim.run()

        model.train.assert_not_called()

    def test_retrain_uses_only_past_data(self, tmp_path):
        """Retraining HistoricalData should not contain future GW data."""
        db = _setup_db(tmp_path, num_gws=5)
        train_calls = []

        class SpyModel(ActionModel):
            def recommend(self, state, data):
                return []
            def train(self, historical_data):
                for sd in historical_data.seasons.values():
                    max_gw = sd.gameweek_performances["gameweek"].max() if len(sd.gameweek_performances) > 0 else 0
                    train_calls.append({"gw": sd.current_gameweek, "max_data_gw": max_gw})

        sim = SeasonSimulator(model=SpyModel(), season="2024-25", db=db, retrain_every_n_gws=2)
        sim.run()

        for call in train_calls:
            # Data available should be strictly less than current GW
            assert call["max_data_gw"] < call["gw"]
```

**Step 2: Run tests to verify they fail**

Run: `/home/node/.local/bin/uv run pytest tests/simulation/test_engine.py::TestMidSeasonRetraining -v`
Expected: FAIL — `SeasonSimulator() got an unexpected keyword argument 'retrain_every_n_gws'`

**Step 3: Implement mid-season retraining**

In `src/fpl_model/simulation/engine.py`:

1. Update `__init__`:
```python
def __init__(
    self,
    model: ActionModel,
    season: str,
    db: Database,
    retrain_every_n_gws: int | None = None,
    train_seasons: list[str] | None = None,
) -> None:
    self.model = model
    self.season = season
    self.db = db
    self.retrain_every_n_gws = retrain_every_n_gws
    self.train_seasons = train_seasons or []
```

2. Add a `_build_training_data` method:
```python
def _build_training_data(self, current_gw: int, gw_perf_df, players_df, fixtures_df, teams_df) -> HistoricalData:
    """Build HistoricalData with data up to (not including) current_gw."""
    from fpl_model.models.base import HistoricalData
    seasons = {}

    # Load full data for any extra training seasons
    for ts in self.train_seasons:
        if ts == self.season:
            continue
        ts_gw = self.db.read("gameweek_performances", where={"season": ts})
        ts_players = self.db.read("players", where={"season": ts})
        ts_fixtures = self.db.read("fixtures", where={"season": ts})
        ts_teams = self.db.read("teams", where={"season": ts})
        max_gw = int(ts_gw["gameweek"].max()) if len(ts_gw) > 0 else 1
        seasons[ts] = SeasonData(
            gameweek_performances=ts_gw,
            fixtures=ts_fixtures,
            players=ts_players,
            teams=ts_teams,
            current_gameweek=max_gw,
            season=ts,
        )

    # Current season: only data before current_gw
    past_perf = gw_perf_df[gw_perf_df["gameweek"] < current_gw]
    seasons[self.season] = SeasonData(
        gameweek_performances=past_perf,
        fixtures=fixtures_df,
        players=players_df,
        teams=teams_df,
        current_gameweek=current_gw,
        season=self.season,
    )

    return HistoricalData(seasons=seasons)
```

3. In the `run()` GW loop, before `actions = self.model.recommend(state, season_data)`, add:
```python
# Mid-season retraining
if (
    self.retrain_every_n_gws is not None
    and gw > gameweeks[0]
    and (gw - gameweeks[0]) % self.retrain_every_n_gws == 0
):
    historical = self._build_training_data(gw, gw_perf_df, players_df, fixtures_df, teams_df)
    self.model.train(historical)
```

**Step 4: Run tests to verify they pass**

Run: `/home/node/.local/bin/uv run pytest tests/simulation/test_engine.py -v`
Expected: ALL PASS

**Step 5: Run full test suite**

Run: `/home/node/.local/bin/uv run pytest tests/ -q`
Expected: ALL PASS (existing tests unaffected since new params are optional)

**Step 6: Commit**

```bash
git add src/fpl_model/simulation/engine.py tests/simulation/test_engine.py
git commit -m "feat: add mid-season retraining to SeasonSimulator"
```

---

### Task 4: Update notebook with retrained model variants

**Files:**
- Modify: `notebooks/model_comparison.ipynb`

**Step 1: Update the training cell**

After the existing model training, add retrained variants to the `models` dict used for simulation. These use the same predictors with `recency_decay` set, and the `SeasonSimulator` is called with `retrain_every_n_gws`.

In the **train cell** (cell with `registry = get_default_registry()`), add after existing training:

```python
from fpl_model.models.base import PredictOptimizeModel
from fpl_model.models.predictors.xgboost_predictor import XGBoostPredictor
from fpl_model.models.optimizers.greedy import GreedyOptimizer

# Create retrained variants with recency decay
xgb_retrain = PredictOptimizeModel(
    XGBoostPredictor(recency_decay=0.95),
    GreedyOptimizer(enable_transfers=True),
)
print("Training xgb-greedy-retrain...", end=" ", flush=True)
t0 = time.time()
xgb_retrain.train(historical)
print(f"done ({time.time() - t0:.1f}s)")
```

**Step 2: Update the simulation cell**

For retrained models, pass `retrain_every_n_gws` to `SeasonSimulator`:

```python
# Add retrained models with mid-season retraining
retrain_models = {"xgb-greedy-retrain": xgb_retrain}

for name, model in retrain_models.items():
    print(f"Simulating {name} (with retraining)...", end=" ", flush=True)
    t0 = time.time()
    sim = SeasonSimulator(
        model=model, season=EVAL_SEASON, db=db,
        retrain_every_n_gws=5, train_seasons=TRAIN_SEASONS,
    )
    result = sim.run()
    elapsed = time.time() - t0
    results[name] = result
    print(f"{result.total_points} pts ({len(result.gameweek_points)} GWs, {elapsed:.1f}s)")
```

**Step 3: Verify notebook runs**

Run all cells in the notebook. The new `xgb-greedy-retrain` model should appear in all comparison charts and tables.

**Step 4: Commit**

```bash
git add notebooks/model_comparison.ipynb
git commit -m "feat: add retrained model variant to comparison notebook"
```
