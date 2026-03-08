# Advanced ML Models Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build sophisticated ML predictors (XGBoost, LSTM), transfer-capable optimizers (greedy, LP), and an end-to-end PPO RL agent for FPL team management.

**Architecture:** Three tiers of models that all conform to the existing `ActionModel` interface. A shared `features.py` module provides feature engineering used by all models. Models are registered in `defaults.py` and evaluated via the existing `SeasonSimulator` backtesting pipeline.

**Tech Stack:** XGBoost, PyTorch (LSTM + PPO), PuLP (LP solver), pandas, numpy

---

### Task 1: Update dependencies

**Files:**
- Modify: `pyproject.toml:12-20`

**Step 1: Update the ml optional dependency group**

```toml
ml = [
    "scikit-learn>=1.4",
    "xgboost>=2.0",
    "pulp>=2.7",
    "torch>=2.0",
]
```

**Step 2: Install and verify**

Run: `cd /home/node/fpl-model && /home/node/.local/bin/uv sync --extra dev --extra ml`
Expected: All packages install successfully

**Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat: add xgboost, pulp, torch to ml dependencies"
```

---

### Task 2: Shared feature engineering module

**Files:**
- Create: `src/fpl_model/models/features.py`
- Test: `tests/models/test_features.py`

**Step 1: Write failing tests**

```python
# tests/models/test_features.py
import numpy as np
import pandas as pd
import pytest

from fpl_model.models.features import build_feature_matrix, build_player_features, build_sequence_features


def _make_season_data():
    """Build minimal SeasonData-like DataFrames for feature tests."""
    from fpl_model.models.base import SeasonData

    players = pd.DataFrame({
        "code": [100, 200, 300],
        "element_type": [2, 3, 4],
        "team_code": [1, 2, 3],
        "now_cost": [50, 80, 100],
    })

    gw_rows = []
    for gw in range(1, 11):
        for code, et in [(100, 2), (200, 3), (300, 4)]:
            gw_rows.append({
                "player_code": code,
                "gameweek": gw,
                "total_points": gw + code // 100,  # varies by player and GW
                "minutes": 90 if gw % 3 != 0 else 0,  # benched every 3rd GW
                "goals_scored": 1 if code == 300 and gw % 2 == 0 else 0,
                "assists": 1 if code == 200 and gw % 3 == 0 else 0,
                "clean_sheets": 1 if code == 100 and gw % 2 == 0 else 0,
                "bonus": gw % 4,
                "bps": 20 + gw,
                "expected_goals": 0.3 if code >= 200 else 0.05,
                "expected_assists": 0.2 if code == 200 else 0.1,
                "expected_goal_involvements": 0.5 if code >= 200 else 0.15,
                "expected_goals_conceded": 1.0,
                "value": 50 + gw,
                "was_home": gw % 2 == 0,
                "opponent_team": (gw % 20) + 1,
                "fixture_id": gw * 10 + code // 100,
                "transfers_balance": 100 * (gw % 3 - 1),
                "selected": 10000 + gw * 100,
                "kickoff_time": f"2024-08-{10 + gw}T15:00:00Z",
            })
    gw_perf = pd.DataFrame(gw_rows)

    fixtures = pd.DataFrame({
        "season": ["2024-25"] * 5,
        "fixture_id": [101, 102, 103, 104, 105],
        "gameweek": [11, 11, 11, 12, 12],
        "team_h": [1, 2, 3, 1, 2],
        "team_a": [4, 5, 6, 3, 4],
        "team_h_difficulty": [2, 3, 4, 2, 3],
        "team_a_difficulty": [3, 2, 2, 4, 3],
        "finished": [0, 0, 0, 0, 0],
        "kickoff_time": ["2024-09-01T15:00:00Z"] * 5,
    })

    teams = pd.DataFrame({
        "season": ["2024-25"] * 6,
        "team_code": [1, 2, 3, 4, 5, 6],
        "name": [f"Team{i}" for i in range(1, 7)],
        "strength_attack_home": [1200, 1100, 1300, 1000, 1150, 1050],
        "strength_attack_away": [1150, 1050, 1250, 950, 1100, 1000],
        "strength_defence_home": [1300, 1200, 1100, 1050, 1250, 1150],
        "strength_defence_away": [1250, 1150, 1050, 1000, 1200, 1100],
    })

    return SeasonData(
        gameweek_performances=gw_perf,
        fixtures=fixtures,
        players=players,
        teams=teams,
        current_gameweek=11,
        season="2024-25",
    )


class TestBuildPlayerFeatures:
    def test_returns_dict_with_expected_keys(self):
        data = _make_season_data()
        features = build_player_features(data, player_code=200, gw=11)
        assert isinstance(features, dict)
        assert "form_3" in features
        assert "form_5" in features
        assert "form_10" in features
        assert "xg_rolling" in features
        assert "xa_rolling" in features
        assert "minutes_rolling" in features
        assert "bps_rolling" in features
        assert "is_home" in features
        assert "element_type" in features

    def test_form_values_are_reasonable(self):
        data = _make_season_data()
        features = build_player_features(data, player_code=200, gw=11)
        # Player 200 scores gw + 2 points per GW, form_3 should be mean of GWs 8,9,10
        assert features["form_3"] > 0

    def test_missing_player_returns_defaults(self):
        data = _make_season_data()
        features = build_player_features(data, player_code=999, gw=11)
        assert features["form_3"] == 0.0
        assert features["minutes_rolling"] == 0.0


class TestBuildFeatureMatrix:
    def test_returns_dataframe_with_all_players(self):
        data = _make_season_data()
        matrix = build_feature_matrix(data)
        assert isinstance(matrix, pd.DataFrame)
        assert len(matrix) == 3  # 3 players
        assert "player_code" in matrix.columns
        assert "form_5" in matrix.columns

    def test_feature_columns_are_numeric(self):
        data = _make_season_data()
        matrix = build_feature_matrix(data)
        feature_cols = [c for c in matrix.columns if c != "player_code"]
        for col in feature_cols:
            assert pd.api.types.is_numeric_dtype(matrix[col]), f"{col} is not numeric"


class TestBuildSequenceFeatures:
    def test_returns_correct_shape(self):
        data = _make_season_data()
        seq = build_sequence_features(data, player_code=200, gw=11, seq_len=5)
        assert isinstance(seq, np.ndarray)
        assert seq.shape[0] == 5  # seq_len
        assert seq.shape[1] > 0  # feature dimension

    def test_pads_short_history(self):
        data = _make_season_data()
        seq = build_sequence_features(data, player_code=200, gw=3, seq_len=10)
        assert seq.shape[0] == 10
        # First rows should be zero-padded
        assert np.all(seq[0] == 0.0)

    def test_unknown_player_returns_zeros(self):
        data = _make_season_data()
        seq = build_sequence_features(data, player_code=999, gw=11, seq_len=5)
        assert np.all(seq == 0.0)
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/node/fpl-model && /home/node/.local/bin/uv run pytest tests/models/test_features.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fpl_model.models.features'`

**Step 3: Implement features.py**

Create `src/fpl_model/models/features.py`:

```python
"""Shared feature engineering for FPL ML models.

Used by XGBoost predictor, sequence predictor, and RL state encoder.
All features are computed from SeasonData without future data leakage —
only data from gameweeks strictly before the target GW is used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fpl_model.models.base import SeasonData

# Per-GW feature columns extracted from gameweek_performances
_GW_STAT_COLS = [
    "total_points", "minutes", "goals_scored", "assists", "clean_sheets",
    "bonus", "bps", "expected_goals", "expected_assists",
    "expected_goal_involvements", "expected_goals_conceded",
]

# Rolling window sizes for form features
_WINDOWS = [3, 5, 10]

# Feature vector columns for sequence model (per timestep)
_SEQ_COLS = [
    "total_points", "minutes", "expected_goals", "expected_assists",
    "bps", "was_home",
]


def build_player_features(
    data: SeasonData,
    player_code: int,
    gw: int,
) -> dict[str, float]:
    """Build a feature dict for a single player at a specific gameweek.

    Only uses data from GWs strictly before *gw* to prevent leakage.
    Returns a dict of feature_name -> float. Unknown players get all zeros.
    """
    gw_perf = data.gameweek_performances
    past = gw_perf[(gw_perf["player_code"] == player_code) & (gw_perf["gameweek"] < gw)]
    past = past.sort_values("gameweek")

    features: dict[str, float] = {}

    # Rolling form (mean total_points over last N GWs)
    for w in _WINDOWS:
        tail = past.tail(w)
        features[f"form_{w}"] = float(tail["total_points"].mean()) if len(tail) > 0 else 0.0
        features[f"form_std_{w}"] = float(tail["total_points"].std()) if len(tail) > 1 else 0.0

    # Rolling expected stats (per 90)
    for col in ["expected_goals", "expected_assists", "expected_goal_involvements", "expected_goals_conceded"]:
        tail = past.tail(5)
        if len(tail) > 0 and col in tail.columns:
            features[f"{col.replace('expected_', 'x')}_rolling"] = float(tail[col].mean())
        else:
            features[f"{col.replace('expected_', 'x')}_rolling"] = 0.0

    # Minutes trend
    tail = past.tail(5)
    features["minutes_rolling"] = float(tail["minutes"].mean()) if len(tail) > 0 else 0.0

    # BPS trend
    features["bps_rolling"] = float(tail["bps"].mean()) if len(tail) > 0 and "bps" in tail.columns else 0.0

    # Market momentum
    if len(tail) > 0 and "transfers_balance" in tail.columns:
        features["transfers_balance_avg"] = float(tail["transfers_balance"].mean())
    else:
        features["transfers_balance_avg"] = 0.0
    if len(tail) > 0 and "selected" in tail.columns:
        features["selected_latest"] = float(tail["selected"].iloc[-1])
    else:
        features["selected_latest"] = 0.0

    # Player metadata from players df
    player_row = data.players[data.players["code"] == player_code]
    if len(player_row) > 0:
        et = int(player_row.iloc[0]["element_type"])
        features["element_type"] = float(et)
        features["is_gk"] = 1.0 if et == 1 else 0.0
        features["is_def"] = 1.0 if et == 2 else 0.0
        features["is_mid"] = 1.0 if et == 3 else 0.0
        features["is_fwd"] = 1.0 if et == 4 else 0.0
        team_code = int(player_row.iloc[0]["team_code"])
        features["now_cost"] = float(player_row.iloc[0].get("now_cost", 0))
    else:
        features["element_type"] = 0.0
        features["is_gk"] = 0.0
        features["is_def"] = 0.0
        features["is_mid"] = 0.0
        features["is_fwd"] = 0.0
        team_code = 0
        features["now_cost"] = 0.0

    # Fixture context for upcoming GW
    is_home, opp_team_code, fixture_difficulty = _get_fixture_context(data, team_code, gw)
    features["is_home"] = float(is_home)
    features["fixture_difficulty"] = float(fixture_difficulty)

    # Opponent strength
    opp_strength = _get_opponent_strength(data, opp_team_code, is_home)
    features.update(opp_strength)

    return features


def build_feature_matrix(data: SeasonData) -> pd.DataFrame:
    """Build a feature matrix for all players in data.players at the current GW.

    Returns a DataFrame with one row per player and columns for each feature
    plus 'player_code'.
    """
    gw = data.current_gameweek
    rows = []
    for code in data.players["code"]:
        feats = build_player_features(data, int(code), gw)
        feats["player_code"] = int(code)
        rows.append(feats)
    return pd.DataFrame(rows)


def build_sequence_features(
    data: SeasonData,
    player_code: int,
    gw: int,
    seq_len: int = 10,
) -> np.ndarray:
    """Build a (seq_len, num_features) array of per-GW features for a player.

    Returns the most recent *seq_len* GWs before *gw*. If fewer GWs exist,
    the earlier positions are zero-padded. Unknown players return all zeros.
    """
    gw_perf = data.gameweek_performances
    past = gw_perf[(gw_perf["player_code"] == player_code) & (gw_perf["gameweek"] < gw)]
    past = past.sort_values("gameweek")

    # Select feature columns that exist
    available_cols = [c for c in _SEQ_COLS if c in past.columns]
    if not available_cols:
        available_cols = _SEQ_COLS  # will produce zeros anyway

    num_features = len(_SEQ_COLS)

    if len(past) == 0:
        return np.zeros((seq_len, num_features), dtype=np.float32)

    # Extract feature values
    values = []
    for col in _SEQ_COLS:
        if col in past.columns:
            values.append(past[col].astype(float).values)
        else:
            values.append(np.zeros(len(past)))
    arr = np.stack(values, axis=1).astype(np.float32)  # (num_gws, num_features)

    # Take last seq_len rows, pad if needed
    if len(arr) >= seq_len:
        return arr[-seq_len:]
    else:
        padded = np.zeros((seq_len, num_features), dtype=np.float32)
        padded[seq_len - len(arr):] = arr
        return padded


def _get_fixture_context(
    data: SeasonData, team_code: int, gw: int
) -> tuple[bool, int, float]:
    """Return (is_home, opponent_team_code, difficulty) for a team's next fixture."""
    fixtures = data.fixtures
    if fixtures.empty or team_code == 0:
        return False, 0, 3.0

    gw_fixtures = fixtures[fixtures["gameweek"] == gw]
    home_match = gw_fixtures[gw_fixtures["team_h"] == team_code]
    if len(home_match) > 0:
        row = home_match.iloc[0]
        difficulty = row.get("team_h_difficulty", 3)
        return True, int(row["team_a"]), float(difficulty) if pd.notna(difficulty) else 3.0

    away_match = gw_fixtures[gw_fixtures["team_a"] == team_code]
    if len(away_match) > 0:
        row = away_match.iloc[0]
        difficulty = row.get("team_a_difficulty", 3)
        return False, int(row["team_h"]), float(difficulty) if pd.notna(difficulty) else 3.0

    return False, 0, 3.0


def _get_opponent_strength(
    data: SeasonData, opp_team_code: int, is_home: bool
) -> dict[str, float]:
    """Return opponent strength features."""
    result = {"opp_attack": 0.0, "opp_defence": 0.0}
    if opp_team_code == 0 or data.teams.empty:
        return result

    opp = data.teams[data.teams["team_code"] == opp_team_code]
    if len(opp) == 0:
        return result

    row = opp.iloc[0]
    if is_home:
        # We're home, opponent is away
        result["opp_attack"] = float(row.get("strength_attack_away", 0) or 0)
        result["opp_defence"] = float(row.get("strength_defence_away", 0) or 0)
    else:
        result["opp_attack"] = float(row.get("strength_attack_home", 0) or 0)
        result["opp_defence"] = float(row.get("strength_defence_home", 0) or 0)
    return result
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/node/fpl-model && /home/node/.local/bin/uv run pytest tests/models/test_features.py -v`
Expected: All pass

**Step 5: Run full test suite**

Run: `cd /home/node/fpl-model && /home/node/.local/bin/uv run pytest -x -q`
Expected: All pass

**Step 6: Commit**

```bash
git add src/fpl_model/models/features.py tests/models/test_features.py
git commit -m "feat: add shared feature engineering module for ML models"
```

---

### Task 3: XGBoost predictor

**Files:**
- Create: `src/fpl_model/models/predictors/xgboost_predictor.py`
- Test: `tests/models/test_xgboost_predictor.py`

**Step 1: Write failing tests**

```python
# tests/models/test_xgboost_predictor.py
import pandas as pd
import pytest

from fpl_model.models.base import HistoricalData, PlayerPredictions, SeasonData
from fpl_model.models.predictors.xgboost_predictor import XGBoostPredictor
from fpl_model.simulation.actions import ChipType
from fpl_model.simulation.state import PlayerInSquad, SquadState


def _make_training_data():
    """Build multi-season data for training."""
    seasons = {}
    for season_name in ["2022-23", "2023-24"]:
        players = pd.DataFrame({
            "code": [100, 200, 300],
            "element_type": [2, 3, 4],
            "team_code": [1, 2, 3],
            "now_cost": [50, 80, 100],
        })

        gw_rows = []
        for gw in range(1, 39):
            for code in [100, 200, 300]:
                gw_rows.append({
                    "player_code": code,
                    "gameweek": gw,
                    "total_points": (code // 100) + (gw % 5),
                    "minutes": 90 if gw % 4 != 0 else 0,
                    "goals_scored": 1 if code == 300 and gw % 3 == 0 else 0,
                    "assists": 1 if code == 200 and gw % 4 == 0 else 0,
                    "clean_sheets": 1 if code == 100 and gw % 2 == 0 else 0,
                    "bonus": gw % 4,
                    "bps": 20 + gw % 10,
                    "expected_goals": 0.3 if code >= 200 else 0.05,
                    "expected_assists": 0.2 if code == 200 else 0.1,
                    "expected_goal_involvements": 0.5 if code >= 200 else 0.15,
                    "expected_goals_conceded": 1.0,
                    "value": 50 + gw % 10,
                    "was_home": gw % 2 == 0,
                    "opponent_team": (gw % 6) + 1,
                    "fixture_id": gw * 10,
                    "transfers_balance": 100,
                    "selected": 10000,
                    "kickoff_time": f"2024-08-{10 + (gw % 20)}T15:00:00Z",
                })
        gw_perf = pd.DataFrame(gw_rows)

        fixtures = pd.DataFrame({
            "season": [season_name] * 3,
            "fixture_id": [1, 2, 3],
            "gameweek": [1, 1, 2],
            "team_h": [1, 2, 3],
            "team_a": [4, 5, 6],
            "team_h_difficulty": [2, 3, 4],
            "team_a_difficulty": [3, 2, 2],
            "finished": [1, 1, 1],
        })

        teams = pd.DataFrame({
            "season": [season_name] * 6,
            "team_code": [1, 2, 3, 4, 5, 6],
            "name": [f"Team{i}" for i in range(1, 7)],
            "strength_attack_home": [1200, 1100, 1300, 1000, 1150, 1050],
            "strength_attack_away": [1150, 1050, 1250, 950, 1100, 1000],
            "strength_defence_home": [1300, 1200, 1100, 1050, 1250, 1150],
            "strength_defence_away": [1250, 1150, 1050, 1000, 1200, 1100],
        })

        seasons[season_name] = SeasonData(
            gameweek_performances=gw_perf,
            fixtures=fixtures,
            players=players,
            teams=teams,
            current_gameweek=38,
            season=season_name,
        )
    return HistoricalData(seasons=seasons)


def _make_squad():
    players = [
        PlayerInSquad(code=100, element_type=2, buy_price=50, sell_price=50),
        PlayerInSquad(code=200, element_type=3, buy_price=80, sell_price=80),
        PlayerInSquad(code=300, element_type=4, buy_price=100, sell_price=100),
    ]
    return SquadState(
        players=players,
        budget=100,
        free_transfers=1,
        chips_available={ct: 2 for ct in ChipType},
        current_gameweek=10,
    )


class TestXGBoostPredictor:
    def test_train_and_predict(self):
        hist = _make_training_data()
        predictor = XGBoostPredictor()
        predictor.train(hist)

        # Use one of the training seasons for prediction
        data = hist.seasons["2023-24"]
        data = SeasonData(
            gameweek_performances=data.gameweek_performances[data.gameweek_performances["gameweek"] <= 10],
            fixtures=data.fixtures,
            players=data.players,
            teams=data.teams,
            current_gameweek=11,
            season="2023-24",
        )

        state = _make_squad()
        preds = predictor.predict(state, data)
        assert isinstance(preds, PlayerPredictions)
        assert len(preds.predictions) == 3
        for code in [100, 200, 300]:
            assert code in preds.predictions
            assert isinstance(preds.predictions[code], float)

    def test_predict_without_training_uses_fallback(self):
        predictor = XGBoostPredictor()
        hist = _make_training_data()
        data = hist.seasons["2023-24"]
        data = SeasonData(
            gameweek_performances=data.gameweek_performances[data.gameweek_performances["gameweek"] <= 10],
            fixtures=data.fixtures,
            players=data.players,
            teams=data.teams,
            current_gameweek=11,
            season="2023-24",
        )
        state = _make_squad()
        preds = predictor.predict(state, data)
        # Without training, should still return predictions (form-based fallback)
        assert len(preds.predictions) == 3
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/node/fpl-model && /home/node/.local/bin/uv run pytest tests/models/test_xgboost_predictor.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement XGBoost predictor**

Create `src/fpl_model/models/predictors/xgboost_predictor.py`:

```python
"""XGBoost-based predictor: predicts next-GW points using engineered features."""

from __future__ import annotations

from fpl_model.models.base import HistoricalData, PlayerPredictions, Predictor, SeasonData
from fpl_model.models.features import build_feature_matrix, build_player_features
from fpl_model.simulation.state import SquadState


class XGBoostPredictor(Predictor):
    """Predict player points using a gradient-boosted tree model.

    Features include rolling form, expected stats, fixture difficulty,
    ICT metrics, and market momentum. Falls back to form-based prediction
    if the model has not been trained.
    """

    def __init__(self, lookback_start: int = 5) -> None:
        self.lookback_start = lookback_start
        self.model = None
        self._feature_cols: list[str] | None = None

    def train(self, historical_data: HistoricalData) -> None:
        """Train XGBoost on all (player, GW) samples from historical seasons."""
        import xgboost as xgb

        rows_X = []
        rows_y = []

        for season_name, season_data in historical_data.seasons.items():
            gw_perf = season_data.gameweek_performances
            gameweeks = sorted(gw_perf["gameweek"].unique())

            for gw in gameweeks:
                if gw < self.lookback_start + 1:
                    continue  # need history to build features

                # Build SeasonData up to (but not including) this GW for features
                visible_data = SeasonData(
                    gameweek_performances=gw_perf[gw_perf["gameweek"] < gw],
                    fixtures=season_data.fixtures,
                    players=season_data.players,
                    teams=season_data.teams,
                    current_gameweek=gw,
                    season=season_name,
                )

                # Actual points for this GW (target)
                gw_actual = gw_perf[gw_perf["gameweek"] == gw]
                actual_map = dict(zip(gw_actual["player_code"], gw_actual["total_points"]))

                for code in season_data.players["code"]:
                    code = int(code)
                    if code not in actual_map:
                        continue
                    feats = build_player_features(visible_data, code, gw)
                    rows_X.append(feats)
                    rows_y.append(float(actual_map[code]))

        if not rows_X:
            return

        import pandas as pd
        X = pd.DataFrame(rows_X)
        self._feature_cols = list(X.columns)
        y = pd.Series(rows_y)

        self.model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            verbosity=0,
        )
        self.model.fit(X, y)

    def predict(self, state: SquadState, data: SeasonData) -> PlayerPredictions:
        """Predict next-GW points for all players."""
        matrix = build_feature_matrix(data)

        if self.model is not None and self._feature_cols is not None:
            codes = matrix["player_code"].tolist()
            X = matrix[self._feature_cols]
            y_pred = self.model.predict(X)
            predictions = {int(c): float(p) for c, p in zip(codes, y_pred)}
        else:
            # Fallback: use form_5 as prediction
            predictions = {}
            for _, row in matrix.iterrows():
                code = int(row["player_code"])
                predictions[code] = float(row.get("form_5", 2.0))

        return PlayerPredictions(predictions=predictions)
```

**Step 4: Run tests**

Run: `cd /home/node/fpl-model && /home/node/.local/bin/uv run pytest tests/models/test_xgboost_predictor.py -v`
Expected: All pass

**Step 5: Run full test suite**

Run: `cd /home/node/fpl-model && /home/node/.local/bin/uv run pytest -x -q`
Expected: All pass

**Step 6: Commit**

```bash
git add src/fpl_model/models/predictors/xgboost_predictor.py tests/models/test_xgboost_predictor.py
git commit -m "feat: add XGBoost predictor with engineered features"
```

---

### Task 4: Sequence (LSTM) predictor

**Files:**
- Create: `src/fpl_model/models/predictors/sequence_predictor.py`
- Test: `tests/models/test_sequence_predictor.py`

**Step 1: Write failing tests**

```python
# tests/models/test_sequence_predictor.py
import pandas as pd
import pytest

from fpl_model.models.base import HistoricalData, PlayerPredictions, SeasonData
from fpl_model.models.predictors.sequence_predictor import SequencePredictor
from fpl_model.simulation.actions import ChipType
from fpl_model.simulation.state import PlayerInSquad, SquadState


def _make_training_data():
    """Build multi-season data for training."""
    seasons = {}
    for season_name in ["2022-23", "2023-24"]:
        players = pd.DataFrame({
            "code": [100, 200],
            "element_type": [3, 4],
            "team_code": [1, 2],
            "now_cost": [80, 100],
        })

        gw_rows = []
        for gw in range(1, 39):
            for code in [100, 200]:
                gw_rows.append({
                    "player_code": code,
                    "gameweek": gw,
                    "total_points": (code // 100) + (gw % 5),
                    "minutes": 90,
                    "expected_goals": 0.3,
                    "expected_assists": 0.2,
                    "bps": 25,
                    "was_home": gw % 2 == 0,
                    "fixture_id": gw * 10,
                })
        gw_perf = pd.DataFrame(gw_rows)

        seasons[season_name] = SeasonData(
            gameweek_performances=gw_perf,
            fixtures=pd.DataFrame(),
            players=players,
            teams=pd.DataFrame(),
            current_gameweek=38,
            season=season_name,
        )
    return HistoricalData(seasons=seasons)


def _make_squad():
    players = [
        PlayerInSquad(code=100, element_type=3, buy_price=80, sell_price=80),
        PlayerInSquad(code=200, element_type=4, buy_price=100, sell_price=100),
    ]
    return SquadState(
        players=players,
        budget=100,
        free_transfers=1,
        chips_available={ct: 2 for ct in ChipType},
        current_gameweek=10,
    )


class TestSequencePredictor:
    def test_train_and_predict(self):
        hist = _make_training_data()
        predictor = SequencePredictor(seq_len=5, hidden_size=16, num_layers=1, epochs=2)
        predictor.train(hist)

        data = hist.seasons["2023-24"]
        data = SeasonData(
            gameweek_performances=data.gameweek_performances[data.gameweek_performances["gameweek"] <= 10],
            fixtures=pd.DataFrame(),
            players=data.players,
            teams=pd.DataFrame(),
            current_gameweek=11,
            season="2023-24",
        )
        state = _make_squad()
        preds = predictor.predict(state, data)
        assert isinstance(preds, PlayerPredictions)
        assert 100 in preds.predictions
        assert 200 in preds.predictions

    def test_predict_without_training(self):
        predictor = SequencePredictor(seq_len=5, hidden_size=16, num_layers=1)
        hist = _make_training_data()
        data = hist.seasons["2023-24"]
        data = SeasonData(
            gameweek_performances=data.gameweek_performances[data.gameweek_performances["gameweek"] <= 10],
            fixtures=pd.DataFrame(),
            players=data.players,
            teams=pd.DataFrame(),
            current_gameweek=11,
            season="2023-24",
        )
        state = _make_squad()
        preds = predictor.predict(state, data)
        assert len(preds.predictions) == 2
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/node/fpl-model && /home/node/.local/bin/uv run pytest tests/models/test_sequence_predictor.py -v`
Expected: FAIL

**Step 3: Implement sequence predictor**

Create `src/fpl_model/models/predictors/sequence_predictor.py`:

```python
"""LSTM-based sequence predictor for FPL player points."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from fpl_model.models.base import HistoricalData, PlayerPredictions, Predictor, SeasonData
from fpl_model.models.features import _SEQ_COLS, build_sequence_features
from fpl_model.simulation.state import SquadState


class _PointsLSTM(nn.Module):
    """LSTM that takes a sequence of per-GW features and predicts next-GW points."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_positions: int = 4):
        super().__init__()
        self.pos_embedding = nn.Embedding(num_positions + 1, 4)  # 0=unknown, 1-4=positions
        self.lstm = nn.LSTM(input_size + 4, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size), pos: (batch,) int
        pos_emb = self.pos_embedding(pos)  # (batch, 4)
        pos_emb = pos_emb.unsqueeze(1).expand(-1, x.size(1), -1)  # (batch, seq_len, 4)
        combined = torch.cat([x, pos_emb], dim=2)
        out, _ = self.lstm(combined)
        return self.fc(out[:, -1, :]).squeeze(-1)  # (batch,)


class SequencePredictor(Predictor):
    """Predict player points using an LSTM on recent GW sequences.

    Falls back to simple mean of recent points if the model is not trained.
    """

    def __init__(
        self,
        seq_len: int = 10,
        hidden_size: int = 64,
        num_layers: int = 2,
        epochs: int = 20,
        lr: float = 1e-3,
        batch_size: int = 64,
    ) -> None:
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.model: _PointsLSTM | None = None

    def train(self, historical_data: HistoricalData) -> None:
        """Train LSTM on all (player, GW) sequences from historical seasons."""
        input_size = len(_SEQ_COLS)
        self.model = _PointsLSTM(input_size, self.hidden_size, self.num_layers)

        X_list, pos_list, y_list = [], [], []

        for season_data in historical_data.seasons.values():
            gw_perf = season_data.gameweek_performances
            gameweeks = sorted(gw_perf["gameweek"].unique())

            for gw in gameweeks:
                if gw < 2:
                    continue
                gw_actual = gw_perf[gw_perf["gameweek"] == gw]
                actual_map = dict(zip(gw_actual["player_code"], gw_actual["total_points"]))

                visible = SeasonData(
                    gameweek_performances=gw_perf[gw_perf["gameweek"] < gw],
                    fixtures=season_data.fixtures,
                    players=season_data.players,
                    teams=season_data.teams,
                    current_gameweek=gw,
                    season=season_data.season,
                )

                for code in season_data.players["code"]:
                    code = int(code)
                    if code not in actual_map:
                        continue
                    seq = build_sequence_features(visible, code, gw, self.seq_len)
                    player_row = season_data.players[season_data.players["code"] == code]
                    et = int(player_row.iloc[0]["element_type"]) if len(player_row) > 0 else 0
                    X_list.append(seq)
                    pos_list.append(et)
                    y_list.append(float(actual_map[code]))

        if not X_list:
            return

        X = torch.tensor(np.array(X_list), dtype=torch.float32)
        pos = torch.tensor(pos_list, dtype=torch.long)
        y = torch.tensor(y_list, dtype=torch.float32)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        self.model.train()
        n = len(X)
        for epoch in range(self.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                pred = self.model(X[idx], pos[idx])
                loss = loss_fn(pred, y[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def predict(self, state: SquadState, data: SeasonData) -> PlayerPredictions:
        """Predict next-GW points for all players in data.players."""
        predictions: dict[int, float] = {}
        gw = data.current_gameweek

        if self.model is not None:
            self.model.eval()
            X_list, pos_list, codes = [], [], []
            for code in data.players["code"]:
                code = int(code)
                seq = build_sequence_features(data, code, gw, self.seq_len)
                player_row = data.players[data.players["code"] == code]
                et = int(player_row.iloc[0]["element_type"]) if len(player_row) > 0 else 0
                X_list.append(seq)
                pos_list.append(et)
                codes.append(code)

            with torch.no_grad():
                X = torch.tensor(np.array(X_list), dtype=torch.float32)
                pos = torch.tensor(pos_list, dtype=torch.long)
                preds = self.model(X, pos).numpy()

            for code, pred in zip(codes, preds):
                predictions[code] = float(pred)
        else:
            # Fallback: mean of recent points
            gw_perf = data.gameweek_performances
            for code in data.players["code"]:
                code = int(code)
                past = gw_perf[(gw_perf["player_code"] == code) & (gw_perf["gameweek"] < gw)]
                if len(past) > 0:
                    predictions[code] = float(past.tail(5)["total_points"].mean())
                else:
                    predictions[code] = 2.0

        return PlayerPredictions(predictions=predictions)
```

**Step 4: Run tests**

Run: `cd /home/node/fpl-model && /home/node/.local/bin/uv run pytest tests/models/test_sequence_predictor.py -v`
Expected: All pass

**Step 5: Full suite**

Run: `cd /home/node/fpl-model && /home/node/.local/bin/uv run pytest -x -q`
Expected: All pass

**Step 6: Commit**

```bash
git add src/fpl_model/models/predictors/sequence_predictor.py tests/models/test_sequence_predictor.py
git commit -m "feat: add LSTM sequence predictor for player point prediction"
```

---

### Task 5: Extend greedy optimizer with transfer evaluation

**Files:**
- Modify: `src/fpl_model/models/optimizers/greedy.py`
- Test: `tests/models/test_greedy_transfers.py`

**Step 1: Write failing tests**

```python
# tests/models/test_greedy_transfers.py
import pandas as pd
import pytest

from fpl_model.models.base import PlayerPredictions, SeasonData
from fpl_model.models.optimizers.greedy import GreedyOptimizer
from fpl_model.simulation.actions import ChipType, Transfer
from fpl_model.simulation.state import PlayerInSquad, SquadState


def _make_state_and_data():
    """Create a squad and data where a transfer is clearly beneficial."""
    # Squad: 2 GK, 5 DEF, 5 MID, 3 FWD
    players = []
    code = 1
    for et, count in [(1, 2), (2, 5), (3, 5), (4, 3)]:
        for _ in range(count):
            players.append(PlayerInSquad(code=code, element_type=et, buy_price=50, sell_price=50))
            code += 1

    state = SquadState(
        players=players,
        budget=100,
        free_transfers=1,
        chips_available={ct: 2 for ct in ChipType},
        current_gameweek=5,
        starting_xi=[p.code for p in players[:11]],
        bench_order=[p.code for p in players[11:]],
        captain=8,
        vice_captain=9,
    )

    # Available players include a much better option (code=99)
    all_players = pd.DataFrame({
        "code": [p.code for p in players] + [99],
        "element_type": [p.element_type for p in players] + [3],  # midfielder
        "team_code": list(range(1, 16)) + [20],
        "now_cost": [50] * 15 + [50],
    })

    data = SeasonData(
        gameweek_performances=pd.DataFrame(),
        fixtures=pd.DataFrame(),
        players=all_players,
        teams=pd.DataFrame(),
        current_gameweek=5,
        season="2024-25",
    )

    return state, data


class TestGreedyTransfers:
    def test_makes_transfer_when_gain_exceeds_threshold(self):
        state, data = _make_state_and_data()
        # Player 99 (available MID) predicted at 10.0, worst squad MID at 1.0
        preds = {p.code: 3.0 for p in state.players}
        preds[12] = 1.0  # a midfielder in squad, predicted low
        preds[99] = 10.0  # available midfielder, predicted high
        predictions = PlayerPredictions(predictions=preds)

        optimizer = GreedyOptimizer(enable_transfers=True, transfer_gain_threshold=2.0)
        actions = optimizer.optimize(predictions, state, data)

        transfers = [a for a in actions if isinstance(a, Transfer)]
        assert len(transfers) == 1
        assert transfers[0].player_out == 12
        assert transfers[0].player_in == 99

    def test_no_transfer_when_gain_below_threshold(self):
        state, data = _make_state_and_data()
        # All players predicted similarly
        preds = {p.code: 5.0 for p in state.players}
        preds[99] = 5.5  # marginal improvement
        predictions = PlayerPredictions(predictions=preds)

        optimizer = GreedyOptimizer(enable_transfers=True, transfer_gain_threshold=2.0)
        actions = optimizer.optimize(predictions, state, data)

        transfers = [a for a in actions if isinstance(a, Transfer)]
        assert len(transfers) == 0

    def test_no_transfer_when_disabled(self):
        state, data = _make_state_and_data()
        preds = {p.code: 3.0 for p in state.players}
        preds[12] = 1.0
        preds[99] = 10.0
        predictions = PlayerPredictions(predictions=preds)

        optimizer = GreedyOptimizer(enable_transfers=False)
        actions = optimizer.optimize(predictions, state, data)
        transfers = [a for a in actions if isinstance(a, Transfer)]
        assert len(transfers) == 0

    def test_respects_budget(self):
        state, data = _make_state_and_data()
        # Make player 99 very expensive
        data.players.loc[data.players["code"] == 99, "now_cost"] = 9999
        preds = {p.code: 3.0 for p in state.players}
        preds[12] = 1.0
        preds[99] = 10.0
        predictions = PlayerPredictions(predictions=preds)

        optimizer = GreedyOptimizer(enable_transfers=True, transfer_gain_threshold=0.0)
        actions = optimizer.optimize(predictions, state, data)
        transfers = [a for a in actions if isinstance(a, Transfer)]
        # Should not transfer in an unaffordable player
        assert all(t.player_in != 99 for t in transfers)
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/node/fpl-model && /home/node/.local/bin/uv run pytest tests/models/test_greedy_transfers.py -v`
Expected: FAIL — `TypeError: GreedyOptimizer.__init__() got an unexpected keyword argument 'enable_transfers'`

**Step 3: Extend GreedyOptimizer**

Modify `src/fpl_model/models/optimizers/greedy.py` to add `__init__` with `enable_transfers` and `transfer_gain_threshold` params, and a `_evaluate_transfers` method that runs before lineup selection. The transfer logic:

1. For each squad player, compute `worst_in_position = min predicted in that position among squad`
2. For each available player NOT in squad, if same position and affordable, compute `gain = pred_in - pred_worst`
3. Pick the (out, in) pair with highest gain
4. If gain > threshold (+ 4 if it's a paid transfer), make the transfer
5. Repeat up to `free_transfers` times (only free transfers for greedy)
6. Return the list of `Transfer` actions

The existing lineup/captain logic stays unchanged and runs after transfers.

Important details for the implementation:
- Check max 3 per team constraint
- Check budget: `state.budget + out_player.sell_price >= in_player.now_cost`
- Only swap same position (to maintain squad composition)

**Step 4: Run tests**

Run: `cd /home/node/fpl-model && /home/node/.local/bin/uv run pytest tests/models/test_greedy_transfers.py -v`
Expected: All pass

**Step 5: Full suite**

Run: `cd /home/node/fpl-model && /home/node/.local/bin/uv run pytest -x -q`
Expected: All pass (ensure existing greedy tests still pass since `enable_transfers` defaults to False)

**Step 6: Commit**

```bash
git add src/fpl_model/models/optimizers/greedy.py tests/models/test_greedy_transfers.py
git commit -m "feat: add transfer evaluation to greedy optimizer"
```

---

### Task 6: LP optimizer

**Files:**
- Create: `src/fpl_model/models/optimizers/lp_optimizer.py`
- Test: `tests/models/test_lp_optimizer.py`

**Step 1: Write failing tests**

```python
# tests/models/test_lp_optimizer.py
import pandas as pd
import pytest

from fpl_model.models.base import PlayerPredictions, SeasonData
from fpl_model.models.optimizers.lp_optimizer import LPOptimizer
from fpl_model.simulation.actions import (
    ChipType,
    PlayChip,
    SetCaptain,
    SetLineup,
    SetViceCaptain,
    Transfer,
)
from fpl_model.simulation.state import PlayerInSquad, SquadState


def _make_state_and_data():
    """Create a full 15-player squad and available players pool."""
    # Current squad: 2 GK, 5 DEF, 5 MID, 3 FWD
    squad_players = []
    code = 1
    for et, count in [(1, 2), (2, 5), (3, 5), (4, 3)]:
        for _ in range(count):
            squad_players.append(
                PlayerInSquad(code=code, element_type=et, buy_price=50, sell_price=50)
            )
            code += 1

    state = SquadState(
        players=squad_players,
        budget=100,  # 10.0m remaining
        free_transfers=1,
        chips_available={ct: 2 for ct in ChipType},
        current_gameweek=5,
        starting_xi=[p.code for p in squad_players[:11]],
        bench_order=[p.code for p in squad_players[11:]],
        captain=8,
        vice_captain=9,
    )

    # Available pool: squad players + 5 extras (one per position + extras)
    extra_players = [
        {"code": 100, "element_type": 1, "team_code": 16, "now_cost": 45},
        {"code": 101, "element_type": 2, "team_code": 17, "now_cost": 55},
        {"code": 102, "element_type": 3, "team_code": 18, "now_cost": 60},
        {"code": 103, "element_type": 4, "team_code": 19, "now_cost": 70},
        {"code": 104, "element_type": 3, "team_code": 20, "now_cost": 50},
    ]
    all_players_data = [
        {"code": p.code, "element_type": p.element_type, "team_code": (p.code % 15) + 1, "now_cost": 50}
        for p in squad_players
    ] + extra_players
    all_players = pd.DataFrame(all_players_data)

    data = SeasonData(
        gameweek_performances=pd.DataFrame(),
        fixtures=pd.DataFrame(),
        players=all_players,
        teams=pd.DataFrame(),
        current_gameweek=5,
        season="2024-25",
    )

    return state, data


class TestLPOptimizer:
    def test_returns_valid_lineup(self):
        state, data = _make_state_and_data()
        preds = {p.code: float(p.code) for p in state.players}
        # Also predict available players
        for code in [100, 101, 102, 103, 104]:
            preds[code] = 1.0  # low predictions, shouldn't trigger transfers
        predictions = PlayerPredictions(predictions=preds)

        optimizer = LPOptimizer()
        actions = optimizer.optimize(predictions, state, data)

        lineups = [a for a in actions if isinstance(a, SetLineup)]
        assert len(lineups) == 1
        assert len(lineups[0].starting_xi) == 11
        assert len(lineups[0].bench_order) == 4

    def test_sets_captain_and_vice(self):
        state, data = _make_state_and_data()
        preds = {p.code: float(p.code) for p in state.players}
        for code in [100, 101, 102, 103, 104]:
            preds[code] = 1.0
        predictions = PlayerPredictions(predictions=preds)

        optimizer = LPOptimizer()
        actions = optimizer.optimize(predictions, state, data)

        captains = [a for a in actions if isinstance(a, SetCaptain)]
        vices = [a for a in actions if isinstance(a, SetViceCaptain)]
        assert len(captains) == 1
        assert len(vices) == 1

    def test_makes_beneficial_transfer(self):
        state, data = _make_state_and_data()
        # Squad MID code=12 predicted at 0.0, available MID code=102 predicted at 15.0
        preds = {p.code: 5.0 for p in state.players}
        preds[12] = 0.0  # worst midfielder
        for code in [100, 101, 102, 103, 104]:
            preds[code] = 1.0
        preds[102] = 15.0  # great available midfielder
        predictions = PlayerPredictions(predictions=preds)

        optimizer = LPOptimizer()
        actions = optimizer.optimize(predictions, state, data)

        transfers = [a for a in actions if isinstance(a, Transfer)]
        assert len(transfers) >= 1
        # Should transfer in player 102
        transfer_in_codes = [t.player_in for t in transfers]
        assert 102 in transfer_in_codes

    def test_no_transfer_when_not_beneficial(self):
        state, data = _make_state_and_data()
        # All squad players predicted high, available players low
        preds = {p.code: 10.0 for p in state.players}
        for code in [100, 101, 102, 103, 104]:
            preds[code] = 1.0
        predictions = PlayerPredictions(predictions=preds)

        optimizer = LPOptimizer()
        actions = optimizer.optimize(predictions, state, data)
        transfers = [a for a in actions if isinstance(a, Transfer)]
        assert len(transfers) == 0
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/node/fpl-model && /home/node/.local/bin/uv run pytest tests/models/test_lp_optimizer.py -v`
Expected: FAIL

**Step 3: Implement LP optimizer**

Create `src/fpl_model/models/optimizers/lp_optimizer.py`. The optimizer:

1. Collects all candidate players (current squad + available from `data.players`)
2. Formulates the ILP with PuLP:
   - Binary variables for squad membership, starting XI, captain, transfers
   - Objective: maximize sum of predicted points for starters + captain bonus - transfer penalty
   - Constraints: squad size, formation, budget, max 3 per team, transfer linking
3. Solves and extracts the optimal squad, lineup, captain
4. Compares to current squad to produce Transfer actions
5. Builds SetLineup, SetCaptain, SetViceCaptain actions
6. Optionally evaluates chip heuristics

Key implementation details:
- Transfer cost is linearized: introduce binary `did_transfer[i]` and auxiliary variables to count paid transfers = max(0, total_transfers - free_transfers)
- For simplicity in v1, limit to evaluating up to `free_transfers` transfers (no paid hits) to keep the ILP tractable. Paid hits can be added later.
- Use `pulp.PULP_CBC_CMD(msg=0)` as the solver (bundled with PuLP, no external solver needed)

**Step 4: Run tests**

Run: `cd /home/node/fpl-model && /home/node/.local/bin/uv run pytest tests/models/test_lp_optimizer.py -v`
Expected: All pass

**Step 5: Full suite**

Run: `cd /home/node/fpl-model && /home/node/.local/bin/uv run pytest -x -q`
Expected: All pass

**Step 6: Commit**

```bash
git add src/fpl_model/models/optimizers/lp_optimizer.py tests/models/test_lp_optimizer.py
git commit -m "feat: add LP optimizer using PuLP for transfer and lineup optimization"
```

---

### Task 7: RL environment

**Files:**
- Create: `src/fpl_model/models/rl/__init__.py`
- Create: `src/fpl_model/models/rl/environment.py`
- Test: `tests/models/test_rl_environment.py`

**Step 1: Write failing tests**

```python
# tests/models/test_rl_environment.py
import pandas as pd
import pytest

from fpl_model.data.db import Database
from fpl_model.models.rl.environment import FPLEnvironment


def _setup_db(tmp_path, num_gws=5):
    """Create a test database with minimal season data."""
    db = Database(tmp_path / "test.db")
    db.create_tables()

    players_data = []
    for i in range(1, 16):
        et = 1 if i <= 2 else (2 if i <= 7 else (3 if i <= 12 else 4))
        players_data.append({
            "season": "2024-25",
            "code": i,
            "first_name": f"Player{i}",
            "second_name": f"Last{i}",
            "web_name": f"P{i}",
            "element_type": et,
            "team_code": (i % 15) + 1,
            "now_cost": 50,
        })
    db.write("players", pd.DataFrame(players_data))

    gw_rows = []
    for gw in range(1, num_gws + 1):
        for i in range(1, 16):
            gw_rows.append({
                "season": "2024-25",
                "player_code": i,
                "gameweek": gw,
                "total_points": 5,
                "minutes": 90,
                "goals_scored": 0,
                "assists": 0,
                "value": 50,
                "was_home": True,
                "opponent_team": 1,
                "expected_goals": 0.2,
                "expected_assists": 0.1,
                "bps": 20,
            })
    db.write("gameweek_performances", pd.DataFrame(gw_rows))

    fixture_rows = []
    for gw in range(1, num_gws + 1):
        fixture_rows.append({
            "season": "2024-25",
            "fixture_id": gw,
            "gameweek": gw,
            "team_h": 1,
            "team_a": 2,
            "team_h_score": 1,
            "team_a_score": 0,
            "finished": True,
        })
    db.write("fixtures", pd.DataFrame(fixture_rows))

    db.write("teams", pd.DataFrame([
        {"season": "2024-25", "team_code": i, "name": f"Team{i}", "short_name": f"T{i}"}
        for i in range(1, 21)
    ]))
    return db


class TestFPLEnvironment:
    def test_reset_returns_state(self, tmp_path):
        db = _setup_db(tmp_path)
        env = FPLEnvironment(season="2024-25", db=db)
        state = env.reset()
        assert state.shape[0] > 0  # non-empty state vector

    def test_step_returns_state_reward_done(self, tmp_path):
        db = _setup_db(tmp_path)
        env = FPLEnvironment(season="2024-25", db=db)
        env.reset()
        state, reward, done, info = env.step(env.null_action())
        assert state.shape[0] > 0
        assert isinstance(reward, float)
        assert isinstance(done, bool)

    def test_episode_terminates(self, tmp_path):
        db = _setup_db(tmp_path, num_gws=3)
        env = FPLEnvironment(season="2024-25", db=db)
        env.reset()
        done = False
        steps = 0
        while not done:
            _, _, done, _ = env.step(env.null_action())
            steps += 1
            if steps > 50:
                break
        assert done
        assert steps == 3

    def test_state_dimension(self, tmp_path):
        db = _setup_db(tmp_path)
        env = FPLEnvironment(season="2024-25", db=db)
        state = env.reset()
        assert env.state_dim == state.shape[0]
        assert env.state_dim > 0
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/node/fpl-model && /home/node/.local/bin/uv run pytest tests/models/test_rl_environment.py -v`
Expected: FAIL

**Step 3: Implement RL environment**

Create `src/fpl_model/models/rl/__init__.py` (empty) and `src/fpl_model/models/rl/environment.py`.

The environment:
- `__init__(season, db)`: loads all season data from DB
- `reset() -> np.ndarray`: creates initial squad (reusing `_initial_state` logic from engine), returns state vector
- `step(action_dict) -> (state, reward, done, info)`: applies actions for current GW, scores, advances
- `null_action()`: returns a no-op action dict (no transfers, keep lineup)
- `_encode_state() -> np.ndarray`: converts SquadState + SeasonData into fixed-size float vector

State vector structure:
- 15 player slots × features (element_type one-hot 4, normalized price, form_3/5, minutes_rolling, is_in_xi) ≈ 15 × 10 = 150
- Global: budget/1000, free_transfers/5, gw/38, 4 chip counts ≈ 7
- Total ≈ 157 dimensions

The environment internally uses the rules from `simulation/rules.py` (score_gameweek, apply_auto_subs, advance_gameweek, etc.) rather than the full SeasonSimulator, for direct control over the step loop.

**Step 4: Run tests**

Run: `cd /home/node/fpl-model && /home/node/.local/bin/uv run pytest tests/models/test_rl_environment.py -v`
Expected: All pass

**Step 5: Full suite**

Run: `cd /home/node/fpl-model && /home/node/.local/bin/uv run pytest -x -q`
Expected: All pass

**Step 6: Commit**

```bash
git add src/fpl_model/models/rl/__init__.py src/fpl_model/models/rl/environment.py tests/models/test_rl_environment.py
git commit -m "feat: add RL environment wrapping FPL season simulation"
```

---

### Task 8: PPO agent

**Files:**
- Create: `src/fpl_model/models/rl/ppo.py`
- Test: `tests/models/test_ppo.py`

**Step 1: Write failing tests**

```python
# tests/models/test_ppo.py
import pandas as pd
import pytest

from fpl_model.data.db import Database
from fpl_model.models.base import SeasonData
from fpl_model.models.rl.ppo import PPOAgent
from fpl_model.simulation.actions import ChipType, SetCaptain, SetLineup
from fpl_model.simulation.state import PlayerInSquad, SquadState


def _setup_db(tmp_path, num_gws=5):
    """Create a test database with minimal season data."""
    db = Database(tmp_path / "test.db")
    db.create_tables()

    players_data = []
    for i in range(1, 16):
        et = 1 if i <= 2 else (2 if i <= 7 else (3 if i <= 12 else 4))
        players_data.append({
            "season": "2024-25",
            "code": i,
            "first_name": f"Player{i}",
            "second_name": f"Last{i}",
            "web_name": f"P{i}",
            "element_type": et,
            "team_code": (i % 15) + 1,
            "now_cost": 50,
        })
    db.write("players", pd.DataFrame(players_data))

    gw_rows = []
    for gw in range(1, num_gws + 1):
        for i in range(1, 16):
            gw_rows.append({
                "season": "2024-25",
                "player_code": i,
                "gameweek": gw,
                "total_points": i,  # player code determines points
                "minutes": 90,
                "goals_scored": 0,
                "assists": 0,
                "value": 50,
                "was_home": True,
                "opponent_team": 1,
                "expected_goals": 0.2,
                "expected_assists": 0.1,
                "bps": 20,
            })
    db.write("gameweek_performances", pd.DataFrame(gw_rows))

    fixture_rows = []
    for gw in range(1, num_gws + 1):
        fixture_rows.append({
            "season": "2024-25",
            "fixture_id": gw,
            "gameweek": gw,
            "team_h": 1,
            "team_a": 2,
            "team_h_score": 1,
            "team_a_score": 0,
            "finished": True,
        })
    db.write("fixtures", pd.DataFrame(fixture_rows))

    db.write("teams", pd.DataFrame([
        {"season": "2024-25", "team_code": i, "name": f"Team{i}", "short_name": f"T{i}"}
        for i in range(1, 21)
    ]))
    return db


class TestPPOAgent:
    def test_recommend_returns_actions(self, tmp_path):
        """PPO agent can produce valid actions without training."""
        db = _setup_db(tmp_path)
        agent = PPOAgent(db=db, seasons=["2024-25"])

        players = []
        code = 1
        for et, count in [(1, 2), (2, 5), (3, 5), (4, 3)]:
            for _ in range(count):
                players.append(PlayerInSquad(code=code, element_type=et, buy_price=50, sell_price=50))
                code += 1

        state = SquadState(
            players=players,
            budget=100,
            free_transfers=1,
            chips_available={ct: 2 for ct in ChipType},
            current_gameweek=3,
            starting_xi=[p.code for p in players[:11]],
            bench_order=[p.code for p in players[11:]],
            captain=8,
            vice_captain=9,
        )

        gw_perf = db.read("gameweek_performances", where={"season": "2024-25"})
        data = SeasonData(
            gameweek_performances=gw_perf[gw_perf["gameweek"] <= 2],
            fixtures=db.read("fixtures", where={"season": "2024-25"}),
            players=db.read("players", where={"season": "2024-25"}),
            teams=db.read("teams", where={"season": "2024-25"}),
            current_gameweek=3,
            season="2024-25",
        )

        actions = agent.recommend(state, data)
        assert len(actions) > 0
        has_lineup = any(isinstance(a, SetLineup) for a in actions)
        has_captain = any(isinstance(a, SetCaptain) for a in actions)
        assert has_lineup
        assert has_captain

    def test_train_runs_without_error(self, tmp_path):
        """PPO training loop completes (even with minimal data)."""
        db = _setup_db(tmp_path, num_gws=5)
        agent = PPOAgent(
            db=db,
            seasons=["2024-25"],
            hidden_size=32,
            train_epochs=1,
            episodes_per_update=1,
        )
        # Should not raise
        from fpl_model.models.base import HistoricalData
        agent.train(HistoricalData())  # training uses internal env, not HistoricalData
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/node/fpl-model && /home/node/.local/bin/uv run pytest tests/models/test_ppo.py -v`
Expected: FAIL

**Step 3: Implement PPO agent**

Create `src/fpl_model/models/rl/ppo.py`. The PPO agent:

- Subclasses `ActionModel` directly
- Contains `PolicyNet` and `ValueNet` as inner `nn.Module` classes
- `__init__(db, seasons, hidden_size=128, ...)`: stores config, lazily creates networks on first use
- `recommend(state, data)`: encodes state, runs policy forward pass, decodes action heads into `Action` list
- `train(historical_data)`: runs PPO training loop using `FPLEnvironment`:
  1. For each episode: reset env, collect (state, action, reward, value) trajectory
  2. Compute GAE advantages
  3. PPO clipped surrogate update for K epochs
  4. Repeat for N update rounds

Action decoding from network outputs:
- **Lineup head** (15 logits): sigmoid → top 11 with formation mask → SetLineup
- **Captain head** (15 logits): softmax over starters → argmax → SetCaptain
- **Transfer head**: for v1, skip transfers (just lineup/captain). Transfer learning can be added later.
- **Chip head**: for v1, always output "no chip"

Key: the agent must handle the fact that `recommend()` is called by the simulator with a `SquadState` and `SeasonData`, so it needs to encode these into the state vector format the policy network expects. Reuse feature engineering from `features.py`.

**Step 4: Run tests**

Run: `cd /home/node/fpl-model && /home/node/.local/bin/uv run pytest tests/models/test_ppo.py -v`
Expected: All pass

**Step 5: Full suite**

Run: `cd /home/node/fpl-model && /home/node/.local/bin/uv run pytest -x -q`
Expected: All pass

**Step 6: Commit**

```bash
git add src/fpl_model/models/rl/ppo.py tests/models/test_ppo.py
git commit -m "feat: add PPO reinforcement learning agent for end-to-end FPL"
```

---

### Task 9: Register all models and update defaults

**Files:**
- Modify: `src/fpl_model/models/defaults.py`
- Test: `tests/models/test_defaults.py`

**Step 1: Write failing tests**

```python
# tests/models/test_defaults.py
from fpl_model.models.defaults import get_default_registry


class TestDefaultRegistry:
    def test_all_models_registered(self):
        registry = get_default_registry()
        names = registry.list()
        assert "form-greedy" in names
        assert "xgb-greedy" in names
        assert "xgb-lp" in names
        assert "sequence-lp" in names

    def test_models_are_action_models(self):
        from fpl_model.models.base import ActionModel
        registry = get_default_registry()
        for name in registry.list():
            model = registry.get(name)
            assert isinstance(model, ActionModel)
```

Note: `ppo-agent` is NOT registered in defaults because it requires a `db` and `seasons` parameter. It must be created explicitly.

**Step 2: Run tests to verify they fail**

Run: `cd /home/node/fpl-model && /home/node/.local/bin/uv run pytest tests/models/test_defaults.py -v`
Expected: FAIL — `"xgb-greedy" not in names`

**Step 3: Update defaults.py**

```python
"""Default model registry with built-in baseline models."""

from __future__ import annotations

from fpl_model.models.registry import ModelRegistry


def get_default_registry() -> ModelRegistry:
    """Create a registry pre-loaded with built-in models."""
    from fpl_model.models.base import PredictOptimizeModel
    from fpl_model.models.optimizers.greedy import GreedyOptimizer
    from fpl_model.models.optimizers.lp_optimizer import LPOptimizer
    from fpl_model.models.predictors.form import FormPredictor
    from fpl_model.models.predictors.sequence_predictor import SequencePredictor
    from fpl_model.models.predictors.xgboost_predictor import XGBoostPredictor

    registry = ModelRegistry()

    # Baseline: simple form average + lineup-only optimizer
    registry.register("form-greedy", PredictOptimizeModel(FormPredictor(), GreedyOptimizer()))

    # XGBoost + greedy with transfers
    registry.register(
        "xgb-greedy",
        PredictOptimizeModel(XGBoostPredictor(), GreedyOptimizer(enable_transfers=True)),
    )

    # XGBoost + LP optimizer
    registry.register("xgb-lp", PredictOptimizeModel(XGBoostPredictor(), LPOptimizer()))

    # LSTM sequence + LP optimizer
    registry.register("sequence-lp", PredictOptimizeModel(SequencePredictor(), LPOptimizer()))

    return registry
```

**Step 4: Run tests**

Run: `cd /home/node/fpl-model && /home/node/.local/bin/uv run pytest tests/models/test_defaults.py -v`
Expected: All pass

**Step 5: Also remove the duplicate tests from test_base.py**

The old `TestDefaultRegistry` class in `tests/models/test_base.py` tests `form-greedy` only. Update it to delegate to the new test file or remove the duplicate.

**Step 6: Full suite**

Run: `cd /home/node/fpl-model && /home/node/.local/bin/uv run pytest -x -q`
Expected: All pass

**Step 7: Commit**

```bash
git add src/fpl_model/models/defaults.py tests/models/test_defaults.py tests/models/test_base.py
git commit -m "feat: register all ML models in default registry"
```

---

### Task 10: Integration test — end-to-end backtest

**Files:**
- Create: `tests/test_ml_integration.py`

**Step 1: Write integration test**

```python
# tests/test_ml_integration.py
"""End-to-end integration tests for ML models running through SeasonSimulator."""

import pandas as pd
import pytest

from fpl_model.data.db import Database
from fpl_model.models.base import HistoricalData, PredictOptimizeModel, SeasonData
from fpl_model.models.defaults import get_default_registry
from fpl_model.models.optimizers.greedy import GreedyOptimizer
from fpl_model.models.predictors.xgboost_predictor import XGBoostPredictor
from fpl_model.simulation.engine import SeasonSimulator, SimulationResult


def _setup_db(tmp_path, num_gws=5):
    """Create a test database with enough data for ML models."""
    db = Database(tmp_path / "test.db")
    db.create_tables()

    players_data = []
    for i in range(1, 16):
        et = 1 if i <= 2 else (2 if i <= 7 else (3 if i <= 12 else 4))
        players_data.append({
            "season": "2024-25",
            "code": i,
            "first_name": f"Player{i}",
            "second_name": f"Last{i}",
            "web_name": f"P{i}",
            "element_type": et,
            "team_code": (i % 15) + 1,
            "now_cost": 50,
        })
    db.write("players", pd.DataFrame(players_data))

    gw_rows = []
    for gw in range(1, num_gws + 1):
        for i in range(1, 16):
            gw_rows.append({
                "season": "2024-25",
                "player_code": i,
                "gameweek": gw,
                "total_points": i + (gw % 3),
                "minutes": 90,
                "goals_scored": 0,
                "assists": 0,
                "value": 50,
                "was_home": gw % 2 == 0,
                "opponent_team": 1,
                "expected_goals": 0.2,
                "expected_assists": 0.1,
                "bps": 20,
                "fixture_id": gw * 10 + i,
            })
    db.write("gameweek_performances", pd.DataFrame(gw_rows))

    fixture_rows = []
    for gw in range(1, num_gws + 1):
        fixture_rows.append({
            "season": "2024-25",
            "fixture_id": gw,
            "gameweek": gw,
            "team_h": 1,
            "team_a": 2,
            "team_h_score": 1,
            "team_a_score": 0,
            "finished": 1,
        })
    db.write("fixtures", pd.DataFrame(fixture_rows))

    db.write("teams", pd.DataFrame([
        {"season": "2024-25", "team_code": i, "name": f"Team{i}", "short_name": f"T{i}"}
        for i in range(1, 21)
    ]))
    return db


class TestMLIntegration:
    def test_xgb_greedy_simulation(self, tmp_path):
        """XGBoost predictor + greedy optimizer runs through full simulation."""
        db = _setup_db(tmp_path, num_gws=5)

        # Train on the data
        predictor = XGBoostPredictor()
        gw_perf = db.read("gameweek_performances", where={"season": "2024-25"})
        players = db.read("players", where={"season": "2024-25"})
        fixtures = db.read("fixtures", where={"season": "2024-25"})
        teams = db.read("teams", where={"season": "2024-25"})

        hist = HistoricalData(seasons={
            "2024-25": SeasonData(
                gameweek_performances=gw_perf,
                fixtures=fixtures,
                players=players,
                teams=teams,
                current_gameweek=5,
                season="2024-25",
            )
        })
        predictor.train(hist)

        model = PredictOptimizeModel(predictor, GreedyOptimizer(enable_transfers=True))
        sim = SeasonSimulator(model=model, season="2024-25", db=db)
        result = sim.run()

        assert isinstance(result, SimulationResult)
        assert result.total_points > 0
        assert len(result.gameweek_points) == 5

    def test_form_greedy_baseline_simulation(self, tmp_path):
        """Form-greedy baseline runs through simulation (no training needed)."""
        db = _setup_db(tmp_path, num_gws=3)
        registry = get_default_registry()
        model = registry.get("form-greedy")
        sim = SeasonSimulator(model=model, season="2024-25", db=db)
        result = sim.run()
        assert result.total_points > 0
```

**Step 2: Run integration tests**

Run: `cd /home/node/fpl-model && /home/node/.local/bin/uv run pytest tests/test_ml_integration.py -v`
Expected: All pass

**Step 3: Full suite**

Run: `cd /home/node/fpl-model && /home/node/.local/bin/uv run pytest -x -q`
Expected: All pass

**Step 4: Commit**

```bash
git add tests/test_ml_integration.py
git commit -m "test: add end-to-end ML integration tests"
```

---

### Summary of tasks

| Task | Component | Dependencies |
|------|-----------|-------------|
| 1 | Update pyproject.toml dependencies | None |
| 2 | Shared feature engineering module | Task 1 |
| 3 | XGBoost predictor | Tasks 1, 2 |
| 4 | Sequence (LSTM) predictor | Tasks 1, 2 |
| 5 | Extend greedy optimizer with transfers | Task 1 |
| 6 | LP optimizer | Task 1 |
| 7 | RL environment | Tasks 1, 2 |
| 8 | PPO agent | Tasks 1, 2, 7 |
| 9 | Register models + update defaults | Tasks 3, 4, 5, 6 |
| 10 | Integration tests | Tasks 3, 5, 9 |

Tasks 3, 4, 5, 6 can be parallelized after Task 2 completes.
Tasks 7, 8 can be parallelized with Tasks 3-6 (independent RL track).
