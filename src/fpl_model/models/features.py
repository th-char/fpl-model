"""Shared feature engineering for FPL ML models.

Used by XGBoost predictor, sequence predictor, and RL state encoder.
All features are computed from SeasonData without future data leakage —
only data from gameweeks strictly before the target GW is used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fpl_model.models.base import SeasonData

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
    _xstat_name_map = {
        "expected_goals": "xg",
        "expected_assists": "xa",
        "expected_goal_involvements": "xgi",
        "expected_goals_conceded": "xgc",
    }
    for col, short_name in _xstat_name_map.items():
        tail = past.tail(5)
        if len(tail) > 0 and col in tail.columns:
            features[f"{short_name}_rolling"] = float(tail[col].mean())
        else:
            features[f"{short_name}_rolling"] = 0.0

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
