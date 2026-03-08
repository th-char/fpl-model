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
