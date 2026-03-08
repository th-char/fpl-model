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
