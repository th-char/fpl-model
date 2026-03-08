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
