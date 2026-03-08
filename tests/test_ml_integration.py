"""End-to-end integration tests for ML models running through SeasonSimulator."""

import pandas as pd

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
