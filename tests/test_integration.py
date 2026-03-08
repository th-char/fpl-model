"""End-to-end integration test: ingest mock data, run simulation, evaluate."""

import pandas as pd
import pytest

from fpl_model.data.db import Database
from fpl_model.models.base import ActionModel, SeasonData
from fpl_model.simulation.actions import SetCaptain, SetLineup, SetViceCaptain
from fpl_model.simulation.state import SquadState
from fpl_model.simulation.engine import SeasonSimulator
from fpl_model.evaluation.metrics import compute_metrics
from fpl_model.evaluation.comparison import compare_results


class AlwaysFirstModel(ActionModel):
    """Picks the first valid squad, captains a midfielder."""

    def recommend(self, state, data):
        xi = [p.code for p in state.players[:11]]
        bench = [p.code for p in state.players[11:]]
        return [
            SetLineup(starting_xi=xi, bench_order=bench),
            SetCaptain(player_id=state.players[7].code),
            SetViceCaptain(player_id=state.players[8].code),
        ]


class TestEndToEnd:
    def _populate_db(self, db: Database, num_gws: int = 5):
        players = []
        for i in range(1, 16):
            et = 1 if i <= 2 else (2 if i <= 7 else (3 if i <= 12 else 4))
            players.append(
                {
                    "season": "2023-24",
                    "code": i,
                    "first_name": f"F{i}",
                    "second_name": f"L{i}",
                    "web_name": f"P{i}",
                    "element_type": et,
                    "team_code": (i % 20) + 1,
                    "now_cost": 50,
                }
            )
        db.write("players", pd.DataFrame(players))

        gw_rows = []
        for gw in range(1, num_gws + 1):
            for i in range(1, 16):
                gw_rows.append(
                    {
                        "season": "2023-24",
                        "player_code": i,
                        "gameweek": gw,
                        "total_points": i,
                        "minutes": 90,
                        "goals_scored": 0,
                        "assists": 0,
                        "value": 50,
                        "was_home": 1,
                        "opponent_team": 1,
                    }
                )
        db.write("gameweek_performances", pd.DataFrame(gw_rows))

        fixtures = [
            {
                "season": "2023-24",
                "fixture_id": gw,
                "gameweek": gw,
                "team_h": 1,
                "team_a": 2,
                "finished": 1,
            }
            for gw in range(1, num_gws + 1)
        ]
        db.write("fixtures", pd.DataFrame(fixtures))

        teams = [
            {"season": "2023-24", "team_code": i, "name": f"Team{i}"}
            for i in range(1, 21)
        ]
        db.write("teams", pd.DataFrame(teams))

    def test_full_pipeline(self, tmp_path):
        db = Database(tmp_path / "test.db")
        db.create_tables()
        self._populate_db(db, num_gws=5)

        model = AlwaysFirstModel()
        sim = SeasonSimulator(model=model, season="2023-24", db=db)
        result = sim.run()

        metrics = compute_metrics(result)
        assert metrics["total_points"] > 0
        assert metrics["num_gameweeks"] == 5

    def test_compare_two_models(self, tmp_path):
        db = Database(tmp_path / "test.db")
        db.create_tables()
        self._populate_db(db, num_gws=3)

        model_a = AlwaysFirstModel()
        model_b = AlwaysFirstModel()

        sim_a = SeasonSimulator(model=model_a, season="2023-24", db=db)
        sim_b = SeasonSimulator(model=model_b, season="2023-24", db=db)

        result_a = sim_a.run()
        result_b = sim_b.run()

        comparison = compare_results({"model_a": result_a, "model_b": result_b})
        assert len(comparison) == 2
        assert comparison[0]["total_points"] == comparison[1]["total_points"]
