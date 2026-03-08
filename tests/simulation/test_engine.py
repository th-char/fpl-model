"""Tests for the season simulation engine."""

import pandas as pd

from fpl_model.data.db import Database
from fpl_model.models.base import ActionModel, SeasonData
from fpl_model.simulation.actions import SetCaptain, SetLineup, SetViceCaptain
from fpl_model.simulation.engine import SeasonSimulator, SimulationResult
from fpl_model.simulation.state import SquadState


class DummyModel(ActionModel):
    """Always picks same lineup, no transfers."""

    def recommend(self, state: SquadState, data: SeasonData) -> list:
        xi = [p.code for p in state.players[:11]]
        bench = [p.code for p in state.players[11:]]
        return [
            SetLineup(starting_xi=xi, bench_order=bench),
            SetCaptain(player_id=state.players[7].code),
            SetViceCaptain(player_id=state.players[8].code),
        ]


class TestSeasonSimulator:
    def _setup_db(self, tmp_path, num_gws=2):
        db = Database(tmp_path / "test.db")
        db.create_tables()

        # 15 players: 2 GK (codes 1-2), 5 DEF (3-7), 5 MID (8-12), 3 FWD (13-15)
        players_data = []
        for i in range(1, 16):
            et = 1 if i <= 2 else (2 if i <= 7 else (3 if i <= 12 else 4))
            players_data.append(
                {
                    "season": "2024-25",
                    "code": i,
                    "first_name": f"Player{i}",
                    "second_name": f"Last{i}",
                    "web_name": f"P{i}",
                    "element_type": et,
                    "team_code": (i % 20) + 1,
                    "now_cost": 50,
                }
            )
        db.write("players", pd.DataFrame(players_data))

        # GW performances: all players score 5 points, play 90 mins
        gw_rows = []
        for gw in range(1, num_gws + 1):
            for i in range(1, 16):
                gw_rows.append(
                    {
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
                    }
                )
        db.write("gameweek_performances", pd.DataFrame(gw_rows))

        fixture_rows = []
        for gw in range(1, num_gws + 1):
            fixture_rows.append(
                {
                    "season": "2024-25",
                    "fixture_id": gw,
                    "gameweek": gw,
                    "team_h": 1,
                    "team_a": 2,
                    "team_h_score": 1,
                    "team_a_score": 0,
                    "finished": True,
                }
            )
        db.write("fixtures", pd.DataFrame(fixture_rows))

        db.write(
            "teams",
            pd.DataFrame(
                [
                    {
                        "season": "2024-25",
                        "team_code": i,
                        "name": f"Team{i}",
                        "short_name": f"T{i}",
                    }
                    for i in range(1, 21)
                ]
            ),
        )
        return db

    def test_simulation_runs_and_returns_result(self, tmp_path):
        db = self._setup_db(tmp_path, num_gws=2)
        model = DummyModel()
        sim = SeasonSimulator(model=model, season="2024-25", db=db)
        result = sim.run()
        assert isinstance(result, SimulationResult)
        assert result.total_points > 0
        assert len(result.gameweek_points) == 2

    def test_captain_gets_double_points(self, tmp_path):
        db = self._setup_db(tmp_path, num_gws=1)
        model = DummyModel()
        sim = SeasonSimulator(model=model, season="2024-25", db=db)
        result = sim.run()
        # 11 starters * 5 pts = 55, captain (player 8) gets double so +5 = 60
        assert result.gameweek_points[1] == 60
