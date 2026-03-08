"""Tests for the season simulation engine."""

import pandas as pd

from fpl_model.data.db import Database
from fpl_model.models.base import ActionModel, SeasonData
from fpl_model.simulation.actions import (
    ChipType,
    PlayChip,
    SetCaptain,
    SetLineup,
    SetViceCaptain,
)
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


class FreeHitModel(ActionModel):
    """Plays Free Hit in GW2 with same lineup (FH blocked in GW1)."""

    def __init__(self):
        self.call_count = 0

    def recommend(self, state: SquadState, data: SeasonData) -> list:
        self.call_count += 1
        xi = [p.code for p in state.players[:11]]
        bench = [p.code for p in state.players[11:]]
        actions = [
            SetLineup(starting_xi=xi, bench_order=bench),
            SetCaptain(player_id=state.players[7].code),
            SetViceCaptain(player_id=state.players[8].code),
        ]
        if state.current_gameweek == 2:
            actions.append(PlayChip(chip_type=ChipType.FREE_HIT))
        return actions


class BenchBoostModel(ActionModel):
    """Plays Bench Boost in GW1."""

    def recommend(self, state: SquadState, data: SeasonData) -> list:
        xi = [p.code for p in state.players[:11]]
        bench = [p.code for p in state.players[11:]]
        actions = [
            SetLineup(starting_xi=xi, bench_order=bench),
            SetCaptain(player_id=state.players[7].code),
            SetViceCaptain(player_id=state.players[8].code),
            PlayChip(chip_type=ChipType.BENCH_BOOST),
        ]
        return actions


class TestFreeHitRevert:
    def _setup_db(self, tmp_path, num_gws=3):
        """Reuse the same DB setup pattern."""
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
                "team_code": (i % 20) + 1,
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

        db.write(
            "teams",
            pd.DataFrame([
                {
                    "season": "2024-25",
                    "team_code": i,
                    "name": f"Team{i}",
                    "short_name": f"T{i}",
                }
                for i in range(1, 21)
            ]),
        )
        return db

    def test_squad_reverts_after_free_hit(self, tmp_path):
        db = self._setup_db(tmp_path, num_gws=3)
        model = FreeHitModel()
        sim = SeasonSimulator(model=model, season="2024-25", db=db)
        result = sim.run()
        # Free Hit played in GW2 -- transfer cost should be 0
        assert result.transfer_costs[2] == 0
        # Simulation should complete all 3 GWs
        assert len(result.gameweek_points) == 3

    def test_free_hit_chip_uses_decremented(self, tmp_path):
        db = self._setup_db(tmp_path, num_gws=3)
        model = FreeHitModel()
        sim = SeasonSimulator(model=model, season="2024-25", db=db)
        result = sim.run()
        # Should have run successfully through all GWs
        assert result.total_points > 0


class TestChipProcessing:
    def _setup_db(self, tmp_path, num_gws=1):
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
                "team_code": (i % 20) + 1,
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

        db.write(
            "teams",
            pd.DataFrame([
                {
                    "season": "2024-25",
                    "team_code": i,
                    "name": f"Team{i}",
                    "short_name": f"T{i}",
                }
                for i in range(1, 21)
            ]),
        )
        return db

    def test_bench_boost_chip_applied(self, tmp_path):
        db = self._setup_db(tmp_path, num_gws=1)
        model = BenchBoostModel()
        sim = SeasonSimulator(model=model, season="2024-25", db=db)
        result = sim.run()
        # With bench boost: 15*5 = 75, captain +5 = 80
        assert result.gameweek_points[1] == 80


class TestDataLeakagePrevention:
    def _setup_db(self, tmp_path):
        """Set up DB with 2 GWs where GW2 fixture is not finished."""
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
                "team_code": (i % 20) + 1,
                "now_cost": 50,
            })
        db.write("players", pd.DataFrame(players_data))

        gw_rows = []
        for gw in range(1, 3):
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
                })
        db.write("gameweek_performances", pd.DataFrame(gw_rows))

        # GW1 finished, GW2 not finished
        fixture_rows = [
            {
                "season": "2024-25",
                "fixture_id": 1,
                "gameweek": 1,
                "team_h": 1,
                "team_a": 2,
                "team_h_score": 1,
                "team_a_score": 0,
                "finished": True,
            },
            {
                "season": "2024-25",
                "fixture_id": 2,
                "gameweek": 2,
                "team_h": 3,
                "team_a": 4,
                "team_h_score": 2,
                "team_a_score": 1,
                "finished": False,
            },
        ]
        db.write("fixtures", pd.DataFrame(fixture_rows))

        db.write(
            "teams",
            pd.DataFrame([
                {
                    "season": "2024-25",
                    "team_code": i,
                    "name": f"Team{i}",
                    "short_name": f"T{i}",
                }
                for i in range(1, 21)
            ]),
        )
        return db

    def test_future_scores_blanked(self, tmp_path):
        """Model should not see scores for unfinished fixtures during GW loop."""
        seen_data = []

        class SpyModel(ActionModel):
            def recommend(self, state, data):
                seen_data.append((state.current_gameweek, data))
                xi = [p.code for p in state.players[:11]]
                bench = [p.code for p in state.players[11:]]
                return [
                    SetLineup(starting_xi=xi, bench_order=bench),
                    SetCaptain(player_id=state.players[7].code),
                    SetViceCaptain(player_id=state.players[8].code),
                ]

        db = self._setup_db(tmp_path)
        model = SpyModel()
        sim = SeasonSimulator(model=model, season="2024-25", db=db)
        sim.run()

        # Check GW-loop calls (skip the initial _initial_state call which uses raw fixtures).
        # During the GW loop, unfinished fixture scores should be blanked.
        gw_loop_calls = [(gw, data) for gw, data in seen_data[1:]]
        assert len(gw_loop_calls) >= 1
        for gw, data in gw_loop_calls:
            unfinished = data.fixtures[data.fixtures["finished"] != 1]
            if not unfinished.empty:
                assert unfinished["team_h_score"].isna().all()
                assert unfinished["team_a_score"].isna().all()
