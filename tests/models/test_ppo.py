"""Tests for PPO reinforcement learning agent."""

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

    def test_train_updates_policy_weights(self, tmp_path):
        """Training actually updates the policy network weights."""
        import torch

        db = _setup_db(tmp_path, num_gws=5)
        agent = PPOAgent(
            db=db,
            seasons=["2024-25"],
            hidden_size=32,
            train_epochs=2,
            episodes_per_update=1,
        )

        # Snapshot weights before training
        before = {
            name: param.clone()
            for name, param in agent._policy.named_parameters()
        }

        from fpl_model.models.base import HistoricalData
        agent.train(HistoricalData())

        # At least some weights should have changed
        changed = any(
            not torch.equal(before[name], param)
            for name, param in agent._policy.named_parameters()
        )
        assert changed, "Policy weights should change after training"

    def test_create_ppo_agent_factory(self, tmp_path):
        """Factory function creates a valid PPOAgent."""
        db = _setup_db(tmp_path)
        from fpl_model.models.defaults import create_ppo_agent

        agent = create_ppo_agent(db=db, seasons=["2024-25"], hidden_size=32)
        assert isinstance(agent, PPOAgent)
