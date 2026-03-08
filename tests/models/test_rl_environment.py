"""Tests for RL environment wrapping FPL season simulation."""

import numpy as np
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

    def test_state_vector_dtype_is_float(self, tmp_path):
        db = _setup_db(tmp_path)
        env = FPLEnvironment(season="2024-25", db=db)
        state = env.reset()
        assert state.dtype == np.float32

    def test_state_dim_is_consistent_across_steps(self, tmp_path):
        db = _setup_db(tmp_path)
        env = FPLEnvironment(season="2024-25", db=db)
        state0 = env.reset()
        state1, _, _, _ = env.step(env.null_action())
        assert state0.shape == state1.shape

    def test_null_action_is_dict(self, tmp_path):
        db = _setup_db(tmp_path)
        env = FPLEnvironment(season="2024-25", db=db)
        env.reset()
        action = env.null_action()
        assert isinstance(action, dict)

    def test_reward_is_nonnegative_with_playing_squad(self, tmp_path):
        """With all players playing 90 min and scoring 5 pts, reward should be positive."""
        db = _setup_db(tmp_path)
        env = FPLEnvironment(season="2024-25", db=db)
        env.reset()
        _, reward, _, _ = env.step(env.null_action())
        # 11 starters * 5 pts + captain bonus (5 extra) = 60
        assert reward > 0

    def test_info_contains_gw_and_points(self, tmp_path):
        db = _setup_db(tmp_path)
        env = FPLEnvironment(season="2024-25", db=db)
        env.reset()
        _, _, _, info = env.step(env.null_action())
        assert "gw" in info
        assert "points" in info

    def test_initial_squad_has_correct_positions(self, tmp_path):
        db = _setup_db(tmp_path)
        env = FPLEnvironment(season="2024-25", db=db)
        env.reset()
        state = env._state
        # Check position counts
        types = [p.element_type for p in state.players]
        assert types.count(1) == 2  # GK
        assert types.count(2) == 5  # DEF
        assert types.count(3) == 5  # MID
        assert types.count(4) == 3  # FWD
