import sqlite3

import pandas as pd
import pytest

from fpl_model.data.db import Database
from fpl_model.data.etl.schemas import TABLES


class TestDatabase:
    def test_create_tables(self, tmp_path):
        db = Database(tmp_path / "test.db")
        db.create_tables()
        conn = sqlite3.connect(tmp_path / "test.db")
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        assert "players" in tables
        assert "gameweek_performances" in tables
        assert "fixtures" in tables
        assert "teams" in tables
        assert "gameweeks" in tables
        conn.close()

    def test_write_and_read_players(self, tmp_path):
        db = Database(tmp_path / "test.db")
        db.create_tables()
        df = pd.DataFrame({
            "season": ["2024-25"],
            "code": [1],
            "first_name": ["Mohamed"],
            "second_name": ["Salah"],
            "web_name": ["Salah"],
            "element_type": [3],
            "team_code": [14],
            "now_cost": [130],
        })
        db.write("players", df)
        result = db.read("players")
        assert len(result) == 1
        assert result.iloc[0]["web_name"] == "Salah"

    def test_write_and_read_gameweek_performances(self, tmp_path):
        db = Database(tmp_path / "test.db")
        db.create_tables()
        df = pd.DataFrame({
            "season": ["2024-25"],
            "player_code": [1],
            "gameweek": [1],
            "fixture_id": [1],
            "total_points": [10],
            "minutes": [90],
            "goals_scored": [1],
            "assists": [1],
            "clean_sheets": [0],
            "bonus": [3],
            "bps": [45],
            "value": [130],
            "was_home": [True],
            "opponent_team": [2],
        })
        db.write("gameweek_performances", df)
        result = db.read("gameweek_performances")
        assert len(result) == 1
        assert result.iloc[0]["total_points"] == 10

    def test_read_with_query(self, tmp_path):
        db = Database(tmp_path / "test.db")
        db.create_tables()
        df = pd.DataFrame({
            "season": ["2024-25", "2024-25"],
            "code": [1, 2],
            "first_name": ["Mohamed", "Erling"],
            "second_name": ["Salah", "Haaland"],
            "web_name": ["Salah", "Haaland"],
            "element_type": [3, 4],
            "team_code": [14, 131],
            "now_cost": [130, 140],
        })
        db.write("players", df)
        result = db.read("players", where="element_type = 4")
        assert len(result) == 1
        assert result.iloc[0]["web_name"] == "Haaland"


class TestSchemas:
    def test_tables_dict_has_required_keys(self):
        assert "players" in TABLES
        assert "gameweek_performances" in TABLES
        assert "fixtures" in TABLES
        assert "teams" in TABLES
        assert "gameweeks" in TABLES

    def test_players_schema_has_key_columns(self):
        cols = TABLES["players"]
        assert "season" in cols
        assert "code" in cols
        assert "web_name" in cols
        assert "element_type" in cols
        assert "now_cost" in cols

    def test_gameweek_performances_schema_has_key_columns(self):
        cols = TABLES["gameweek_performances"]
        assert "season" in cols
        assert "player_code" in cols
        assert "gameweek" in cols
        assert "total_points" in cols
        assert "minutes" in cols
