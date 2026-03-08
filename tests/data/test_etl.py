# tests/data/test_etl.py
import pandas as pd
import pytest
from fpl_model.data.etl.schemas import TABLES
from fpl_model.data.etl.transformers import VaastavTransformer, FPLApiTransformer
from fpl_model.data.etl.unifier import unify_to_schema


class TestUnifyToSchema:
    def test_adds_missing_columns_as_null(self):
        df = pd.DataFrame({"season": ["2024-25"], "code": [1], "web_name": ["Salah"]})
        result = unify_to_schema(df, "players")
        assert "element_type" in result.columns
        assert pd.isna(result.iloc[0]["element_type"])

    def test_drops_extra_columns(self):
        df = pd.DataFrame({
            "season": ["2024-25"],
            "code": [1],
            "web_name": ["Salah"],
            "extra_column": ["drop_me"],
        })
        result = unify_to_schema(df, "players")
        assert "extra_column" not in result.columns

    def test_preserves_existing_values(self):
        df = pd.DataFrame({
            "season": ["2024-25"],
            "code": [1],
            "web_name": ["Salah"],
            "element_type": [3],
            "now_cost": [130],
        })
        result = unify_to_schema(df, "players")
        assert result.iloc[0]["element_type"] == 3
        assert result.iloc[0]["now_cost"] == 130


class TestVaastavTransformer:
    def test_transform_players(self):
        raw = pd.DataFrame({
            "code": [118748],
            "first_name": ["Mohamed"],
            "second_name": ["Salah"],
            "web_name": ["Salah"],
            "element_type": [3],
            "team_code": [14],
            "now_cost": [130],
        })
        transformer = VaastavTransformer(season="2024-25")
        result = transformer.transform_players(raw)
        assert result.iloc[0]["season"] == "2024-25"
        assert "code" in result.columns

    def test_transform_gameweek_performances_maps_element_to_player_code(self):
        raw_gw = pd.DataFrame({
            "element": [1],
            "round": [1],
            "fixture": [1],
            "total_points": [10],
            "minutes": [90],
            "goals_scored": [1],
            "assists": [0],
            "value": [130],
            "was_home": [True],
            "opponent_team": [2],
        })
        id_to_code = {1: 118748}
        transformer = VaastavTransformer(season="2024-25", id_to_code=id_to_code)
        result = transformer.transform_gameweek_performances(raw_gw)
        assert result.iloc[0]["player_code"] == 118748
        assert result.iloc[0]["gameweek"] == 1

    def test_transform_fixtures(self):
        raw = pd.DataFrame({
            "id": [1],
            "event": [1],
            "kickoff_time": ["2024-08-17T14:00:00Z"],
            "team_h": [14],
            "team_a": [131],
            "team_h_score": [2],
            "team_a_score": [0],
            "team_h_difficulty": [2],
            "team_a_difficulty": [4],
            "finished": [True],
            "started": [True],
        })
        transformer = VaastavTransformer(season="2024-25")
        result = transformer.transform_fixtures(raw)
        assert result.iloc[0]["season"] == "2024-25"
        assert result.iloc[0]["fixture_id"] == 1
