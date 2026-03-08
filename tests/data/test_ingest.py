# tests/data/test_ingest.py
from unittest.mock import AsyncMock, MagicMock, patch
import pandas as pd
import pytest
from fpl_model.data.ingest import Ingester


class TestIngester:
    def test_init(self, tmp_path):
        ingester = Ingester(db_path=tmp_path / "test.db", cache_dir=tmp_path / "cache")
        assert ingester.db is not None
        assert ingester.cache is not None

    @pytest.mark.asyncio
    async def test_ingest_vaastav_season(self, tmp_path):
        ingester = Ingester(db_path=tmp_path / "test.db", cache_dir=tmp_path / "cache")

        mock_players = pd.DataFrame({
            "id": [1],
            "code": [118748],
            "first_name": ["Mohamed"],
            "second_name": ["Salah"],
            "web_name": ["Salah"],
            "element_type": [3],
            "team_code": [14],
            "now_cost": [130],
        })
        mock_gw = pd.DataFrame({
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
        mock_fixtures = pd.DataFrame({
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
        mock_teams = pd.DataFrame({
            "code": [14],
            "name": ["Liverpool"],
            "short_name": ["LIV"],
            "strength": [5],
        })

        mock_source = AsyncMock()
        mock_source.fetch_players = AsyncMock(return_value=mock_players)
        mock_source.fetch_gameweek_performances = AsyncMock(return_value=mock_gw)
        mock_source.fetch_fixtures = AsyncMock(return_value=mock_fixtures)
        mock_source.fetch_teams = AsyncMock(return_value=mock_teams)

        await ingester.ingest_season("2024-25", source=mock_source)

        players = ingester.db.read("players")
        assert len(players) == 1
        assert players.iloc[0]["web_name"] == "Salah"

        gw_perf = ingester.db.read("gameweek_performances")
        assert len(gw_perf) == 1
        assert gw_perf.iloc[0]["player_code"] == 118748
