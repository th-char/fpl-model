# tests/data/test_fpl_api.py
from fpl_model.data.sources.base import DataSource
from fpl_model.data.sources.fpl_api import FPLApiSource


class TestFPLApiSource:
    def test_is_data_source(self):
        source = FPLApiSource.__new__(FPLApiSource)
        assert isinstance(source, DataSource)

    def test_parse_elements_to_players(self):
        source = FPLApiSource.__new__(FPLApiSource)
        bootstrap = {
            "elements": [
                {
                    "code": 118748,
                    "first_name": "Mohamed",
                    "second_name": "Salah",
                    "web_name": "Salah",
                    "element_type": 3,
                    "team_code": 14,
                    "now_cost": 130,
                    "total_points": 200,
                    "minutes": 2700,
                    "goals_scored": 15,
                    "assists": 12,
                }
            ]
        }
        df = source._parse_players(bootstrap)
        assert len(df) == 1
        assert df.iloc[0]["code"] == 118748

    def test_parse_teams(self):
        source = FPLApiSource.__new__(FPLApiSource)
        bootstrap = {
            "teams": [
                {
                    "code": 14,
                    "name": "Liverpool",
                    "short_name": "LIV",
                    "strength": 5,
                    "strength_overall_home": 1300,
                    "strength_overall_away": 1300,
                    "strength_attack_home": 1300,
                    "strength_attack_away": 1300,
                    "strength_defence_home": 1300,
                    "strength_defence_away": 1300,
                }
            ]
        }
        df = source._parse_teams(bootstrap)
        assert len(df) == 1
        assert df.iloc[0]["name"] == "Liverpool"

    def test_parse_events_to_gameweeks(self):
        source = FPLApiSource.__new__(FPLApiSource)
        bootstrap = {
            "events": [
                {
                    "id": 1,
                    "deadline_time": "2025-08-16T10:00:00Z",
                    "finished": True,
                    "average_entry_score": 55,
                    "highest_score": 120,
                    "most_selected": 1,
                    "most_captained": 1,
                    "most_transferred_in": 1,
                }
            ]
        }
        df = source._parse_gameweeks(bootstrap)
        assert len(df) == 1
        assert df.iloc[0]["gameweek"] == 1
