# tests/data/test_vaastav.py
from fpl_model.data.sources.base import DataSource
from fpl_model.data.sources.vaastav import VaastavSource


class TestDataSourceInterface:
    def test_vaastav_is_data_source(self):
        source = VaastavSource.__new__(VaastavSource)
        assert isinstance(source, DataSource)


class TestVaastavSource:
    def test_build_url_players_raw(self):
        source = VaastavSource.__new__(VaastavSource)
        url = source._build_url("2024-25", "players_raw.csv")
        assert "vaastav/Fantasy-Premier-League" in url
        assert "2024-25" in url
        assert "players_raw.csv" in url

    def test_build_url_merged_gw(self):
        source = VaastavSource.__new__(VaastavSource)
        url = source._build_url("2024-25", "gws/merged_gw.csv")
        assert "gws/merged_gw.csv" in url

    def test_parse_players_raw(self):
        source = VaastavSource.__new__(VaastavSource)
        csv_content = (
            "code,first_name,second_name,web_name,element_type,team_code,now_cost,"
            "total_points,minutes,goals_scored,assists\n"
            "118748,Mohamed,Salah,Salah,3,14,130,200,2700,15,12\n"
        )
        df = source._parse_csv(csv_content)
        assert len(df) == 1
        assert df.iloc[0]["code"] == 118748
        assert df.iloc[0]["web_name"] == "Salah"
