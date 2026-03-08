"""Transform raw source data into canonical schema format."""

import pandas as pd

from fpl_model.data.etl.unifier import unify_to_schema


class VaastavTransformer:
    """Transform CSVs from the vaastav/Fantasy-Premier-League GitHub repo."""

    def __init__(self, season: str, id_to_code: dict[int, int] | None = None):
        self.season = season
        self.id_to_code = id_to_code or {}

    def transform_players(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.copy()
        df["season"] = self.season
        return unify_to_schema(df, "players")

    def transform_gameweek_performances(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.copy()
        df["season"] = self.season
        if "element" in df.columns:
            df["player_code"] = df["element"].map(self.id_to_code)
        rename_map = {}
        if "round" in df.columns:
            rename_map["round"] = "gameweek"
        if "fixture" in df.columns:
            rename_map["fixture"] = "fixture_id"
        if rename_map:
            df = df.rename(columns=rename_map)
        return unify_to_schema(df, "gameweek_performances")

    def transform_fixtures(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.copy()
        df["season"] = self.season
        rename_map = {}
        if "id" in df.columns:
            rename_map["id"] = "fixture_id"
        if "event" in df.columns:
            rename_map["event"] = "gameweek"
        if rename_map:
            df = df.rename(columns=rename_map)
        return unify_to_schema(df, "fixtures")

    def transform_teams(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.copy()
        df["season"] = self.season
        if "code" in df.columns and "team_code" not in df.columns:
            df = df.rename(columns={"code": "team_code"})
        return unify_to_schema(df, "teams")


class FPLApiTransformer:
    """Transform data fetched from the live FPL API."""

    def __init__(self, season: str):
        self.season = season

    def transform_players(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.copy()
        df["season"] = self.season
        return unify_to_schema(df, "players")

    def transform_gameweek_performances(
        self, raw: pd.DataFrame, id_to_code: dict[int, int]
    ) -> pd.DataFrame:
        df = raw.copy()
        df["season"] = self.season
        if "element" in df.columns:
            df["player_code"] = df["element"].map(id_to_code)
        rename_map = {}
        if "round" in df.columns:
            rename_map["round"] = "gameweek"
        if "fixture" in df.columns:
            rename_map["fixture"] = "fixture_id"
        if rename_map:
            df = df.rename(columns=rename_map)
        return unify_to_schema(df, "gameweek_performances")

    def transform_fixtures(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.copy()
        df["season"] = self.season
        rename_map = {}
        if "id" in df.columns:
            rename_map["id"] = "fixture_id"
        if "event" in df.columns:
            rename_map["event"] = "gameweek"
        if rename_map:
            df = df.rename(columns=rename_map)
        return unify_to_schema(df, "fixtures")

    def transform_teams(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.copy()
        df["season"] = self.season
        if "code" in df.columns and "team_code" not in df.columns:
            df = df.rename(columns={"code": "team_code"})
        return unify_to_schema(df, "teams")
