"""Transform raw source data into canonical schema format."""

import pandas as pd

from fpl_model.data.etl.unifier import unify_to_schema


class BaseTransformer:
    """Shared transformation logic for all data sources."""

    def __init__(self, season: str):
        self.season = season

    def _add_season(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["season"] = self.season
        return df

    def transform_fixtures(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = self._add_season(raw)
        rename_map = {}
        if "id" in df.columns:
            rename_map["id"] = "fixture_id"
        if "event" in df.columns:
            rename_map["event"] = "gameweek"
        if rename_map:
            df = df.rename(columns=rename_map)
        return unify_to_schema(df, "fixtures")

    def transform_teams(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = self._add_season(raw)
        if "code" in df.columns and "team_code" not in df.columns:
            df = df.rename(columns={"code": "team_code"})
        return unify_to_schema(df, "teams")

    def transform_gameweeks(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = self._add_season(raw)
        if "id" in df.columns and "gameweek" not in df.columns:
            df = df.rename(columns={"id": "gameweek"})
        return unify_to_schema(df, "gameweeks")


class VaastavTransformer(BaseTransformer):
    """Transform CSVs from the vaastav/Fantasy-Premier-League GitHub repo."""

    def __init__(self, season: str, id_to_code: dict[int, int] | None = None):
        super().__init__(season)
        self.id_to_code = id_to_code or {}

    def transform_players(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = self._add_season(raw)
        return unify_to_schema(df, "players")

    def transform_gameweek_performances(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = self._add_season(raw)
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


class FPLApiTransformer(BaseTransformer):
    """Transform data fetched from the live FPL API."""

    def transform_players(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = self._add_season(raw)
        return unify_to_schema(df, "players")

    def transform_gameweek_performances(
        self, raw: pd.DataFrame, id_to_code: dict[int, int]
    ) -> pd.DataFrame:
        df = self._add_season(raw)
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
