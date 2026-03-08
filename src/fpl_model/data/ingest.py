"""Orchestrates data ingestion from sources into the SQLite database."""

from pathlib import Path

import pandas as pd

from fpl_model.data.cache import FileCache
from fpl_model.data.db import Database
from fpl_model.data.etl.transformers import FPLApiTransformer, VaastavTransformer
from fpl_model.data.sources.base import DataSource
from fpl_model.data.sources.fpl_api import FPLApiSource
from fpl_model.data.sources.vaastav import VaastavSource


class Ingester:
    """Main entry point for getting data into the system."""

    def __init__(
        self, db_path: str | Path = "data/fpl.db", cache_dir: str | Path = "data/raw"
    ) -> None:
        self.db = Database(db_path)
        self.db.create_tables()
        self.cache = FileCache(cache_dir)

    async def ingest_season(
        self, season: str, source: DataSource | None = None
    ) -> None:
        """Ingest a single season from a data source (defaults to Vaastav)."""
        src = source or VaastavSource(cache=self.cache)
        try:
            await self._ingest_from_source(season, src)
        finally:
            if source is None:
                await src.close()

    async def ingest_current(self, season: str) -> None:
        """Ingest the current season from the live FPL API."""
        src = FPLApiSource(cache=self.cache)
        try:
            await self._ingest_from_source(season, src)
        finally:
            await src.close()

    async def _ingest_from_source(
        self, season: str, source: DataSource
    ) -> None:
        """Fetch, transform, and write all tables for a season."""
        for table in ["players", "gameweek_performances", "fixtures", "teams"]:
            self.db.clear_table(table, where=f"season = '{season}'")

        raw_players = await source.fetch_players(season)
        raw_gw = await source.fetch_gameweek_performances(season)
        raw_fixtures = await source.fetch_fixtures(season)
        raw_teams = await source.fetch_teams(season)

        # Build id-to-code mapping for player cross-referencing
        id_to_code: dict[int, int] = {}
        if "id" in raw_players.columns and "code" in raw_players.columns:
            id_to_code = dict(zip(raw_players["id"], raw_players["code"]))

        if isinstance(source, FPLApiSource):
            transformer = FPLApiTransformer(season)
            gw_perf = transformer.transform_gameweek_performances(raw_gw, id_to_code)
        else:
            transformer = VaastavTransformer(season, id_to_code=id_to_code)
            gw_perf = transformer.transform_gameweek_performances(raw_gw)

        players = transformer.transform_players(raw_players)
        fixtures = transformer.transform_fixtures(raw_fixtures)
        teams = transformer.transform_teams(raw_teams)

        self.db.write("players", players)
        self.db.write("gameweek_performances", gw_perf)
        self.db.write("fixtures", fixtures)
        self.db.write("teams", teams)

    async def ingest_seasons(self, seasons: list[str]) -> None:
        """Ingest multiple historical seasons, reusing a single source connection."""
        source = VaastavSource(cache=self.cache)
        try:
            for season in seasons:
                await self.ingest_season(season, source=source)
        finally:
            await source.close()
