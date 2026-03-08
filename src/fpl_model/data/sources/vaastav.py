"""Data source for historical FPL data from vaastav/Fantasy-Premier-League GitHub repo."""

from io import StringIO
import httpx
import pandas as pd
from fpl_model.data.cache import FileCache
from fpl_model.data.sources.base import DataSource

BASE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"


class VaastavSource(DataSource):
    def __init__(self, cache: FileCache | None = None):
        self.cache = cache
        self.client = httpx.AsyncClient(timeout=30.0)

    def _build_url(self, season: str, filename: str) -> str:
        return f"{BASE_URL}/{season}/{filename}"

    async def _download(self, season: str, filename: str) -> str:
        if self.cache and self.cache.has("vaastav", season, filename):
            return self.cache.get("vaastav", season, filename).decode("utf-8")
        url = self._build_url(season, filename)
        response = await self.client.get(url)
        response.raise_for_status()
        content = response.text
        if self.cache:
            self.cache.put("vaastav", season, filename, content.encode("utf-8"))
        return content

    def _parse_csv(self, content: str) -> pd.DataFrame:
        return pd.read_csv(StringIO(content))

    async def fetch_players(self, season: str) -> pd.DataFrame:
        content = await self._download(season, "players_raw.csv")
        return self._parse_csv(content)

    async def fetch_gameweek_performances(self, season: str) -> pd.DataFrame:
        content = await self._download(season, "gws/merged_gw.csv")
        return self._parse_csv(content)

    async def fetch_fixtures(self, season: str) -> pd.DataFrame:
        content = await self._download(season, "fixtures.csv")
        return self._parse_csv(content)

    async def fetch_teams(self, season: str) -> pd.DataFrame:
        content = await self._download(season, "teams.csv")
        return self._parse_csv(content)

    async def fetch_master_team_list(self) -> pd.DataFrame:
        url = f"{BASE_URL}/master_team_list.csv"
        if self.cache and self.cache.has("vaastav", "_global", "master_team_list.csv"):
            content = self.cache.get("vaastav", "_global", "master_team_list.csv").decode("utf-8")
        else:
            response = await self.client.get(url)
            response.raise_for_status()
            content = response.text
            if self.cache:
                self.cache.put("vaastav", "_global", "master_team_list.csv", content.encode("utf-8"))
        return self._parse_csv(content)

    async def close(self) -> None:
        await self.client.aclose()
