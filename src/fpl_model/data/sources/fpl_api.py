"""Data source for the current season from the official FPL API."""

import json

import httpx
import pandas as pd

from fpl_model.data.cache import FileCache
from fpl_model.data.sources.base import DataSource

BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"
LIVE_GW_URL = "https://fantasy.premierleague.com/api/event/{gw}/live/"


class FPLApiSource(DataSource):
    def __init__(self, cache: FileCache | None = None):
        self.cache = cache
        self.client = httpx.AsyncClient(timeout=30.0)
        self._bootstrap: dict | None = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def _get_bootstrap(self) -> dict:
        if self._bootstrap is not None:
            return self._bootstrap
        cache_key = "bootstrap-static.json"
        if self.cache and self.cache.has("fpl_api", "_current", cache_key):
            content = self.cache.get("fpl_api", "_current", cache_key).decode("utf-8")
            self._bootstrap = json.loads(content)
            return self._bootstrap
        response = await self.client.get(BOOTSTRAP_URL)
        response.raise_for_status()
        self._bootstrap = response.json()
        if self.cache:
            self.cache.put(
                "fpl_api",
                "_current",
                cache_key,
                json.dumps(self._bootstrap).encode("utf-8"),
            )
        return self._bootstrap

    def _parse_players(self, bootstrap: dict) -> pd.DataFrame:
        return pd.DataFrame(bootstrap["elements"])

    def _parse_teams(self, bootstrap: dict) -> pd.DataFrame:
        return pd.DataFrame(bootstrap["teams"])

    def _parse_gameweeks(self, bootstrap: dict) -> pd.DataFrame:
        df = pd.DataFrame(bootstrap["events"])
        df = df.rename(columns={"id": "gameweek"})
        return df

    async def fetch_players(self, season: str) -> pd.DataFrame:
        bootstrap = await self._get_bootstrap()
        return self._parse_players(bootstrap)

    async def fetch_gameweek_performances(self, season: str) -> pd.DataFrame:
        bootstrap = await self._get_bootstrap()
        finished_gws = [e["id"] for e in bootstrap["events"] if e["finished"]]
        all_gw_data = []
        for gw in finished_gws:
            cache_key = f"event-{gw}-live.json"
            if self.cache and self.cache.has("fpl_api", "_current", cache_key):
                data = json.loads(self.cache.get("fpl_api", "_current", cache_key).decode("utf-8"))
            else:
                response = await self.client.get(LIVE_GW_URL.format(gw=gw))
                response.raise_for_status()
                data = response.json()
                if self.cache:
                    self.cache.put(
                        "fpl_api",
                        "_current",
                        cache_key,
                        json.dumps(data).encode("utf-8"),
                    )
            for element in data["elements"]:
                row = element["stats"]
                row["element"] = element["id"]
                row["round"] = gw
                all_gw_data.append(row)
        return pd.DataFrame(all_gw_data) if all_gw_data else pd.DataFrame()

    async def fetch_fixtures(self, season: str) -> pd.DataFrame:
        response = await self.client.get(FIXTURES_URL)
        response.raise_for_status()
        return pd.DataFrame(response.json())

    async def fetch_teams(self, season: str) -> pd.DataFrame:
        bootstrap = await self._get_bootstrap()
        return self._parse_teams(bootstrap)

    async def fetch_gameweeks(self, season: str) -> pd.DataFrame:
        bootstrap = await self._get_bootstrap()
        return self._parse_gameweeks(bootstrap)

    async def close(self) -> None:
        await self.client.aclose()
