"""Abstract base class for data sources."""

from abc import ABC, abstractmethod
import pandas as pd


class DataSource(ABC):
    """Interface for fetching raw FPL data from a source."""

    @abstractmethod
    async def fetch_players(self, season: str) -> pd.DataFrame:
        ...

    @abstractmethod
    async def fetch_gameweek_performances(self, season: str) -> pd.DataFrame:
        ...

    @abstractmethod
    async def fetch_fixtures(self, season: str) -> pd.DataFrame:
        ...

    @abstractmethod
    async def fetch_teams(self, season: str) -> pd.DataFrame:
        ...
