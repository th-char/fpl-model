# FPL ML Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the FPL ML pipeline from design doc: data ingestion, model abstractions, simulation engine, evaluation framework, and CLI.

**Architecture:** Layered library in `src/fpl_model/` with data, models, simulation, evaluation, and CLI subpackages. SQLite persistence. UV-managed Python 3.12+ project.

**Tech Stack:** Python 3.12+, UV, httpx, pandas, click, pytest, ruff, SQLite

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/fpl_model/__init__.py`
- Create: `.gitignore`
- Create: `.python-version`

**Step 1: Initialize UV project with Python 3.12**

```bash
cd /home/node/fpl-model
/home/node/.local/bin/uv init --lib --python 3.12
```

If `pyproject.toml` already exists from `uv init`, edit it. Otherwise create it:

```toml
[project]
name = "fpl-model"
version = "0.1.0"
description = "FPL ML pipeline for team recommendations"
requires-python = ">=3.12"
dependencies = [
    "httpx>=0.27",
    "pandas>=2.2",
    "click>=8.1",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "ruff>=0.4",
    "pytest-asyncio>=0.23",
]
ml = [
    "scikit-learn>=1.4",
]

[project.scripts]
fpl-model = "fpl_model.cli.main:cli"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

**Step 2: Create .gitignore**

```
data/raw/
data/*.db
__pycache__/
*.pyc
.venv/
*.egg-info/
dist/
.ruff_cache/
.pytest_cache/
```

**Step 3: Create directory structure**

```bash
mkdir -p src/fpl_model/{data/{sources,etl},models/{predictors,optimizers},simulation,evaluation,cli}
mkdir -p tests/{data,models,simulation,evaluation}
mkdir -p notebooks
mkdir -p data/raw
touch src/fpl_model/__init__.py
touch src/fpl_model/{data,models,simulation,evaluation,cli}/__init__.py
touch src/fpl_model/data/{sources,etl}/__init__.py
touch src/fpl_model/models/{predictors,optimizers}/__init__.py
touch tests/__init__.py
touch tests/{data,models,simulation,evaluation}/__init__.py
```

**Step 4: Install dependencies**

```bash
/home/node/.local/bin/uv sync --all-extras
```

**Step 5: Verify setup**

```bash
/home/node/.local/bin/uv run python -c "import fpl_model; print('OK')"
/home/node/.local/bin/uv run pytest --co -q  # collect 0 tests, no errors
```

**Step 6: Commit**

```bash
git add -A
git commit -m "feat: scaffold project with UV, directory structure, and dependencies"
```

---

### Task 2: Data Layer — SQLite Database & Canonical Schemas

**Files:**
- Create: `src/fpl_model/data/etl/schemas.py`
- Create: `src/fpl_model/data/db.py`
- Create: `tests/data/test_db.py`

**Step 1: Write the failing test for DB and schemas**

```python
# tests/data/test_db.py
import sqlite3
import pandas as pd
import pytest
from fpl_model.data.db import Database
from fpl_model.data.etl.schemas import TABLES


class TestDatabase:
    def test_create_tables(self, tmp_path):
        db = Database(tmp_path / "test.db")
        db.create_tables()
        conn = sqlite3.connect(tmp_path / "test.db")
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        assert "players" in tables
        assert "gameweek_performances" in tables
        assert "fixtures" in tables
        assert "teams" in tables
        assert "gameweeks" in tables
        conn.close()

    def test_write_and_read_players(self, tmp_path):
        db = Database(tmp_path / "test.db")
        db.create_tables()
        df = pd.DataFrame({
            "season": ["2024-25"],
            "code": [1],
            "first_name": ["Mohamed"],
            "second_name": ["Salah"],
            "web_name": ["Salah"],
            "element_type": [3],
            "team_code": [14],
            "now_cost": [130],
        })
        db.write("players", df)
        result = db.read("players")
        assert len(result) == 1
        assert result.iloc[0]["web_name"] == "Salah"

    def test_write_and_read_gameweek_performances(self, tmp_path):
        db = Database(tmp_path / "test.db")
        db.create_tables()
        df = pd.DataFrame({
            "season": ["2024-25"],
            "player_code": [1],
            "gameweek": [1],
            "fixture_id": [1],
            "total_points": [10],
            "minutes": [90],
            "goals_scored": [1],
            "assists": [1],
            "clean_sheets": [0],
            "bonus": [3],
            "bps": [45],
            "value": [130],
            "was_home": [True],
            "opponent_team": [2],
        })
        db.write("gameweek_performances", df)
        result = db.read("gameweek_performances")
        assert len(result) == 1
        assert result.iloc[0]["total_points"] == 10

    def test_read_with_query(self, tmp_path):
        db = Database(tmp_path / "test.db")
        db.create_tables()
        df = pd.DataFrame({
            "season": ["2024-25", "2024-25"],
            "code": [1, 2],
            "first_name": ["Mohamed", "Erling"],
            "second_name": ["Salah", "Haaland"],
            "web_name": ["Salah", "Haaland"],
            "element_type": [3, 4],
            "team_code": [14, 131],
            "now_cost": [130, 140],
        })
        db.write("players", df)
        result = db.read("players", where="element_type = 4")
        assert len(result) == 1
        assert result.iloc[0]["web_name"] == "Haaland"


class TestSchemas:
    def test_tables_dict_has_required_keys(self):
        assert "players" in TABLES
        assert "gameweek_performances" in TABLES
        assert "fixtures" in TABLES
        assert "teams" in TABLES
        assert "gameweeks" in TABLES

    def test_players_schema_has_key_columns(self):
        cols = TABLES["players"]
        assert "season" in cols
        assert "code" in cols
        assert "web_name" in cols
        assert "element_type" in cols
        assert "now_cost" in cols

    def test_gameweek_performances_schema_has_key_columns(self):
        cols = TABLES["gameweek_performances"]
        assert "season" in cols
        assert "player_code" in cols
        assert "gameweek" in cols
        assert "total_points" in cols
        assert "minutes" in cols
```

**Step 2: Run tests to verify they fail**

```bash
/home/node/.local/bin/uv run pytest tests/data/test_db.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'fpl_model.data.db'`

**Step 3: Implement schemas**

```python
# src/fpl_model/data/etl/schemas.py
"""Canonical table schemas for the unified FPL database.

Each table is defined as a dict of {column_name: sqlite_type}.
These represent the superset of columns across all seasons.
Missing columns in older seasons become NULL.
"""

TABLES: dict[str, dict[str, str]] = {
    "players": {
        "season": "TEXT NOT NULL",
        "code": "INTEGER NOT NULL",          # stable cross-season identifier
        "first_name": "TEXT",
        "second_name": "TEXT",
        "web_name": "TEXT",
        "element_type": "INTEGER",           # 1=GK, 2=DEF, 3=MID, 4=FWD
        "team_code": "INTEGER",
        "now_cost": "INTEGER",               # in 0.1m units
        "total_points": "INTEGER",
        "minutes": "INTEGER",
        "goals_scored": "INTEGER",
        "assists": "INTEGER",
        "clean_sheets": "INTEGER",
        "goals_conceded": "INTEGER",
        "own_goals": "INTEGER",
        "penalties_saved": "INTEGER",
        "penalties_missed": "INTEGER",
        "yellow_cards": "INTEGER",
        "red_cards": "INTEGER",
        "saves": "INTEGER",
        "bonus": "INTEGER",
        "bps": "INTEGER",
        "influence": "REAL",
        "creativity": "REAL",
        "threat": "REAL",
        "ict_index": "REAL",
        "expected_goals": "REAL",
        "expected_assists": "REAL",
        "expected_goal_involvements": "REAL",
        "expected_goals_conceded": "REAL",
        "starts": "INTEGER",
        "form": "REAL",
        "points_per_game": "REAL",
        "selected_by_percent": "REAL",
        "transfers_in": "INTEGER",
        "transfers_out": "INTEGER",
        "status": "TEXT",                    # a=available, i=injured, etc.
        "chance_of_playing_next_round": "INTEGER",
        "news": "TEXT",
        "cost_change_start": "INTEGER",
        "penalties_order": "INTEGER",
        "direct_freekicks_order": "INTEGER",
        "corners_and_indirect_freekicks_order": "INTEGER",
    },
    "gameweek_performances": {
        "season": "TEXT NOT NULL",
        "player_code": "INTEGER NOT NULL",   # matches players.code
        "gameweek": "INTEGER NOT NULL",
        "fixture_id": "INTEGER",
        "total_points": "INTEGER",
        "minutes": "INTEGER",
        "goals_scored": "INTEGER",
        "assists": "INTEGER",
        "clean_sheets": "INTEGER",
        "goals_conceded": "INTEGER",
        "own_goals": "INTEGER",
        "penalties_saved": "INTEGER",
        "penalties_missed": "INTEGER",
        "yellow_cards": "INTEGER",
        "red_cards": "INTEGER",
        "saves": "INTEGER",
        "bonus": "INTEGER",
        "bps": "INTEGER",
        "influence": "REAL",
        "creativity": "REAL",
        "threat": "REAL",
        "ict_index": "REAL",
        "expected_goals": "REAL",
        "expected_assists": "REAL",
        "expected_goal_involvements": "REAL",
        "expected_goals_conceded": "REAL",
        "starts": "INTEGER",
        "value": "INTEGER",                 # player price at time of GW
        "was_home": "INTEGER",              # boolean as 0/1
        "opponent_team": "INTEGER",
        "kickoff_time": "TEXT",
        "team_h_score": "INTEGER",
        "team_a_score": "INTEGER",
        "transfers_in": "INTEGER",
        "transfers_out": "INTEGER",
        "selected": "INTEGER",
    },
    "fixtures": {
        "season": "TEXT NOT NULL",
        "fixture_id": "INTEGER NOT NULL",
        "gameweek": "INTEGER",
        "kickoff_time": "TEXT",
        "team_h": "INTEGER",                # team code
        "team_a": "INTEGER",                # team code
        "team_h_score": "INTEGER",
        "team_a_score": "INTEGER",
        "team_h_difficulty": "INTEGER",      # FDR 1-5
        "team_a_difficulty": "INTEGER",      # FDR 1-5
        "finished": "INTEGER",              # boolean
        "started": "INTEGER",               # boolean
    },
    "teams": {
        "season": "TEXT NOT NULL",
        "team_code": "INTEGER NOT NULL",
        "name": "TEXT",
        "short_name": "TEXT",
        "strength": "INTEGER",
        "strength_overall_home": "INTEGER",
        "strength_overall_away": "INTEGER",
        "strength_attack_home": "INTEGER",
        "strength_attack_away": "INTEGER",
        "strength_defence_home": "INTEGER",
        "strength_defence_away": "INTEGER",
        "position": "INTEGER",              # league position
        "played": "INTEGER",
        "win": "INTEGER",
        "draw": "INTEGER",
        "loss": "INTEGER",
        "points": "INTEGER",
    },
    "gameweeks": {
        "season": "TEXT NOT NULL",
        "gameweek": "INTEGER NOT NULL",
        "deadline_time": "TEXT",
        "finished": "INTEGER",
        "average_entry_score": "INTEGER",
        "highest_score": "INTEGER",
        "most_selected": "INTEGER",         # player id
        "most_captained": "INTEGER",        # player id
        "most_transferred_in": "INTEGER",   # player id
    },
}
```

**Step 4: Implement Database class**

```python
# src/fpl_model/data/db.py
"""SQLite database for persisting unified FPL data."""

import sqlite3
from pathlib import Path

import pandas as pd

from fpl_model.data.etl.schemas import TABLES


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def create_tables(self) -> None:
        conn = self._connect()
        try:
            for table_name, columns in TABLES.items():
                cols_sql = ", ".join(f"{col} {dtype}" for col, dtype in columns.items())
                conn.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({cols_sql})")
            conn.commit()
        finally:
            conn.close()

    def write(self, table: str, df: pd.DataFrame) -> None:
        if table not in TABLES:
            raise ValueError(f"Unknown table: {table}")
        conn = self._connect()
        try:
            # Only write columns that exist in schema
            schema_cols = set(TABLES[table].keys())
            df_cols = [c for c in df.columns if c in schema_cols]
            df[df_cols].to_sql(table, conn, if_exists="append", index=False)
        finally:
            conn.close()

    def read(self, table: str, where: str | None = None) -> pd.DataFrame:
        if table not in TABLES:
            raise ValueError(f"Unknown table: {table}")
        conn = self._connect()
        try:
            query = f"SELECT * FROM {table}"  # noqa: S608
            if where:
                query += f" WHERE {where}"
            return pd.read_sql_query(query, conn)
        finally:
            conn.close()

    def query(self, sql: str) -> pd.DataFrame:
        conn = self._connect()
        try:
            return pd.read_sql_query(sql, conn)
        finally:
            conn.close()

    def clear_table(self, table: str, where: str | None = None) -> None:
        if table not in TABLES:
            raise ValueError(f"Unknown table: {table}")
        conn = self._connect()
        try:
            query = f"DELETE FROM {table}"
            if where:
                query += f" WHERE {where}"
            conn.execute(query)
            conn.commit()
        finally:
            conn.close()
```

**Step 5: Run tests to verify they pass**

```bash
/home/node/.local/bin/uv run pytest tests/data/test_db.py -v
```

Expected: All PASS

**Step 6: Commit**

```bash
git add src/fpl_model/data/etl/schemas.py src/fpl_model/data/db.py tests/data/test_db.py
git commit -m "feat: add canonical schemas and SQLite database layer"
```

---

### Task 3: Data Layer — File Cache

**Files:**
- Create: `src/fpl_model/data/cache.py`
- Create: `tests/data/test_cache.py`

**Step 1: Write the failing test**

```python
# tests/data/test_cache.py
from pathlib import Path
from fpl_model.data.cache import FileCache


class TestFileCache:
    def test_cache_miss(self, tmp_path):
        cache = FileCache(tmp_path)
        assert cache.get("vaastav", "2024-25", "players_raw.csv") is None

    def test_cache_put_and_get(self, tmp_path):
        cache = FileCache(tmp_path)
        content = b"id,name\n1,Salah"
        cache.put("vaastav", "2024-25", "players_raw.csv", content)
        result = cache.get("vaastav", "2024-25", "players_raw.csv")
        assert result == content

    def test_cache_path_structure(self, tmp_path):
        cache = FileCache(tmp_path)
        cache.put("vaastav", "2024-25", "players_raw.csv", b"data")
        expected = tmp_path / "vaastav" / "2024-25" / "players_raw.csv"
        assert expected.exists()

    def test_cache_clear_season(self, tmp_path):
        cache = FileCache(tmp_path)
        cache.put("vaastav", "2024-25", "players_raw.csv", b"data")
        cache.clear("vaastav", "2024-25")
        assert cache.get("vaastav", "2024-25", "players_raw.csv") is None

    def test_has(self, tmp_path):
        cache = FileCache(tmp_path)
        assert not cache.has("vaastav", "2024-25", "players_raw.csv")
        cache.put("vaastav", "2024-25", "players_raw.csv", b"data")
        assert cache.has("vaastav", "2024-25", "players_raw.csv")
```

**Step 2: Run test to verify it fails**

```bash
/home/node/.local/bin/uv run pytest tests/data/test_cache.py -v
```

**Step 3: Implement FileCache**

```python
# src/fpl_model/data/cache.py
"""Local file cache for raw downloaded data."""

import shutil
from pathlib import Path


class FileCache:
    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)

    def _path(self, source: str, season: str, filename: str) -> Path:
        return self.base_dir / source / season / filename

    def has(self, source: str, season: str, filename: str) -> bool:
        return self._path(source, season, filename).exists()

    def get(self, source: str, season: str, filename: str) -> bytes | None:
        path = self._path(source, season, filename)
        if path.exists():
            return path.read_bytes()
        return None

    def put(self, source: str, season: str, filename: str, content: bytes) -> None:
        path = self._path(source, season, filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def clear(self, source: str, season: str | None = None) -> None:
        if season:
            target = self.base_dir / source / season
        else:
            target = self.base_dir / source
        if target.exists():
            shutil.rmtree(target)
```

**Step 4: Run tests to verify they pass**

```bash
/home/node/.local/bin/uv run pytest tests/data/test_cache.py -v
```

**Step 5: Commit**

```bash
git add src/fpl_model/data/cache.py tests/data/test_cache.py
git commit -m "feat: add file cache for raw data downloads"
```

---

### Task 4: Data Layer — DataSource Interface & Vaastav Source

**Files:**
- Create: `src/fpl_model/data/sources/base.py`
- Create: `src/fpl_model/data/sources/vaastav.py`
- Create: `tests/data/test_vaastav.py`

**Step 1: Write the failing test**

```python
# tests/data/test_vaastav.py
from unittest.mock import AsyncMock, patch
import pandas as pd
import pytest
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
```

**Step 2: Run tests to verify they fail**

```bash
/home/node/.local/bin/uv run pytest tests/data/test_vaastav.py -v
```

**Step 3: Implement DataSource base and VaastavSource**

```python
# src/fpl_model/data/sources/base.py
"""Abstract base class for data sources."""

from abc import ABC, abstractmethod

import pandas as pd


class DataSource(ABC):
    """Interface for fetching raw FPL data from a source."""

    @abstractmethod
    async def fetch_players(self, season: str) -> pd.DataFrame:
        """Fetch player data for a season."""
        ...

    @abstractmethod
    async def fetch_gameweek_performances(self, season: str) -> pd.DataFrame:
        """Fetch per-player per-gameweek data for a season."""
        ...

    @abstractmethod
    async def fetch_fixtures(self, season: str) -> pd.DataFrame:
        """Fetch fixture data for a season."""
        ...

    @abstractmethod
    async def fetch_teams(self, season: str) -> pd.DataFrame:
        """Fetch team data for a season."""
        ...
```

```python
# src/fpl_model/data/sources/vaastav.py
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
```

**Step 4: Run tests to verify they pass**

```bash
/home/node/.local/bin/uv run pytest tests/data/test_vaastav.py -v
```

**Step 5: Commit**

```bash
git add src/fpl_model/data/sources/base.py src/fpl_model/data/sources/vaastav.py tests/data/test_vaastav.py
git commit -m "feat: add DataSource interface and vaastav GitHub source"
```

---

### Task 5: Data Layer — FPL API Source

**Files:**
- Create: `src/fpl_model/data/sources/fpl_api.py`
- Create: `tests/data/test_fpl_api.py`

**Step 1: Write the failing test**

```python
# tests/data/test_fpl_api.py
import json
from unittest.mock import AsyncMock, MagicMock, patch
import pandas as pd
import pytest
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
```

**Step 2: Run tests to verify they fail**

```bash
/home/node/.local/bin/uv run pytest tests/data/test_fpl_api.py -v
```

**Step 3: Implement FPLApiSource**

```python
# src/fpl_model/data/sources/fpl_api.py
"""Data source for the current season from the official FPL API."""

import httpx
import pandas as pd

from fpl_model.data.cache import FileCache
from fpl_model.data.sources.base import DataSource

BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"
LIVE_GW_URL = "https://fantasy.premierleague.com/api/event/{gw}/live/"
PLAYER_URL = "https://fantasy.premierleague.com/api/element-summary/{player_id}/"


class FPLApiSource(DataSource):
    def __init__(self, cache: FileCache | None = None):
        self.cache = cache
        self.client = httpx.AsyncClient(timeout=30.0)
        self._bootstrap: dict | None = None

    async def _get_bootstrap(self) -> dict:
        if self._bootstrap is not None:
            return self._bootstrap

        cache_key = "bootstrap-static.json"
        if self.cache and self.cache.has("fpl_api", "_current", cache_key):
            import json
            content = self.cache.get("fpl_api", "_current", cache_key).decode("utf-8")
            self._bootstrap = json.loads(content)
            return self._bootstrap

        response = await self.client.get(BOOTSTRAP_URL)
        response.raise_for_status()
        self._bootstrap = response.json()

        if self.cache:
            import json
            self.cache.put(
                "fpl_api", "_current", cache_key,
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
        # Fetches live data for all finished GWs
        bootstrap = await self._get_bootstrap()
        finished_gws = [
            e["id"] for e in bootstrap["events"] if e["finished"]
        ]

        all_gw_data = []
        for gw in finished_gws:
            response = await self.client.get(LIVE_GW_URL.format(gw=gw))
            response.raise_for_status()
            data = response.json()
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

    async def close(self) -> None:
        await self.client.aclose()
```

**Step 4: Run tests to verify they pass**

```bash
/home/node/.local/bin/uv run pytest tests/data/test_fpl_api.py -v
```

**Step 5: Commit**

```bash
git add src/fpl_model/data/sources/fpl_api.py tests/data/test_fpl_api.py
git commit -m "feat: add FPL API data source for current season"
```

---

### Task 6: Data Layer — ETL Transformers & Unifier

**Files:**
- Create: `src/fpl_model/data/etl/transformers.py`
- Create: `src/fpl_model/data/etl/unifier.py`
- Create: `tests/data/test_etl.py`

**Step 1: Write the failing test**

```python
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
        # vaastav merged_gw.csv uses 'element' column (the per-season ID)
        # We need a player_id_map to convert to code
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
        # Mapping from season element ID to stable code
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
```

**Step 2: Run tests to verify they fail**

```bash
/home/node/.local/bin/uv run pytest tests/data/test_etl.py -v
```

**Step 3: Implement transformers and unifier**

```python
# src/fpl_model/data/etl/unifier.py
"""Unify DataFrames to canonical schemas."""

import pandas as pd

from fpl_model.data.etl.schemas import TABLES


def unify_to_schema(df: pd.DataFrame, table: str) -> pd.DataFrame:
    """Map a DataFrame to the canonical schema for a table.

    - Drops columns not in the schema
    - Adds missing columns as NaN
    - Orders columns to match schema
    """
    if table not in TABLES:
        raise ValueError(f"Unknown table: {table}")

    schema_cols = list(TABLES[table].keys())

    # Add missing columns as NaN
    for col in schema_cols:
        if col not in df.columns:
            df[col] = pd.NA

    # Keep only schema columns, in order
    return df[schema_cols].copy()
```

```python
# src/fpl_model/data/etl/transformers.py
"""Transform raw source data into canonical schema format."""

import pandas as pd

from fpl_model.data.etl.unifier import unify_to_schema


class VaastavTransformer:
    """Transform vaastav GitHub CSV data to canonical schemas."""

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

        # Map element (per-season ID) to stable code
        if "element" in df.columns:
            df["player_code"] = df["element"].map(self.id_to_code)

        # Rename round -> gameweek, fixture -> fixture_id
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

        # vaastav uses 'code' for team code
        if "code" in df.columns and "team_code" not in df.columns:
            df = df.rename(columns={"code": "team_code"})

        return unify_to_schema(df, "teams")


class FPLApiTransformer:
    """Transform FPL API JSON data to canonical schemas."""

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
```

**Step 4: Run tests to verify they pass**

```bash
/home/node/.local/bin/uv run pytest tests/data/test_etl.py -v
```

**Step 5: Commit**

```bash
git add src/fpl_model/data/etl/transformers.py src/fpl_model/data/etl/unifier.py tests/data/test_etl.py
git commit -m "feat: add ETL transformers and schema unifier"
```

---

### Task 7: Data Layer — Ingest Orchestrator

**Files:**
- Create: `src/fpl_model/data/ingest.py`
- Create: `tests/data/test_ingest.py`

**Step 1: Write the failing test**

```python
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

        # Mock the vaastav source
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

        with patch.object(ingester, "_source") as mock_source:
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
```

**Step 2: Run test to verify it fails**

```bash
/home/node/.local/bin/uv run pytest tests/data/test_ingest.py -v
```

**Step 3: Implement Ingester**

```python
# src/fpl_model/data/ingest.py
"""Orchestrates data ingestion from sources into the SQLite database."""

from pathlib import Path

import pandas as pd

from fpl_model.data.cache import FileCache
from fpl_model.data.db import Database
from fpl_model.data.etl.transformers import VaastavTransformer, FPLApiTransformer
from fpl_model.data.sources.base import DataSource
from fpl_model.data.sources.vaastav import VaastavSource
from fpl_model.data.sources.fpl_api import FPLApiSource


class Ingester:
    def __init__(
        self,
        db_path: str | Path = "data/fpl.db",
        cache_dir: str | Path = "data/raw",
    ):
        self.db = Database(db_path)
        self.db.create_tables()
        self.cache = FileCache(cache_dir)
        self._source: DataSource | None = None

    async def ingest_season(self, season: str, source: DataSource | None = None) -> None:
        src = source or VaastavSource(cache=self.cache)
        try:
            await self._ingest_from_source(season, src)
        finally:
            if source is None:
                await src.close()

    async def ingest_current(self, season: str) -> None:
        src = FPLApiSource(cache=self.cache)
        try:
            await self._ingest_from_source(season, src)
        finally:
            await src.close()

    async def _ingest_from_source(self, season: str, source: DataSource) -> None:
        # Clear existing data for this season
        for table in ["players", "gameweek_performances", "fixtures", "teams"]:
            self.db.clear_table(table, where=f"season = '{season}'")

        # Fetch raw data
        raw_players = await source.fetch_players(season)
        raw_gw = await source.fetch_gameweek_performances(season)
        raw_fixtures = await source.fetch_fixtures(season)
        raw_teams = await source.fetch_teams(season)

        # Build id-to-code mapping from players
        id_to_code = {}
        if "id" in raw_players.columns and "code" in raw_players.columns:
            id_to_code = dict(zip(raw_players["id"], raw_players["code"]))

        # Determine transformer based on source type
        if isinstance(source, FPLApiSource):
            transformer = FPLApiTransformer(season)
            gw_perf = transformer.transform_gameweek_performances(raw_gw, id_to_code)
        else:
            transformer = VaastavTransformer(season, id_to_code=id_to_code)
            gw_perf = transformer.transform_gameweek_performances(raw_gw)

        players = transformer.transform_players(raw_players)
        fixtures = transformer.transform_fixtures(raw_fixtures)
        teams = transformer.transform_teams(raw_teams)

        # Write to database
        self.db.write("players", players)
        self.db.write("gameweek_performances", gw_perf)
        self.db.write("fixtures", fixtures)
        self.db.write("teams", teams)

    async def ingest_seasons(self, seasons: list[str]) -> None:
        source = VaastavSource(cache=self.cache)
        try:
            for season in seasons:
                await self.ingest_season(season, source=source)
        finally:
            await source.close()
```

**Step 4: Run tests to verify they pass**

```bash
/home/node/.local/bin/uv run pytest tests/data/test_ingest.py -v
```

**Step 5: Commit**

```bash
git add src/fpl_model/data/ingest.py tests/data/test_ingest.py
git commit -m "feat: add ingest orchestrator for multi-source ETL"
```

---

### Task 8: Model Layer — Base Abstractions & Actions

**Files:**
- Create: `src/fpl_model/simulation/actions.py`
- Create: `src/fpl_model/simulation/state.py`
- Create: `src/fpl_model/models/base.py`
- Create: `src/fpl_model/models/registry.py`
- Create: `tests/models/test_base.py`

**Step 1: Write the failing test**

```python
# tests/models/test_base.py
import pytest
from fpl_model.simulation.actions import (
    Transfer, SetCaptain, SetViceCaptain, SetLineup, PlayChip, ChipType,
)
from fpl_model.simulation.state import SquadState, PlayerInSquad
from fpl_model.models.base import ActionModel, PredictOptimizeModel, Predictor, Optimizer
from fpl_model.models.registry import ModelRegistry


class TestActions:
    def test_transfer_creation(self):
        t = Transfer(player_out=1, player_in=2)
        assert t.player_out == 1
        assert t.player_in == 2

    def test_set_captain(self):
        c = SetCaptain(player_id=1)
        assert c.player_id == 1

    def test_play_chip(self):
        c = PlayChip(chip_type=ChipType.BENCH_BOOST)
        assert c.chip_type == ChipType.BENCH_BOOST

    def test_set_lineup(self):
        sl = SetLineup(starting_xi=[1, 2, 3], bench_order=[4, 5])
        assert len(sl.starting_xi) == 3


class TestSquadState:
    def test_create_squad_state(self):
        players = [
            PlayerInSquad(code=i, element_type=(i % 4) + 1, buy_price=50, sell_price=50)
            for i in range(15)
        ]
        state = SquadState(
            players=players,
            budget=0,
            free_transfers=1,
            chips_available={ChipType.WILDCARD: 2, ChipType.BENCH_BOOST: 2},
            current_gameweek=1,
        )
        assert len(state.players) == 15
        assert state.free_transfers == 1


class TestModelRegistry:
    def test_register_and_get(self):
        registry = ModelRegistry()

        class DummyModel(ActionModel):
            def recommend(self, state, data):
                return []

        model = DummyModel()
        registry.register("dummy", model)
        assert registry.get("dummy") is model

    def test_get_unknown_raises(self):
        registry = ModelRegistry()
        with pytest.raises(KeyError):
            registry.get("nonexistent")

    def test_list_models(self):
        registry = ModelRegistry()

        class DummyModel(ActionModel):
            def recommend(self, state, data):
                return []

        registry.register("dummy", DummyModel())
        assert "dummy" in registry.list()
```

**Step 2: Run tests to verify they fail**

```bash
/home/node/.local/bin/uv run pytest tests/models/test_base.py -v
```

**Step 3: Implement actions, state, base models, and registry**

```python
# src/fpl_model/simulation/actions.py
"""Action dataclasses for FPL decisions."""

from dataclasses import dataclass
from enum import Enum


class ChipType(Enum):
    WILDCARD = "wildcard"
    FREE_HIT = "free_hit"
    BENCH_BOOST = "bench_boost"
    TRIPLE_CAPTAIN = "triple_captain"


@dataclass(frozen=True)
class Transfer:
    player_out: int  # player code
    player_in: int   # player code


@dataclass(frozen=True)
class SetCaptain:
    player_id: int   # player code


@dataclass(frozen=True)
class SetViceCaptain:
    player_id: int   # player code


@dataclass(frozen=True)
class SetLineup:
    starting_xi: list[int]   # player codes
    bench_order: list[int]   # player codes, ordered


@dataclass(frozen=True)
class PlayChip:
    chip_type: ChipType


# Union type for all actions
Action = Transfer | SetCaptain | SetViceCaptain | SetLineup | PlayChip
```

```python
# src/fpl_model/simulation/state.py
"""Squad state tracking for simulation."""

from dataclasses import dataclass, field

from fpl_model.simulation.actions import ChipType


@dataclass
class PlayerInSquad:
    code: int
    element_type: int        # 1=GK, 2=DEF, 3=MID, 4=FWD
    buy_price: int           # price when bought, in 0.1m units
    sell_price: int           # current sell price


@dataclass
class SquadState:
    players: list[PlayerInSquad]
    budget: int                                    # remaining budget in 0.1m units
    free_transfers: int
    chips_available: dict[ChipType, int]           # chip -> uses remaining
    current_gameweek: int
    captain: int | None = None                     # player code
    vice_captain: int | None = None                # player code
    starting_xi: list[int] = field(default_factory=list)    # player codes
    bench_order: list[int] = field(default_factory=list)    # player codes
    active_chip: ChipType | None = None
    # For Free Hit: store the pre-FH squad to revert to
    pre_free_hit_state: "SquadState | None" = None
```

```python
# src/fpl_model/models/base.py
"""Abstract base classes for FPL models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd

from fpl_model.simulation.actions import Action
from fpl_model.simulation.state import SquadState


@dataclass
class SeasonData:
    """All data available to a model for making decisions."""
    gameweek_performances: pd.DataFrame   # historical player performances
    fixtures: pd.DataFrame                 # all fixtures (past + upcoming)
    players: pd.DataFrame                  # player metadata
    teams: pd.DataFrame                    # team data
    current_gameweek: int
    season: str


@dataclass
class PlayerPredictions:
    """Predicted points per player for upcoming gameweek(s)."""
    # Maps player_code -> predicted points (for next GW)
    predictions: dict[int, float]
    # Optional: multi-GW predictions (player_code -> list of predicted points)
    multi_gw: dict[int, list[float]] | None = None


@dataclass
class HistoricalData:
    """Historical data for training models."""
    seasons: dict[str, SeasonData]


class ActionModel(ABC):
    """Universal interface — every model implements this."""

    @abstractmethod
    def recommend(self, state: SquadState, data: SeasonData) -> list[Action]:
        """Given current squad state and available data, return actions."""
        ...

    def train(self, historical_data: HistoricalData) -> None:
        """Optional: train on historical data. No-op for rule-based models."""
        pass


class Predictor(ABC):
    """Predicts future player performance."""

    @abstractmethod
    def predict(self, state: SquadState, data: SeasonData) -> PlayerPredictions:
        ...

    def train(self, historical_data: HistoricalData) -> None:
        pass


class Optimizer(ABC):
    """Given predictions + constraints, outputs optimal actions."""

    @abstractmethod
    def optimize(self, predictions: PlayerPredictions, state: SquadState, data: SeasonData) -> list[Action]:
        ...


class PredictOptimizeModel(ActionModel):
    """Composes a Predictor + Optimizer."""

    def __init__(self, predictor: Predictor, optimizer: Optimizer):
        self.predictor = predictor
        self.optimizer = optimizer

    def recommend(self, state: SquadState, data: SeasonData) -> list[Action]:
        predictions = self.predictor.predict(state, data)
        return self.optimizer.optimize(predictions, state, data)

    def train(self, historical_data: HistoricalData) -> None:
        self.predictor.train(historical_data)
```

```python
# src/fpl_model/models/registry.py
"""Model registration and discovery."""

from fpl_model.models.base import ActionModel


class ModelRegistry:
    def __init__(self):
        self._models: dict[str, ActionModel] = {}

    def register(self, name: str, model: ActionModel) -> None:
        self._models[name] = model

    def get(self, name: str) -> ActionModel:
        if name not in self._models:
            raise KeyError(f"Model '{name}' not found. Available: {list(self._models.keys())}")
        return self._models[name]

    def list(self) -> list[str]:
        return list(self._models.keys())
```

**Step 4: Run tests to verify they pass**

```bash
/home/node/.local/bin/uv run pytest tests/models/test_base.py -v
```

**Step 5: Commit**

```bash
git add src/fpl_model/simulation/actions.py src/fpl_model/simulation/state.py \
    src/fpl_model/models/base.py src/fpl_model/models/registry.py tests/models/test_base.py
git commit -m "feat: add model abstractions, actions, squad state, and registry"
```

---

### Task 9: Simulation Engine — FPL Rules

**Files:**
- Create: `src/fpl_model/simulation/rules.py`
- Create: `tests/simulation/test_rules.py`

**Step 1: Write the failing test**

```python
# tests/simulation/test_rules.py
import pandas as pd
import pytest
from fpl_model.simulation.actions import (
    Transfer, SetCaptain, SetLineup, PlayChip, ChipType,
)
from fpl_model.simulation.state import SquadState, PlayerInSquad
from fpl_model.simulation.rules import (
    validate_formation,
    apply_transfers,
    calculate_transfer_cost,
    apply_auto_subs,
    score_gameweek,
    advance_gameweek,
)


def make_squad(gk=2, defs=5, mids=5, fwds=3, budget=0):
    """Helper to create a basic squad state."""
    players = []
    code = 1
    for _ in range(gk):
        players.append(PlayerInSquad(code=code, element_type=1, buy_price=45, sell_price=45))
        code += 1
    for _ in range(defs):
        players.append(PlayerInSquad(code=code, element_type=2, buy_price=50, sell_price=50))
        code += 1
    for _ in range(mids):
        players.append(PlayerInSquad(code=code, element_type=3, buy_price=60, sell_price=60))
        code += 1
    for _ in range(fwds):
        players.append(PlayerInSquad(code=code, element_type=4, buy_price=70, sell_price=70))
        code += 1
    return SquadState(
        players=players,
        budget=budget,
        free_transfers=1,
        chips_available={ct: 2 for ct in ChipType},
        current_gameweek=1,
        captain=players[7].code,  # a midfielder
        vice_captain=players[12].code,  # a forward
        starting_xi=[p.code for p in players[:11]],
        bench_order=[p.code for p in players[11:]],
    )


class TestValidateFormation:
    def test_valid_442(self):
        # 1 GK, 4 DEF, 4 MID, 2 FWD
        xi_types = [1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4]
        assert validate_formation(xi_types) is True

    def test_valid_343(self):
        xi_types = [1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4]
        assert validate_formation(xi_types) is True

    def test_invalid_too_few_defenders(self):
        xi_types = [1, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4]
        assert validate_formation(xi_types) is False

    def test_invalid_no_forward(self):
        xi_types = [1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
        assert validate_formation(xi_types) is False


class TestTransferCost:
    def test_one_free_transfer_no_cost(self):
        assert calculate_transfer_cost(num_transfers=1, free_transfers=1) == 0

    def test_extra_transfer_costs_4(self):
        assert calculate_transfer_cost(num_transfers=2, free_transfers=1) == 4

    def test_multiple_extras(self):
        assert calculate_transfer_cost(num_transfers=4, free_transfers=2) == 8

    def test_zero_transfers(self):
        assert calculate_transfer_cost(num_transfers=0, free_transfers=1) == 0


class TestAutoSubs:
    def test_auto_sub_replaces_non_playing_starter(self):
        squad = make_squad()
        # GW data: all starters played except player code=3 (a defender)
        gw_data = pd.DataFrame({
            "player_code": [p.code for p in squad.players],
            "minutes": [90 if p.code != 3 else 0 for p in squad.players],
            "total_points": [5 if p.code != 3 else 0 for p in squad.players],
        })
        player_types = {p.code: p.element_type for p in squad.players}
        final_xi, final_bench = apply_auto_subs(
            squad.starting_xi, squad.bench_order, gw_data, player_types
        )
        assert 3 not in final_xi
        # First valid bench player should come in
        assert any(b_code in final_xi for b_code in squad.bench_order)

    def test_no_sub_if_all_played(self):
        squad = make_squad()
        gw_data = pd.DataFrame({
            "player_code": [p.code for p in squad.players],
            "minutes": [90] * 15,
            "total_points": [5] * 15,
        })
        player_types = {p.code: p.element_type for p in squad.players}
        final_xi, final_bench = apply_auto_subs(
            squad.starting_xi, squad.bench_order, gw_data, player_types
        )
        assert final_xi == squad.starting_xi


class TestAdvanceGameweek:
    def test_accrues_free_transfer(self):
        squad = make_squad()
        squad.free_transfers = 1
        new_state = advance_gameweek(squad, transfers_made=0)
        assert new_state.free_transfers == 2

    def test_caps_free_transfers_at_5(self):
        squad = make_squad()
        squad.free_transfers = 5
        new_state = advance_gameweek(squad, transfers_made=0)
        assert new_state.free_transfers == 5

    def test_resets_to_1_if_transfers_made(self):
        squad = make_squad()
        squad.free_transfers = 2
        new_state = advance_gameweek(squad, transfers_made=2)
        assert new_state.free_transfers == 1
```

**Step 2: Run tests to verify they fail**

```bash
/home/node/.local/bin/uv run pytest tests/simulation/test_rules.py -v
```

**Step 3: Implement rules**

```python
# src/fpl_model/simulation/rules.py
"""FPL game rules: formation validation, transfers, auto-subs, scoring."""

from copy import deepcopy

import pandas as pd

from fpl_model.simulation.state import SquadState, PlayerInSquad


# Formation constraints
MIN_GK = 1
MAX_GK = 1
MIN_DEF = 3
MIN_MID = 2
MIN_FWD = 1


def validate_formation(xi_types: list[int]) -> bool:
    """Check if a starting XI satisfies formation constraints."""
    from collections import Counter
    counts = Counter(xi_types)
    return (
        counts.get(1, 0) == MIN_GK
        and counts.get(2, 0) >= MIN_DEF
        and counts.get(3, 0) >= MIN_MID
        and counts.get(4, 0) >= MIN_FWD
        and sum(counts.values()) == 11
    )


def calculate_transfer_cost(num_transfers: int, free_transfers: int) -> int:
    """Calculate point cost of transfers."""
    extra = max(0, num_transfers - free_transfers)
    return extra * 4


def apply_transfers(
    state: SquadState,
    transfers: list,
    player_data: pd.DataFrame,
) -> SquadState:
    """Apply transfers to squad state, updating players and budget."""
    new_state = deepcopy(state)

    for transfer in transfers:
        # Find and remove outgoing player
        out_player = next(p for p in new_state.players if p.code == transfer.player_out)
        new_state.players.remove(out_player)
        new_state.budget += out_player.sell_price

        # Find incoming player price from player_data
        in_row = player_data[player_data["code"] == transfer.player_in].iloc[0]
        in_price = int(in_row["now_cost"])
        new_state.budget -= in_price

        new_state.players.append(PlayerInSquad(
            code=transfer.player_in,
            element_type=int(in_row["element_type"]),
            buy_price=in_price,
            sell_price=in_price,
        ))

    return new_state


def apply_auto_subs(
    starting_xi: list[int],
    bench_order: list[int],
    gw_data: pd.DataFrame,
    player_types: dict[int, int],
) -> tuple[list[int], list[int]]:
    """Apply automatic substitutions for non-playing starters."""
    from collections import Counter

    minutes_map = dict(zip(gw_data["player_code"], gw_data["minutes"]))
    final_xi = list(starting_xi)
    final_bench = list(bench_order)

    for starter in starting_xi:
        if minutes_map.get(starter, 0) > 0:
            continue

        # Try to sub in from bench
        xi_types = [player_types[p] for p in final_xi if p != starter]
        for bench_player in list(final_bench):
            if minutes_map.get(bench_player, 0) == 0:
                continue

            # Check formation validity with this sub
            candidate_types = xi_types + [player_types[bench_player]]
            if validate_formation(candidate_types):
                final_xi[final_xi.index(starter)] = bench_player
                final_bench.remove(bench_player)
                break

    return final_xi, final_bench


def score_gameweek(
    starting_xi: list[int],
    bench_order: list[int],
    captain: int | None,
    vice_captain: int | None,
    gw_data: pd.DataFrame,
    active_chip: "ChipType | None" = None,
) -> int:
    """Calculate total points for a gameweek after auto-subs."""
    from fpl_model.simulation.actions import ChipType

    points_map = dict(zip(gw_data["player_code"], gw_data["total_points"]))
    minutes_map = dict(zip(gw_data["player_code"], gw_data["minutes"]))

    total = sum(points_map.get(p, 0) for p in starting_xi)

    # Bench Boost: add bench player points
    if active_chip == ChipType.BENCH_BOOST:
        total += sum(points_map.get(p, 0) for p in bench_order)

    # Captain logic
    captain_multiplier = 2
    if active_chip == ChipType.TRIPLE_CAPTAIN:
        captain_multiplier = 3

    if captain and minutes_map.get(captain, 0) > 0:
        # Captain played: add extra points (1x for double, 2x for triple)
        total += points_map.get(captain, 0) * (captain_multiplier - 1)
    elif vice_captain and minutes_map.get(vice_captain, 0) > 0:
        # Vice-captain activates with double (not triple, even with TC)
        total += points_map.get(vice_captain, 0)

    return total


def advance_gameweek(state: SquadState, transfers_made: int) -> SquadState:
    """Advance state to next gameweek: accrue free transfers, increment GW."""
    new_state = deepcopy(state)
    new_state.current_gameweek += 1
    new_state.active_chip = None

    if transfers_made > 0:
        # Used transfers: reset to 1 for next week, then accrue
        remaining = max(0, state.free_transfers - transfers_made)
        new_state.free_transfers = min(remaining + 1, 5)
    else:
        # Banked: accrue +1, cap at 5
        new_state.free_transfers = min(state.free_transfers + 1, 5)

    return new_state
```

**Step 4: Run tests to verify they pass**

```bash
/home/node/.local/bin/uv run pytest tests/simulation/test_rules.py -v
```

**Step 5: Commit**

```bash
git add src/fpl_model/simulation/rules.py tests/simulation/test_rules.py
git commit -m "feat: implement FPL rules engine (formation, transfers, auto-subs, scoring)"
```

---

### Task 10: Simulation Engine — SeasonSimulator

**Files:**
- Create: `src/fpl_model/simulation/engine.py`
- Create: `tests/simulation/test_engine.py`

**Step 1: Write the failing test**

```python
# tests/simulation/test_engine.py
import pandas as pd
import pytest
from fpl_model.simulation.engine import SeasonSimulator, SimulationResult
from fpl_model.simulation.actions import SetCaptain, SetViceCaptain, SetLineup, ChipType
from fpl_model.simulation.state import SquadState, PlayerInSquad
from fpl_model.models.base import ActionModel, SeasonData
from fpl_model.data.db import Database


class DummyModel(ActionModel):
    """Always picks same lineup, no transfers."""
    def recommend(self, state, data):
        xi = [p.code for p in state.players[:11]]
        bench = [p.code for p in state.players[11:]]
        return [
            SetLineup(starting_xi=xi, bench_order=bench),
            SetCaptain(player_id=state.players[7].code),
            SetViceCaptain(player_id=state.players[8].code),
        ]


class TestSeasonSimulator:
    def _setup_db(self, tmp_path, num_gws=2):
        """Create a minimal DB with fixture and GW data."""
        db = Database(tmp_path / "test.db")
        db.create_tables()

        # 15 players
        players_data = []
        for i in range(1, 16):
            et = 1 if i <= 2 else (2 if i <= 7 else (3 if i <= 12 else 4))
            players_data.append({
                "season": "2024-25", "code": i, "first_name": f"Player{i}",
                "second_name": f"Last{i}", "web_name": f"P{i}",
                "element_type": et, "team_code": (i % 20) + 1, "now_cost": 50,
            })
        db.write("players", pd.DataFrame(players_data))

        # GW performances: all players score 5 points, play 90 mins
        gw_rows = []
        for gw in range(1, num_gws + 1):
            for i in range(1, 16):
                gw_rows.append({
                    "season": "2024-25", "player_code": i, "gameweek": gw,
                    "total_points": 5, "minutes": 90, "goals_scored": 0,
                    "assists": 0, "value": 50, "was_home": True, "opponent_team": 1,
                })
        db.write("gameweek_performances", pd.DataFrame(gw_rows))

        # Fixtures
        fixture_rows = []
        for gw in range(1, num_gws + 1):
            fixture_rows.append({
                "season": "2024-25", "fixture_id": gw, "gameweek": gw,
                "team_h": 1, "team_a": 2, "team_h_score": 1, "team_a_score": 0,
                "finished": True,
            })
        db.write("fixtures", pd.DataFrame(fixture_rows))

        # Teams
        db.write("teams", pd.DataFrame([
            {"season": "2024-25", "team_code": i, "name": f"Team{i}", "short_name": f"T{i}"}
            for i in range(1, 21)
        ]))

        return db

    def test_simulation_runs_and_returns_result(self, tmp_path):
        db = self._setup_db(tmp_path, num_gws=2)
        model = DummyModel()
        sim = SeasonSimulator(model=model, season="2024-25", db=db)
        result = sim.run()

        assert isinstance(result, SimulationResult)
        assert result.total_points > 0
        assert len(result.gameweek_points) == 2

    def test_captain_gets_double_points(self, tmp_path):
        db = self._setup_db(tmp_path, num_gws=1)
        model = DummyModel()
        sim = SeasonSimulator(model=model, season="2024-25", db=db)
        result = sim.run()

        # 11 starters * 5 pts = 55, captain gets double so +5 = 60
        assert result.gameweek_points[1] == 60
```

**Step 2: Run tests to verify they fail**

```bash
/home/node/.local/bin/uv run pytest tests/simulation/test_engine.py -v
```

**Step 3: Implement SeasonSimulator**

```python
# src/fpl_model/simulation/engine.py
"""Season simulation engine."""

from dataclasses import dataclass, field

import pandas as pd

from fpl_model.data.db import Database
from fpl_model.models.base import ActionModel, SeasonData
from fpl_model.simulation.actions import (
    Action, Transfer, SetCaptain, SetViceCaptain, SetLineup, PlayChip, ChipType,
)
from fpl_model.simulation.state import SquadState, PlayerInSquad
from fpl_model.simulation.rules import (
    apply_auto_subs,
    apply_transfers,
    calculate_transfer_cost,
    score_gameweek,
    advance_gameweek,
)


@dataclass
class SimulationResult:
    total_points: int
    gameweek_points: dict[int, int]          # gw -> points
    actions_log: dict[int, list[Action]]     # gw -> actions taken
    transfer_costs: dict[int, int]           # gw -> point hits
    budget_history: dict[int, int]           # gw -> budget after GW


class SeasonSimulator:
    def __init__(self, model: ActionModel, season: str, db: Database):
        self.model = model
        self.season = season
        self.db = db

    def run(self) -> SimulationResult:
        # Load all data
        players_df = self.db.read("players", where=f"season = '{self.season}'")
        gw_perf_df = self.db.read("gameweek_performances", where=f"season = '{self.season}'")
        fixtures_df = self.db.read("fixtures", where=f"season = '{self.season}'")
        teams_df = self.db.read("teams", where=f"season = '{self.season}'")

        gameweeks = sorted(gw_perf_df["gameweek"].unique())
        if not gameweeks:
            raise ValueError(f"No gameweek data for season {self.season}")

        # Initialize state for GW1
        state = self._initial_state(players_df, gameweeks[0])
        result = SimulationResult(
            total_points=0,
            gameweek_points={},
            actions_log={},
            transfer_costs={},
            budget_history={},
        )

        for gw in gameweeks:
            state.current_gameweek = gw

            # Build season data visible to the model
            season_data = SeasonData(
                gameweek_performances=gw_perf_df[gw_perf_df["gameweek"] <= gw],
                fixtures=fixtures_df,
                players=players_df,
                teams=teams_df,
                current_gameweek=gw,
                season=self.season,
            )

            # Get model recommendations
            actions = self.model.recommend(state, season_data)
            result.actions_log[gw] = actions

            # Apply actions
            transfers = [a for a in actions if isinstance(a, Transfer)]
            transfer_cost = calculate_transfer_cost(len(transfers), state.free_transfers)
            result.transfer_costs[gw] = transfer_cost

            if transfers:
                state = apply_transfers(state, transfers, players_df)

            for action in actions:
                if isinstance(action, SetCaptain):
                    state.captain = action.player_id
                elif isinstance(action, SetViceCaptain):
                    state.vice_captain = action.player_id
                elif isinstance(action, SetLineup):
                    state.starting_xi = action.starting_xi
                    state.bench_order = action.bench_order
                elif isinstance(action, PlayChip):
                    state.active_chip = action.chip_type
                    state.chips_available[action.chip_type] -= 1

            # Get this GW's performance data
            gw_data = gw_perf_df[gw_perf_df["gameweek"] == gw]

            # Apply auto-subs
            player_types = {p.code: p.element_type for p in state.players}
            final_xi, final_bench = apply_auto_subs(
                state.starting_xi, state.bench_order, gw_data, player_types
            )

            # Score
            gw_points = score_gameweek(
                final_xi, final_bench,
                state.captain, state.vice_captain,
                gw_data, state.active_chip,
            )
            gw_points -= transfer_cost

            result.gameweek_points[gw] = gw_points
            result.total_points += gw_points
            result.budget_history[gw] = state.budget

            # Advance to next GW
            state = advance_gameweek(state, transfers_made=len(transfers))

        return result

    def _initial_state(self, players_df: pd.DataFrame, first_gw: int) -> SquadState:
        """Create initial squad state. Model picks the squad via recommend()."""
        # For GW1, provide all players so the model can pick
        season_data = SeasonData(
            gameweek_performances=pd.DataFrame(),
            fixtures=self.db.read("fixtures", where=f"season = '{self.season}'"),
            players=players_df,
            teams=self.db.read("teams", where=f"season = '{self.season}'"),
            current_gameweek=first_gw,
            season=self.season,
        )

        # Start with empty state - model must set lineup
        # Pick 15 cheapest as placeholder for initial recommend() call
        sorted_players = players_df.sort_values("now_cost").head(15)
        initial_players = [
            PlayerInSquad(
                code=int(row["code"]),
                element_type=int(row["element_type"]),
                buy_price=int(row["now_cost"]),
                sell_price=int(row["now_cost"]),
            )
            for _, row in sorted_players.iterrows()
        ]

        state = SquadState(
            players=initial_players,
            budget=1000 - sum(p.buy_price for p in initial_players),
            free_transfers=999,  # unlimited for GW1
            chips_available={ct: 2 for ct in ChipType},
            current_gameweek=first_gw,
        )

        # Let model pick initial squad via recommend with transfers
        actions = self.model.recommend(state, season_data)

        for action in actions:
            if isinstance(action, SetLineup):
                state.starting_xi = action.starting_xi
                state.bench_order = action.bench_order
            elif isinstance(action, SetCaptain):
                state.captain = action.player_id
            elif isinstance(action, SetViceCaptain):
                state.vice_captain = action.player_id

        # Reset free transfers for the actual season
        state.free_transfers = 1

        return state
```

**Step 4: Run tests to verify they pass**

```bash
/home/node/.local/bin/uv run pytest tests/simulation/test_engine.py -v
```

**Step 5: Commit**

```bash
git add src/fpl_model/simulation/engine.py tests/simulation/test_engine.py
git commit -m "feat: implement season simulator with full GW loop"
```

---

### Task 11: Evaluation Framework

**Files:**
- Create: `src/fpl_model/evaluation/metrics.py`
- Create: `src/fpl_model/evaluation/comparison.py`
- Create: `src/fpl_model/evaluation/reports.py`
- Create: `tests/evaluation/test_evaluation.py`

**Step 1: Write the failing test**

```python
# tests/evaluation/test_evaluation.py
import pytest
from fpl_model.simulation.engine import SimulationResult
from fpl_model.evaluation.metrics import compute_metrics
from fpl_model.evaluation.comparison import compare_results
from fpl_model.evaluation.reports import format_report


class TestMetrics:
    def test_compute_metrics(self):
        result = SimulationResult(
            total_points=200,
            gameweek_points={1: 60, 2: 70, 3: 70},
            actions_log={},
            transfer_costs={1: 0, 2: 4, 3: 0},
            budget_history={1: 100, 2: 95, 3: 90},
        )
        metrics = compute_metrics(result)
        assert metrics["total_points"] == 200
        assert metrics["num_gameweeks"] == 3
        assert metrics["avg_points_per_gw"] == pytest.approx(200 / 3, rel=0.01)
        assert metrics["total_transfer_cost"] == 4
        assert metrics["best_gw"] == (2, 70)  # first occurrence of max
        assert metrics["worst_gw"] == (1, 60)


class TestComparison:
    def test_compare_two_models(self):
        r1 = SimulationResult(
            total_points=200,
            gameweek_points={1: 60, 2: 70, 3: 70},
            actions_log={}, transfer_costs={1: 0, 2: 0, 3: 0},
            budget_history={},
        )
        r2 = SimulationResult(
            total_points=180,
            gameweek_points={1: 50, 2: 60, 3: 70},
            actions_log={}, transfer_costs={1: 0, 2: 4, 3: 0},
            budget_history={},
        )
        comparison = compare_results({"model_a": r1, "model_b": r2})
        assert comparison[0]["name"] == "model_a"  # sorted by points desc
        assert comparison[0]["total_points"] == 200


class TestReports:
    def test_format_report_returns_string(self):
        metrics = {
            "total_points": 200,
            "num_gameweeks": 3,
            "avg_points_per_gw": 66.7,
            "total_transfer_cost": 4,
            "best_gw": (2, 70),
            "worst_gw": (1, 60),
        }
        report = format_report("test_model", metrics)
        assert "test_model" in report
        assert "200" in report
```

**Step 2: Run tests to verify they fail**

```bash
/home/node/.local/bin/uv run pytest tests/evaluation/test_evaluation.py -v
```

**Step 3: Implement evaluation modules**

```python
# src/fpl_model/evaluation/metrics.py
"""Compute evaluation metrics from simulation results."""

from fpl_model.simulation.engine import SimulationResult


def compute_metrics(result: SimulationResult) -> dict:
    gw_points = result.gameweek_points
    num_gws = len(gw_points)

    best_gw = max(gw_points.items(), key=lambda x: x[1])
    worst_gw = min(gw_points.items(), key=lambda x: x[1])

    return {
        "total_points": result.total_points,
        "num_gameweeks": num_gws,
        "avg_points_per_gw": result.total_points / num_gws if num_gws > 0 else 0,
        "total_transfer_cost": sum(result.transfer_costs.values()),
        "best_gw": best_gw,
        "worst_gw": worst_gw,
        "points_std": _std(list(gw_points.values())),
    }


def _std(values: list[int | float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return variance ** 0.5
```

```python
# src/fpl_model/evaluation/comparison.py
"""Compare multiple model simulation results."""

from fpl_model.simulation.engine import SimulationResult
from fpl_model.evaluation.metrics import compute_metrics


def compare_results(
    results: dict[str, SimulationResult],
) -> list[dict]:
    """Compare multiple models, return sorted by total points descending."""
    rows = []
    for name, result in results.items():
        metrics = compute_metrics(result)
        rows.append({"name": name, **metrics})

    rows.sort(key=lambda r: r["total_points"], reverse=True)
    return rows
```

```python
# src/fpl_model/evaluation/reports.py
"""Format evaluation reports for display."""


def format_report(model_name: str, metrics: dict) -> str:
    lines = [
        f"=== {model_name} ===",
        f"Total Points: {metrics['total_points']}",
        f"Gameweeks: {metrics['num_gameweeks']}",
        f"Avg Points/GW: {metrics['avg_points_per_gw']:.1f}",
        f"Transfer Costs: {metrics['total_transfer_cost']}",
        f"Best GW: GW{metrics['best_gw'][0]} ({metrics['best_gw'][1]} pts)",
        f"Worst GW: GW{metrics['worst_gw'][0]} ({metrics['worst_gw'][1]} pts)",
    ]
    return "\n".join(lines)


def format_comparison(comparison: list[dict]) -> str:
    lines = ["=== Model Comparison ===", ""]
    header = f"{'Rank':<6}{'Model':<20}{'Points':<10}{'Avg/GW':<10}{'Hits':<8}"
    lines.append(header)
    lines.append("-" * len(header))
    for i, row in enumerate(comparison, 1):
        lines.append(
            f"{i:<6}{row['name']:<20}{row['total_points']:<10}"
            f"{row['avg_points_per_gw']:<10.1f}{row['total_transfer_cost']:<8}"
        )
    return "\n".join(lines)
```

**Step 4: Run tests to verify they pass**

```bash
/home/node/.local/bin/uv run pytest tests/evaluation/test_evaluation.py -v
```

**Step 5: Commit**

```bash
git add src/fpl_model/evaluation/metrics.py src/fpl_model/evaluation/comparison.py \
    src/fpl_model/evaluation/reports.py tests/evaluation/test_evaluation.py
git commit -m "feat: add evaluation framework with metrics, comparison, and reports"
```

---

### Task 12: CLI

**Files:**
- Create: `src/fpl_model/cli/main.py`
- Create: `tests/test_cli.py`

**Step 1: Write the failing test**

```python
# tests/test_cli.py
from click.testing import CliRunner
from fpl_model.cli.main import cli


class TestCLI:
    def test_cli_group_exists(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "FPL Model" in result.output

    def test_ingest_command_exists(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["ingest", "--help"])
        assert result.exit_code == 0
        assert "--seasons" in result.output

    def test_simulate_command_exists(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["simulate", "--help"])
        assert result.exit_code == 0

    def test_evaluate_command_exists(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["evaluate", "--help"])
        assert result.exit_code == 0
```

**Step 2: Run tests to verify they fail**

```bash
/home/node/.local/bin/uv run pytest tests/test_cli.py -v
```

**Step 3: Implement CLI**

```python
# src/fpl_model/cli/main.py
"""CLI entry points for fpl-model."""

import asyncio
from pathlib import Path

import click


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """FPL Model — ML pipeline for Fantasy Premier League recommendations."""
    pass


@cli.command()
@click.option("--seasons", type=str, help="Comma-separated seasons, e.g. '2022-23,2023-24'")
@click.option("--current", is_flag=True, help="Fetch current season from FPL API")
@click.option("--db-path", default="data/fpl.db", help="Path to SQLite database")
@click.option("--cache-dir", default="data/raw", help="Path to raw file cache")
def ingest(seasons, current, db_path, cache_dir):
    """Download and ingest FPL data into the database."""
    from fpl_model.data.ingest import Ingester

    ingester = Ingester(db_path=db_path, cache_dir=cache_dir)

    async def _run():
        if current:
            click.echo("Ingesting current season from FPL API...")
            await ingester.ingest_current(season="2025-26")
            click.echo("Done.")

        if seasons:
            season_list = [s.strip() for s in seasons.split(",")]
            click.echo(f"Ingesting seasons: {season_list}")
            await ingester.ingest_seasons(season_list)
            click.echo("Done.")

        if not current and not seasons:
            click.echo("Specify --seasons or --current. Use --help for details.")

    asyncio.run(_run())


@cli.command()
@click.argument("model_name")
@click.option("--season", required=True, help="Season to simulate, e.g. '2024-25'")
@click.option("--db-path", default="data/fpl.db", help="Path to SQLite database")
def simulate(model_name, season, db_path):
    """Simulate a model over a historical season."""
    from fpl_model.data.db import Database
    from fpl_model.models.registry import ModelRegistry
    from fpl_model.simulation.engine import SeasonSimulator
    from fpl_model.evaluation.metrics import compute_metrics
    from fpl_model.evaluation.reports import format_report

    db = Database(db_path)
    registry = ModelRegistry()
    # TODO: register models from discovered modules
    model = registry.get(model_name)
    sim = SeasonSimulator(model=model, season=season, db=db)
    result = sim.run()
    metrics = compute_metrics(result)
    click.echo(format_report(model_name, metrics))


@cli.command()
@click.argument("model_name")
@click.option("--seasons", required=True, help="Comma-separated seasons to evaluate")
@click.option("--db-path", default="data/fpl.db", help="Path to SQLite database")
def evaluate(model_name, seasons, db_path):
    """Evaluate a model across multiple seasons."""
    from fpl_model.data.db import Database
    from fpl_model.models.registry import ModelRegistry
    from fpl_model.simulation.engine import SeasonSimulator
    from fpl_model.evaluation.metrics import compute_metrics
    from fpl_model.evaluation.reports import format_report

    db = Database(db_path)
    registry = ModelRegistry()
    model = registry.get(model_name)

    season_list = [s.strip() for s in seasons.split(",")]
    for season in season_list:
        sim = SeasonSimulator(model=model, season=season, db=db)
        result = sim.run()
        metrics = compute_metrics(result)
        click.echo(format_report(f"{model_name} ({season})", metrics))
        click.echo()


@cli.command()
@click.argument("model_names", nargs=-1, required=True)
@click.option("--seasons", required=True, help="Comma-separated seasons to compare on")
@click.option("--db-path", default="data/fpl.db", help="Path to SQLite database")
def compare(model_names, seasons, db_path):
    """Compare multiple models on the same season(s)."""
    from fpl_model.data.db import Database
    from fpl_model.models.registry import ModelRegistry
    from fpl_model.simulation.engine import SeasonSimulator
    from fpl_model.evaluation.comparison import compare_results
    from fpl_model.evaluation.reports import format_comparison

    db = Database(db_path)
    registry = ModelRegistry()
    season_list = [s.strip() for s in seasons.split(",")]

    all_results = {}
    for name in model_names:
        model = registry.get(name)
        for season in season_list:
            sim = SeasonSimulator(model=model, season=season, db=db)
            result = sim.run()
            all_results[f"{name} ({season})"] = result

    comparison = compare_results(all_results)
    click.echo(format_comparison(comparison))
```

**Step 4: Run tests to verify they pass**

```bash
/home/node/.local/bin/uv run pytest tests/test_cli.py -v
```

**Step 5: Commit**

```bash
git add src/fpl_model/cli/main.py tests/test_cli.py
git commit -m "feat: add CLI with ingest, simulate, evaluate, and compare commands"
```

---

### Task 13: Integration Test — End-to-End Pipeline

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write the integration test**

```python
# tests/test_integration.py
"""End-to-end integration test: ingest mock data, run simulation, evaluate."""

import pandas as pd
import pytest
from fpl_model.data.db import Database
from fpl_model.models.base import ActionModel, SeasonData
from fpl_model.simulation.actions import SetCaptain, SetViceCaptain, SetLineup
from fpl_model.simulation.state import SquadState
from fpl_model.simulation.engine import SeasonSimulator
from fpl_model.evaluation.metrics import compute_metrics
from fpl_model.evaluation.comparison import compare_results


class AlwaysFirstModel(ActionModel):
    """Picks the first valid squad alphabetically, captains player with most points."""
    def recommend(self, state, data):
        xi = [p.code for p in state.players[:11]]
        bench = [p.code for p in state.players[11:]]
        return [
            SetLineup(starting_xi=xi, bench_order=bench),
            SetCaptain(player_id=state.players[7].code),
            SetViceCaptain(player_id=state.players[8].code),
        ]


class TestEndToEnd:
    def _populate_db(self, db: Database, num_gws: int = 5):
        players = []
        for i in range(1, 16):
            et = 1 if i <= 2 else (2 if i <= 7 else (3 if i <= 12 else 4))
            players.append({
                "season": "2023-24", "code": i,
                "first_name": f"F{i}", "second_name": f"L{i}", "web_name": f"P{i}",
                "element_type": et, "team_code": (i % 20) + 1, "now_cost": 50,
            })
        db.write("players", pd.DataFrame(players))

        gw_rows = []
        for gw in range(1, num_gws + 1):
            for i in range(1, 16):
                gw_rows.append({
                    "season": "2023-24", "player_code": i, "gameweek": gw,
                    "total_points": i, "minutes": 90, "goals_scored": 0,
                    "assists": 0, "value": 50, "was_home": True, "opponent_team": 1,
                })
        db.write("gameweek_performances", pd.DataFrame(gw_rows))

        fixtures = [
            {"season": "2023-24", "fixture_id": gw, "gameweek": gw,
             "team_h": 1, "team_a": 2, "finished": True}
            for gw in range(1, num_gws + 1)
        ]
        db.write("fixtures", pd.DataFrame(fixtures))

        teams = [
            {"season": "2023-24", "team_code": i, "name": f"Team{i}"}
            for i in range(1, 21)
        ]
        db.write("teams", pd.DataFrame(teams))

    def test_full_pipeline(self, tmp_path):
        # 1. Set up DB with data
        db = Database(tmp_path / "test.db")
        db.create_tables()
        self._populate_db(db, num_gws=5)

        # 2. Run simulation
        model = AlwaysFirstModel()
        sim = SeasonSimulator(model=model, season="2023-24", db=db)
        result = sim.run()

        # 3. Evaluate
        metrics = compute_metrics(result)
        assert metrics["total_points"] > 0
        assert metrics["num_gameweeks"] == 5

    def test_compare_two_models(self, tmp_path):
        db = Database(tmp_path / "test.db")
        db.create_tables()
        self._populate_db(db, num_gws=3)

        model_a = AlwaysFirstModel()
        model_b = AlwaysFirstModel()  # same model, just testing comparison infra

        sim_a = SeasonSimulator(model=model_a, season="2023-24", db=db)
        sim_b = SeasonSimulator(model=model_b, season="2023-24", db=db)

        result_a = sim_a.run()
        result_b = sim_b.run()

        comparison = compare_results({"model_a": result_a, "model_b": result_b})
        assert len(comparison) == 2
        assert comparison[0]["total_points"] == comparison[1]["total_points"]
```

**Step 2: Run the integration test**

```bash
/home/node/.local/bin/uv run pytest tests/test_integration.py -v
```

**Step 3: Run all tests**

```bash
/home/node/.local/bin/uv run pytest -v
```

**Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add end-to-end integration test for full pipeline"
```

---

### Task 14: Lint and Final Cleanup

**Step 1: Run ruff**

```bash
/home/node/.local/bin/uv run ruff check src/ tests/ --fix
```

**Step 2: Run full test suite**

```bash
/home/node/.local/bin/uv run pytest -v
```

**Step 3: Fix any issues, then commit**

```bash
git add -A
git commit -m "chore: lint fixes and cleanup"
```

---

## Task Dependency Graph

```
Task 1 (Scaffolding)
  ├── Task 2 (DB & Schemas)
  │     └── Task 7 (Ingest Orchestrator)
  ├── Task 3 (Cache)
  │     ├── Task 4 (Vaastav Source)
  │     └── Task 5 (FPL API Source)
  ├── Task 6 (ETL Transformers) ← depends on Task 2
  │     └── Task 7 (Ingest Orchestrator) ← depends on Tasks 4, 5, 6
  ├── Task 8 (Model Abstractions & Actions)
  │     └── Task 9 (Rules) ← depends on Task 8
  │           └── Task 10 (Simulator) ← depends on Tasks 2, 8, 9
  └── Task 11 (Evaluation) ← depends on Task 10
        └── Task 12 (CLI) ← depends on Tasks 7, 10, 11
              └── Task 13 (Integration Test) ← depends on all
                    └── Task 14 (Lint & Cleanup)
```

**Parallelizable groups:**
- After Task 1: Tasks 2, 3, 8 can run in parallel
- After Task 2+3: Tasks 4, 5, 6 can run in parallel
- After Task 8: Task 9 can start
