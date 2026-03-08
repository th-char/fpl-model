"""SQLite database interface for FPL data."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from fpl_model.data.etl.schemas import TABLES


class Database:
    """Single interface for reading and writing FPL data to SQLite."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def create_tables(self) -> None:
        """Create all tables defined in the canonical schema."""
        from fpl_model.data.etl.schemas import TABLE_KEYS

        conn = self._connect()
        try:
            for table_name, columns in TABLES.items():
                col_defs = ", ".join(f"{col} {dtype}" for col, dtype in columns.items())
                keys = TABLE_KEYS.get(table_name, [])
                if keys:
                    col_defs += f", PRIMARY KEY ({', '.join(keys)})"
                conn.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({col_defs})")
            conn.commit()
        finally:
            conn.close()

    def write(self, table: str, df: pd.DataFrame) -> None:
        """Append a DataFrame to a table, keeping only columns in the schema."""
        if table not in TABLES:
            raise ValueError(f"Unknown table: {table}")
        schema_cols = list(TABLES[table].keys())
        cols_to_write = [c for c in schema_cols if c in df.columns]
        conn = self._connect()
        try:
            # SQLite has a max of 999 SQL variables; chunk to stay under that limit.
            num_cols = len(cols_to_write)
            chunksize = max(1, 999 // num_cols) if num_cols > 0 else None
            df[cols_to_write].to_sql(
                table, conn, if_exists="append", index=False,
                method="multi", chunksize=chunksize,
            )
        finally:
            conn.close()

    def read(self, table: str, where: dict[str, object] | None = None) -> pd.DataFrame:
        """Read a table into a DataFrame, with an optional parameterized WHERE clause."""
        if table not in TABLES:
            raise ValueError(f"Unknown table: {table}")
        sql = f"SELECT * FROM {table}"
        params: list[object] = []
        if where:
            clauses = [f"{col} = ?" for col in where]
            sql += " WHERE " + " AND ".join(clauses)
            params = list(where.values())
        conn = self._connect()
        try:
            return pd.read_sql_query(sql, conn, params=params)
        finally:
            conn.close()

    def query(self, sql: str) -> pd.DataFrame:
        """Execute a raw SQL query and return results as a DataFrame."""
        conn = self._connect()
        try:
            return pd.read_sql_query(sql, conn)
        finally:
            conn.close()

    def clear_table(self, table: str, where: dict[str, object] | None = None) -> None:
        """Delete rows from a table, optionally filtered by a parameterized WHERE clause."""
        if table not in TABLES:
            raise ValueError(f"Unknown table: {table}")
        sql = f"DELETE FROM {table}"
        params: list[object] = []
        if where:
            clauses = [f"{col} = ?" for col in where]
            sql += " WHERE " + " AND ".join(clauses)
            params = list(where.values())
        conn = self._connect()
        try:
            conn.execute(sql, params)
            conn.commit()
        finally:
            conn.close()
