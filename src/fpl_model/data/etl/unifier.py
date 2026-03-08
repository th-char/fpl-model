"""Unify DataFrames to canonical schemas."""

import pandas as pd

from fpl_model.data.etl.schemas import TABLES


def unify_to_schema(df: pd.DataFrame, table: str) -> pd.DataFrame:
    """Align a DataFrame to a canonical schema: add missing cols as NA, drop extras."""
    if table not in TABLES:
        raise ValueError(f"Unknown table: {table}")
    schema_cols = list(TABLES[table].keys())
    result = df.copy()
    for col in schema_cols:
        if col not in result.columns:
            result[col] = pd.NA
    return result[schema_cols].copy()
