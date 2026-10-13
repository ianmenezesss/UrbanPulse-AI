"""Data loading and filtering helpers for the local UrbanPulse dashboard."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "hour.parquet"
REQUIRED_COLUMNS = ("datetime", "yr", "workingday", "hr", "weekday", "mnth", "temp", "cnt")
WEEKDAY_NAMES_PT = {
    0: "Segunda-feira",
    1: "Terça-feira",
    2: "Quarta-feira",
    3: "Quinta-feira",
    4: "Sexta-feira",
    5: "Sábado",
    6: "Domingo",
}


def load_dashboard_data(path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Read the prepared Parquet and check columns needed by the dashboard."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Processed dataset not found: {path}")

    frame = pd.read_parquet(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Processed dataset is missing required columns: {', '.join(missing)}")

    if not pd.api.types.is_datetime64_any_dtype(frame["datetime"]):
        frame = frame.copy()
        frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    if frame["datetime"].isna().any():
        raise ValueError("Processed dataset contains missing or invalid datetime values.")
    return frame


def filter_dashboard_data(
    frame: pd.DataFrame,
    *,
    year: int | None = None,
    workingday: bool | None = None,
) -> pd.DataFrame:
    """Filter rows by calendar year and UCI's workingday flag.

    ``workingday=0`` represents both weekends and holidays in this dataset.
    Passing ``None`` for either filter leaves that dimension unrestricted.
    """
    if year not in (None, 2011, 2012):
        raise ValueError("year must be 2011, 2012, or None")

    mask = pd.Series(True, index=frame.index)
    if year is not None:
        mask &= frame["datetime"].dt.year.eq(year)
    if workingday is not None:
        mask &= frame["workingday"].eq(int(workingday))
    return frame.loc[mask].copy()
