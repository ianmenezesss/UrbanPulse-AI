"""Initial, read-only validation for the UCI hourly bike-sharing dataset."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import pandas as pd

DEFAULT_SOURCE = Path(__file__).resolve().parents[2] / "data" / "raw" / "hour.csv"
EXPECTED_COLUMNS = (
    "instant", "dteday", "season", "yr", "mnth", "hr", "holiday", "weekday",
    "workingday", "weathersit", "temp", "atemp", "hum", "windspeed", "casual",
    "registered", "cnt",
)
INTEGER_COLUMNS = (
    "instant", "season", "yr", "mnth", "hr", "holiday", "weekday", "workingday",
    "weathersit", "casual", "registered", "cnt",
)
FLOAT_COLUMNS = ("temp", "atemp", "hum", "windspeed")
DOMAINS: dict[str, tuple[float, float]] = {
    "instant": (1, float("inf")),
    "season": (1, 4),
    "yr": (0, 1),
    "mnth": (1, 12),
    "hr": (0, 23),
    "holiday": (0, 1),
    "weekday": (0, 6),
    "workingday": (0, 1),
    "weathersit": (1, 4),
    "temp": (0, 1),
    "atemp": (0, 1),
    "hum": (0, 1),
    "windspeed": (0, 1),
    "casual": (0, float("inf")),
    "registered": (0, float("inf")),
    "cnt": (0, float("inf")),
}
LOGGER = logging.getLogger(__name__)


class DataValidationError(RuntimeError):
    """Raised when the source cannot be read for validation."""


@dataclass
class ValidationReport:
    """Summary of schema, type, completeness, duplicate, and domain checks."""

    source: Path
    row_count: int
    column_count: int
    dtypes: dict[str, str] = field(default_factory=dict)
    missing_values: dict[str, int] = field(default_factory=dict)
    duplicate_rows: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors


def _check_numeric_column(
    frame: pd.DataFrame,
    column: str,
    report: ValidationReport,
    *,
    integer: bool,
) -> pd.Series | None:
    if column not in frame.columns:
        return None
    source = frame[column]
    converted = pd.to_numeric(source, errors="coerce")
    unparseable = source.notna() & converted.isna()
    if unparseable.any():
        report.errors.append(f"Column '{column}' contains non-numeric values ({int(unparseable.sum())} rows).")
        return None
    if integer:
        fractional = converted.notna() & (converted % 1 != 0)
        if fractional.any():
            report.errors.append(f"Column '{column}' must contain integers ({int(fractional.sum())} fractional values).")
    return converted


def validate_hour_csv(source: Path = DEFAULT_SOURCE) -> ValidationReport:
    """Read and validate the hourly CSV without modifying it."""
    source = Path(source)
    try:
        frame = pd.read_csv(source, low_memory=False)
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise DataValidationError(f"Could not read input CSV '{source}': {exc}") from exc

    report = ValidationReport(
        source=source,
        row_count=len(frame),
        column_count=len(frame.columns),
        dtypes={str(name): str(dtype) for name, dtype in frame.dtypes.items()},
    )
    if frame.empty:
        report.errors.append("Dataset contains no data rows.")

    missing_columns = [name for name in EXPECTED_COLUMNS if name not in frame.columns]
    extra_columns = [str(name) for name in frame.columns if name not in EXPECTED_COLUMNS]
    if missing_columns:
        report.errors.append(f"Missing required columns: {', '.join(missing_columns)}.")
    if extra_columns:
        report.warnings.append(f"Unexpected columns are present: {', '.join(extra_columns)}.")

    report.missing_values = {
        str(name): int(count)
        for name, count in frame.isna().sum().items()
        if count > 0
    }
    if report.missing_values:
        details = ", ".join(f"{name}={count}" for name, count in report.missing_values.items())
        report.errors.append(f"Missing values found: {details}.")

    report.duplicate_rows = int(frame.duplicated().sum())
    if report.duplicate_rows:
        report.warnings.append(f"Found {report.duplicate_rows} fully duplicated row(s); review before downstream use.")

    parsed: dict[str, pd.Series] = {}
    for column in INTEGER_COLUMNS:
        values = _check_numeric_column(frame, column, report, integer=True)
        if values is not None:
            parsed[column] = values
    for column in FLOAT_COLUMNS:
        values = _check_numeric_column(frame, column, report, integer=False)
        if values is not None:
            parsed[column] = values

    for column, (minimum, maximum) in DOMAINS.items():
        values = parsed.get(column)
        if values is None:
            continue
        invalid = values.notna() & ((values < minimum) | (values > maximum))
        if invalid.any():
            report.errors.append(
                f"Column '{column}' has {int(invalid.sum())} value(s) outside the allowed range [{minimum}, {maximum}]."
            )

    if "instant" in parsed:
        duplicate_ids = parsed["instant"].dropna().duplicated()
        if duplicate_ids.any():
            report.errors.append(f"Column 'instant' contains {int(duplicate_ids.sum())} duplicate identifier(s).")

    dates: pd.Series | None = None
    if "dteday" in frame.columns:
        dates = pd.to_datetime(frame["dteday"], format="%Y-%m-%d", errors="coerce")
        invalid_dates = frame["dteday"].notna() & dates.isna()
        if invalid_dates.any():
            report.errors.append(f"Column 'dteday' contains {int(invalid_dates.sum())} invalid date value(s); expected YYYY-MM-DD.")

    if dates is not None and "mnth" in parsed:
        mismatch = dates.notna() & parsed["mnth"].notna() & (dates.dt.month != parsed["mnth"])
        if mismatch.any():
            report.errors.append(f"Column 'mnth' disagrees with 'dteday' in {int(mismatch.sum())} row(s).")
    if dates is not None and "yr" in parsed:
        expected_year = parsed["yr"] + 2011
        mismatch = dates.notna() & expected_year.notna() & (dates.dt.year != expected_year)
        if mismatch.any():
            report.errors.append(f"Column 'yr' disagrees with 'dteday' in {int(mismatch.sum())} row(s).")

    if all(column in parsed for column in ("casual", "registered", "cnt")):
        mismatch = (
            parsed["casual"].notna()
            & parsed["registered"].notna()
            & parsed["cnt"].notna()
            & (parsed["casual"] + parsed["registered"] != parsed["cnt"])
        )
        if mismatch.any():
            report.errors.append(f"Column 'cnt' differs from casual + registered in {int(mismatch.sum())} row(s).")

    return report


def format_report(report: ValidationReport) -> str:
    lines = [
        f"Source: {report.source}",
        f"Rows: {report.row_count}",
        f"Columns: {report.column_count}",
        "Data types: " + (", ".join(f"{name}={dtype}" for name, dtype in report.dtypes.items()) or "none"),
        "Missing values: " + (", ".join(f"{name}={count}" for name, count in report.missing_values.items()) or "none"),
        f"Fully duplicated rows: {report.duplicate_rows}",
    ]
    lines.extend(f"ERROR: {message}" for message in report.errors)
    lines.extend(f"WARNING: {message}" for message in report.warnings)
    lines.append("Validation: " + ("PASSED" if report.is_valid else "FAILED"))
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the UCI hourly Bike Sharing CSV (read-only).")
    parser.add_argument("--input", type=Path, default=DEFAULT_SOURCE, help="CSV path (default: data/raw/hour.csv).")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        report = validate_hour_csv(args.input)
    except DataValidationError as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    print(format_report(report))
    LOGGER.info("Validation finished for %s", report.source)
    return 0 if report.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())


