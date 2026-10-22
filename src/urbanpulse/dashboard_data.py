"""Data loading and filtering helpers for the local UrbanPulse dashboard."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

DEFAULT_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "hour.parquet"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FINAL_METRICS_PATH = PROJECT_ROOT / "reports" / "model_comparison" / "final_test_metrics.csv"
DEFAULT_TEMPORAL_REPORT_DIR = PROJECT_ROOT / "reports" / "temporal_validation"
REQUIRED_COLUMNS = ("datetime", "yr", "workingday", "hr", "weekday", "mnth", "temp", "cnt")
FINAL_MODEL_NAMES = frozenset(
    {"training_mean", "training_hour_mean", "ridge", "random_forest"}
)
TEMPORAL_MODEL_NAMES = frozenset({"Ridge", "Random Forest"})
WEEKDAY_NAMES_PT = {
    0: "Segunda-feira",
    1: "Terça-feira",
    2: "Quarta-feira",
    3: "Quinta-feira",
    4: "Sexta-feira",
    5: "Sábado",
    6: "Domingo",
}
FINAL_METRIC_COLUMNS = (
    "model", "evaluation_split", "train_rows", "test_rows",
    "test_start", "test_end", "mae", "rmse", "r2",
)
TEMPORAL_REPORT_FILES = {
    "window_metrics": "window_metrics.csv",
    "summary_metrics": "summary_metrics.csv",
    "predictions": "out_of_sample_predictions.csv",
    "errors_by_hour": "error_by_hour.csv",
    "errors_by_weekday": "error_by_weekday.csv",
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


def load_final_model_metrics(
    path: Path = DEFAULT_FINAL_METRICS_PATH,
) -> pd.DataFrame:
    """Load and validate metrics for the reserved final 20% test set."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Final test metrics report not found: {path}. "
            "Generate it with urbanpulse-model-comparison."
        )
    frame = pd.read_csv(path)
    missing = [column for column in FINAL_METRIC_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Final metrics report is missing columns: {', '.join(missing)}")
    if set(frame["model"]) != FINAL_MODEL_NAMES or frame["model"].duplicated().any():
        raise ValueError("Final metrics report must contain each of the four supported models exactly once.")
    if set(frame["evaluation_split"]) != {"final_test_20pct"}:
        raise ValueError("Final metrics report must contain only the reserved final 20% test set.")
    for column in ("mae", "rmse", "r2"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[column].isna().any():
            raise ValueError(f"Final metrics report contains invalid values in '{column}'.")
    if frame[["train_rows", "test_rows", "test_start", "test_end"]].nunique().max() != 1:
        raise ValueError("Final metrics rows do not describe the same train/test split.")
    return frame


def load_temporal_validation_reports(
    report_dir: Path = DEFAULT_TEMPORAL_REPORT_DIR,
) -> dict[str, pd.DataFrame]:
    """Load temporal CV tables, keeping them separate from the final test report."""
    report_dir = Path(report_dir)
    reports: dict[str, pd.DataFrame] = {}
    for key, filename in TEMPORAL_REPORT_FILES.items():
        path = report_dir / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"Temporal validation report not found: {path}. "
                "Generate the reports with urbanpulse-temporal-validation."
            )
        reports[key] = pd.read_csv(path)

    window_metrics = reports["window_metrics"]
    required_metrics = {"window", "model", "mae", "rmse", "r2"}
    if not required_metrics.issubset(window_metrics.columns):
        raise ValueError("Temporal window metrics report is missing required columns.")
    if set(window_metrics["model"]) != TEMPORAL_MODEL_NAMES:
        raise ValueError("Temporal window metrics must contain Ridge and Random Forest only.")
    windows_by_model = window_metrics.groupby("model")["window"].agg(list)
    if any(len(windows) != 5 or sorted(windows) != [1, 2, 3, 4, 5] for windows in windows_by_model):
        raise ValueError("Temporal report must contain five validation windows per model.")
    if not {"model", "mae_mean", "mae_std", "rmse_mean", "rmse_std", "r2_mean", "r2_std"}.issubset(
        reports["summary_metrics"].columns
    ):
        raise ValueError("Temporal summary report is missing required columns.")
    summary = reports["summary_metrics"]
    if set(summary["model"]) != TEMPORAL_MODEL_NAMES or summary["model"].duplicated().any():
        raise ValueError("Temporal summary must contain Ridge and Random Forest exactly once.")
    expected_summary = window_metrics.groupby("model")[["mae", "rmse", "r2"]].agg(["mean", "std"])
    for _, summary_row in summary.iterrows():
        model = summary_row["model"]
        for metric in ("mae", "rmse", "r2"):
            for statistic in ("mean", "std"):
                value = pd.to_numeric(summary_row[f"{metric}_{statistic}"], errors="coerce")
                expected = expected_summary.loc[model, (metric, statistic)]
                if pd.isna(value) or not math.isclose(
                    float(value), float(expected), rel_tol=1e-5, abs_tol=1e-5
                ):
                    raise ValueError(
                        f"Temporal summary {metric}_{statistic} for {model} "
                        "does not match the per-window metrics."
                    )
    prediction_columns = {"window", "datetime", "model", "actual", "predicted", "error"}
    if not prediction_columns.issubset(reports["predictions"].columns):
        raise ValueError("Temporal out-of-sample predictions report is missing required columns.")
    if set(reports["predictions"]["model"]) != TEMPORAL_MODEL_NAMES:
        raise ValueError("Temporal predictions must contain Ridge and Random Forest only.")
    if not {"model", "hour", "mae"}.issubset(reports["errors_by_hour"].columns):
        raise ValueError("Temporal hourly error report is missing required columns.")
    if not {"model", "weekday", "weekday_name", "mae"}.issubset(
        reports["errors_by_weekday"].columns
    ):
        raise ValueError("Temporal weekday error report is missing required columns.")
    for key, category, expected_values in (
        ("errors_by_hour", "hour", set(range(24))),
        ("errors_by_weekday", "weekday", set(range(7))),
    ):
        table = reports[key]
        if (
            set(table["model"]) != TEMPORAL_MODEL_NAMES
            or table.duplicated(["model", category]).any()
            or any(set(group[category]) != expected_values for _, group in table.groupby("model"))
        ):
            raise ValueError(f"Temporal {category} errors must cover each category once for both models.")
    return reports
