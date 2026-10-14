"""Chronological evaluation of simple hourly bike-demand baselines."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DEFAULT_DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "processed" / "hour.parquet"
TARGET = "cnt"


@dataclass(frozen=True)
class RegressionMetrics:
    mae: float
    rmse: float
    r2: float


@dataclass
class BaselineEvaluation:
    train_rows: int
    test_rows: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    predictions: pd.DataFrame
    metrics: dict[str, RegressionMetrics]


def load_processed_data(path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the prepared Parquet needed for baseline evaluation."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Processed dataset not found: {path}")
    frame = pd.read_parquet(path)
    missing = [column for column in ("datetime", TARGET) if column not in frame.columns]
    if missing:
        raise ValueError(f"Processed dataset is missing required columns: {', '.join(missing)}")
    return frame


def temporal_train_test_split(
    frame: pd.DataFrame,
    train_fraction: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sort chronologically and split by row count without shuffling."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be strictly between 0 and 1")
    missing = [column for column in ("datetime", TARGET) if column not in frame.columns]
    if missing:
        raise ValueError(f"Input data is missing required columns: {', '.join(missing)}")

    ordered = frame.copy()
    if not pd.api.types.is_datetime64_any_dtype(ordered["datetime"]):
        ordered["datetime"] = pd.to_datetime(ordered["datetime"], errors="coerce")
    if ordered["datetime"].isna().any():
        raise ValueError("datetime contains missing or invalid values")
    ordered = ordered.sort_values("datetime", kind="stable").reset_index(drop=True)

    split_index = int(len(ordered) * train_fraction)
    if split_index < 1 or len(ordered) - split_index < 2:
        raise ValueError("The dataset must provide at least one training row and two test rows")
    return ordered.iloc[:split_index].copy(), ordered.iloc[split_index:].copy()


def predict_training_baselines(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:
    """Predict from training targets only: one global mean and one mean per hour.

    If a test hour is absent from training, its prediction falls back to the
    global training mean. ``casual``, ``registered`` and test ``cnt`` are never
    consulted when generating predictions.
    """
    required = ("datetime", TARGET)
    missing_train = [column for column in required if column not in train.columns]
    missing_test = ["datetime"] if "datetime" not in test.columns else []
    if missing_train or missing_test:
        missing = missing_train + missing_test
        raise ValueError(f"Data is missing required columns: {', '.join(missing)}")
    if train.empty:
        raise ValueError("Training data must not be empty")

    global_mean = float(train[TARGET].mean())
    hourly_means = train.groupby(train["datetime"].dt.hour)[TARGET].mean()
    hourly_predictions = test["datetime"].dt.hour.map(hourly_means).fillna(global_mean)
    return pd.DataFrame(
        {
            "training_mean": global_mean,
            "training_hour_mean": hourly_predictions.to_numpy(dtype=float),
        },
        index=test.index,
    )


def calculate_metrics(actual: pd.Series, predicted: pd.Series) -> RegressionMetrics:
    """Calculate MAE, RMSE, and R² for one set of numeric predictions."""
    if len(actual) != len(predicted) or len(actual) < 2:
        raise ValueError("actual and predicted must have equal lengths and at least two observations")
    return RegressionMetrics(
        mae=float(mean_absolute_error(actual, predicted)),
        rmse=float(mean_squared_error(actual, predicted) ** 0.5),
        r2=float(r2_score(actual, predicted)),
    )


def evaluate_baselines(
    frame: pd.DataFrame,
    train_fraction: float = 0.8,
) -> BaselineEvaluation:
    """Run both baselines on a chronological holdout and report their metrics."""
    train, test = temporal_train_test_split(frame, train_fraction=train_fraction)
    predictions = predict_training_baselines(train, test)
    metrics = {
        name: calculate_metrics(test[TARGET], predictions[name])
        for name in predictions.columns
    }
    return BaselineEvaluation(
        train_rows=len(train),
        test_rows=len(test),
        train_start=pd.Timestamp(train["datetime"].iloc[0]),
        train_end=pd.Timestamp(train["datetime"].iloc[-1]),
        test_start=pd.Timestamp(test["datetime"].iloc[0]),
        test_end=pd.Timestamp(test["datetime"].iloc[-1]),
        predictions=predictions,
        metrics=metrics,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate chronological hourly-demand baselines.")
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_PATH, help="Prepared Parquet path.")
    args = parser.parse_args(argv)
    try:
        result = evaluate_baselines(load_processed_data(args.input))
    except (OSError, ValueError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")

    print(f"Training rows: {result.train_rows:,} ({result.train_start} — {result.train_end})")
    print(f"Test rows: {result.test_rows:,} ({result.test_start} — {result.test_end})")
    print("Metrics (test set):")
    for name, metrics in result.metrics.items():
        print(f"  {name}: MAE={metrics.mae:.4f}, RMSE={metrics.rmse:.4f}, R²={metrics.r2:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
