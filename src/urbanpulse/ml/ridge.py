"""Train and evaluate a Ridge regression for hourly bike demand."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from urbanpulse.ml.baseline import (
    DEFAULT_DATA_PATH,
    TARGET,
    RegressionMetrics,
    calculate_metrics,
    load_processed_data,
    temporal_train_test_split,
)

CATEGORICAL_FEATURES = (
    "season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit",
)
NUMERIC_FEATURES = ("temp", "atemp", "hum", "windspeed")
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
FORBIDDEN_FEATURES = frozenset({"instant", "casual", "registered", TARGET})
RIDGE_ALPHA = 1.0


@dataclass
class RidgeEvaluation:
    train_rows: int
    test_rows: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    predictions: np.ndarray
    metrics: RegressionMetrics


def select_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return only the declared calendar and weather predictors."""
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Input data is missing required feature columns: {', '.join(missing)}")
    return frame.loc[:, FEATURE_COLUMNS].copy()


def build_ridge_pipeline(alpha: float = RIDGE_ALPHA) -> Pipeline:
    """Create preprocessing and Ridge steps; preprocessing fits on train only."""
    if alpha <= 0:
        raise ValueError("alpha must be greater than zero")
    preprocessing = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), list(CATEGORICAL_FEATURES)),
            ("numeric", StandardScaler(), list(NUMERIC_FEATURES)),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessing),
            ("regressor", Ridge(alpha=alpha)),
        ]
    )


def fit_predict_ridge(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    alpha: float = RIDGE_ALPHA,
) -> np.ndarray:
    """Fit on training rows and return predictions for test features only."""
    if TARGET not in train.columns:
        raise ValueError(f"Training data is missing target column: {TARGET}")
    if train.empty:
        raise ValueError("Training data must not be empty")
    x_train = select_features(train)
    x_test = select_features(test)
    model = build_ridge_pipeline(alpha=alpha)
    model.fit(x_train, train[TARGET])
    return np.asarray(model.predict(x_test), dtype=float)


def evaluate_ridge(
    frame: pd.DataFrame,
    train_fraction: float = 0.8,
    *,
    alpha: float = RIDGE_ALPHA,
) -> RidgeEvaluation:
    """Evaluate Ridge on the same chronological split used by the baselines."""
    train, test = temporal_train_test_split(frame, train_fraction=train_fraction)
    predictions = fit_predict_ridge(train, test, alpha=alpha)
    metrics = calculate_metrics(test[TARGET], pd.Series(predictions, index=test.index))
    return RidgeEvaluation(
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
    parser = argparse.ArgumentParser(description="Evaluate Ridge regression for hourly bike demand.")
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_PATH, help="Prepared Parquet path.")
    args = parser.parse_args(argv)
    try:
        result = evaluate_ridge(load_processed_data(args.input))
    except (OSError, ValueError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")

    print(f"Training rows: {result.train_rows:,} ({result.train_start} — {result.train_end})")
    print(f"Test rows: {result.test_rows:,} ({result.test_start} — {result.test_end})")
    print(f"Ridge (alpha={RIDGE_ALPHA:g}): MAE={result.metrics.mae:.4f}, "
          f"RMSE={result.metrics.rmse:.4f}, R²={result.metrics.r2:.4f}")
    print(
        "Weather-feature assumption: weather forecasts for the target hour are available "
        "at prediction time."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
