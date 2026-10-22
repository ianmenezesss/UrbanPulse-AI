"""Train and evaluate a random forest for hourly bike demand."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from urbanpulse.ml.baseline import (
    DEFAULT_DATA_PATH,
    TARGET,
    RegressionMetrics,
    calculate_metrics,
    load_processed_data,
    temporal_train_test_split,
)
from urbanpulse.ml.ridge import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    select_features,
)

RANDOM_STATE = 42
N_ESTIMATORS = 100
MIN_SAMPLES_LEAF = 2
N_JOBS = -1


@dataclass
class RandomForestEvaluation:
    train_rows: int
    test_rows: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    predictions: np.ndarray
    metrics: RegressionMetrics


def build_random_forest_pipeline() -> Pipeline:
    """Build a train-fitted encoder and a fixed-parameter random forest.

    Numeric variables are passed through without scaling: tree splits do not
    depend on feature scales. One-hot encoding avoids treating category codes
    as ordered distances, and is fitted as part of the training pipeline.
    """
    preprocessing = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), list(CATEGORICAL_FEATURES)),
            ("numeric", "passthrough", list(NUMERIC_FEATURES)),
        ],
        remainder="drop",
    )
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )
    return Pipeline(steps=[("preprocessor", preprocessing), ("regressor", model)])


def fit_predict_random_forest(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Fit only on the training partition and predict using allowed features."""
    if TARGET not in train.columns:
        raise ValueError(f"Training data is missing target column: {TARGET}")
    if train.empty:
        raise ValueError("Training data must not be empty")
    x_train = select_features(train)
    x_test = select_features(test)
    model = build_random_forest_pipeline()
    model.fit(x_train, train[TARGET])
    return np.asarray(model.predict(x_test), dtype=float)


def evaluate_random_forest(
    frame: pd.DataFrame,
    train_fraction: float = 0.8,
) -> RandomForestEvaluation:
    """Evaluate on the baseline/Ridge chronological holdout, without shuffling."""
    train, test = temporal_train_test_split(frame, train_fraction=train_fraction)
    predictions = fit_predict_random_forest(train, test)
    metrics = calculate_metrics(test[TARGET], pd.Series(predictions, index=test.index))
    return RandomForestEvaluation(
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
    parser = argparse.ArgumentParser(description="Evaluate a random forest for hourly bike demand.")
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_PATH, help="Prepared Parquet path.")
    args = parser.parse_args(argv)
    try:
        result = evaluate_random_forest(load_processed_data(args.input))
    except (OSError, ValueError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")

    print(f"Training rows: {result.train_rows:,} ({result.train_start} — {result.train_end})")
    print(f"Test rows: {result.test_rows:,} ({result.test_start} — {result.test_end})")
    print(
        "Random Forest (n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=-1): "
        f"MAE={result.metrics.mae:.4f}, RMSE={result.metrics.rmse:.4f}, "
        f"R²={result.metrics.r2:.4f}"
    )
    print(
        "Weather-feature assumption: weather forecasts for the target hour are available "
        "at prediction time."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
