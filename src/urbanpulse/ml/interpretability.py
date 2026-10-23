"""Temporal, out-of-sample permutation importance for Random Forest."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from urbanpulse.ml.baseline import DEFAULT_DATA_PATH, TARGET, load_processed_data, temporal_train_test_split
from urbanpulse.ml.random_forest import build_random_forest_pipeline, select_features
from urbanpulse.ml.temporal_validation import expanding_windows

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_IMPORTANCE_REPORT = PROJECT_ROOT / "reports" / "interpretability" / "permutation_importance.csv"
N_REPEATS = 5
RANDOM_STATE = 42
SCORING = "neg_mean_absolute_error"


@dataclass(frozen=True)
class PermutationImportanceResult:
    report: pd.DataFrame
    output_path: Path | None = None


def calculate_temporal_permutation_importance(frame: pd.DataFrame) -> pd.DataFrame:
    """Fit one RF per expanding window, scoring only its later validation block.

    The final 20% is isolated first. The report aggregates the five validation
    windows within the first 80%; it never reads the reserved final targets.
    """
    development, _reserved_final_test = temporal_train_test_split(frame)
    windows = expanding_windows(development, n_splits=5, initial_train_fraction=0.5)
    rows: list[dict[str, object]] = []
    for window in windows:
        model = build_random_forest_pipeline()
        train_features = select_features(window.train)
        validation_features = select_features(window.validation)
        model.fit(train_features, window.train[TARGET])
        result = permutation_importance(
            model,
            validation_features,
            window.validation[TARGET],
            scoring=SCORING,
            n_repeats=N_REPEATS,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        for index, feature in enumerate(train_features.columns):
            rows.append(
                {
                    "scope": "window",
                    "window": window.number,
                    "feature": feature,
                    "importance_mean": float(result.importances_mean[index]),
                    "importance_std": float(result.importances_std[index]),
                    "train_rows": len(window.train),
                    "validation_rows": len(window.validation),
                    "train_start": window.train["datetime"].iloc[0],
                    "train_end": window.train["datetime"].iloc[-1],
                    "validation_start": window.validation["datetime"].iloc[0],
                    "validation_end": window.validation["datetime"].iloc[-1],
                }
            )

    window_report = pd.DataFrame(rows)
    aggregates = (
        window_report.groupby("feature", sort=False)["importance_mean"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "importance_mean", "std": "importance_std"})
    )
    aggregates.insert(0, "scope", "aggregate")
    aggregates.insert(1, "window", pd.NA)
    aggregates["train_rows"] = pd.NA
    aggregates["validation_rows"] = pd.NA
    for column in ("train_start", "train_end", "validation_start", "validation_end"):
        aggregates[column] = pd.NaT
    report = pd.concat([aggregates, window_report], ignore_index=True)
    report["window"] = report["window"].astype("Int64")
    return report


def save_permutation_importance_report(
    report: pd.DataFrame,
    output_path: Path = DEFAULT_IMPORTANCE_REPORT,
) -> Path:
    """Write aggregate and per-window feature importance to one CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)
    return output_path


def run_permutation_importance(
    frame: pd.DataFrame,
    output_path: Path | None = DEFAULT_IMPORTANCE_REPORT,
) -> PermutationImportanceResult:
    report = calculate_temporal_permutation_importance(frame)
    saved = save_permutation_importance_report(report, output_path) if output_path else None
    return PermutationImportanceResult(report=report, output_path=saved)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Calculate Random Forest permutation importance on temporal validation windows."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_PATH, help="Prepared Parquet path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_IMPORTANCE_REPORT, help="CSV report destination.")
    args = parser.parse_args(argv)
    try:
        result = run_permutation_importance(load_processed_data(args.input), args.output)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")

    aggregate = result.report.loc[result.report["scope"] == "aggregate"]
    print("Mean increase in validation MAE after permutation, across five windows (rentals/hour):")
    print(aggregate.sort_values("importance_mean", ascending=False)[
        ["feature", "importance_mean", "importance_std"]
    ].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Saved report: {result.output_path}")
    print("Only expanding-window validation blocks within the initial 80% were used; final 20% remained reserved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
