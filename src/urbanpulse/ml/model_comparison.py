"""Generate a reusable report for all models on the reserved final test set."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

from urbanpulse.ml.baseline import (
    DEFAULT_DATA_PATH,
    evaluate_baselines,
    load_processed_data,
)
from urbanpulse.ml.random_forest import evaluate_random_forest
from urbanpulse.ml.ridge import evaluate_ridge

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "reports" / "model_comparison" / "final_test_metrics.csv"
FINAL_TEST_SPLIT = "final_test_20pct"


def evaluate_final_test_models(frame: pd.DataFrame) -> pd.DataFrame:
    """Evaluate the four established approaches on the same chronological holdout."""
    evaluations = [
        ("training_mean", None),
        ("training_hour_mean", None),
        ("ridge", evaluate_ridge(frame)),
        ("random_forest", evaluate_random_forest(frame)),
    ]
    baseline = evaluate_baselines(frame)
    shared = {
        "evaluation_split": FINAL_TEST_SPLIT,
        "train_rows": baseline.train_rows,
        "test_rows": baseline.test_rows,
        "test_start": baseline.test_start,
        "test_end": baseline.test_end,
    }
    rows: list[dict[str, object]] = []
    for name, result in evaluations:
        metrics = baseline.metrics[name] if result is None else result.metrics
        rows.append(
            {
                **shared,
                "model": name,
                "mae": metrics.mae,
                "rmse": metrics.rmse,
                "r2": metrics.r2,
            }
        )
    report = pd.DataFrame(rows)
    for result in evaluations[2:]:
        evaluation = result[1]
        assert evaluation is not None
        if (
            evaluation.train_rows != shared["train_rows"]
            or evaluation.test_rows != shared["test_rows"]
            or evaluation.test_start != shared["test_start"]
            or evaluation.test_end != shared["test_end"]
        ):
            raise RuntimeError("Model evaluations do not use the same final test split.")
    return report


def save_final_test_metrics(
    frame: pd.DataFrame,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Evaluate, save, and return the final-test metrics table."""
    report = evaluate_final_test_models(frame)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate final-test metrics for the four UrbanPulse demand approaches."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_PATH, help="Prepared Parquet path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output CSV path.")
    args = parser.parse_args(argv)
    try:
        report = save_final_test_metrics(load_processed_data(args.input), args.output)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    print(report.to_string(index=False))
    print(f"Saved final-test metrics to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
