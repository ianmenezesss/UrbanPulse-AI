from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from urbanpulse.dashboard_data import (
    FINAL_MODEL_NAMES,
    load_final_model_metrics,
    load_temporal_validation_reports,
)
from urbanpulse.ml.model_comparison import save_final_test_metrics


def model_frame(rows: int = 40) -> pd.DataFrame:
    index = np.arange(rows)
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2011-01-01", periods=rows, freq="h"),
            "instant": index + 1,
            "season": index % 4 + 1,
            "yr": index // 20,
            "mnth": index % 12 + 1,
            "hr": index % 24,
            "holiday": index % 2,
            "weekday": index % 7,
            "workingday": (index + 1) % 2,
            "weathersit": index % 4 + 1,
            "temp": index / rows,
            "atemp": (index + 1) / (rows + 1),
            "hum": 0.3 + index / (rows * 2),
            "windspeed": index / (rows * 3),
            "casual": index + 20,
            "registered": index + 30,
            "cnt": index * 5 + 10,
        }
    )


def write_temporal_reports(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    models = ["Ridge", "Random Forest"]
    pd.DataFrame(
        [
            {"window": window, "model": model, "mae": window + 1, "rmse": window + 2, "r2": 0.5}
            for model in models
            for window in range(1, 6)
        ]
    ).to_csv(directory / "window_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": model,
                "mae_mean": 4.0,
                "mae_std": 1.5811388301,
                "rmse_mean": 5.0,
                "rmse_std": 1.5811388301,
                "r2_mean": 0.5,
                "r2_std": 0.0,
            }
            for model in models
        ]
    ).to_csv(directory / "summary_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "window": 1,
                "datetime": "2011-01-01 00:00:00",
                "model": model,
                "actual": 10,
                "predicted": 9,
                "error": -1,
            }
            for model in models
        ]
    ).to_csv(directory / "out_of_sample_predictions.csv", index=False)
    pd.DataFrame(
        [{"model": model, "hour": hour, "mae": 1.0} for model in models for hour in range(24)]
    ).to_csv(directory / "error_by_hour.csv", index=False)
    pd.DataFrame(
        [
            {"model": model, "weekday": day, "weekday_name": str(day), "mae": 1.0}
            for model in models
            for day in range(7)
        ]
    ).to_csv(directory / "error_by_weekday.csv", index=False)


class FinalModelMetricsReportTests(unittest.TestCase):
    def test_generates_and_loads_consistent_metrics_for_four_models(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "final_test_metrics.csv"
            generated = save_final_test_metrics(model_frame(), path)
            loaded = load_final_model_metrics(path)

        self.assertEqual(set(loaded["model"]), FINAL_MODEL_NAMES)
        self.assertEqual(set(loaded["evaluation_split"]), {"final_test_20pct"})
        self.assertEqual(len(loaded), 4)
        pd.testing.assert_frame_equal(
            loaded[["model", "mae", "rmse", "r2"]].sort_values("model").reset_index(drop=True),
            generated[["model", "mae", "rmse", "r2"]].sort_values("model").reset_index(drop=True),
        )
        self.assertEqual(loaded["test_rows"].nunique(), 1)
        self.assertEqual(loaded["test_start"].nunique(), 1)

    def test_rejects_report_mixing_final_test_with_another_split(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            metrics = save_final_test_metrics(model_frame(), path)
            metrics.loc[0, "evaluation_split"] = "temporal_validation"
            metrics.to_csv(path, index=False)

            with self.assertRaisesRegex(ValueError, "reserved final 20%"):
                load_final_model_metrics(path)

    def test_missing_report_has_generation_instruction(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "urbanpulse-model-comparison"):
                load_final_model_metrics(Path(directory) / "missing.csv")


class TemporalValidationReportTests(unittest.TestCase):
    def test_loads_and_checks_temporal_report_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            report_dir = Path(directory)
            write_temporal_reports(report_dir)

            reports = load_temporal_validation_reports(report_dir)

        self.assertEqual(set(reports), {
            "window_metrics",
            "summary_metrics",
            "predictions",
            "errors_by_hour",
            "errors_by_weekday",
        })
        self.assertEqual(set(reports["window_metrics"]["model"]), {"Ridge", "Random Forest"})

    def test_missing_temporal_report_has_generation_instruction(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "urbanpulse-temporal-validation"):
                load_temporal_validation_reports(Path(directory))

    def test_rejects_temporal_report_with_wrong_model_set(self):
        with tempfile.TemporaryDirectory() as directory:
            report_dir = Path(directory)
            write_temporal_reports(report_dir)
            window_metrics = pd.read_csv(report_dir / "window_metrics.csv")
            window_metrics.loc[0, "model"] = "training_mean"
            window_metrics.to_csv(report_dir / "window_metrics.csv", index=False)

            with self.assertRaisesRegex(ValueError, "Ridge and Random Forest"):
                load_temporal_validation_reports(report_dir)

    def test_rejects_summary_metrics_inconsistent_with_window_results(self):
        with tempfile.TemporaryDirectory() as directory:
            report_dir = Path(directory)
            write_temporal_reports(report_dir)
            summary = pd.read_csv(report_dir / "summary_metrics.csv")
            summary.loc[0, "mae_mean"] = 999
            summary.to_csv(report_dir / "summary_metrics.csv", index=False)

            with self.assertRaisesRegex(ValueError, "does not match"):
                load_temporal_validation_reports(report_dir)


if __name__ == "__main__":
    unittest.main()
