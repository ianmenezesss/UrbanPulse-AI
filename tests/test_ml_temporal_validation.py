from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from urbanpulse.ml.baseline import temporal_train_test_split
from urbanpulse.ml.temporal_validation import (
    evaluate_temporal_validation,
    expanding_windows,
    summarize_window_metrics,
)


def small_frame(rows: int = 120) -> pd.DataFrame:
    index = np.arange(rows)
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2011-01-01", periods=rows, freq="h"),
            "instant": index + 1,
            "season": index % 4 + 1,
            "yr": index // 60,
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
            "casual": index + 40,
            "registered": index + 80,
            "cnt": index * 10 + 10,
        }
    )


class ExpandingWindowTests(unittest.TestCase):
    def test_windows_are_ordered_disjoint_and_expand_training_history(self):
        development, _ = temporal_train_test_split(small_frame())
        windows = expanding_windows(development, n_splits=4, initial_train_fraction=0.5)

        self.assertEqual([len(window.train) for window in windows], [48, 60, 72, 84])
        self.assertEqual([len(window.validation) for window in windows], [12, 12, 12, 12])
        for previous, current in zip(windows, windows[1:]):
            self.assertLess(previous.validation["datetime"].max(), current.validation["datetime"].min())
            self.assertGreater(len(current.train), len(previous.train))
        for window in windows:
            self.assertLess(window.train["datetime"].max(), window.validation["datetime"].min())
        validated = pd.concat([window.validation for window in windows])
        self.assertEqual(len(validated), len(development) - 48)
        self.assertTrue(validated["datetime"].is_monotonic_increasing)

    def test_rejects_not_enough_rows_for_requested_windows(self):
        with self.assertRaisesRegex(ValueError, "Not enough rows"):
            expanding_windows(small_frame(8), n_splits=5)


class TemporalValidationEvaluationTests(unittest.TestCase):
    def test_final_twenty_percent_does_not_affect_validation_results(self):
        frame = small_frame()
        _, final_test = temporal_train_test_split(frame)
        original = evaluate_temporal_validation(frame, n_splits=3, initial_train_fraction=0.5)

        changed = frame.copy()
        changed.loc[final_test.index, "cnt"] = 1_000_000
        changed.loc[final_test.index, "temp"] = 100_000
        repeated = evaluate_temporal_validation(changed, n_splits=3, initial_train_fraction=0.5)

        pd.testing.assert_frame_equal(original.window_metrics, repeated.window_metrics)
        pd.testing.assert_frame_equal(original.predictions, repeated.predictions)

    def test_metrics_and_summary_match_out_of_sample_predictions(self):
        result = evaluate_temporal_validation(small_frame(), n_splits=3, initial_train_fraction=0.5)
        first_row = result.window_metrics.iloc[0]
        subset = result.predictions[
            (result.predictions["window"] == first_row["window"])
            & (result.predictions["model"] == first_row["model"])
        ]
        expected_mae = np.abs(subset["error"]).mean()
        expected_rmse = np.sqrt(np.square(subset["error"]).mean())

        self.assertAlmostEqual(first_row["mae"], expected_mae)
        self.assertAlmostEqual(first_row["rmse"], expected_rmse)
        summary = summarize_window_metrics(result.window_metrics)
        ridge_rows = result.window_metrics[result.window_metrics["model"] == "Ridge"]
        ridge_summary = summary[summary["model"] == "Ridge"].iloc[0]
        self.assertAlmostEqual(ridge_summary["mae_mean"], ridge_rows["mae"].mean())
        self.assertAlmostEqual(ridge_summary["mae_std"], ridge_rows["mae"].std())

    def test_error_tables_cover_hours_and_weekdays_for_both_models(self):
        result = evaluate_temporal_validation(small_frame(500), n_splits=3, initial_train_fraction=0.5)

        self.assertEqual(set(result.errors_by_hour["hour"]), set(range(24)))
        self.assertEqual(set(result.errors_by_weekday["weekday"]), set(range(7)))
        self.assertEqual(set(result.errors_by_hour["model"]), {"Ridge", "Random Forest"})
        self.assertEqual(set(result.errors_by_weekday["model"]), {"Ridge", "Random Forest"})


if __name__ == "__main__":
    unittest.main()
