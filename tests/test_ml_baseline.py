from __future__ import annotations

import unittest

import pandas as pd

from urbanpulse.ml.baseline import (
    calculate_metrics,
    evaluate_baselines,
    predict_training_baselines,
    temporal_train_test_split,
)


def small_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2011-01-01 02:00",
                    "2011-01-01 00:00",
                    "2011-01-01 01:00",
                    "2011-01-01 03:00",
                    "2011-01-01 04:00",
                    "2011-01-01 05:00",
                    "2011-01-01 06:00",
                    "2011-01-01 07:00",
                    "2011-01-01 08:00",
                    "2011-01-01 09:00",
                ]
            ),
            "cnt": [30, 10, 20, 40, 50, 60, 70, 80, 90, 100],
            "casual": [3] * 10,
            "registered": [7] * 10,
        }
    )


class TemporalSplitTests(unittest.TestCase):
    def test_split_sorts_chronologically_and_uses_eighty_twenty_row_counts(self):
        train, test = temporal_train_test_split(small_frame())

        self.assertEqual((len(train), len(test)), (8, 2))
        self.assertTrue(train["datetime"].is_monotonic_increasing)
        self.assertTrue(test["datetime"].is_monotonic_increasing)
        self.assertLess(train["datetime"].max(), test["datetime"].min())
        self.assertEqual(train["datetime"].iloc[0], pd.Timestamp("2011-01-01 00:00"))
        self.assertEqual(test["datetime"].iloc[0], pd.Timestamp("2011-01-01 08:00"))

    def test_invalid_fraction_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "train_fraction"):
            temporal_train_test_split(small_frame(), train_fraction=1.0)


class BaselinePredictionTests(unittest.TestCase):
    def test_predictions_use_only_training_mean_and_hourly_training_means(self):
        train = pd.DataFrame(
            {
                "datetime": pd.to_datetime(
                    ["2011-01-01 08:00", "2011-01-02 08:00", "2011-01-01 09:00"]
                ),
                "cnt": [10, 30, 20],
            }
        )
        test = pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2011-01-03 08:00", "2011-01-03 09:00"]),
                "cnt": [5000, 7000],
            }
        )

        predictions = predict_training_baselines(train, test)

        self.assertEqual(predictions["training_mean"].tolist(), [20.0, 20.0])
        self.assertEqual(predictions["training_hour_mean"].tolist(), [20.0, 20.0])

    def test_changing_test_targets_and_user_counts_does_not_change_predictions(self):
        train, test = temporal_train_test_split(small_frame())
        baseline_predictions = predict_training_baselines(train, test)
        changed_test = test.copy()
        changed_test["cnt"] = 1_000_000
        changed_test["casual"] = -100
        changed_test["registered"] = 999_999

        changed_predictions = predict_training_baselines(train, changed_test)

        pd.testing.assert_frame_equal(baseline_predictions, changed_predictions)

    def test_hour_missing_from_training_falls_back_to_training_global_mean(self):
        train = pd.DataFrame(
            {"datetime": pd.to_datetime(["2011-01-01 08:00", "2011-01-01 09:00"]), "cnt": [10, 30]}
        )
        test = pd.DataFrame({"datetime": pd.to_datetime(["2011-01-01 10:00"])})

        predictions = predict_training_baselines(train, test)

        self.assertEqual(predictions.loc[0, "training_hour_mean"], 20.0)


class MetricTests(unittest.TestCase):
    def test_calculates_mae_rmse_and_r2(self):
        metrics = calculate_metrics(pd.Series([1, 3, 5]), pd.Series([2, 2, 4]))

        self.assertAlmostEqual(metrics.mae, 1.0)
        self.assertAlmostEqual(metrics.rmse, 1.0)
        self.assertAlmostEqual(metrics.r2, 0.625)

    def test_evaluation_metrics_match_prediction_columns(self):
        result = evaluate_baselines(small_frame())

        self.assertEqual(result.train_rows, 8)
        self.assertEqual(result.test_rows, 2)
        self.assertEqual(set(result.metrics), {"training_mean", "training_hour_mean"})
        self.assertEqual(set(result.predictions.columns), set(result.metrics))


if __name__ == "__main__":
    unittest.main()
