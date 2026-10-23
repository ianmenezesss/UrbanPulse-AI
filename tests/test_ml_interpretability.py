from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from urbanpulse.dashboard_data import load_permutation_importance_report
from urbanpulse.ml import interpretability
from urbanpulse.ml.interpretability import (
    calculate_temporal_permutation_importance,
    save_permutation_importance_report,
)
from urbanpulse.ml.ridge import CATEGORICAL_FEATURES
from urbanpulse.ml.ridge import FEATURE_COLUMNS


def sample_frame(rows: int = 100) -> pd.DataFrame:
    index = np.arange(rows)
    temperature = (index % 47) / 47
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2011-01-01", periods=rows, freq="h"),
            "instant": index + 1,
            "season": index // 12 % 4 + 1,
            "yr": 0,
            "mnth": index // 8 % 12 + 1,
            "hr": index % 24,
            "holiday": index % 13 == 0,
            "weekday": index // 24 % 7,
            "workingday": index % 7 < 5,
            "weathersit": index % 4 + 1,
            "temp": temperature,
            "atemp": temperature * 0.9,
            "hum": (index % 100) / 100,
            "windspeed": (index % 30) / 67,
            "casual": index % 20,
            "registered": index % 40,
            "cnt": (temperature * 500 + index % 5).astype(int),
        }
    )


class TemporalPermutationImportanceTests(unittest.TestCase):
    def test_missing_dashboard_report_includes_generation_command(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "urbanpulse-interpretability"):
                load_permutation_importance_report(Path(directory) / "missing.csv")

    def test_calculates_each_model_feature_for_all_five_ordered_windows(self):
        frame = sample_frame()
        original_function = interpretability.permutation_importance
        with patch.object(
            interpretability,
            "permutation_importance",
            wraps=original_function,
        ) as mocked_importance:
            report = calculate_temporal_permutation_importance(frame)

        self.assertEqual(set(report["scope"]), {"aggregate", "window"})
        aggregate = report.loc[report["scope"] == "aggregate"]
        windows = report.loc[report["scope"] == "window"]
        self.assertEqual(set(aggregate["feature"]), set(FEATURE_COLUMNS))
        self.assertEqual(set(windows["window"].dropna().astype(int)), {1, 2, 3, 4, 5})
        self.assertEqual(len(windows), 5 * len(FEATURE_COLUMNS))
        self.assertTrue(np.isfinite(report["importance_mean"]).all())
        self.assertTrue((report["importance_std"] >= 0).all())
        self.assertEqual(mocked_importance.call_count, 5)
        for call in mocked_importance.call_args_list:
            self.assertEqual(call.kwargs["scoring"], "neg_mean_absolute_error")
            self.assertEqual(call.kwargs["n_repeats"], 5)
            self.assertEqual(call.kwargs["random_state"], 42)

        for _, group in windows.groupby("window"):
            self.assertLess(
                pd.Timestamp(group["train_end"].iloc[0]),
                pd.Timestamp(group["validation_start"].iloc[0]),
            )
        first_validation = windows.loc[windows["window"] == 1, "validation_start"].iloc[0]
        self.assertLess(
            windows.loc[windows["window"] == 1, "train_rows"].iloc[0],
            windows.loc[windows["window"] == 5, "train_rows"].iloc[0],
        )
        self.assertTrue(pd.Timestamp(first_validation) < pd.Timestamp(windows["validation_end"].max()))
        first_pipeline = mocked_importance.call_args_list[0].args[0]
        encoded_categories = first_pipeline.named_steps["preprocessor"].named_transformers_[
            "categorical"
        ].categories_
        month_index = CATEGORICAL_FEATURES.index("mnth")
        self.assertNotIn(7, encoded_categories[month_index])

    def test_reserved_final_twenty_percent_cannot_change_importance(self):
        frame = sample_frame()
        changed_final = frame.copy()
        changed_final.loc[80:, "cnt"] = 9_999_999
        for feature in FEATURE_COLUMNS:
            changed_final.loc[80:, feature] = False if changed_final[feature].dtype == bool else 0

        original_report = calculate_temporal_permutation_importance(frame)
        changed_report = calculate_temporal_permutation_importance(changed_final)

        pd.testing.assert_frame_equal(original_report, changed_report)

    def test_saved_report_loads_and_rejects_inconsistent_aggregate(self):
        report = calculate_temporal_permutation_importance(sample_frame())
        with tempfile.TemporaryDirectory() as directory:
            path = save_permutation_importance_report(
                report, Path(directory) / "permutation_importance.csv"
            )
            loaded = load_permutation_importance_report(path)
            self.assertEqual(len(loaded), 6 * len(FEATURE_COLUMNS))

            loaded.loc[
                (loaded["scope"] == "aggregate") & (loaded["feature"] == FEATURE_COLUMNS[0]),
                "importance_mean",
            ] += 1
            loaded.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_permutation_importance_report(path)


if __name__ == "__main__":
    unittest.main()
