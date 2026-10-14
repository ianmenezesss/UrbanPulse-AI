from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from urbanpulse.ml.baseline import temporal_train_test_split
from urbanpulse.ml.ridge import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    FORBIDDEN_FEATURES,
    NUMERIC_FEATURES,
    build_ridge_pipeline,
    evaluate_ridge,
    fit_predict_ridge,
    select_features,
)


def sample_frame(rows: int = 20) -> pd.DataFrame:
    index = np.arange(rows)
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2011-01-01", periods=rows, freq="h"),
            "instant": index + 1,
            "season": index % 4 + 1,
            "yr": index // 10,
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
            "casual": index + 50,
            "registered": index + 100,
            "cnt": index * 10 + 20,
        }
    )


class RidgeFeatureTests(unittest.TestCase):
    def test_select_features_uses_calendar_and_weather_only(self):
        selected = select_features(sample_frame())

        self.assertEqual(tuple(selected.columns), FEATURE_COLUMNS)
        self.assertTrue(set(CATEGORICAL_FEATURES).issubset(selected.columns))
        self.assertTrue(set(NUMERIC_FEATURES).issubset(selected.columns))
        self.assertTrue(FORBIDDEN_FEATURES.isdisjoint(selected.columns))

    def test_missing_predictor_has_clear_error(self):
        with self.assertRaisesRegex(ValueError, "temp"):
            select_features(sample_frame().drop(columns="temp"))

    def test_pipeline_has_explicit_preprocessing_and_ridge_alpha(self):
        pipeline = build_ridge_pipeline(alpha=2.5)

        self.assertEqual(pipeline.named_steps["regressor"].alpha, 2.5)
        self.assertEqual(set(pipeline.named_steps["preprocessor"].transformers[0][2]), set(CATEGORICAL_FEATURES))
        self.assertEqual(set(pipeline.named_steps["preprocessor"].transformers[1][2]), set(NUMERIC_FEATURES))


class RidgeTemporalAndLeakageTests(unittest.TestCase):
    def test_evaluation_reuses_chronological_eighty_twenty_split(self):
        frame = sample_frame()
        train, test = temporal_train_test_split(frame)
        result = evaluate_ridge(frame)

        self.assertEqual((result.train_rows, result.test_rows), (len(train), len(test)))
        self.assertEqual(result.test_start, test["datetime"].iloc[0])
        self.assertEqual(result.test_end, test["datetime"].iloc[-1])

    def test_test_targets_and_user_counts_do_not_change_model_predictions(self):
        train, test = temporal_train_test_split(sample_frame())
        first = fit_predict_ridge(train, test)
        modified_test = test.copy()
        modified_test["cnt"] = -999_999
        modified_test["casual"] = 1_000_000
        modified_test["registered"] = 2_000_000
        second = fit_predict_ridge(train, modified_test)

        np.testing.assert_allclose(first, second)

    def test_preprocessing_is_fitted_only_on_training_values(self):
        train, test = temporal_train_test_split(sample_frame())
        pipeline = build_ridge_pipeline()
        pipeline.fit(select_features(train), train["cnt"])
        scaler = pipeline.named_steps["preprocessor"].named_transformers_["numeric"]
        expected_train_means = train.loc[:, NUMERIC_FEATURES].mean().to_numpy()

        np.testing.assert_allclose(scaler.mean_, expected_train_means)
        self.assertTrue(test.loc[:, NUMERIC_FEATURES].mean().ne(pd.Series(scaler.mean_, index=NUMERIC_FEATURES)).any())


if __name__ == "__main__":
    unittest.main()
