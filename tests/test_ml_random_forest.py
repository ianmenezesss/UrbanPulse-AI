from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from urbanpulse.ml.baseline import temporal_train_test_split
from urbanpulse.ml.random_forest import (
    CATEGORICAL_FEATURES,
    MIN_SAMPLES_LEAF,
    N_ESTIMATORS,
    N_JOBS,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    build_random_forest_pipeline,
    evaluate_random_forest,
    fit_predict_random_forest,
    select_features,
)
from urbanpulse.ml.ridge import FEATURE_COLUMNS


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


class RandomForestFeatureTests(unittest.TestCase):
    def test_selection_matches_ridge_calendar_and_weather_features(self):
        selected = select_features(sample_frame())

        self.assertEqual(tuple(selected.columns), FEATURE_COLUMNS)
        self.assertTrue(set(CATEGORICAL_FEATURES).issubset(selected.columns))
        self.assertTrue(set(NUMERIC_FEATURES).issubset(selected.columns))
        self.assertTrue({"instant", "casual", "registered", "cnt"}.isdisjoint(selected.columns))

    def test_pipeline_has_required_forest_parameters_and_no_scaler(self):
        pipeline = build_random_forest_pipeline()
        forest = pipeline.named_steps["regressor"]
        preprocessing = pipeline.named_steps["preprocessor"]

        self.assertIsInstance(forest, RandomForestRegressor)
        self.assertEqual(forest.random_state, RANDOM_STATE)
        self.assertEqual(forest.n_estimators, N_ESTIMATORS)
        self.assertEqual(forest.min_samples_leaf, MIN_SAMPLES_LEAF)
        self.assertEqual(forest.n_jobs, N_JOBS)
        self.assertEqual(preprocessing.transformers[0][0], "categorical")
        self.assertEqual(preprocessing.transformers[1][1], "passthrough")


class RandomForestTemporalAndLeakageTests(unittest.TestCase):
    def test_evaluation_reuses_chronological_eighty_twenty_split(self):
        frame = sample_frame()
        train, test = temporal_train_test_split(frame)
        result = evaluate_random_forest(frame)

        self.assertEqual((result.train_rows, result.test_rows), (len(train), len(test)))
        self.assertEqual(result.test_start, test["datetime"].iloc[0])
        self.assertEqual(result.test_end, test["datetime"].iloc[-1])

    def test_test_targets_and_user_counts_do_not_change_predictions(self):
        train, test = temporal_train_test_split(sample_frame())
        first = fit_predict_random_forest(train, test)
        changed_test = test.copy()
        changed_test["cnt"] = -999_999
        changed_test["casual"] = 1_000_000
        changed_test["registered"] = 2_000_000
        second = fit_predict_random_forest(train, changed_test)

        np.testing.assert_allclose(first, second)

    def test_encoder_categories_are_fitted_on_training_rows_only(self):
        train, test = temporal_train_test_split(sample_frame())
        pipeline = build_random_forest_pipeline()
        pipeline.fit(select_features(train), train["cnt"])
        encoder = pipeline.named_steps["preprocessor"].named_transformers_["categorical"]
        hour_position = CATEGORICAL_FEATURES.index("hr")

        self.assertEqual(set(encoder.categories_[hour_position]), set(train["hr"].unique()))
        self.assertTrue(set(test["hr"].unique()).difference(train["hr"].unique()))


if __name__ == "__main__":
    unittest.main()
