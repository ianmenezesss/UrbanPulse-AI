from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

from urbanpulse.ml.ridge import FEATURE_COLUMNS
from urbanpulse.ml.artifact_metadata import metadata_path_for_artifact
from urbanpulse.ml.train_artifact import train_and_save_artifact
from urbanpulse.simulator import (
    build_daily_scenario_features,
    build_scenario_features,
    convert_weather_to_dataset_scale,
    load_simulator_model,
    predict_daily_demand,
    predict_hourly_demand,
    uci_holiday_for_date,
    uci_season_for_date,
)

UCI_RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "hour.csv"


def sample_frame(rows: int = 30) -> pd.DataFrame:
    index = np.arange(rows)
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2011-01-01", periods=rows, freq="h"),
            "instant": index + 1,
            "season": index // 7 % 4 + 1,
            "yr": 0,
            "mnth": index // 7 % 12 + 1,
            "hr": index % 24,
            "holiday": index % 2,
            "weekday": index % 7,
            "workingday": (index + 1) % 2,
            "weathersit": index % 4 + 1,
            "temp": index / rows,
            "atemp": (index + 1) / (rows + 1),
            "hum": 0.3 + index / (rows * 2),
            "windspeed": index / (rows * 3),
            "casual": index + 2,
            "registered": index + 10,
            "cnt": index * 4 + 20,
        }
    )


class WeatherConversionTests(unittest.TestCase):
    def test_converts_original_weather_scales_to_uci_normalized_features(self):
        converted = convert_weather_to_dataset_scale(-8, -16, 100, 67)

        self.assertEqual(converted, {"temp": 0.0, "atemp": 0.0, "hum": 1.0, "windspeed": 1.0})

    def test_rejects_weather_values_outside_documented_scale(self):
        with self.assertRaisesRegex(ValueError, "temperature_c"):
            convert_weather_to_dataset_scale(-8.1, 0, 50, 20)
        with self.assertRaisesRegex(ValueError, "humidity_percent"):
            convert_weather_to_dataset_scale(20, 20, 101, 20)


class ScenarioValidationTests(unittest.TestCase):
    def test_builds_exact_training_schema_and_uci_calendar_codes(self):
        features = build_scenario_features(
            date(2012, 1, 7),
            8,
            temperature_c=15,
            apparent_temperature_c=18,
            humidity_percent=60,
            wind_speed_dataset_units=20,
            weathersit=2,
            holiday=False,
        )

        self.assertEqual(tuple(features.columns), FEATURE_COLUMNS)
        row = features.iloc[0]
        self.assertEqual((row["season"], row["yr"], row["mnth"], row["weekday"]), (1, 1, 1, 6))
        self.assertEqual((row["workingday"], row["holiday"], row["hr"]), (0, 0, 8))
        self.assertAlmostEqual(row["temp"], 23 / 47)
        self.assertAlmostEqual(row["atemp"], 34 / 66)
        self.assertAlmostEqual(row["hum"], 0.6)
        self.assertAlmostEqual(row["windspeed"], 20 / 67)

    def test_season_codes_follow_observed_uci_transition_dates(self):
        for year in (2011, 2012):
            cases = (
                (date(year, 3, 20), 1), (date(year, 3, 21), 2),
                (date(year, 6, 20), 2), (date(year, 6, 21), 3),
                (date(year, 9, 22), 3), (date(year, 9, 23), 4),
                (date(year, 12, 20), 4), (date(year, 12, 21), 1),
            )
            for day, expected in cases:
                with self.subTest(day=day):
                    self.assertEqual(uci_season_for_date(day), expected)

    def test_calendar_fields_match_dataset_encoding_and_holiday_lookup(self):
        day = date(2011, 11, 11)  # holiday recorded in the UCI hourly file
        row = build_scenario_features(
            day, 12, temperature_c=20, apparent_temperature_c=20,
            humidity_percent=50, wind_speed_dataset_units=15,
            weathersit=1,
        ).iloc[0]
        self.assertEqual(row["weekday"], (day.weekday() + 1) % 7)
        self.assertEqual((row["yr"], row["mnth"]), (0, 11))
        self.assertEqual((row["holiday"], row["workingday"]), (1, 0))
        self.assertTrue(uci_holiday_for_date(day))

        forced = build_scenario_features(
            date(2011, 11, 10), 12, temperature_c=20,
            apparent_temperature_c=20, humidity_percent=50,
            wind_speed_dataset_units=15, weathersit=1, holiday=True,
        ).iloc[0]
        self.assertEqual((forced["holiday"], forced["workingday"]), (1, 0))

    @unittest.skipUnless(UCI_RAW_PATH.is_file(), "local UCI source CSV is not available")
    def test_derived_calendar_matches_every_source_row(self):
        source = pd.read_csv(
            UCI_RAW_PATH,
            usecols=["dteday", "season", "yr", "mnth", "holiday", "weekday", "workingday"],
        )
        dates = pd.to_datetime(source["dteday"])
        expected_season = dates.dt.date.map(uci_season_for_date)
        expected_holiday = dates.dt.date.map(uci_holiday_for_date).astype(int)
        expected_weekday = ((dates.dt.dayofweek + 1) % 7).astype(int)
        expected_year = (dates.dt.year - 2011).astype(int)
        expected_workingday = ((~expected_weekday.isin([0, 6])) & expected_holiday.eq(0)).astype(int)

        pd.testing.assert_series_equal(source["season"], expected_season.rename("season"))
        pd.testing.assert_series_equal(source["yr"], expected_year.rename("yr"))
        pd.testing.assert_series_equal(
            source["mnth"], dates.dt.month.astype(int).rename("mnth")
        )
        pd.testing.assert_series_equal(source["holiday"], expected_holiday.rename("holiday"))
        pd.testing.assert_series_equal(source["weekday"], expected_weekday.rename("weekday"))
        pd.testing.assert_series_equal(
            source["workingday"], expected_workingday.rename("workingday")
        )

    def test_holiday_is_non_working_and_rejects_unsupported_calendar_inputs(self):
        features = build_scenario_features(
            date(2011, 7, 4), 12,
            temperature_c=20, apparent_temperature_c=20, humidity_percent=50,
            wind_speed_dataset_units=15, weathersit=1, holiday=True,
        )
        self.assertEqual(features.iloc[0]["workingday"], 0)
        with self.assertRaisesRegex(ValueError, "2011 or 2012"):
            build_scenario_features(
                date(2013, 1, 1), 12,
                temperature_c=20, apparent_temperature_c=20, humidity_percent=50,
                wind_speed_dataset_units=15, weathersit=1, holiday=False,
            )
        with self.assertRaisesRegex(ValueError, "hour"):
            build_scenario_features(
                date(2011, 1, 1), 24,
                temperature_c=20, apparent_temperature_c=20, humidity_percent=50,
                wind_speed_dataset_units=15, weathersit=1, holiday=False,
            )
        with self.assertRaisesRegex(ValueError, "weathersit"):
            build_scenario_features(
                date(2011, 1, 1), 12,
                temperature_c=20, apparent_temperature_c=20, humidity_percent=50,
                wind_speed_dataset_units=15, weathersit=5, holiday=False,
            )


class ArtifactAndPredictionTests(unittest.TestCase):
    def test_trains_reproducible_pipeline_using_only_initial_eighty_percent(self):
        frame = sample_frame()
        changed_test = frame.copy()
        changed_test.loc[24:, "cnt"] = 999999
        changed_test.loc[24:, FEATURE_COLUMNS] = 0

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_one, input_two = root / "one.parquet", root / "two.parquet"
            frame.to_parquet(input_one, index=False)
            changed_test.to_parquet(input_two, index=False)
            artifact_one, artifact_two = root / "one.joblib", root / "two.joblib"

            result = train_and_save_artifact(input_one, artifact_one)
            train_and_save_artifact(input_two, artifact_two)
            metadata_path = metadata_path_for_artifact(artifact_one)
            self.assertTrue(metadata_path.is_file())
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            model_one = load_simulator_model(artifact_one)
            model_two = load_simulator_model(artifact_two)

        self.assertEqual((result.train_rows, result.test_rows_reserved), (24, 6))
        self.assertEqual(result.metadata_path, metadata_path)
        self.assertEqual(metadata["features"], list(FEATURE_COLUMNS))
        self.assertEqual(metadata["train_rows"], 24)
        self.assertEqual(metadata["total_rows"], 30)
        self.assertEqual(metadata["training_period"]["start"], "2011-01-01 00:00:00")
        self.assertEqual(set(metadata["versions"]), {"python", "scikit_learn", "pandas", "joblib"})
        self.assertEqual(metadata["versions"]["scikit_learn"], __import__("sklearn").__version__)
        self.assertEqual(metadata["model_parameters"]["random_state"], 42)
        self.assertEqual(len(metadata["dataset_sha256"]), 64)
        self.assertEqual(len(metadata["artifact_sha256"]), 64)
        self.assertEqual(tuple(model_one.feature_names_in_), FEATURE_COLUMNS)
        np.testing.assert_allclose(
            model_one.predict(frame.loc[24:, FEATURE_COLUMNS]),
            model_two.predict(frame.loc[24:, FEATURE_COLUMNS]),
        )

    def test_daily_predictions_are_ordered_batched_and_match_selected_hour(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "sample.parquet"
            artifact_path = Path(directory) / "model.joblib"
            sample_frame().to_parquet(input_path, index=False)
            train_and_save_artifact(input_path, artifact_path)
            model = load_simulator_model(artifact_path)

        features = build_daily_scenario_features(
            date(2011, 1, 3),
            temperature_c=18,
            apparent_temperature_c=20,
            humidity_percent=55,
            wind_speed_dataset_units=12,
            weathersit=1,
            holiday=False,
        )
        hourly = predict_daily_demand(model, features)
        selected_hour = 8
        individual = predict_hourly_demand(model, features.loc[features["hr"] == selected_hour])

        self.assertEqual(len(hourly), 24)
        self.assertEqual(hourly["hr"].tolist(), list(range(24)))
        self.assertTrue((hourly["demand_estimated"] >= 0).all())
        self.assertAlmostEqual(
            hourly.loc[hourly["hr"] == selected_hour, "demand_estimated"].iloc[0],
            individual,
        )

    def test_daily_prediction_rejects_invalid_inputs_and_nonordered_hours(self):
        with self.assertRaisesRegex(ValueError, "temperature_c"):
            build_daily_scenario_features(
                date(2011, 1, 3),
                temperature_c=50,
                apparent_temperature_c=20,
                humidity_percent=55,
                wind_speed_dataset_units=12,
                weathersit=1,
                holiday=False,
            )
        wrong_hours = pd.DataFrame([{column: 0 for column in FEATURE_COLUMNS}] * 24)
        with self.assertRaisesRegex(ValueError, "ordered by hour"):
            predict_daily_demand(object(), wrong_hours)

    def test_loader_rejects_missing_and_incompatible_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.joblib"
            with self.assertRaises(FileNotFoundError):
                load_simulator_model(path)
            joblib.dump({"wrong": "artifact"}, path)
            from urbanpulse.ml.artifact_metadata import current_runtime_versions, sha256_file

            with self.assertRaisesRegex(FileNotFoundError, "Model metadata not found"):
                load_simulator_model(path)
            metadata_path_for_artifact(path).write_text(
                json.dumps({
                    "metadata_schema": 1,
                    "versions": current_runtime_versions(),
                    "artifact_sha256": sha256_file(path),
                    "dataset_sha256": "0" * 64,
                    "features": list(FEATURE_COLUMNS),
                    "train_rows": 1,
                    "training_period": {"start": "2011-01-01", "end": "2011-01-01"},
                    "model_parameters": {},
                }),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "compatible"):
                load_simulator_model(path)

    def test_loader_rejects_runtime_version_mismatch_before_deserialization(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "sample.parquet"
            artifact_path = Path(directory) / "model.joblib"
            sample_frame().to_parquet(input_path, index=False)
            train_and_save_artifact(input_path, artifact_path)
            metadata_path = metadata_path_for_artifact(artifact_path)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["versions"]["scikit_learn"] = "0.0"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Incompatible model runtime versions"):
                load_simulator_model(artifact_path)

    def test_loader_rejects_artifact_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "sample.parquet"
            artifact_path = Path(directory) / "model.joblib"
            sample_frame().to_parquet(input_path, index=False)
            train_and_save_artifact(input_path, artifact_path)
            artifact_path.write_bytes(artifact_path.read_bytes() + b"tamper")
            with self.assertRaisesRegex(ValueError, "artifact hash"):
                load_simulator_model(artifact_path)

    def test_prediction_is_nonnegative_and_requires_exactly_one_schema_row(self):
        negative_model = Pipeline([("regressor", DummyRegressor(strategy="constant", constant=-4))])
        negative_model.fit(np.array([[0], [1]]), [0, 0])
        features = pd.DataFrame([{column: 0 for column in FEATURE_COLUMNS}], columns=FEATURE_COLUMNS)

        self.assertEqual(predict_hourly_demand(negative_model, features), 0.0)
        with self.assertRaisesRegex(ValueError, "exactly one row"):
            predict_hourly_demand(negative_model, pd.concat([features, features], ignore_index=True))


if __name__ == "__main__":
    unittest.main()
