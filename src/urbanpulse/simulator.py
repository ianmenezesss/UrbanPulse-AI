"""Input conversion and prediction helpers for the demand simulator."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from urbanpulse.ml.ridge import FEATURE_COLUMNS
from urbanpulse.ml.artifact_metadata import (
    metadata_path_for_artifact,
    read_artifact_metadata,
    sha256_file,
    validate_artifact_compatibility,
)
from urbanpulse.ml.train_artifact import DEFAULT_MODEL_PATH

YEAR_CODES = {2011: 0, 2012: 1}
# Calendar labels transcribed from the official UCI hourly file for supported years.
# The observed transition dates are identical in 2011 and 2012.
UCI_SEASON_TRANSITIONS = ((3, 21, 2), (6, 21, 3), (9, 23, 4), (12, 21, 1))
UCI_HOLIDAY_DATES = frozenset(
    date.fromisoformat(value)
    for value in (
        "2011-01-17", "2011-02-21", "2011-04-15", "2011-05-30", "2011-07-04",
        "2011-09-05", "2011-10-10", "2011-11-11", "2011-11-24", "2011-12-26",
        "2012-01-02", "2012-01-16", "2012-02-20", "2012-04-16", "2012-05-28",
        "2012-07-04", "2012-09-03", "2012-10-08", "2012-11-12", "2012-11-22",
        "2012-12-25",
    )
)


def uci_season_for_date(scenario_date: date) -> int:
    """Return the UCI season code using the transitions observed in hour.csv.

    The source file changes labels on March 21, June 21, September 23, and
    December 21 in both supported years. This date rule is verified against all
    rows in the supplied 2011–2012 source data; it is not a general climate rule.
    """
    if scenario_date.year not in YEAR_CODES:
        raise ValueError("Season mapping is supported only for 2011 and 2012.")
    season = 1
    for month, day, new_season in UCI_SEASON_TRANSITIONS:
        if (scenario_date.month, scenario_date.day) >= (month, day):
            season = new_season
    return season


def uci_holiday_for_date(scenario_date: date) -> bool:
    """Return the holiday flag observed in the UCI file for supported years."""
    if scenario_date.year not in YEAR_CODES:
        raise ValueError("Holiday mapping is supported only for 2011 and 2012.")
    return scenario_date in UCI_HOLIDAY_DATES


def convert_weather_to_dataset_scale(
    temperature_c: float,
    apparent_temperature_c: float,
    humidity_percent: float,
    wind_speed_dataset_units: float,
) -> dict[str, float]:
    """Convert physical inputs using the UCI hourly dataset's normalization."""
    values = {
        "temperature_c": (temperature_c, -8, 39),
        "apparent_temperature_c": (apparent_temperature_c, -16, 50),
        "humidity_percent": (humidity_percent, 0, 100),
        "wind_speed_dataset_units": (wind_speed_dataset_units, 0, 67),
    }
    for name, (value, minimum, maximum) in values.items():
        if not np.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return {
        "temp": (float(temperature_c) + 8.0) / 47.0,
        "atemp": (float(apparent_temperature_c) + 16.0) / 66.0,
        "hum": float(humidity_percent) / 100.0,
        "windspeed": float(wind_speed_dataset_units) / 67.0,
    }


def build_scenario_features(
    scenario_date: date,
    hour: int,
    *,
    temperature_c: float,
    apparent_temperature_c: float,
    humidity_percent: float,
    wind_speed_dataset_units: float,
    weathersit: int,
    holiday: bool | None = None,
) -> pd.DataFrame:
    """Build one row with exactly the features and encodings used in training."""
    if scenario_date.year not in YEAR_CODES:
        raise ValueError("Scenario date must be in 2011 or 2012, the years represented by the model.")
    if isinstance(hour, bool) or not isinstance(hour, (int, np.integer)) or not 0 <= hour <= 23:
        raise ValueError("hour must be an integer from 0 to 23.")
    if isinstance(weathersit, bool) or weathersit not in (1, 2, 3, 4):
        raise ValueError("weathersit must be one of 1, 2, 3, or 4.")

    weekday = (scenario_date.weekday() + 1) % 7  # UCI: Sunday=0, Saturday=6
    is_weekend = weekday in (0, 6)
    is_holiday = uci_holiday_for_date(scenario_date) if holiday is None else bool(holiday)
    weather = convert_weather_to_dataset_scale(
        temperature_c,
        apparent_temperature_c,
        humidity_percent,
        wind_speed_dataset_units,
    )
    row = {
        "season": uci_season_for_date(scenario_date),
        "yr": YEAR_CODES[scenario_date.year],
        "mnth": scenario_date.month,
        "hr": int(hour),
        "holiday": int(is_holiday),
        "weekday": weekday,
        "workingday": int(not is_weekend and not is_holiday),
        "weathersit": int(weathersit),
        **weather,
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def build_daily_scenario_features(
    scenario_date: date,
    *,
    temperature_c: float,
    apparent_temperature_c: float,
    humidity_percent: float,
    wind_speed_dataset_units: float,
    weathersit: int,
    holiday: bool | None = None,
) -> pd.DataFrame:
    """Build one chronologically ordered feature row for every hour of a day."""
    features = build_scenario_features(
        scenario_date,
        0,
        temperature_c=temperature_c,
        apparent_temperature_c=apparent_temperature_c,
        humidity_percent=humidity_percent,
        wind_speed_dataset_units=wind_speed_dataset_units,
        weathersit=weathersit,
        holiday=holiday,
    ).loc[np.repeat(0, 24)].reset_index(drop=True)
    features["hr"] = np.arange(24, dtype=int)
    return features


def load_simulator_model(path: Path = DEFAULT_MODEL_PATH) -> Pipeline:
    """Load and verify the persisted, fully fitted Random Forest pipeline."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Random Forest artifact not found: {path}")
    metadata_path = metadata_path_for_artifact(path)
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Model metadata not found: {metadata_path}. Regenerate the trusted artifact "
            "with urbanpulse-train-random-forest."
        )
    metadata = read_artifact_metadata(metadata_path)
    validate_artifact_compatibility(metadata)
    if metadata["artifact_sha256"] != sha256_file(path):
        raise ValueError("The model artifact hash does not match its metadata.")
    if tuple(metadata["features"]) != FEATURE_COLUMNS:
        raise ValueError("The artifact metadata feature order does not match simulator inputs.")
    # joblib/pickle can execute code while loading. Only deserialize a trusted artifact.
    model = joblib.load(path)
    if not isinstance(model, Pipeline) or not {"preprocessor", "regressor"}.issubset(model.named_steps):
        raise ValueError("The saved artifact is not a compatible UrbanPulse Random Forest pipeline.")
    feature_names = tuple(getattr(model, "feature_names_in_", ()))
    if feature_names != FEATURE_COLUMNS:
        raise ValueError("The saved model feature schema does not match the simulator inputs.")
    actual_parameters = model.named_steps["regressor"].get_params()
    expected_parameters = metadata["model_parameters"]
    for name in ("n_estimators", "min_samples_leaf", "random_state", "n_jobs"):
        if actual_parameters.get(name) != expected_parameters.get(name):
            raise ValueError(f"The saved model parameter '{name}' does not match its metadata.")
    return model


def predict_hourly_demand(model: Pipeline, features: pd.DataFrame) -> float:
    """Predict nonnegative rentals for one scenario row."""
    if tuple(features.columns) != FEATURE_COLUMNS or len(features) != 1:
        raise ValueError("features must contain exactly one row with the simulator feature schema.")
    prediction = float(np.asarray(model.predict(features), dtype=float)[0])
    if not np.isfinite(prediction):
        raise ValueError("The model returned a non-finite prediction.")
    return max(0.0, prediction)


def predict_daily_demand(model: Pipeline, features: pd.DataFrame) -> pd.DataFrame:
    """Predict all 24 hours in one model call and clamp negative estimates."""
    if tuple(features.columns) != FEATURE_COLUMNS or len(features) != 24:
        raise ValueError("features must contain 24 rows with the simulator feature schema.")
    if features["hr"].tolist() != list(range(24)):
        raise ValueError("features must be ordered by hour from 0 through 23.")
    predictions = np.asarray(model.predict(features), dtype=float)
    if predictions.shape != (24,) or not np.isfinite(predictions).all():
        raise ValueError("The model must return 24 finite hourly predictions.")
    return pd.DataFrame(
        {"hr": np.arange(24, dtype=int), "demand_estimated": np.maximum(predictions, 0.0)}
    )
