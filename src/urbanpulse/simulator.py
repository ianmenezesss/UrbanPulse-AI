"""Input conversion and prediction helpers for the demand simulator."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from urbanpulse.ml.ridge import FEATURE_COLUMNS
from urbanpulse.ml.train_artifact import DEFAULT_MODEL_PATH

YEAR_CODES = {2011: 0, 2012: 1}


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
    holiday: bool,
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
    weather = convert_weather_to_dataset_scale(
        temperature_c,
        apparent_temperature_c,
        humidity_percent,
        wind_speed_dataset_units,
    )
    row = {
        "season": (scenario_date.month - 1) // 3 + 1,
        "yr": YEAR_CODES[scenario_date.year],
        "mnth": scenario_date.month,
        "hr": int(hour),
        "holiday": int(bool(holiday)),
        "weekday": weekday,
        "workingday": int(not is_weekend and not holiday),
        "weathersit": int(weathersit),
        **weather,
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def load_simulator_model(path: Path = DEFAULT_MODEL_PATH) -> Pipeline:
    """Load and verify the persisted, fully fitted Random Forest pipeline."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Random Forest artifact not found: {path}")
    model = joblib.load(path)
    if not isinstance(model, Pipeline) or not {"preprocessor", "regressor"}.issubset(model.named_steps):
        raise ValueError("The saved artifact is not a compatible UrbanPulse Random Forest pipeline.")
    feature_names = tuple(getattr(model, "feature_names_in_", ()))
    if feature_names != FEATURE_COLUMNS:
        raise ValueError("The saved model feature schema does not match the simulator inputs.")
    return model


def predict_hourly_demand(model: Pipeline, features: pd.DataFrame) -> float:
    """Predict nonnegative rentals for one scenario row."""
    if tuple(features.columns) != FEATURE_COLUMNS or len(features) != 1:
        raise ValueError("features must contain exactly one row with the simulator feature schema.")
    prediction = float(np.asarray(model.predict(features), dtype=float)[0])
    if not np.isfinite(prediction):
        raise ValueError("The model returned a non-finite prediction.")
    return max(0.0, prediction)
