from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from urbanpulse.prepare import PreparationError, prepare_hour_dataset


def valid_rows() -> list[dict[str, object]]:
    common: dict[str, object] = {
        "season": 1,
        "yr": 0,
        "mnth": 1,
        "holiday": 0,
        "weekday": 6,
        "workingday": 0,
        "weathersit": 1,
        "temp": 0.24,
        "atemp": 0.28,
        "hum": 0.81,
        "windspeed": 0.0,
        "casual": 3,
        "registered": 13,
        "cnt": 16,
    }
    # Deliberately out of chronological order; all target columns remain present.
    return [
        {**common, "instant": 3, "dteday": "2011-01-02", "hr": 0},
        {**common, "instant": 1, "dteday": "2011-01-01", "hr": 2},
        {**common, "instant": 2, "dteday": "2011-01-01", "hr": 0},
    ]


class PrepareHourDatasetTests(unittest.TestCase):
    def write_source(self, directory: str, rows: list[dict[str, object]] | None = None) -> Path:
        source = Path(directory) / "hour.csv"
        pd.DataFrame(valid_rows() if rows is None else rows).to_csv(source, index=False)
        return source

    def test_creates_sorted_parquet_preserving_columns_rows_and_types(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.write_source(directory)
            original_bytes = source.read_bytes()
            destination = Path(directory) / "processed" / "hour.parquet"

            result = prepare_hour_dataset(source, destination)
            prepared = pd.read_parquet(result, engine="pyarrow")

            self.assertEqual(result, destination)
            self.assertEqual(len(prepared), 3)
            self.assertEqual(
                list(prepared["datetime"].astype(str)),
                ["2011-01-01 00:00:00", "2011-01-01 02:00:00", "2011-01-02 00:00:00"],
            )
            self.assertIn("casual", prepared.columns)
            self.assertIn("registered", prepared.columns)
            self.assertIn("cnt", prepared.columns)
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(prepared["datetime"]))
            self.assertTrue(pd.api.types.is_integer_dtype(prepared["instant"]))
            self.assertTrue(pd.api.types.is_float_dtype(prepared["temp"]))
            self.assertEqual(source.read_bytes(), original_bytes)

    def test_existing_output_is_not_overwritten_without_force(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.write_source(directory)
            destination = Path(directory) / "hour.parquet"
            destination.write_bytes(b"existing output")

            with self.assertRaisesRegex(FileExistsError, "Pass --force"):
                prepare_hour_dataset(source, destination)

            self.assertEqual(destination.read_bytes(), b"existing output")

    def test_force_replaces_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.write_source(directory)
            destination = Path(directory) / "hour.parquet"
            destination.write_bytes(b"old output")

            prepare_hour_dataset(source, destination, force=True)
            prepared = pd.read_parquet(destination, engine="pyarrow")

            self.assertEqual(len(prepared), 3)
            self.assertNotEqual(destination.read_bytes(), b"old output")

    def test_invalid_data_stops_before_output_is_written(self):
        rows = valid_rows()
        rows[0]["cnt"] = 999
        with tempfile.TemporaryDirectory() as directory:
            source = self.write_source(directory, rows)
            destination = Path(directory) / "hour.parquet"

            with self.assertRaisesRegex(PreparationError, "Input validation failed"):
                prepare_hour_dataset(source, destination)

            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
