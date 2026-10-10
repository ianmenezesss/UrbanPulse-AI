from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from urbanpulse.validation import DataValidationError, validate_hour_csv

VALID_ROW = {
    "instant": 1,
    "dteday": "2011-01-01",
    "season": 1,
    "yr": 0,
    "mnth": 1,
    "hr": 0,
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


class ValidateHourCsvTests(unittest.TestCase):
    def write_csv(self, rows: list[dict], directory: str) -> Path:
        path = Path(directory) / "hour.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def test_valid_dataset_passes_and_reports_shape_and_types(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv([VALID_ROW], directory)
            original = path.read_bytes()
            report = validate_hour_csv(path)

            self.assertTrue(report.is_valid, report.errors)
            self.assertEqual((report.row_count, report.column_count), (1, 17))
            self.assertIn("int", report.dtypes["instant"])
            self.assertEqual(report.missing_values, {})
            self.assertEqual(report.duplicate_rows, 0)
            self.assertEqual(path.read_bytes(), original)

    def test_empty_dataset_is_a_blocking_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hour.csv"
            pd.DataFrame(columns=VALID_ROW.keys()).to_csv(path, index=False)
            report = validate_hour_csv(path)
        self.assertFalse(report.is_valid)
        self.assertTrue(any("no data rows" in error for error in report.errors))
    def test_missing_required_columns_are_errors(self):
        row = {name: value for name, value in VALID_ROW.items() if name != "cnt"}
        with tempfile.TemporaryDirectory() as directory:
            report = validate_hour_csv(self.write_csv([row], directory))
        self.assertFalse(report.is_valid)
        self.assertTrue(any("Missing required columns" in error for error in report.errors))

    def test_missing_values_are_blocking_errors(self):
        row = dict(VALID_ROW, hum=None)
        with tempfile.TemporaryDirectory() as directory:
            report = validate_hour_csv(self.write_csv([row], directory))
        self.assertFalse(report.is_valid)
        self.assertEqual(report.missing_values, {"hum": 1})

    def test_invalid_domains_and_count_relationship_are_errors(self):
        row = dict(VALID_ROW, hr=24, temp=1.2, cnt=17)
        with tempfile.TemporaryDirectory() as directory:
            report = validate_hour_csv(self.write_csv([row], directory))
        self.assertFalse(report.is_valid)
        self.assertTrue(any("outside the allowed range" in error for error in report.errors))
        self.assertTrue(any("casual + registered" in error for error in report.errors))

    def test_invalid_numeric_type_is_an_error(self):
        row = dict(VALID_ROW, season="spring")
        with tempfile.TemporaryDirectory() as directory:
            report = validate_hour_csv(self.write_csv([row], directory))
        self.assertFalse(report.is_valid)
        self.assertTrue(any("non-numeric" in error for error in report.errors))

    def test_duplicate_rows_are_reported_as_warning(self):
        with tempfile.TemporaryDirectory() as directory:
            report = validate_hour_csv(self.write_csv([VALID_ROW, VALID_ROW], directory))
        self.assertFalse(report.is_valid)
        self.assertEqual(report.duplicate_rows, 1)
        self.assertTrue(any("duplicated row" in warning for warning in report.warnings))
        self.assertTrue(any("duplicate identifier" in error for error in report.errors))

    def test_invalid_date_and_calendar_mismatch_are_errors(self):
        row = dict(VALID_ROW, dteday="2011-02-01")
        with tempfile.TemporaryDirectory() as directory:
            report = validate_hour_csv(self.write_csv([row], directory))
        self.assertFalse(report.is_valid)
        self.assertTrue(any("mnth" in error for error in report.errors))

    def test_unreadable_path_raises_clear_error(self):
        with self.assertRaisesRegex(DataValidationError, "Could not read input CSV"):
            validate_hour_csv(Path("missing-hour.csv"))

    def test_extra_columns_are_informational_warnings(self):
        row = dict(VALID_ROW, note="extra")
        with tempfile.TemporaryDirectory() as directory:
            report = validate_hour_csv(self.write_csv([row], directory))
        self.assertTrue(report.is_valid)
        self.assertTrue(any("Unexpected columns" in warning for warning in report.warnings))


if __name__ == "__main__":
    unittest.main()




