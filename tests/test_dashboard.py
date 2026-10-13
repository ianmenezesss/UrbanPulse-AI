from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from urbanpulse.dashboard_data import filter_dashboard_data, load_dashboard_data


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2011-01-03 08:00", "2011-01-08 10:00", "2012-01-04 09:00", "2012-01-07 11:00"]
            ),
            "yr": [0, 0, 1, 1],
            "workingday": [1, 0, 1, 0],
            "hr": [8, 10, 9, 11],
            "weekday": [1, 6, 3, 6],
            "mnth": [1, 1, 1, 1],
            "temp": [0.2, 0.3, 0.4, 0.5],
            "cnt": [100, 200, 300, 400],
        }
    )


class DashboardDataTests(unittest.TestCase):
    def test_load_dashboard_data_reads_parquet(self):
        expected = sample_frame()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.parquet"
            expected.to_parquet(path, index=False)

            loaded = load_dashboard_data(path)

        pd.testing.assert_frame_equal(loaded, expected)

    def test_missing_parquet_raises_file_not_found(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load_dashboard_data(Path(directory) / "missing.parquet")

    def test_load_rejects_missing_required_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.parquet"
            pd.DataFrame({"cnt": [1]}).to_parquet(path, index=False)

            with self.assertRaisesRegex(ValueError, "missing required columns"):
                load_dashboard_data(path)

    def test_filter_by_year_and_all_years(self):
        frame = sample_frame()

        self.assertEqual(len(filter_dashboard_data(frame, year=2011)), 2)
        self.assertEqual(len(filter_dashboard_data(frame, year=2012)), 2)
        self.assertEqual(len(filter_dashboard_data(frame)), 4)

    def test_filter_by_workingday_definition(self):
        frame = sample_frame()

        workdays = filter_dashboard_data(frame, workingday=True)
        non_workdays = filter_dashboard_data(frame, workingday=False)

        self.assertEqual(workdays["cnt"].tolist(), [100, 300])
        self.assertEqual(non_workdays["cnt"].tolist(), [200, 400])

    def test_year_and_workingday_filters_combine_without_mutating_input(self):
        frame = sample_frame()
        original = frame.copy(deep=True)

        filtered = filter_dashboard_data(frame, year=2012, workingday=False)

        self.assertEqual(filtered["cnt"].tolist(), [400])
        pd.testing.assert_frame_equal(frame, original)

    def test_rejects_unsupported_year(self):
        with self.assertRaisesRegex(ValueError, "year must be"):
            filter_dashboard_data(sample_frame(), year=2013)


if __name__ == "__main__":
    unittest.main()
