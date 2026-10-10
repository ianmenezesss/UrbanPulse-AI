from __future__ import annotations

import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from urbanpulse.ingest import DATASET_URL, IngestionError, download_hour_csv


def make_archive(csv_bytes: bytes = b"instant,dteday,hr,cnt\n1,2011-01-01,0,16\n") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("Bike-Sharing-Dataset/hour.csv", csv_bytes)
    return buffer.getvalue()


class FakeResponse(io.BytesIO):
    def __init__(self, payload: bytes, status: int = 200):
        super().__init__(payload)
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class DownloadHourCsvTests(unittest.TestCase):
    def test_download_saves_original_csv_and_creates_destination(self):
        original = b"instant,dteday,hr,cnt\n1,2011-01-01,0,16\n"
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "nested" / "hour.csv"
            with patch("urbanpulse.ingest.urllib.request.urlopen", return_value=FakeResponse(make_archive(original))) as urlopen:
                result = download_hour_csv(destination)

            self.assertEqual(result, destination)
            self.assertTrue(destination.is_file())
            self.assertEqual(destination.read_bytes(), original)
            urlopen.assert_called_once()
            self.assertEqual(urlopen.call_args.args[0].full_url, DATASET_URL)

    def test_existing_file_skips_network_without_force(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "hour.csv"
            destination.write_bytes(b"keep original")
            with patch("urbanpulse.ingest.urllib.request.urlopen") as urlopen:
                download_hour_csv(destination)
            urlopen.assert_not_called()
            self.assertEqual(destination.read_bytes(), b"keep original")

    def test_force_replaces_existing_file(self):
        replacement = b"new original bytes"
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "hour.csv"
            destination.write_bytes(b"old data")
            with patch("urbanpulse.ingest.urllib.request.urlopen", return_value=FakeResponse(make_archive(replacement))):
                download_hour_csv(destination, force=True)
            self.assertEqual(destination.read_bytes(), replacement)

    def test_invalid_http_status_raises_and_does_not_create_file(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "hour.csv"
            with patch("urbanpulse.ingest.urllib.request.urlopen", return_value=FakeResponse(b"", status=503)):
                with self.assertRaises(IngestionError):
                    download_hour_csv(destination)
            self.assertFalse(destination.exists())

    def test_connection_failure_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "hour.csv"
            with patch("urbanpulse.ingest.urllib.request.urlopen", side_effect=OSError("connection failed")):
                with self.assertRaisesRegex(IngestionError, "connection failed"):
                    download_hour_csv(destination)

    def test_invalid_zip_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "hour.csv"
            with patch("urbanpulse.ingest.urllib.request.urlopen", return_value=FakeResponse(b"not a zip")):
                with self.assertRaises(IngestionError):
                    download_hour_csv(destination)


if __name__ == "__main__":
    unittest.main()
