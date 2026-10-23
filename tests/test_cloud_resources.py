from __future__ import annotations

import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from urbanpulse.cloud_resources import (
    CloudResourceError,
    _download_to_path,
    _prepare_model,
)


class FakeResponse(io.BytesIO):
    status = 200

    def __init__(self, content: bytes, *, url: str = "https://assets.example/model") -> None:
        super().__init__(content)
        self.url = url

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def geturl(self) -> str:
        return self.url


class VerifiedDownloadTests(unittest.TestCase):
    def test_download_streams_verifies_hash_and_replaces_destination(self) -> None:
        content = b"trusted model bytes"
        digest = hashlib.sha256(content).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "model.joblib"
            destination.write_bytes(b"old")
            with patch(
                "urbanpulse.cloud_resources.urllib.request.urlopen",
                return_value=FakeResponse(content),
            ):
                result = _download_to_path(
                    "https://assets.example/model",
                    destination,
                    digest,
                    max_bytes=1024,
                )
            self.assertEqual(result.read_bytes(), content)

    def test_hash_mismatch_preserves_existing_destination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "model.joblib"
            destination.write_bytes(b"old trusted file")
            with patch(
                "urbanpulse.cloud_resources.urllib.request.urlopen",
                return_value=FakeResponse(b"unexpected bytes"),
            ):
                with self.assertRaisesRegex(CloudResourceError, "SHA-256"):
                    _download_to_path(
                        "https://assets.example/model",
                        destination,
                        hashlib.sha256(b"expected").hexdigest(),
                        max_bytes=1024,
                    )
            self.assertEqual(destination.read_bytes(), b"old trusted file")

    def test_non_https_assets_are_rejected_before_request(self) -> None:
        with patch("urbanpulse.cloud_resources.urllib.request.urlopen") as open_url:
            with self.assertRaisesRegex(CloudResourceError, "HTTPS"):
                _download_to_path(
                    "http://assets.example/model",
                    Path("model.joblib"),
                    "0" * 64,
                    max_bytes=1024,
                )
        open_url.assert_not_called()

    def test_download_size_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "urbanpulse.cloud_resources.urllib.request.urlopen",
                return_value=FakeResponse(b"larger than limit"),
            ):
                with self.assertRaisesRegex(CloudResourceError, "allowed download size"):
                    _download_to_path(
                        "https://assets.example/model",
                        Path(directory) / "model.joblib",
                        hashlib.sha256(b"larger than limit").hexdigest(),
                        max_bytes=4,
                    )

    def test_partial_model_configuration_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(CloudResourceError, "together"):
                _prepare_model(
                    Path(directory) / "model.joblib",
                    model_url="https://assets.example/model",
                    model_sha256="0" * 64,
                    metadata_url=None,
                    metadata_sha256=None,
                )


if __name__ == "__main__":
    unittest.main()
