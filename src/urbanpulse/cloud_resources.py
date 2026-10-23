"""One-time, verified resource preparation for Streamlit Community Cloud."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from urllib.parse import urlparse

from urbanpulse.dashboard_data import (
    DEFAULT_FINAL_METRICS_PATH,
    DEFAULT_IMPORTANCE_REPORT_PATH,
    DEFAULT_TEMPORAL_REPORT_DIR,
    load_final_model_metrics,
    load_permutation_importance_report,
    load_temporal_validation_reports,
)
from urbanpulse.ingest import DEFAULT_DESTINATION as DEFAULT_RAW_PATH
from urbanpulse.ingest import IngestionError, download_hour_csv
from urbanpulse.ml.artifact_metadata import (
    metadata_path_for_artifact,
    read_artifact_metadata,
    sha256_file,
    validate_artifact_compatibility,
)
from urbanpulse.ml.train_artifact import DEFAULT_MODEL_PATH
from urbanpulse.prepare import DEFAULT_DESTINATION as DEFAULT_PROCESSED_PATH
from urbanpulse.prepare import PreparationError, prepare_hour_dataset

MAX_MODEL_BYTES = 128 * 1024 * 1024
MAX_METADATA_BYTES = 1024 * 1024
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class CloudResourceError(RuntimeError):
    """Raised when deployment resources cannot be safely prepared."""


@dataclass(frozen=True)
class CloudPreparationResult:
    processed_data: Path
    available_reports: tuple[str, ...]
    model_available: bool


def _validate_https_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise CloudResourceError("External model and metadata URLs must use HTTPS.")


def _download_to_path(
    url: str,
    destination: Path,
    expected_sha256: str,
    *,
    max_bytes: int,
) -> Path:
    """Stream one HTTPS asset to a temporary file, verify it, then atomically install it."""
    _validate_https_url(url)
    if not SHA256_PATTERN.fullmatch(expected_sha256):
        raise CloudResourceError("An expected SHA-256 digest must contain exactly 64 hexadecimal characters.")

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    digest = hashlib.sha256()
    total_bytes = 0
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".download",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            request = urllib.request.Request(url, headers={"User-Agent": "UrbanPulse-AI/0.1"})
            with urllib.request.urlopen(request, timeout=60) as response:
                status = getattr(response, "status", 200)
                if status != 200 or urlparse(response.geturl()).scheme != "https":
                    raise CloudResourceError("The external resource returned an invalid HTTPS response.")
                while block := response.read(1024 * 1024):
                    total_bytes += len(block)
                    if total_bytes > max_bytes:
                        raise CloudResourceError("The external resource exceeds the allowed download size.")
                    digest.update(block)
                    temporary.write(block)
        if digest.hexdigest() != expected_sha256.lower():
            raise CloudResourceError(f"SHA-256 verification failed for '{destination.name}'.")
        os.replace(temporary_path, destination)
        temporary_path = None
    except CloudResourceError:
        raise
    except (OSError, TimeoutError, urllib.error.URLError) as exc:
        raise CloudResourceError(
            f"Could not download the configured external resource '{destination.name}'."
        ) from exc
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination


def _validate_local_model(
    model_path: Path,
    metadata_path: Path,
    *,
    expected_model_sha256: str | None,
    expected_metadata_sha256: str | None,
) -> None:
    metadata = read_artifact_metadata(metadata_path)
    validate_artifact_compatibility(metadata)
    model_digest = sha256_file(model_path)
    metadata_digest = sha256_file(metadata_path)
    if metadata["artifact_sha256"] != model_digest:
        raise CloudResourceError("The model file does not match the artifact hash in its metadata.")
    if expected_model_sha256 and model_digest != expected_model_sha256.lower():
        raise CloudResourceError("The model file does not match the independently configured SHA-256 pin.")
    if expected_metadata_sha256 and metadata_digest != expected_metadata_sha256.lower():
        raise CloudResourceError("The model metadata does not match the configured SHA-256 pin.")


def _prepare_model(
    model_path: Path,
    *,
    model_url: str | None,
    model_sha256: str | None,
    metadata_url: str | None,
    metadata_sha256: str | None,
) -> bool:
    metadata_path = metadata_path_for_artifact(model_path)
    configured = (model_url, model_sha256, metadata_url, metadata_sha256)
    if any(configured) and not all(configured):
        raise CloudResourceError(
            "Configure model_url, model_sha256, metadata_url, and metadata_sha256 together."
        )

    if model_path.is_file() and metadata_path.is_file():
        _validate_local_model(
            model_path,
            metadata_path,
            expected_model_sha256=model_sha256,
            expected_metadata_sha256=metadata_sha256,
        )
        return True
    if not all(configured):
        return False

    # Keep the pair staged until both independent hashes and the manifest agree.
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=model_path.parent, prefix="urbanpulse-model-") as temporary_dir:
        staged_model = Path(temporary_dir) / model_path.name
        staged_metadata = Path(temporary_dir) / metadata_path.name
        _download_to_path(model_url, staged_model, model_sha256, max_bytes=MAX_MODEL_BYTES)
        _download_to_path(metadata_url, staged_metadata, metadata_sha256, max_bytes=MAX_METADATA_BYTES)
        _validate_local_model(
            staged_model,
            staged_metadata,
            expected_model_sha256=model_sha256,
            expected_metadata_sha256=metadata_sha256,
        )
        os.replace(staged_model, model_path)
        os.replace(staged_metadata, metadata_path)
    return True


def prepare_cloud_resources(
    *,
    raw_path: Path = DEFAULT_RAW_PATH,
    processed_path: Path = DEFAULT_PROCESSED_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    final_metrics_path: Path = DEFAULT_FINAL_METRICS_PATH,
    temporal_report_dir: Path = DEFAULT_TEMPORAL_REPORT_DIR,
    importance_report_path: Path = DEFAULT_IMPORTANCE_REPORT_PATH,
    model_url: str | None = None,
    model_sha256: str | None = None,
    metadata_url: str | None = None,
    metadata_sha256: str | None = None,
) -> CloudPreparationResult:
    """Fetch/prepare missing data and verify precomputed reports and model without training."""
    try:
        download_hour_csv(Path(raw_path))
        if not Path(processed_path).is_file():
            prepare_hour_dataset(Path(raw_path), Path(processed_path))
    except (IngestionError, PreparationError, OSError, FileExistsError) as exc:
        raise CloudResourceError("Could not prepare the official UCI dataset for the dashboard.") from exc

    available_reports: list[str] = []
    if Path(final_metrics_path).is_file():
        load_final_model_metrics(Path(final_metrics_path))
        available_reports.append("final_metrics")
    if Path(temporal_report_dir).is_dir():
        try:
            load_temporal_validation_reports(Path(temporal_report_dir))
        except FileNotFoundError:
            pass
        else:
            available_reports.append("temporal_validation")
    if Path(importance_report_path).is_file():
        load_permutation_importance_report(Path(importance_report_path))
        available_reports.append("interpretability")

    try:
        model_available = _prepare_model(
            Path(model_path),
            model_url=model_url,
            model_sha256=model_sha256,
            metadata_url=metadata_url,
            metadata_sha256=metadata_sha256,
        )
    except (OSError, ValueError) as exc:
        raise CloudResourceError("The configured Random Forest artifact failed compatibility or integrity checks.") from exc

    return CloudPreparationResult(
        processed_data=Path(processed_path),
        available_reports=tuple(available_reports),
        model_available=model_available,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare UrbanPulse deployment resources; this command never trains a model."
    )
    parser.add_argument("--model-url", default=os.getenv("URBANPULSE_MODEL_URL"))
    parser.add_argument("--model-sha256", default=os.getenv("URBANPULSE_MODEL_SHA256"))
    parser.add_argument("--metadata-url", default=os.getenv("URBANPULSE_MODEL_METADATA_URL"))
    parser.add_argument("--metadata-sha256", default=os.getenv("URBANPULSE_MODEL_METADATA_SHA256"))
    args = parser.parse_args(argv)
    try:
        result = prepare_cloud_resources(
            model_url=args.model_url,
            model_sha256=args.model_sha256,
            metadata_url=args.metadata_url,
            metadata_sha256=args.metadata_sha256,
        )
    except (CloudResourceError, OSError, ValueError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")

    print(f"Processed data ready: {result.processed_data}")
    print(f"Validated reports: {', '.join(result.available_reports) or 'none'}")
    print(f"Random Forest artifact available: {result.model_available}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
