"""Metadata and integrity helpers for trusted joblib model artifacts."""

from __future__ import annotations

import hashlib
import json
import platform
import re
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn

ARTIFACT_METADATA_SCHEMA = 1
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def python_versions_compatible(expected: str, current: str) -> bool:
    """Allow patch updates within one Python major/minor series only."""
    version_pattern = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
    expected_match = version_pattern.fullmatch(expected)
    current_match = version_pattern.fullmatch(current)
    if expected_match is None or current_match is None:
        return False
    return expected_match.group(1, 2) == current_match.group(1, 2)


def metadata_path_for_artifact(artifact_path: Path) -> Path:
    """Return the sidecar path associated with a joblib artifact."""
    artifact_path = Path(artifact_path)
    return artifact_path.with_name(f"{artifact_path.name}.metadata.json")


def sha256_file(path: Path) -> str:
    """Calculate a file's SHA-256 digest without loading it into memory at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def current_runtime_versions() -> dict[str, str]:
    """Return runtime versions required to reproduce/load the serialized pipeline."""
    return {
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "pandas": pd.__version__,
        "joblib": joblib.__version__,
    }


def read_artifact_metadata(path: Path) -> dict[str, Any]:
    """Read a model sidecar and reject malformed or unsupported metadata."""
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            metadata = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read model metadata '{path}': {exc}") from exc
    if (
        not isinstance(metadata, dict)
        or type(metadata.get("metadata_schema")) is not int
        or metadata.get("metadata_schema") != ARTIFACT_METADATA_SCHEMA
    ):
        raise ValueError("Model metadata is missing or uses an unsupported schema version.")
    return metadata


def validate_artifact_compatibility(metadata: dict[str, Any]) -> None:
    """Check Python series plus exact serialization-library versions before unpickling."""
    expected = metadata.get("versions")
    if not isinstance(expected, dict):
        raise ValueError("Model metadata does not contain runtime versions.")
    current = current_runtime_versions()
    mismatches: list[str] = []
    for name, current_version in current.items():
        expected_version = expected.get(name)
        if not isinstance(expected_version, str):
            mismatches.append(f"{name}: missing or invalid version in artifact metadata")
        elif name == "python":
            if not python_versions_compatible(expected_version, current_version):
                mismatches.append(
                    f"python: artifact={expected_version!r}, current={current_version!r} "
                    "(only patch changes within the same major/minor series are supported)"
                )
        elif expected_version != current_version:
            mismatches.append(
                f"{name}: artifact={expected_version!r}, current={current_version!r}"
            )
    if mismatches:
        raise ValueError(
            "Incompatible model runtime versions; Python patch updates are accepted only "
            "within the same major/minor series, while serialization library versions "
            "must match exactly. "
            + "; ".join(mismatches)
        )
    required = {
        "artifact_sha256", "dataset_sha256", "features", "train_rows",
        "training_period", "model_parameters",
    }
    missing = sorted(required.difference(metadata))
    if missing:
        raise ValueError(f"Model metadata is missing required fields: {', '.join(missing)}.")
    features = metadata["features"]
    if (
        not isinstance(features, list)
        or not features
        or any(not isinstance(feature, str) or not feature for feature in features)
        or len(features) != len(set(features))
    ):
        raise ValueError("Model metadata contains an invalid feature list.")
    if type(metadata["train_rows"]) is not int or metadata["train_rows"] <= 0:
        raise ValueError("Model metadata contains an invalid training row count.")
    if not isinstance(metadata["training_period"], dict) or not {
        "start", "end"
    }.issubset(metadata["training_period"]):
        raise ValueError("Model metadata contains an invalid training period.")
    if any(
        not isinstance(metadata["training_period"][key], str)
        or not metadata["training_period"][key]
        for key in ("start", "end")
    ):
        raise ValueError("Model metadata contains invalid training period dates.")
    if not isinstance(metadata["model_parameters"], dict):
        raise ValueError("Model metadata contains invalid model parameters.")
    for name in ("dataset_sha256", "artifact_sha256"):
        value = metadata[name]
        if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
            raise ValueError(f"Model metadata contains an invalid '{name}' digest.")
