"""Build Streamlit Community Cloud Secrets values for a trusted model release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence
from urllib.parse import urlsplit, urlunsplit

from urbanpulse.ml.artifact_metadata import (
    metadata_path_for_artifact,
    read_artifact_metadata,
    sha256_file,
    validate_artifact_compatibility,
)
from urbanpulse.ml.train_artifact import DEFAULT_MODEL_PATH

MODEL_FILENAME = "random_forest.joblib"
MANIFEST_FILENAME = "random_forest.joblib.metadata.json"


def _normalize_release_base_url(release_base_url: str) -> str:
    if any(character.isspace() for character in release_base_url):
        raise ValueError("Release base URL must not contain whitespace.")
    parsed = urlsplit(release_base_url)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Release base URL must be an HTTPS URL without credentials, query, or fragment.")
    return urlunsplit(("https", parsed.netloc, parsed.path.rstrip("/"), "", ""))


def build_cloud_secrets_toml(
    release_base_url: str,
    *,
    model_path: Path = DEFAULT_MODEL_PATH,
    manifest_path: Path | None = None,
) -> str:
    """Return TOML for the four required secrets, pinning both release files."""
    base_url = _normalize_release_base_url(release_base_url)
    model_path = Path(model_path)
    manifest_path = Path(manifest_path or metadata_path_for_artifact(model_path))
    if model_path.name != MODEL_FILENAME or manifest_path.name != MANIFEST_FILENAME:
        raise ValueError(
            f"Release assets must be named exactly '{MODEL_FILENAME}' and '{MANIFEST_FILENAME}'."
        )
    if not model_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Both the model artifact and its JSON manifest must exist locally.")

    metadata = read_artifact_metadata(manifest_path)
    validate_artifact_compatibility(metadata)
    model_sha256 = sha256_file(model_path)
    if metadata.get("artifact_sha256") != model_sha256:
        raise ValueError("The model file SHA-256 does not match artifact_sha256 in the JSON manifest.")
    manifest_sha256 = sha256_file(manifest_path)
    model_url = f"{base_url}/{MODEL_FILENAME}"
    manifest_url = f"{base_url}/{MANIFEST_FILENAME}"

    # JSON string literals are valid TOML basic strings for these validated URLs.
    return "\n".join(
        (
            "[cloud_resources]",
            f"model_url = {json.dumps(model_url)}",
            f"model_sha256 = {json.dumps(model_sha256)}",
            f"metadata_url = {json.dumps(manifest_url)}",
            f"metadata_sha256 = {json.dumps(manifest_sha256)}",
            "",
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print Streamlit Cloud Secrets for an existing trusted GitHub Release."
    )
    parser.add_argument(
        "--release-base-url",
        required=True,
        help="HTTPS URL through the release tag, without either asset filename.",
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--manifest-path", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        secrets_toml = build_cloud_secrets_toml(
            args.release_base_url,
            model_path=args.model_path,
            manifest_path=args.manifest_path,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    print(secrets_toml, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
