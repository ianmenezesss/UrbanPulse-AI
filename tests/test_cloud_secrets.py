from __future__ import annotations

import hashlib
import json
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

from urbanpulse.cloud_secrets import build_cloud_secrets_toml


class CloudSecretsConfigTests(unittest.TestCase):
    def test_requirements_install_local_package_and_dashboard_dependencies(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        requirements = (project_root / "requirements.txt").read_text(encoding="utf-8")
        pyproject = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))

        self.assertIn("-e .[cloud]", requirements)
        cloud_dependencies = pyproject["project"]["optional-dependencies"]["cloud"]
        for dependency in ("joblib", "pandas", "plotly", "pyarrow", "scikit-learn", "streamlit"):
            self.assertTrue(any(item.lower().startswith(f"{dependency.lower()}==") for item in cloud_dependencies))
        self.assertTrue((project_root / "src" / "urbanpulse" / "__init__.py").is_file())

    def test_config_uses_release_base_and_hashes_manifest_bytes_separately(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "random_forest.joblib"
            manifest_path = root / "random_forest.joblib.metadata.json"
            model_bytes = b"trusted model content"
            model_path.write_bytes(model_bytes)
            manifest_bytes = (
                json.dumps(
                    {
                        "metadata_schema": 1,
                        "artifact_sha256": hashlib.sha256(model_bytes).hexdigest(),
                    },
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
            manifest_path.write_bytes(manifest_bytes)
            with patch("urbanpulse.cloud_secrets.validate_artifact_compatibility"):
                config = tomllib.loads(
                    build_cloud_secrets_toml(
                        "https://github.com/acme/urbanpulse/releases/download/v1/",
                        model_path=model_path,
                        manifest_path=manifest_path,
                    )
                )

        secrets = config["cloud_resources"]
        self.assertEqual(
            secrets["model_url"],
            "https://github.com/acme/urbanpulse/releases/download/v1/random_forest.joblib",
        )
        self.assertEqual(
            secrets["metadata_url"],
            "https://github.com/acme/urbanpulse/releases/download/v1/"
            "random_forest.joblib.metadata.json",
        )
        self.assertEqual(secrets["model_sha256"], hashlib.sha256(model_bytes).hexdigest())
        self.assertEqual(secrets["metadata_sha256"], hashlib.sha256(manifest_bytes).hexdigest())
        self.assertNotEqual(secrets["metadata_sha256"], secrets["model_sha256"])

    def test_http_release_base_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "HTTPS"):
            build_cloud_secrets_toml("http://github.com/acme/urbanpulse/releases/download/v1")

    def test_manifest_must_match_the_local_model_before_printing_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "random_forest.joblib"
            manifest_path = root / "random_forest.joblib.metadata.json"
            model_path.write_bytes(b"model")
            manifest_path.write_text(
                json.dumps({"metadata_schema": 1, "artifact_sha256": "0" * 64}),
                encoding="utf-8",
            )
            with patch("urbanpulse.cloud_secrets.validate_artifact_compatibility"):
                with self.assertRaisesRegex(ValueError, "does not match"):
                    build_cloud_secrets_toml(
                        "https://github.com/acme/urbanpulse/releases/download/v1",
                        model_path=model_path,
                        manifest_path=manifest_path,
                    )


if __name__ == "__main__":
    unittest.main()
