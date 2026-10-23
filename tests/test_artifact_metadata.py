from __future__ import annotations

import unittest
from unittest.mock import patch

from urbanpulse.ml.artifact_metadata import (
    python_versions_compatible,
    validate_artifact_compatibility,
)


def valid_metadata() -> dict[str, object]:
    return {
        "metadata_schema": 1,
        "versions": {
            "python": "3.12.10",
            "scikit_learn": "1.9.1",
            "pandas": "3.0.6",
            "joblib": "1.6.0",
        },
        "artifact_sha256": "a" * 64,
        "dataset_sha256": "b" * 64,
        "features": ["hr", "temp"],
        "train_rows": 100,
        "training_period": {"start": "2011-01-01", "end": "2011-01-05"},
        "model_parameters": {"n_estimators": 100},
    }


class PythonVersionCompatibilityTests(unittest.TestCase):
    def test_python_patch_updates_in_same_minor_series_are_compatible(self) -> None:
        self.assertTrue(python_versions_compatible("3.12.10", "3.12.0"))
        self.assertTrue(python_versions_compatible("3.12.10", "3.12.99"))

    def test_python_major_or_minor_changes_are_incompatible(self) -> None:
        self.assertFalse(python_versions_compatible("3.12.10", "3.13.0"))
        self.assertFalse(python_versions_compatible("3.12.10", "3.11.9"))
        self.assertFalse(python_versions_compatible("3.12", "3.12.1"))

    def test_manifest_accepts_python_patch_change_but_keeps_library_pins_exact(self) -> None:
        metadata = valid_metadata()
        runtime = {
            "python": "3.12.12",
            "scikit_learn": "1.9.1",
            "pandas": "3.0.6",
            "joblib": "1.6.0",
        }
        with patch("urbanpulse.ml.artifact_metadata.current_runtime_versions", return_value=runtime):
            validate_artifact_compatibility(metadata)

        runtime["scikit_learn"] = "1.9.2"
        with patch("urbanpulse.ml.artifact_metadata.current_runtime_versions", return_value=runtime):
            with self.assertRaisesRegex(ValueError, "scikit_learn"):
                validate_artifact_compatibility(metadata)

    def test_manifest_rejects_python_minor_change(self) -> None:
        metadata = valid_metadata()
        runtime = {
            "python": "3.13.0",
            "scikit_learn": "1.9.1",
            "pandas": "3.0.6",
            "joblib": "1.6.0",
        }
        with patch("urbanpulse.ml.artifact_metadata.current_runtime_versions", return_value=runtime):
            with self.assertRaisesRegex(ValueError, "same major/minor"):
                validate_artifact_compatibility(metadata)


if __name__ == "__main__":
    unittest.main()
