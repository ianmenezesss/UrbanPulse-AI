"""Train and persist the Random Forest pipeline on the initial 80% only."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib

from urbanpulse.ml.baseline import DEFAULT_DATA_PATH, TARGET, load_processed_data, temporal_train_test_split
from urbanpulse.ml.random_forest import build_random_forest_pipeline, select_features

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "random_forest.joblib"


@dataclass(frozen=True)
class TrainingArtifactResult:
    artifact_path: Path
    train_rows: int
    test_rows_reserved: int
    train_start: object
    train_end: object


def train_and_save_artifact(
    input_path: Path = DEFAULT_DATA_PATH,
    output_path: Path = DEFAULT_MODEL_PATH,
) -> TrainingArtifactResult:
    """Fit the full RF pipeline on the chronological first 80% and save it."""
    frame = load_processed_data(input_path)
    train, test_reserved = temporal_train_test_split(frame, train_fraction=0.8)
    pipeline = build_random_forest_pipeline()
    pipeline.fit(select_features(train), train[TARGET])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    return TrainingArtifactResult(
        artifact_path=output_path,
        train_rows=len(train),
        test_rows_reserved=len(test_reserved),
        train_start=train["datetime"].iloc[0],
        train_end=train["datetime"].iloc[-1],
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train and save the UrbanPulse Random Forest simulator model.")
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_PATH, help="Prepared Parquet path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH, help="Joblib artifact destination.")
    args = parser.parse_args(argv)
    try:
        result = train_and_save_artifact(args.input, args.output)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")

    print(f"Saved model: {result.artifact_path}")
    print(f"Training rows: {result.train_rows:,} ({result.train_start} — {result.train_end})")
    print(f"Reserved final test rows (not used for fitting): {result.test_rows_reserved:,}")
    print("Features: season, yr, mnth, hr, holiday, weekday, workingday, weathersit, temp, atemp, hum, windspeed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
