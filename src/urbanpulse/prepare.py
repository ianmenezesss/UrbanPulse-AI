"""Prepare the validated hourly dataset as a chronologically ordered Parquet file."""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
from pathlib import Path
from typing import Sequence

import pandas as pd

from urbanpulse.validation import DEFAULT_SOURCE, validate_hour_csv

DEFAULT_DESTINATION = Path(__file__).resolve().parents[2] / "data" / "processed" / "hour.parquet"
LOGGER = logging.getLogger(__name__)


class PreparationError(RuntimeError):
    """Raised when validation or preparation cannot be completed."""


def prepare_hour_dataset(
    source: Path = DEFAULT_SOURCE,
    destination: Path = DEFAULT_DESTINATION,
    *,
    force: bool = False,
) -> Path:
    """Validate, add a timestamp, sort, and write the hourly dataset to Parquet.

    Source columns are preserved, including ``casual``, ``registered`` and
    ``cnt``. The original CSV is only read and is never modified.
    """
    source = Path(source)
    destination = Path(destination)

    report = validate_hour_csv(source)
    if not report.is_valid:
        details = " ".join(report.errors)
        raise PreparationError(f"Input validation failed; no output was written. {details}")

    if destination.exists() and not force:
        raise FileExistsError(
            f"Output already exists: '{destination}'. Pass --force to replace it."
        )

    try:
        frame = pd.read_csv(source, low_memory=False)
        if len(frame) != report.row_count:
            raise PreparationError(
                "The source row count changed between validation and preparation; run the command again."
            )

        frame["datetime"] = pd.to_datetime(
            frame["dteday"], format="%Y-%m-%d"
        ) + pd.to_timedelta(frame["hr"], unit="h")
        frame = frame.sort_values("datetime", kind="stable", ignore_index=True)
        if len(frame) != report.row_count:
            raise PreparationError("Preparation changed the number of rows; refusing to write output.")
    except PreparationError:
        raise
    except (OSError, ValueError, TypeError, pd.errors.ParserError) as exc:
        raise PreparationError(f"Could not prepare input CSV '{source}': {exc}") from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp.parquet",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        frame.to_parquet(temporary_path, engine="pyarrow", index=False)
        if force:
            os.replace(temporary_path, destination)
        else:
            # On Windows, rename fails if another process created the destination
            # after the initial existence check, preserving the no-overwrite rule.
            os.rename(temporary_path, destination)
    except FileExistsError:
        raise FileExistsError(
            f"Output already exists: '{destination}'. Pass --force to replace it."
        ) from None
    except (OSError, ImportError, ValueError) as exc:
        raise PreparationError(f"Could not write Parquet output '{destination}': {exc}") from exc
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    LOGGER.info(
        "Prepared dataset. source=%s destination=%s rows=%d columns=%d",
        source,
        destination,
        len(frame),
        len(frame.columns),
    )
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate and prepare the hourly bike-sharing dataset as Parquet."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_SOURCE, help="Input CSV path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_DESTINATION, help="Output Parquet path.")
    parser.add_argument("--force", action="store_true", help="Replace an existing output file.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        output = prepare_hour_dataset(args.input, args.output, force=args.force)
    except (PreparationError, FileExistsError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")

    LOGGER.info("Prepared Parquet file is ready at %s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
