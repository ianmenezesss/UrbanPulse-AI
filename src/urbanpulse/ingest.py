"""Download the official UCI hourly bike-sharing CSV without transforming it."""

from __future__ import annotations

import argparse
import io
import logging
import os
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

DATASET_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
DEFAULT_DESTINATION = Path(__file__).resolve().parents[2] / "data" / "raw" / "hour.csv"
LOGGER = logging.getLogger(__name__)


class IngestionError(RuntimeError):
    """Raised when the official dataset cannot be downloaded or read."""


def download_hour_csv(
    destination: Path = DEFAULT_DESTINATION,
    *,
    force: bool = False,
    url: str = DATASET_URL,
    timeout: float = 30,
) -> Path:
    """Download and extract ``hour.csv`` from UCI's ZIP, preserving its bytes.

    Existing files are left untouched unless ``force`` is true. The target is
    atomically replaced only after a successful HTTP response and ZIP extraction.
    """
    destination = Path(destination)
    if destination.exists() and not force:
        LOGGER.info("Dataset already exists; skipping download. source=%s destination=%s", url, destination)
        return destination

    request = urllib.request.Request(url, headers={"User-Agent": "UrbanPulse-AI/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", 200)
            if status != 200:
                raise IngestionError(f"UCI returned unexpected HTTP status {status} for {url}")
            archive_bytes = response.read()
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            member = next((name for name in archive.namelist() if Path(name).name == "hour.csv"), None)
            if member is None:
                raise IngestionError(f"The downloaded UCI archive does not contain hour.csv: {url}")
            csv_bytes = archive.read(member)
    except IngestionError:
        raise
    except (urllib.error.URLError, TimeoutError, OSError, zipfile.BadZipFile, KeyError) as exc:
        raise IngestionError(f"Could not download or read the UCI dataset from {url}: {exc}") from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.", delete=False) as temporary:
            temporary.write(csv_bytes)
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    LOGGER.info(
        "Downloaded dataset. source=%s destination=%s archive_bytes_downloaded=%d",
        url,
        destination,
        len(archive_bytes),
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the official UCI hourly Bike Sharing CSV.")
    parser.add_argument("--force", action="store_true", help="Download again and replace the local CSV.")
    parser.add_argument("--output", type=Path, default=DEFAULT_DESTINATION, help="Destination path (default: data/raw/hour.csv).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        path = download_hour_csv(args.output, force=args.force)
    except IngestionError as exc:
        parser.exit(1, f"Error: {exc}\n")
    LOGGER.info("Hourly dataset ready at %s", path)


if __name__ == "__main__":
    main()

