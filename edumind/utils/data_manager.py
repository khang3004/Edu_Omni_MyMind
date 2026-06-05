"""Data management utilities for EduMIND.

Handles automatic directory creation, path resolution, and disk usage statistics.
"""

from __future__ import annotations

import os
from pathlib import Path

from edumind.config import get_settings
from edumind.core.logging import get_logger

logger = get_logger(__name__)


def ensure_data_dirs() -> None:
    """Creates all required folders inside the data directory structure if they do not exist."""
    settings = get_settings()
    data_dir = settings.DATA_DIR

    dirs = [
        data_dir / "raw",
        data_dir / "interim",
        data_dir / "processed",
        data_dir / "qdrant_db",
    ]

    for d in dirs:
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
            logger.info("created_data_directory", path=str(d))


def get_raw_dir() -> Path:
    """Returns the path to the raw files directory."""
    settings = get_settings()
    return settings.DATA_DIR / "raw"


def get_interim_dir() -> Path:
    """Returns the path to the interim/temp files directory."""
    settings = get_settings()
    return settings.DATA_DIR / "interim"


def get_processed_dir() -> Path:
    """Returns the path to the processed outputs directory."""
    settings = get_settings()
    return settings.DATA_DIR / "processed"


def get_qdrant_db_dir() -> Path:
    """Returns the path to the local Qdrant database folder."""
    settings = get_settings()
    return settings.DATA_DIR / "qdrant_db"


def _get_dir_size(path: Path) -> int:
    """Computes total size of all files inside a directory in bytes."""
    total_size = 0
    if not path.exists():
        return 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # Skip symlinks to avoid infinite loops or incorrect stats
            if not os.path.islink(fp):
                try:
                    total_size += os.path.getsize(fp)
                except OSError:
                    pass
    return total_size


def get_storage_stats() -> dict[str, any]:
    """Gathers storage metrics for reporting in system stats or Streamlit UI.

    Returns:
        A dictionary with sizes (in MB) and counts for different directories.
    """
    settings = get_settings()
    data_dir = settings.DATA_DIR

    raw_dir = get_raw_dir()
    interim_dir = get_interim_dir()
    qdrant_dir = get_qdrant_db_dir()

    # Count raw files
    raw_files_count = 0
    if raw_dir.exists():
        raw_files_count = len([f for f in raw_dir.iterdir() if f.is_file() and not f.name.startswith(".")])

    # Compute sizes in MB
    raw_size_mb = round(_get_dir_size(raw_dir) / (1024 * 1024), 2)
    interim_size_mb = round(_get_dir_size(interim_dir) / (1024 * 1024), 2)
    qdrant_size_mb = round(_get_dir_size(qdrant_dir) / (1024 * 1024), 2)
    total_size_mb = round(_get_dir_size(data_dir) / (1024 * 1024), 2)

    return {
        "raw_files_count": raw_files_count,
        "raw_size_mb": raw_size_mb,
        "interim_size_mb": interim_size_mb,
        "qdrant_size_mb": qdrant_size_mb,
        "total_size_mb": total_size_mb,
        "qdrant_mode": settings.QDRANT_MODE,
    }
