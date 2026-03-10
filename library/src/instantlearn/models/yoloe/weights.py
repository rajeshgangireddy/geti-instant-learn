# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""YOLOE model weights caching.

All YOLOE weights are stored under
``~/.cache/instantlearn/ultralytics/weights/`` so they are downloaded
once and shared across library and application usage.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

WEIGHTS_DIR = Path.home() / ".cache" / "instantlearn" / "ultralytics" / "weights"


def get_weights_path(filename: str) -> Path:
    """Resolve a model weights file, downloading if necessary.

    Looks for the file in the central cache directory
    (``~/.cache/instantlearn/ultralytics/weights/``).  If not found,
    downloads it via ultralytics' asset downloader and saves it there.

    Args:
        filename: Weights filename, e.g. ``"yoloe-26n-seg.pt"``.

    Returns:
        Absolute path to the cached weights file.

    Raises:
        ImportError: If ultralytics is not installed.
    """
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    cached = WEIGHTS_DIR / filename

    if cached.exists():
        logger.debug("Using cached weights: %s", cached)
        return cached

    # Download via ultralytics
    logger.info("Downloading %s to %s ...", filename, WEIGHTS_DIR)
    try:
        from ultralytics.utils.downloads import attempt_download_asset
    except ImportError as e:
        msg = (
            "ultralytics is required for automatic weight downloads. "
            "Install it with: uv pip install ultralytics"
        )
        raise ImportError(msg) from e

    # attempt_download_asset downloads to CWD — download then move
    downloaded = Path(attempt_download_asset(filename))
    if downloaded.resolve() != cached.resolve():
        downloaded.rename(cached)
        logger.info("Moved weights to cache: %s", cached)

    return cached
