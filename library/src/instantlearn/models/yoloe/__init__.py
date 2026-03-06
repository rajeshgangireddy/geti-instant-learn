# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""YOLOE model package.

This package contains the YOLOE model implementation for real-time
open-vocabulary detection and segmentation with visual prompting.
"""

from .yoloe import YOLOE

__all__ = [
    "YOLOE",
]
