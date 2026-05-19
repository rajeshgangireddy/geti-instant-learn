# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""YOLOE model package.

This package contains the YOLOE model implementation for real-time
open-vocabulary detection and segmentation with text and visual prompting,
and an OpenVINO IR inference variant with fixed (exported) classes.
"""

from .config import YoloeOVVariant, YoloePostProcessingConfig, YoloePromptMode
from .yoloe import YOLOE
from .yoloe_openvino import YOLOEOpenVINO

__all__ = [
    "YOLOE",
    "YOLOEOpenVINO",
    "YoloeOVVariant",
    "YoloePostProcessingConfig",
    "YoloePromptMode",
]
