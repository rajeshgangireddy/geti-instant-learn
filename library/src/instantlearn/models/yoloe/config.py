# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Configuration dataclasses for YOLOE models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class YoloePromptMode(str, Enum):
    """Prompt modes for YOLOE inference.

    - TEXT: Text-based class prompting via CLIP embeddings
      (fused into conv weights via ``set_classes``).
    - VISUAL_EXEMPLAR: Visual prompting using bounding boxes
      from reference images.
    """

    TEXT = "text"
    VISUAL_EXEMPLAR = "visual_exemplar"


class YoloeOVVariant(str, Enum):
    """OpenVINO model variants for YOLOE.

    Variants correspond to compression levels applied at export time.
    """

    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"


@dataclass
class YoloePostProcessingConfig:
    """Post-processing configuration for YOLOE models.

    Args:
        confidence_threshold: Minimum detection confidence score.
        iou_threshold: IoU threshold for non-maximum suppression.
        use_nms: Whether to apply non-maximum suppression.
    """

    confidence_threshold: float = 0.25
    iou_threshold: float = 0.7
    use_nms: bool = True
