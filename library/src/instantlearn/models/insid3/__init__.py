# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""INSID3 model package.

Training-free in-context segmentation using frozen DINOv3 features,
based on the paper 'INSID3: Training-Free In-Context Segmentation with DINOv3' (CVPR 2026).
"""

from .insid3 import INSID3

__all__ = ["INSID3"]
