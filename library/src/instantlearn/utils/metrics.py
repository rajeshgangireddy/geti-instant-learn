# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Segmentation metrics for benchmarking."""

import torch


class SegmentationMetrics:
    """Accumulates pixel-level segmentation metrics per class.

    Tracks true positives, false positives, and false negatives across
    all updates, then computes IoU, F1 (Dice), Precision, and Recall.

    Args:
        num_classes: Number of classes in the dataset.
        device: Device to place metric tensors on.
    """

    def __init__(self, num_classes: int, device: torch.device) -> None:
        """Initialize metric accumulators."""
        self.num_classes = num_classes
        self.device = device
        self.tp = torch.zeros(num_classes, dtype=torch.long, device=device)
        self.fp = torch.zeros(num_classes, dtype=torch.long, device=device)
        self.fn = torch.zeros(num_classes, dtype=torch.long, device=device)

    def update(self, pred: torch.Tensor, gt: torch.Tensor) -> None:
        """Update metrics with a single image prediction.

        Args:
            pred: Predicted one-hot tensor of shape (1, C, H, W), boolean.
            gt: Ground truth one-hot tensor of shape (1, C, H, W), boolean.
        """
        pred = pred.squeeze(0).bool()  # (C, H, W)
        gt = gt.squeeze(0).bool()  # (C, H, W)

        for c in range(self.num_classes):
            p = pred[c].flatten()
            g = gt[c].flatten()
            self.tp[c] += (p & g).sum().item()
            self.fp[c] += (p & ~g).sum().item()
            self.fn[c] += (~p & g).sum().item()

    def compute(self) -> dict[str, torch.Tensor]:
        """Compute all metrics per class.

        Returns:
            Dictionary with per-class tensors of shape (num_classes,) for:
            - iou: Intersection over Union
            - f1: F1 score (equivalent to Dice coefficient)
            - precision: Pixel-level precision
            - recall: Pixel-level recall
        """
        tp = self.tp.float()
        fp = self.fp.float()
        fn = self.fn.float()

        iou = tp / (tp + fp + fn).clamp(min=1)
        precision = tp / (tp + fp).clamp(min=1)
        recall = tp / (tp + fn).clamp(min=1)
        f1 = 2 * tp / (2 * tp + fp + fn).clamp(min=1)

        # Zero out metrics for classes with no ground truth and no predictions
        no_data = (tp + fp + fn) == 0
        iou[no_data] = 0.0
        precision[no_data] = 0.0
        recall[no_data] = 0.0
        f1[no_data] = 0.0

        return {
            "iou": iou,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }

    def to(self, device: torch.device) -> "SegmentationMetrics":
        """Move metric tensors to the specified device.

        Args:
            device: Target device.

        Returns:
            Self for chaining.
        """
        self.device = device
        self.tp = self.tp.to(device)
        self.fp = self.fp.to(device)
        self.fn = self.fn.to(device)
        return self

    def reset(self) -> None:
        """Reset all accumulators to zero."""
        self.tp.zero_()
        self.fp.zero_()
        self.fn.zero_()
