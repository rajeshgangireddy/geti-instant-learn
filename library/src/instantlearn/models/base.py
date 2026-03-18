# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base class for all models."""

from __future__ import annotations

import time
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Self, overload

import torch
from torch import nn

from instantlearn.utils.constants import Backend

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from instantlearn.data.base.batch import Batch, Collatable
    from instantlearn.data.base.sample import Sample


@dataclass
class InferenceTiming:
    """Per-component timing breakdown for a single fit() or predict() call.

    Component names are model-defined strings (e.g., "encoder", "prompt_generator",
    "decoder"). Times are in milliseconds.
    """

    component_times: dict[str, float] = field(default_factory=dict)
    total_ms: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Serialize to a flat dictionary."""
        result = dict(self.component_times)
        result["total_ms"] = self.total_ms
        return result


@dataclass
class InferenceResult:
    """Result of a model predict() call, containing predictions and optional timing.

    Supports iteration and indexing over predictions for backward compatibility,
    so existing code like ``for pred in model.predict(batch)`` still works.
    """

    predictions: list[dict[str, torch.Tensor]]
    timing: InferenceTiming | None = None

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """Iterate over predictions."""
        return iter(self.predictions)

    @overload
    def __getitem__(self, index: int) -> dict[str, torch.Tensor]: ...
    @overload
    def __getitem__(self, index: slice) -> list[dict[str, torch.Tensor]]: ...

    def __getitem__(self, index: int | slice) -> dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]:
        """Get prediction(s) by index or slice."""
        return self.predictions[index]

    def __len__(self) -> int:
        """Return number of predictions."""
        return len(self.predictions)


class _TimingContext:
    """Lightweight context manager for timing a named component.

    Usage::

        ctx = _TimingContext("encoder", sync_fn)
        with ctx:
            result = encoder(images)
        # ctx.elapsed_ms is now populated
    """

    def __init__(self, component: str, sync_fn: callable | None = None) -> None:
        self.component = component
        self._sync_fn = sync_fn
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> Self:
        if self._sync_fn:
            self._sync_fn()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._sync_fn:
            self._sync_fn()
        self._end = time.perf_counter()
        self.elapsed_ms = (self._end - self._start) * 1000.0


class Model(nn.Module):
    """This class is the base class for all models."""

    # Stores timing from the last fit() call, so callers can read it.
    last_fit_timing: InferenceTiming | None = None

    @abstractmethod
    def fit(self, reference: Sample | list[Sample] | Batch) -> None:
        """Learn the context from reference samples.

        Args:
            reference: Reference data to learn from. Accepts:
                - Sample: A single reference sample
                - list[Sample]: A list of reference samples
                - Batch: A batch of reference samples
        """

    @abstractmethod
    def predict(self, target: Collatable) -> InferenceResult:
        """Use the learned context to infer object locations.

        Args:
            target: Target data to infer. Accepts:
                - Sample: A single target sample
                - list[Sample]: A list of target samples
                - Batch: A batch of target samples
                - str | Path: A single image path
                - list[str] | list[Path]: Multiple image paths

        Returns:
            InferenceResult containing predictions and per-component timing.
            Each prediction dict contains:
                - "pred_masks": torch.Tensor of shape [num_masks, H, W]
                - "pred_points": torch.Tensor of shape [num_points, 4] with [x, y, score, fg_label]
                - "pred_boxes": torch.Tensor of shape [num_boxes, 5] with [x1, y1, x2, y2, score]
                - "pred_labels": torch.Tensor of shape [num_masks]
        """

    @abstractmethod
    def export(
        self,
        export_dir: str | Path,
        backend: str | Backend = Backend.ONNX,
        **kwargs: object,
    ) -> Path:
        """This method exports the model to a given path.

        Args:
            export_dir: The directory to export the model to.
            backend: The backend to export the model to.
            **kwargs: Additional arguments to pass to the export method.

        Returns:
            The path to the exported model.
        """

    def _get_sync_fn(self) -> callable | None:
        """Return a device synchronization function for accurate GPU timing."""
        try:
            device = next(self.parameters()).device
        except StopIteration:
            return None
        if device.type == "cuda":
            return torch.cuda.synchronize
        if device.type == "xpu":
            return torch.xpu.synchronize
        return None

    def _time_component(self, component: str) -> _TimingContext:
        """Create a timing context manager for a named component.

        Args:
            component: Name of the component being timed (e.g., "encoder").

        Returns:
            A context manager that records elapsed time in milliseconds.
        """
        return _TimingContext(component, self._get_sync_fn())

    @staticmethod
    def _build_timing(contexts: list[_TimingContext], total_ms: float) -> InferenceTiming:
        """Build an InferenceTiming from a list of completed timing contexts.

        Args:
            contexts: List of completed _TimingContext objects.
            total_ms: Wall-clock total in milliseconds.

        Returns:
            Populated InferenceTiming.
        """
        return InferenceTiming(
            component_times={ctx.component: ctx.elapsed_ms for ctx in contexts},
            total_ms=total_ms,
        )
