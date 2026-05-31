# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for EfficientSAM3 precision resolution.

Verifies that ``_resolve_precision`` correctly expands 'auto' based on the
target device and passes other values through unchanged.
"""

from __future__ import annotations

from unittest.mock import patch

import torch

from instantlearn.models.efficient_sam3.efficient_sam3 import _resolve_precision  # noqa: PLC2701


def test_resolve_precision_passthrough() -> None:
    """Non-'auto' values are returned unchanged regardless of device."""
    for value in ("fp32", "fp16", "bf16"):
        assert _resolve_precision(value, "cuda") == value
        assert _resolve_precision(value, "cpu") == value
        assert _resolve_precision(value, "xpu") == value


def test_resolve_precision_auto_cpu_returns_fp32() -> None:
    """'auto' on CPU resolves to 'fp32'."""
    assert _resolve_precision("auto", "cpu") == "fp32"


def test_resolve_precision_auto_cuda_available_returns_bf16() -> None:
    """'auto' on CUDA returns 'bf16' when CUDA is available."""
    with patch.object(torch.cuda, "is_available", return_value=True):
        assert _resolve_precision("auto", "cuda") == "bf16"


def test_resolve_precision_auto_cuda_unavailable_returns_fp32() -> None:
    """'auto' on CUDA falls back to 'fp32' when CUDA is unavailable."""
    with patch.object(torch.cuda, "is_available", return_value=False):
        assert _resolve_precision("auto", "cuda") == "fp32"


def test_resolve_precision_auto_xpu_available_returns_bf16() -> None:
    """'auto' on XPU returns 'bf16' when XPU is available."""
    # Build a stand-in xpu namespace with is_available=True regardless of
    # whether the host actually has an Intel GPU.
    fake_xpu = type("FakeXpu", (), {"is_available": staticmethod(lambda: True)})
    with patch.object(torch, "xpu", fake_xpu, create=True):
        assert _resolve_precision("auto", "xpu") == "bf16"


def test_resolve_precision_auto_xpu_unavailable_returns_fp32() -> None:
    """'auto' on XPU falls back to 'fp32' when XPU is unavailable."""
    fake_xpu = type("FakeXpu", (), {"is_available": staticmethod(lambda: False)})
    with patch.object(torch, "xpu", fake_xpu, create=True):
        assert _resolve_precision("auto", "xpu") == "fp32"
