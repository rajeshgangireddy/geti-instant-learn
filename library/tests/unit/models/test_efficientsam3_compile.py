# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the EfficientSAM3 ``torch.compile`` fallback helper.

Exercises :meth:`EfficientSAM3._try_compile_model` in isolation (without
constructing a full model) to verify it wraps ``self.model`` on success and
keeps the original module when ``torch.compile`` raises.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import patch

import torch
from torch import nn

from instantlearn.models.efficient_sam3.efficient_sam3 import EfficientSAM3

if TYPE_CHECKING:
    import pytest


def _make_bare_wrapper(backbone_type: str = "efficientvit", variant: str = "b2") -> EfficientSAM3:
    """Build a minimal EfficientSAM3-shaped object without running __init__.

    Avoids the HF download / model load path. Only the attributes used by
    ``_try_compile_model`` are populated. ``nn.Module.__init__`` is invoked
    explicitly so that submodule registration via __setattr__ works.
    """
    obj = EfficientSAM3.__new__(EfficientSAM3)
    nn.Module.__init__(obj)
    obj.backbone_type = backbone_type
    obj.variant = variant
    obj.model = nn.Linear(4, 4)
    return obj


def test_try_compile_replaces_model_on_success() -> None:
    """On success, ``self.model`` is replaced with the compile() return value."""
    wrapper = _make_bare_wrapper()
    original = wrapper.model
    sentinel = nn.Linear(4, 4)

    with patch.object(torch, "compile", return_value=sentinel) as mock_compile:
        wrapper._try_compile_model()  # noqa: SLF001

    mock_compile.assert_called_once_with(original)
    assert wrapper.model is sentinel


def test_try_compile_falls_back_on_exception(caplog: pytest.LogCaptureFixture) -> None:
    """On torch.compile failure, ``self.model`` is kept and a warning is logged."""
    wrapper = _make_bare_wrapper()
    original = wrapper.model

    with (
        patch.object(torch, "compile", side_effect=RuntimeError("backend unavailable")),
        caplog.at_level(logging.WARNING, logger="instantlearn.models.efficient_sam3.efficient_sam3"),
    ):
        wrapper._try_compile_model()  # noqa: SLF001

    assert wrapper.model is original
    assert any("torch.compile failed" in record.message for record in caplog.records)


def test_try_compile_tinyvit_logs_extra_warning(caplog: pytest.LogCaptureFixture) -> None:
    """TinyViT backbone triggers an additional pre-compile warning."""
    wrapper = _make_bare_wrapper(backbone_type="tinyvit", variant="11m")

    with (
        patch.object(torch, "compile", return_value=wrapper.model),
        caplog.at_level(logging.WARNING, logger="instantlearn.models.efficient_sam3.efficient_sam3"),
    ):
        wrapper._try_compile_model()  # noqa: SLF001

    assert any("TinyViT backbone" in record.message for record in caplog.records)
