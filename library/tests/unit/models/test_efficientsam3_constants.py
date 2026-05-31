# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the EfficientSAM3 checkpoint filename resolver.

Covers the new ``ft=True`` path that selects fine-tuned checkpoints for the
medium variant of each backbone family. Filenames are verified against the
upstream HuggingFace layout recorded in
``/memories/repo/efficient_sam3_hf_layout.md``.
"""

from __future__ import annotations

import pytest

from instantlearn.models.efficient_sam3.constants import (
    BackboneType,
    get_checkpoint_filename,
)


def test_stage1_default() -> None:
    """Default (ft=False) returns the stage-1 distilled checkpoint."""
    assert (
        get_checkpoint_filename(BackboneType.EFFICIENTVIT, "b2")
        == "efficient_sam3_efficientvit-b2_mobileclip_s1.pth"
    )


@pytest.mark.parametrize(
    ("backbone", "variant", "expected"),
    [
        (BackboneType.EFFICIENTVIT, "b1", "efficient_sam3_efficientvit_b1_mobileclip_s1_ft.pth"),
        (BackboneType.REPVIT, "m1_1", "efficient_sam3_repvit_m1.1_mobileclip_s1_ft.pth"),
        (BackboneType.TINYVIT, "11m", "efficient_sam3_tinyvit_11m_mobileclip_s1_ft.pth"),
    ],
)
def test_ft_supported_variants(backbone: str, variant: str, expected: str) -> None:
    """ft=True returns the correct upstream filename for the 3 medium variants."""
    assert get_checkpoint_filename(backbone, variant, ft=True) == expected


@pytest.mark.parametrize(
    ("backbone", "variant"),
    [
        (BackboneType.EFFICIENTVIT, "b0"),
        (BackboneType.EFFICIENTVIT, "b2"),
        (BackboneType.REPVIT, "m0_9"),
        (BackboneType.REPVIT, "m2_3"),
        (BackboneType.TINYVIT, "5m"),
        (BackboneType.TINYVIT, "21m"),
    ],
)
def test_ft_unsupported_variants_raise(backbone: str, variant: str) -> None:
    """ft=True for any non-medium variant raises ValueError."""
    with pytest.raises(ValueError, match="No fine-tuned checkpoint"):
        get_checkpoint_filename(backbone, variant, ft=True)
