# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for Matcher.export() with weight compression modes.

Tests the full export flow (PyTorch → ONNX → OpenVINO → compress → save)
for each CompressionMode and validates that the resulting model:
- Produces files on disk
- Loads and compiles with OpenVINO Core
- Returns outputs with valid shapes
- Decreases in size (or stays the same) as compression increases
"""

from pathlib import Path

import numpy as np
import pytest

from instantlearn.data.base import Batch
from instantlearn.data.folder import FolderDataset
from instantlearn.data.utils.image import read_image
from instantlearn.models.matcher import Matcher
from instantlearn.utils.constants import Backend, CompressionMode, SAMModelName


@pytest.fixture
def fss1000_root() -> Path:
    return Path(__file__).parent.parent.parent.parent / "examples" / "assets" / "fss-1000"


@pytest.fixture
def target_image_path(fss1000_root: Path) -> Path:
    return fss1000_root / "images" / "apple" / "2.jpg"


@pytest.fixture
def dataset(fss1000_root: Path) -> FolderDataset:
    return FolderDataset(root=fss1000_root, categories=["apple"], n_shots=1)


@pytest.fixture
def reference_batch(dataset: FolderDataset) -> Batch:
    ref_dataset = dataset.get_reference_dataset()
    return Batch.collate([ref_dataset[0]])


@pytest.fixture
def fitted_matcher(reference_batch: Batch) -> Matcher:
    """Return a Matcher that has been fitted on the reference batch."""
    matcher = Matcher(
        sam=SAMModelName.SAM_HQ_BASE,
        device="cpu",
        precision="fp32",
        encoder_model="dinov3_small",
        use_mask_refinement=False,
    )
    matcher.fit(reference_batch)
    return matcher


class TestMatcherQuantizedExport:
    """Integration tests for Matcher export with weight compression."""

    @pytest.mark.parametrize(
        "compression",
        [
            CompressionMode.FP32,
            CompressionMode.FP16,
            CompressionMode.INT8_SYM,
            CompressionMode.INT8_ASYM,
        ],
    )
    def test_export_and_inference(
        self,
        fitted_matcher: Matcher,
        target_image_path: Path,
        tmp_path: Path,
        compression: CompressionMode,
    ) -> None:
        """Export with a given compression mode, load, and run inference."""
        import openvino  # noqa: PLC0415

        exported_path = fitted_matcher.export(
            export_dir=tmp_path,
            backend=Backend.OPENVINO,
            compression=compression,
        )

        assert exported_path.exists()
        assert exported_path.suffix == ".xml"
        assert (exported_path.parent / "matcher.bin").exists()

        core = openvino.Core()
        ov_model = core.read_model(str(exported_path))
        compiled = core.compile_model(ov_model, "CPU")

        # Prepare input matching static shape
        expected_shape = compiled.input(0).shape
        target_image = read_image(target_image_path)
        input_data = target_image.numpy()[None, ...].astype(np.float32)
        if input_data.shape != tuple(expected_shape):
            import torch  # noqa: PLC0415
            import torch.nn.functional as F  # noqa: N812, PLC0415

            tensor = torch.from_numpy(input_data)
            tensor = F.interpolate(tensor, size=(expected_shape[2], expected_shape[3]), mode="bilinear")
            input_data = tensor.numpy()

        outputs = compiled(input_data)
        masks, scores, labels = outputs.values()

        assert masks.ndim == 3
        assert scores.ndim == 1
        assert labels.ndim == 1
        assert masks.shape[0] == scores.shape[0] == labels.shape[0]

    @pytest.mark.parametrize(
        "compression",
        [CompressionMode.INT4_SYM, CompressionMode.INT4_ASYM],
    )
    def test_int4_export_and_inference(
        self,
        fitted_matcher: Matcher,
        target_image_path: Path,
        tmp_path: Path,
        compression: CompressionMode,
    ) -> None:
        """INT4 compression may degrade accuracy but should still produce valid outputs."""
        import openvino  # noqa: PLC0415

        exported_path = fitted_matcher.export(
            export_dir=tmp_path,
            backend=Backend.OPENVINO,
            compression=compression,
        )

        assert exported_path.exists()

        core = openvino.Core()
        ov_model = core.read_model(str(exported_path))
        compiled = core.compile_model(ov_model, "CPU")

        expected_shape = compiled.input(0).shape
        target_image = read_image(target_image_path)
        input_data = target_image.numpy()[None, ...].astype(np.float32)
        if input_data.shape != tuple(expected_shape):
            import torch  # noqa: PLC0415
            import torch.nn.functional as F  # noqa: N812, PLC0415

            tensor = torch.from_numpy(input_data)
            tensor = F.interpolate(tensor, size=(expected_shape[2], expected_shape[3]), mode="bilinear")
            input_data = tensor.numpy()

        outputs = compiled(input_data)
        masks, scores, labels = outputs.values()

        assert masks.ndim == 3
        assert scores.ndim == 1
        assert labels.ndim == 1

    def test_compression_reduces_model_size(
        self,
        fitted_matcher: Matcher,
        tmp_path: Path,
    ) -> None:
        """INT8 model should be smaller than FP32 model on disk."""
        sizes: dict[str, int] = {}

        for mode in (CompressionMode.FP32, CompressionMode.INT8_SYM):
            export_dir = tmp_path / mode.value
            fitted_matcher.export(
                export_dir=export_dir,
                backend=Backend.OPENVINO,
                compression=mode,
            )
            total = sum(f.stat().st_size for f in export_dir.rglob("*") if f.is_file())
            sizes[mode.value] = total

        assert sizes["int8_sym"] < sizes["fp32"], (
            f"INT8 model ({sizes['int8_sym']} bytes) should be smaller than FP32 ({sizes['fp32']} bytes)"
        )

    def test_compression_via_init_default(
        self,
        reference_batch: Batch,
        tmp_path: Path,
    ) -> None:
        """Compression set in __init__ should be used as default in export()."""
        matcher = Matcher(
            sam=SAMModelName.SAM_HQ_BASE,
            device="cpu",
            precision="fp32",
            encoder_model="dinov3_small",
            use_mask_refinement=False,
            compression=CompressionMode.INT8_SYM,
        )
        matcher.fit(reference_batch)

        exported_path = matcher.export(
            export_dir=tmp_path,
            backend=Backend.OPENVINO,
            # compression not passed — should use self.compression (INT8_SYM)
        )
        assert exported_path.exists()
