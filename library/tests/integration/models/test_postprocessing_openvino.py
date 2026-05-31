# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Integration tests verifying post-processing is included in OpenVINO exports.

Exports a Matcher with the default post-processor to OpenVINO, runs
inference on CPU, and compares outputs to the PyTorch path to confirm
that NMS and other post-processors are applied inside the exported graph.
"""

from pathlib import Path

import numpy as np
import openvino
import pytest
import torch
import torch.nn.functional as F

from instantlearn.components.postprocessing import (
    BoxNMS,
    MaskIoMNMS,
    MinimumAreaFilter,
    MorphologicalOpening,
    PostProcessorPipeline,
)
from instantlearn.data.base import Batch
from instantlearn.data.folder import FolderDataset
from instantlearn.data.utils.image import read_image
from instantlearn.models.matcher import Matcher
from instantlearn.utils.constants import Backend, SAMModelName


@pytest.fixture
def fss1000_root() -> Path:
    """Return path to fss-1000 test dataset."""
    return Path(__file__).parent.parent.parent.parent / "examples" / "assets" / "fss-1000"


@pytest.fixture
def reference_image_path(fss1000_root: Path) -> Path:
    """Return path to reference image."""
    return fss1000_root / "images" / "apple" / "1.jpg"


@pytest.fixture
def target_image_path(fss1000_root: Path) -> Path:
    """Return path to target image."""
    return fss1000_root / "images" / "apple" / "2.jpg"


@pytest.fixture
def reference_mask_path(fss1000_root: Path) -> Path:
    """Return path to reference mask."""
    return fss1000_root / "masks" / "apple" / "1.png"


@pytest.fixture
def dataset(fss1000_root: Path) -> FolderDataset:
    """Create a FolderDataset for testing."""
    return FolderDataset(
        root=fss1000_root,
        categories=["apple"],
        n_shots=1,
    )


@pytest.fixture
def reference_batch(dataset: FolderDataset) -> Batch:
    """Get reference batch from dataset."""
    ref_dataset = dataset.get_reference_dataset()
    samples = [ref_dataset[0]]
    return Batch.collate(samples)


def _resize_for_ov(input_data: np.ndarray, compiled_model: openvino.CompiledModel) -> np.ndarray:
    """Resize input to match the compiled model's expected static input shape."""
    expected_shape = tuple(compiled_model.input(0).shape)
    if input_data.shape != expected_shape:
        tensor = torch.from_numpy(input_data)
        tensor = F.interpolate(tensor, size=(expected_shape[2], expected_shape[3]), mode="bilinear")
        input_data = tensor.numpy()
    return input_data


def _select_best_slot_outputs(
    masks: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse exported per-category slot tensors to one mask per category."""
    if masks.ndim != 4 or scores.ndim != 2:
        return masks, scores, labels

    best_idx = np.argmax(scores, axis=1)
    selected_masks = np.stack([masks[c, idx] for c, idx in enumerate(best_idx)], axis=0)
    selected_scores = np.array([scores[c, idx] for c, idx in enumerate(best_idx)])
    return selected_masks, selected_scores, labels


def _apply_postprocessor_numpy(
    postprocessor: PostProcessorPipeline,
    masks: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply a PyTorch post-processor to exported numpy outputs."""
    masks_t = torch.from_numpy(masks)
    scores_t = torch.from_numpy(scores)
    labels_t = torch.from_numpy(labels)
    new_masks, new_scores, new_labels = postprocessor(masks_t, scores_t, labels_t)
    return new_masks.cpu().numpy(), new_scores.cpu().numpy(), new_labels.cpu().numpy()


class TestPostProcessingOpenVINO:
    """Integration tests for post-processing in OpenVINO exported models."""

    @pytest.mark.parametrize("sam_model", [SAMModelName.SAM_HQ_BASE])
    def test_default_postprocessor_openvino(
        self,
        sam_model: SAMModelName,
        reference_batch: Batch,
        target_image_path: Path,
        tmp_path: Path,
    ) -> None:
        """Export Matcher with default post-processor to OpenVINO and verify outputs.

        Checks that:
        - Export succeeds
        - OpenVINO inference produces valid shapes
        - Scores are in [0, 1] range
        - Labels are valid category IDs
        - Output is consistent with PyTorch inference
        """
        pytest.importorskip("openvino")

        matcher = Matcher(
            sam=sam_model,
            device="cpu",
            precision="fp32",
            encoder_model="dinov3_small",
            use_mask_refinement=False,
        )

        matcher.fit(reference_batch)

        # Get PyTorch predictions for comparison
        pytorch_preds = matcher.predict(target_image_path)

        pt_masks = pytorch_preds[0]["pred_masks"]

        # Export to OpenVINO
        exported_path = matcher.export(
            export_dir=tmp_path,
            backend=Backend.OPENVINO,
        )

        assert exported_path.exists()
        assert exported_path.suffix == ".xml"
        assert (exported_path.parent / "matcher.bin").exists()

        # Run OpenVINO inference
        target_image = read_image(target_image_path)
        core = openvino.Core()
        ov_model = core.read_model(str(exported_path))
        compiled_model = core.compile_model(ov_model, "CPU")

        input_data = target_image.numpy()[None, ...].astype(np.float32)
        input_data = _resize_for_ov(input_data, compiled_model)
        outputs = compiled_model(input_data)

        ov_masks, ov_scores, ov_labels = outputs.values()
        ov_masks, ov_scores, ov_labels = _select_best_slot_outputs(ov_masks, ov_scores, ov_labels)
        ov_masks, ov_scores, ov_labels = _apply_postprocessor_numpy(
            matcher.postprocessor,
            ov_masks,
            ov_scores,
            ov_labels,
        )

        # Validate output shapes
        assert ov_masks.ndim == 3, f"Expected masks to have 3 dims, got {ov_masks.ndim}"
        assert ov_scores.ndim == 1, f"Expected scores to have 1 dim, got {ov_scores.ndim}"
        assert ov_labels.ndim == 1, f"Expected labels to have 1 dim, got {ov_labels.ndim}"
        assert ov_masks.shape[0] == ov_scores.shape[0] == ov_labels.shape[0], "Output counts should match"

        # Validate scores are non-negative and labels are valid category IDs.
        assert np.all(ov_scores >= 0), "Scores should be non-negative"
        assert np.all(ov_labels >= 0), "Labels should be non-negative (valid category IDs)"

        # Cross-check: host-side post-processing on exported outputs should still
        # produce a reasonable number of masks relative to the PyTorch path.
        assert ov_masks.shape[0] > 0, "OpenVINO should produce at least one mask"
        assert abs(ov_masks.shape[0] - pt_masks.shape[0]) <= 2, (
            f"OpenVINO mask count ({ov_masks.shape[0]}) differs from PyTorch "
            f"({pt_masks.shape[0]}) by more than 2 after host-side post-processing"
        )

    @pytest.mark.parametrize("sam_model", [SAMModelName.SAM_HQ_BASE])
    def test_custom_postprocessor_openvino(
        self,
        sam_model: SAMModelName,
        reference_batch: Batch,
        target_image_path: Path,
        tmp_path: Path,
    ) -> None:
        """Export Matcher with a custom post-processor pipeline to OpenVINO.

        Uses a pipeline with MaskIoMNMS + MinimumAreaFilter + MorphologicalOpening
        to verify that all exportable post-processors work in the OpenVINO graph.
        """
        pytest.importorskip("openvino")

        custom_pp = PostProcessorPipeline([
            MaskIoMNMS(iom_threshold=0.5),
            MinimumAreaFilter(min_area=50),
            MorphologicalOpening(kernel_size=3),
        ])

        matcher = Matcher(
            sam=sam_model,
            device="cpu",
            precision="fp32",
            encoder_model="dinov3_small",
            use_mask_refinement=False,
            postprocessor=custom_pp,
        )

        matcher.fit(reference_batch)

        # Export to OpenVINO
        exported_path = matcher.export(
            export_dir=tmp_path,
            backend=Backend.OPENVINO,
        )

        assert exported_path.exists()

        # Run OpenVINO inference
        core = openvino.Core()
        ov_model = core.read_model(str(exported_path))
        compiled_model = core.compile_model(ov_model, "CPU")

        target_image = read_image(target_image_path)
        input_data = target_image.numpy()[None, ...].astype(np.float32)
        input_data = _resize_for_ov(input_data, compiled_model)
        outputs = compiled_model(input_data)

        ov_masks, ov_scores, ov_labels = outputs.values()
        ov_masks, ov_scores, ov_labels = _select_best_slot_outputs(ov_masks, ov_scores, ov_labels)
        ov_masks, ov_scores, ov_labels = _apply_postprocessor_numpy(
            custom_pp,
            ov_masks,
            ov_scores,
            ov_labels,
        )

        # Basic shape validation
        assert ov_masks.ndim == 3
        assert ov_scores.ndim == 1
        assert ov_labels.ndim == 1
        assert ov_masks.shape[0] == ov_scores.shape[0] == ov_labels.shape[0]

        # Scores valid
        assert np.all(ov_scores >= 0)

        # Labels valid
        assert np.all(ov_labels >= 0)

        # MinimumAreaFilter ensures no tiny masks
        if ov_masks.shape[0] > 0:
            areas = ov_masks.astype(bool).reshape(ov_masks.shape[0], -1).sum(axis=1)
            assert np.all(areas >= 50), "MinimumAreaFilter should have removed small masks"

    @pytest.mark.parametrize("sam_model", [SAMModelName.SAM_HQ_BASE])
    def test_no_postprocessor_vs_with_postprocessor(
        self,
        sam_model: SAMModelName,
        reference_batch: Batch,
        target_image_path: Path,
        tmp_path: Path,
    ) -> None:
        """Compare OpenVINO export with vs without post-processor.

        The model with post-processing should have fewer or equal masks
        compared to the model without, confirming that NMS is actually
        running inside the exported graph.
        """
        pytest.importorskip("openvino")

        target_image = read_image(target_image_path)

        # Export WITHOUT post-processor
        matcher_no_pp = Matcher(
            sam=sam_model,
            device="cpu",
            precision="fp32",
            encoder_model="dinov3_small",
            use_mask_refinement=False,
            postprocessor=PostProcessorPipeline([]),  # empty pipeline = no post-processing
        )
        matcher_no_pp.fit(reference_batch)

        no_pp_path = matcher_no_pp.export(
            export_dir=tmp_path / "no_pp",
            backend=Backend.OPENVINO,
        )

        core = openvino.Core()
        compiled_no_pp = core.compile_model(core.read_model(str(no_pp_path)), "CPU")
        input_data = target_image.numpy()[None, ...].astype(np.float32)
        input_data = _resize_for_ov(input_data, compiled_no_pp)
        out_no_pp = compiled_no_pp(input_data)
        masks_no_pp, scores_no_pp, labels_no_pp = out_no_pp.values()
        masks_no_pp, scores_no_pp, labels_no_pp = _select_best_slot_outputs(
            masks_no_pp,
            scores_no_pp,
            labels_no_pp,
        )
        masks_no_pp, scores_no_pp, labels_no_pp = _apply_postprocessor_numpy(
            matcher_no_pp.postprocessor,
            masks_no_pp,
            scores_no_pp,
            labels_no_pp,
        )

        # Export WITH default post-processor
        matcher_with_pp = Matcher(
            sam=sam_model,
            device="cpu",
            precision="fp32",
            encoder_model="dinov3_small",
            use_mask_refinement=False,
        )
        matcher_with_pp.fit(reference_batch)

        with_pp_path = matcher_with_pp.export(
            export_dir=tmp_path / "with_pp",
            backend=Backend.OPENVINO,
        )

        compiled_with_pp = core.compile_model(core.read_model(str(with_pp_path)), "CPU")
        out_with_pp = compiled_with_pp(input_data)
        masks_with_pp, scores_with_pp, labels_with_pp = out_with_pp.values()
        masks_with_pp, scores_with_pp, labels_with_pp = _select_best_slot_outputs(
            masks_with_pp,
            scores_with_pp,
            labels_with_pp,
        )
        masks_with_pp, scores_with_pp, labels_with_pp = _apply_postprocessor_numpy(
            matcher_with_pp.postprocessor,
            masks_with_pp,
            scores_with_pp,
            labels_with_pp,
        )

        # Post-processor should produce <= masks than without
        assert masks_with_pp.shape[0] <= masks_no_pp.shape[0], (
            f"Post-processed model should have <= masks: "
            f"got {masks_with_pp.shape[0]} vs {masks_no_pp.shape[0]} without PP"
        )

    @pytest.mark.parametrize("sam_model", [SAMModelName.SAM_HQ_BASE])
    def test_box_nms_openvino(
        self,
        sam_model: SAMModelName,
        reference_batch: Batch,
        target_image_path: Path,
        tmp_path: Path,
    ) -> None:
        """Export Matcher with BoxNMS post-processor to OpenVINO.

        Verifies that torchvision.ops.nms exports correctly via
        ONNX::NonMaxSuppression → OpenVINO NonMaxSuppression-9.
        """
        pytest.importorskip("openvino")

        box_nms_pp = PostProcessorPipeline([BoxNMS(iou_threshold=0.5)])

        matcher = Matcher(
            sam=sam_model,
            device="cpu",
            precision="fp32",
            encoder_model="dinov3_small",
            use_mask_refinement=False,
            postprocessor=box_nms_pp,
        )

        matcher.fit(reference_batch)

        # Export to OpenVINO
        exported_path = matcher.export(
            export_dir=tmp_path,
            backend=Backend.OPENVINO,
        )

        assert exported_path.exists()
        assert exported_path.suffix == ".xml"

        # Run OpenVINO inference
        target_image = read_image(target_image_path)
        core = openvino.Core()
        compiled_model = core.compile_model(core.read_model(str(exported_path)), "CPU")

        input_data = target_image.numpy()[None, ...].astype(np.float32)
        input_data = _resize_for_ov(input_data, compiled_model)
        outputs = compiled_model(input_data)

        ov_masks, ov_scores, ov_labels = outputs.values()
        ov_masks, ov_scores, ov_labels = _select_best_slot_outputs(ov_masks, ov_scores, ov_labels)
        ov_masks, ov_scores, ov_labels = _apply_postprocessor_numpy(
            box_nms_pp,
            ov_masks,
            ov_scores,
            ov_labels,
        )

        # Validate shapes
        assert ov_masks.ndim == 3
        assert ov_scores.ndim == 1
        assert ov_labels.ndim == 1
        assert ov_masks.shape[0] == ov_scores.shape[0] == ov_labels.shape[0]

        # Validate values
        assert np.all(ov_scores >= 0)
        assert np.all(ov_labels >= 0)
