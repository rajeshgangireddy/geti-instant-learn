# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""YOLOE OpenVINO model for inference with pre-exported IR models.

Unlike the PyTorch-based ``YOLOE`` class, this model loads a
pre-exported OpenVINO IR (XML/BIN) where the target classes have
already been baked into the conv weights at export time (via
``set_classes`` + ``_fuse_tp``).  Visual/text prompting is **not**
available at inference time — the categories are fixed.

Typical workflow:
  1. Export a YOLOE model for specific classes using the export script.
  2. Load the exported IR with ``YOLOEOpenVINO``.
  3. Call ``fit()`` to set up category mapping from reference data.
  4. Call ``predict()`` for efficient OpenVINO inference.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import yaml

from instantlearn.data.base.batch import Batch, Collatable
from instantlearn.models.base import Model
from instantlearn.models.yoloe.config import YoloeOVVariant, YoloePostProcessingConfig
from instantlearn.models.yoloe.postprocessing import (
    parse_detections,
    preprocess_image,
    process_mask_protos,
    scale_boxes_to_original,
)
from instantlearn.utils.constants import Backend
from instantlearn.utils.utils import device_to_openvino_device

if TYPE_CHECKING:
    from instantlearn.data.base.sample import Sample

logger = logging.getLogger(__name__)

# Default number of mask prototype coefficients for YOLOE-seg models
_DEFAULT_NM = 32


class YOLOEOpenVINO(Model):
    """YOLOE model using OpenVINO IR for inference.

    Classes are fixed at export time. ``fit()`` only records the
    category-ID mapping from reference data so that ``pred_labels``
    use the same IDs as the rest of the pipeline.

    Args:
        model_dir: Path to the exported OpenVINO model directory
            containing the ``.xml``, ``.bin``, and ``metadata.yaml`` files.
        confidence_threshold: Minimum detection confidence.
        iou_threshold: IoU threshold (unused — model is end2end).
        device: Inference device (``"cpu"``, ``"cuda"``/``"GPU"``, ``"AUTO"``).
        variant: Model variant indicating compression level used at export.
        post_processing: Optional PostProcessingConfig (overrides flat params).

    Examples:
        >>> from instantlearn.models.yoloe import YOLOEOpenVINO
        >>> model = YOLOEOpenVINO(model_dir="exports/yoloe_openvino")
        >>> model.fit(reference_sample)
        >>> results = model.predict(target_sample)
    """

    def __init__(
        self,
        model_dir: str | Path,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        device: str = "cpu",
        variant: str | YoloeOVVariant = YoloeOVVariant.FP16,
        post_processing: YoloePostProcessingConfig | None = None,
    ) -> None:
        """Initialize YOLOE OpenVINO model.

        Args:
            model_dir: Path to the exported OpenVINO IR directory.
            confidence_threshold: Minimum confidence score for detections.
            iou_threshold: IoU threshold for non-maximum suppression.
            device: Device for inference ("cpu", "gpu").
            variant: Model quantization variant.
            post_processing: Optional PostProcessingConfig (overrides
                confidence_threshold/iou_threshold if provided).
        """
        super().__init__()

        self.model_dir = Path(model_dir)
        self.device_name = device
        self.variant = YoloeOVVariant(variant)

        # Post-processing config
        if post_processing is not None:
            self.post_processing = post_processing
        else:
            self.post_processing = YoloePostProcessingConfig(
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold,
            )

        # Loaded at _load_model
        self._compiled_model = None
        self._infer_request = None
        self._metadata: dict[str, Any] = {}
        self._imgsz: tuple[int, int] = (640, 640)
        self._nm: int = _DEFAULT_NM
        self._names: dict[int, str] = {}

        self._load_model()

        # Set after fit()
        self._category_id_map: dict[int, int] | None = None

    @property
    def confidence_threshold(self) -> float:
        """Minimum confidence score for detections."""
        return self.post_processing.confidence_threshold

    @property
    def iou_threshold(self) -> float:
        """IoU threshold for non-maximum suppression."""
        return self.post_processing.iou_threshold

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        """Load the OpenVINO IR model and parse metadata.

        Raises:
            FileNotFoundError: If no .xml file is found in model_dir.
        """
        import openvino as ov  # noqa: PLC0415

        xml_files = list(self.model_dir.glob("*.xml"))
        if not xml_files:
            msg = f"No .xml file found in {self.model_dir}"
            raise FileNotFoundError(msg)

        xml_path = xml_files[0]
        logger.info("Loading YOLOE OpenVINO model from %s", xml_path)

        core = ov.Core()
        model = core.read_model(str(xml_path))

        ov_device = device_to_openvino_device(self.device_name)

        # GPU optimizations (aligned with SAM3OpenVINO)
        compile_props: dict[str, str] = {}
        if ov_device != "CPU":
            compile_props["PERFORMANCE_HINT"] = "LATENCY"

        # Enable model caching for GPU (faster subsequent loads)
        cache_dir = self.model_dir / ".ov_cache"
        if ov_device != "CPU":
            core.set_property({"CACHE_DIR": str(cache_dir)})

        self._compiled_model = core.compile_model(model, ov_device, compile_props)

        # Pre-create infer request (avoid per-call allocation on GPU)
        self._infer_request = self._compiled_model.create_infer_request()

        # Parse metadata
        meta_path = self.model_dir / "metadata.yaml"
        if meta_path.exists():
            with meta_path.open() as f:
                self._metadata = yaml.safe_load(f) or {}
        else:
            meta_json = self.model_dir / "metadata.json"
            if meta_json.exists():
                with meta_json.open() as f:
                    self._metadata = json.load(f)

        # Extract model parameters from metadata
        imgsz = self._metadata.get("imgsz", [640, 640])
        if isinstance(imgsz, int):
            imgsz = [imgsz, imgsz]
        self._imgsz = (imgsz[0], imgsz[1])

        names = self._metadata.get("names", {})
        self._names = {int(k): v for k, v in names.items()}

        # Determine nm from output shape
        output0_shape = self._compiled_model.output(0).partial_shape
        if len(output0_shape) == 3:
            total_cols = output0_shape[2].get_length()
            self._nm = total_cols - 6  # subtract box(4) + score(1) + class(1)
        else:
            self._nm = _DEFAULT_NM

        logger.info(
            "YOLOE OpenVINO model loaded: imgsz=%s, names=%s, nm=%d, device=%s, variant=%s",
            self._imgsz,
            self._names,
            self._nm,
            ov_device,
            self.variant.value,
        )

    # ------------------------------------------------------------------
    # fit / predict
    # ------------------------------------------------------------------
    def fit(self, reference: Sample | list[Sample] | Batch) -> None:
        """Record category mapping from reference data.

        Since classes are baked into the model at export time, ``fit()``
        only extracts category IDs from the reference to ensure
        ``pred_labels`` use the correct integer IDs downstream.

        Args:
            reference: Reference data (used only for category mapping).
        """
        reference_batch = Batch.collate(reference)

        category_id_map: dict[int, int] = {}

        for cat_ids in reference_batch.category_ids:
            if cat_ids is None:
                continue
            ids = (
                cat_ids
                if isinstance(cat_ids, (list, np.ndarray, torch.Tensor))
                else [cat_ids]
            )
            for cid in ids:
                cid_int = int(cid)
                category_id_map[cid_int] = cid_int

        self._category_id_map = category_id_map or None

        logger.info(
            "YOLOE OpenVINO fit: recorded %d category IDs",
            len(category_id_map),
        )

    def predict(self, target: Collatable) -> list[dict[str, torch.Tensor]]:
        """Run inference on target images.

        Args:
            target: Target data to infer.

        Returns:
            List of predictions per image, each containing:
                "pred_masks": [num_masks, H, W] bool
                "pred_boxes": [num_boxes, 5] with [x1, y1, x2, y2, score]
                "pred_labels": [num_masks] int64 — category IDs
        """
        target_batch = Batch.collate(target)
        all_results: list[dict[str, torch.Tensor]] = []

        for image in target_batch.images:
            prediction = self._infer_single(image)
            all_results.append(prediction)

        return all_results

    def _infer_single(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run inference on a single image tensor."""
        # Convert to HWC numpy
        img_np = image.permute(1, 2, 0).cpu().numpy()
        if img_np.dtype != np.uint8:
            img_np = (
                (img_np * 255).astype(np.uint8)
                if img_np.max() <= 1.0
                else img_np.astype(np.uint8)
            )

        orig_h, orig_w = img_np.shape[:2]
        original_shape = (orig_h, orig_w)

        # Preprocess
        input_tensor, scale, pad = preprocess_image(img_np, self._imgsz)

        # Inference (reuse pre-created request)
        self._infer_request.infer({0: input_tensor})
        det_output = self._infer_request.get_output_tensor(0).data.copy()
        proto_output = self._infer_request.get_output_tensor(1).data.copy()

        # Parse detections
        boxes, scores, class_ids, mask_coeffs = parse_detections(
            det_output,
            confidence_threshold=self.confidence_threshold,
            nm=self._nm,
        )

        if len(boxes) == 0:
            return {
                "pred_masks": torch.zeros((0, orig_h, orig_w), dtype=torch.bool),
                "pred_boxes": torch.zeros((0, 5), dtype=torch.float32),
                "pred_labels": torch.zeros(0, dtype=torch.long),
            }

        # Generate instance masks
        masks = process_mask_protos(
            mask_coeffs=mask_coeffs,
            protos=proto_output,
            boxes=boxes,
            original_shape=original_shape,
            scale=scale,
            pad=pad,
        )

        # Scale boxes back to original coordinates
        boxes_orig = scale_boxes_to_original(boxes, scale, pad, original_shape)

        # Map class IDs if category mapping exists
        if self._category_id_map is not None:
            class_ids = np.array(
                [self._category_id_map.get(cid, cid) for cid in class_ids],
                dtype=np.int64,
            )

        # Build output tensors
        pred_boxes = np.concatenate(
            [boxes_orig, scores[:, None]], axis=1,
        )

        return {
            "pred_masks": torch.from_numpy(masks),
            "pred_boxes": torch.from_numpy(pred_boxes).float(),
            "pred_labels": torch.from_numpy(class_ids).long(),
        }

    def export(
        self,
        export_dir: str | Path = Path("./exports/yoloe_openvino"),  # noqa: ARG002
        backend: str | Backend = Backend.OPENVINO,  # noqa: ARG002
        **kwargs: object,  # noqa: ARG002
    ) -> Path:
        """Export is not applicable — model is already an OpenVINO IR.

        Returns:
            Path to the model directory.
        """
        return self.model_dir
