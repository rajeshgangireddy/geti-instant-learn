# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""YOLOE model for real-time open-vocabulary detection and segmentation.

Based on "YOLOE: Real-Time Seeing Anything" — an end-to-end model
that supports text and visual prompting for zero/few-shot detection
and instance segmentation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from instantlearn.data.base.batch import Batch, Collatable
from instantlearn.data.base.sample import Sample
from instantlearn.models.base import Model
from instantlearn.utils.constants import Backend

if TYPE_CHECKING:
    from ultralytics import YOLO as UltralyticsYOLO
    from ultralytics.engine.results import Results as UltralyticsResults
    from ultralytics.engine.predictor import BasePredictor

logger = logging.getLogger(__name__)

YOLOE_MODELS: dict[str, str] = {
    "yoloe-v8s-seg": "yoloe-v8s-seg.pt",
    "yoloe-v8m-seg": "yoloe-v8m-seg.pt",
    "yoloe-v8l-seg": "yoloe-v8l-seg.pt",
    "yoloe-11s-seg": "yoloe-11s-seg.pt",
    "yoloe-11m-seg": "yoloe-11m-seg.pt",
    "yoloe-11l-seg": "yoloe-11l-seg.pt",
    "yoloe-26n-seg": "yoloe-26n-seg.pt",
    "yoloe-26s-seg": "yoloe-26s-seg.pt",
    "yoloe-26m-seg": "yoloe-26m-seg.pt",
    "yoloe-26l-seg": "yoloe-26l-seg.pt",
    "yoloe-26x-seg": "yoloe-26x-seg.pt",
}


class YOLOE(Model):
    """YOLOE model for open-vocabulary detection and instance segmentation.

    End-to-end model that supports visual prompting for zero/few-shot
    object detection and segmentation. Unlike Matcher/PerDINO, YOLOE
    does not require a separate encoder + SAM decoder pipeline.

    Examples:
        >>> from instantlearn.models import YOLOE
        >>> from instantlearn.data.base.sample import Sample
        >>> import torch
        >>> import numpy as np

        >>> model = YOLOE(model_name="yoloe-v8s-seg")

        >>> # Create reference sample
        >>> ref_sample = Sample(
        ...     image=torch.zeros((3, 640, 640)),
        ...     masks=torch.ones(30, 30, dtype=torch.bool).unsqueeze(0),
        ...     category_ids=np.array([1]),
        ...     is_reference=[True],
        ...     categories=["object"],
        ... )

        >>> # Create target sample
        >>> target_sample = Sample(
        ...     image=torch.zeros((3, 640, 640)),
        ...     is_reference=[False],
        ...     categories=["object"],
        ... )

        >>> model.fit(ref_sample)
        >>> results = model.predict(target_sample)
    """

    def __init__(
        self,
        model_name: str = "yoloe-v8s-seg",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        imgsz: int = 640,
        use_nms: bool = True,
        precision: str = "fp16",
        device: str = "cuda",
    ) -> None:
        """Initialize the YOLOE model.

        Args:
            model_name: YOLOE model variant to use.
            confidence_threshold: Minimum confidence score for detections.
            iou_threshold: IoU threshold for non-maximum suppression.
            imgsz: Input image size for the model.
            use_nms: Whether to apply non-maximum suppression.
            precision: Model precision ("fp16", "fp32", "bf16").
            device: Device for inference ("cuda", "cpu").
        """
        super().__init__()

        if model_name not in YOLOE_MODELS:
            available = ", ".join(YOLOE_MODELS.keys())
            msg = f"Unknown YOLOE model '{model_name}'. Available: {available}"
            raise ValueError(msg)

        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.use_nms = use_nms
        self.precision = precision
        self.device_name = device

        self._model = self._load_model()
        self._predictor_cls = self._get_predictor_cls()
        self._visual_prompts_set = False

    def _load_model(self) -> UltralyticsYOLO:
        """Load the YOLOE model from ultralytics.

        Returns:
            Loaded YOLO model instance.
        """
        try:
            from ultralytics import YOLO

            logging.getLogger("ultralytics").setLevel(logging.WARNING)
        except ImportError as e:
            msg = (
                "ultralytics is required for YOLOE. "
                "Install it with: uv pip install ultralytics"
            )
            raise ImportError(msg) from e

        from instantlearn.models.yoloe.weights import get_weights_path

        model_file = YOLOE_MODELS[self.model_name]
        model_path = get_weights_path(model_file)
        logger.info("Loading YOLOE model: %s", model_path)
        model = YOLO(str(model_path))
        model.to(self.device_name)

        return model

    def _get_predictor_cls(self) -> type[BasePredictor]:
        """Return the correct YOLOE visual-prompt predictor class.

        Uses ``YOLOEVPSegPredictor`` for segmentation model variants
        and ``YOLOEVPDetectPredictor`` for detection-only variants.

        Returns:
            Predictor class appropriate for the loaded model.
        """
        from ultralytics.models.yolo.yoloe.predict import (
            YOLOEVPDetectPredictor,
            YOLOEVPSegPredictor,
        )

        if "seg" in self.model_name:
            return YOLOEVPSegPredictor
        return YOLOEVPDetectPredictor

    @staticmethod
    def _image_to_numpy(image: torch.Tensor) -> np.ndarray:
        """Convert a CHW torch image tensor to an HWC uint8 numpy array.

        Args:
            image: Image tensor of shape [3, H, W].

        Returns:
            Numpy array of shape [H, W, 3] with dtype uint8.
        """
        img_np = image.permute(1, 2, 0).cpu().numpy()
        if img_np.dtype != np.uint8:
            img_np = (
                (img_np * 255).astype(np.uint8)
                if img_np.max() <= 1.0
                else img_np.astype(np.uint8)
            )
        return img_np

    def _prepare_visual_prompts(
        self,
        images: list[torch.Tensor],
        masks: list[torch.Tensor | None],
        category_ids: list[torch.Tensor],
    ) -> None:
        """Prepare and store visual prompts from reference data.

        Extracts bounding boxes from masks and builds the
        ``visual_prompts`` dict expected by the ultralytics YOLOE
        ``predict()`` API.  Only the **first** reference image is
        used as ``refer_image`` (ultralytics limitation).

        Args:
            images: Reference images, each of shape [3, H, W].
            masks: Reference masks, each of shape [N, H, W] (or None).
            category_ids: Integer category ID tensors aligned with masks.
        """
        all_bboxes: list[list[float]] = []
        all_cls: list[int] = []

        for mask_set, cat_ids in zip(masks, category_ids):
            if mask_set is None:
                continue

            if mask_set.dim() == 2:
                mask_set = mask_set.unsqueeze(0)

            cat_ids_tensor = (
                torch.atleast_1d(cat_ids)
                if isinstance(cat_ids, torch.Tensor)
                else torch.atleast_1d(torch.tensor(cat_ids))
            )

            for mask, cat_id in zip(mask_set, cat_ids_tensor):
                ys, xs = torch.where(mask > 0)
                if len(ys) == 0:
                    continue
                x1, y1 = xs.min().item(), ys.min().item()
                x2, y2 = xs.max().item(), ys.max().item()
                all_bboxes.append([x1, y1, x2, y2])
                all_cls.append(int(cat_id.item()))

        if not all_bboxes:
            logger.warning("No valid bounding boxes extracted from reference masks.")
            return

        # Store for use in predict()
        self._refer_image = self._image_to_numpy(images[0])
        self._visual_prompts: dict[str, list] = {
            "bboxes": all_bboxes,
            "cls": all_cls,
        }
        self._visual_prompts_set = True

        logger.info(
            "Visual prompts prepared with %d bounding box(es) from %d reference image(s)",
            len(all_bboxes),
            len(images),
        )

    def fit(self, reference: Sample | list[Sample] | Batch) -> None:
        """Learn from reference images by setting visual prompts.

        Args:
            reference: Reference data to learn from. Accepts:
                - Sample: A single reference sample
                - list[Sample]: A list of reference samples
                - Batch: A batch of reference samples
        """
        reference_batch = Batch.collate(reference)

        self._prepare_visual_prompts(
            images=reference_batch.images,
            masks=reference_batch.masks,
            category_ids=reference_batch.category_ids,
        )

    def predict(self, target: Collatable) -> list[dict[str, torch.Tensor]]:
        """Detect objects in target images using visual prompts.

        Args:
            target: Target data to infer. Accepts:
                - Sample: A single target sample
                - list[Sample]: A list of target samples
                - Batch: A batch of target samples
                - str | Path: A single image path
                - list[str] | list[Path]: Multiple image paths

        Returns:
            List of predictions per image, each containing:
                "pred_masks": [num_masks, H, W]
                "pred_boxes": [num_boxes, 5] with [x1, y1, x2, y2, score]
                "pred_labels": [num_masks] - category IDs

        Raises:
            RuntimeError: If fit() has not been called before predict().
        """
        if not self._visual_prompts_set:
            msg = "No visual prompts set. Call fit() first."
            raise RuntimeError(msg)

        target_batch = Batch.collate(target)
        all_results = []

        for image in target_batch.images:
            img_np = self._image_to_numpy(image)

            results = self._model.predict(
                source=img_np,
                visual_prompts=self._visual_prompts,
                refer_image=self._refer_image,
                predictor=self._predictor_cls,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                imgsz=self.imgsz,
                device=self.device_name,
                verbose=False,
            )

            result = results[0] if results else None
            prediction = self._parse_result(result, image.shape[-2:])
            all_results.append(prediction)

        return all_results

    def _parse_result(
        self,
        result: UltralyticsResults | None,
        original_size: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        """Parse ultralytics result into the standard prediction format.

        Args:
            result: Ultralytics detection result.
            original_size: Original image size (H, W).

        Returns:
            Dictionary with pred_masks, pred_boxes, and pred_labels.
        """
        if result is None or result.boxes is None or len(result.boxes) == 0:
            h, w = original_size
            return {
                "pred_masks": torch.zeros((0, h, w)),
                "pred_boxes": torch.zeros((0, 5)),
                "pred_labels": torch.zeros(0, dtype=torch.long),
            }

        boxes = result.boxes
        pred_boxes = torch.cat([
            boxes.xyxy,
            boxes.conf.unsqueeze(1),
        ], dim=1)
        pred_labels = boxes.cls.long()

        if result.masks is not None:
            masks_data = result.masks.data
            h, w = original_size
            if masks_data.shape[-2:] != (h, w):
                masks_data = torch.nn.functional.interpolate(
                    masks_data.unsqueeze(0).float(),
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            pred_masks = (masks_data > 0.5).bool()
        else:
            h, w = original_size
            pred_masks = torch.zeros((len(pred_labels), h, w), dtype=torch.bool)

        return {
            "pred_masks": pred_masks,
            "pred_boxes": pred_boxes,
            "pred_labels": pred_labels,
        }

    @torch.no_grad()
    def export(
        self,
        export_dir: str | Path = Path("./exports/yoloe"),
        backend: str | Backend = Backend.ONNX,
        **kwargs,
    ) -> Path:
        """Export YOLOE model.

        Args:
            export_dir: Directory to save exported model.
            backend: Export backend (ONNX, OpenVINO).
            **kwargs: Additional export parameters.

        Returns:
            Path to the exported model file.

        Raises:
            ImportError: If OpenVINO is requested but not installed.
        """
        export_path = Path(export_dir)
        export_path.mkdir(parents=True, exist_ok=True)

        if backend == Backend.ONNX:
            result = self._model.export(
                format="onnx",
                imgsz=self.imgsz,
                half=self.precision == "fp16",
                simplify=True,
            )
            onnx_path = Path(result)
            target = export_path / "yoloe.onnx"
            onnx_path.rename(target)
            return target

        if backend == Backend.OPENVINO:
            try:
                result = self._model.export(
                    format="openvino",
                    imgsz=self.imgsz,
                    half=self.precision == "fp16",
                )
                ov_path = Path(result)
                return ov_path
            except ImportError as e:
                msg = "OpenVINO is not installed. Install it to use OpenVINO export."
                raise ImportError(msg) from e

        return export_path
