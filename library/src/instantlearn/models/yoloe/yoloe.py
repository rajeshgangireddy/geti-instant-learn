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
from instantlearn.models.base import Model
from instantlearn.models.yoloe.config import YoloePostProcessingConfig, YoloePromptMode
from instantlearn.utils.constants import Backend

if TYPE_CHECKING:
    from ultralytics import YOLO as UltralyticsYOLO  # noqa: N811
    from ultralytics.engine.predictor import BasePredictor
    from ultralytics.engine.results import Results as UltralyticsResults

    from instantlearn.data.base.sample import Sample

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

    End-to-end model that supports both text and visual prompting for
    zero/few-shot object detection and segmentation. Unlike Matcher/PerDINO,
    YOLOE does not require a separate encoder + SAM decoder pipeline.

    Prompt Modes:
        - TEXT: Text-based class prompting via CLIP embeddings.
          Categories are extracted from reference samples and fused
          into the model weights via ``set_classes``.
        - VISUAL_EXEMPLAR: Visual prompting using bounding boxes
          extracted from reference masks.

    Examples:
        Visual prompting (default):

        >>> from instantlearn.models import YOLOE
        >>> model = YOLOE(model_name="yoloe-v8s-seg")
        >>> model.fit(ref_sample)
        >>> results = model.predict(target_sample)

        Text prompting:

        >>> from instantlearn.models.yoloe.config import YoloePromptMode
        >>> model = YOLOE(model_name="yoloe-v8s-seg", prompt_mode=YoloePromptMode.TEXT)
        >>> model.fit(ref_sample)  # categories extracted from sample
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
        prompt_mode: str | YoloePromptMode = YoloePromptMode.VISUAL_EXEMPLAR,
        post_processing: YoloePostProcessingConfig | None = None,
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
            prompt_mode: Prompting strategy — "text" or "visual_exemplar".
            post_processing: Optional PostProcessingConfig (overrides
                confidence_threshold/iou_threshold/use_nms if provided).

        Raises:
            ValueError: If model_name is not a known YOLOE model.
        """
        super().__init__()

        if model_name not in YOLOE_MODELS:
            available = ", ".join(YOLOE_MODELS.keys())
            msg = f"Unknown YOLOE model '{model_name}'. Available: {available}"
            raise ValueError(msg)

        self.model_name = model_name
        self.imgsz = imgsz
        self.precision = precision
        self.device_name = device
        self.prompt_mode = YoloePromptMode(prompt_mode)

        # Post-processing config (flat params used as defaults)
        if post_processing is not None:
            self.post_processing = post_processing
        else:
            self.post_processing = YoloePostProcessingConfig(
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold,
                use_nms=use_nms,
            )

        self._model = self._load_model()
        self._predictor_cls = self._get_predictor_cls()

        # State populated by fit()
        self._visual_prompts_set = False
        self._text_prompts_set = False
        self._category_mapping: dict[str, int] = {}

    @property
    def confidence_threshold(self) -> float:
        """Minimum confidence score for detections."""
        return self.post_processing.confidence_threshold

    @property
    def iou_threshold(self) -> float:
        """IoU threshold for non-maximum suppression."""
        return self.post_processing.iou_threshold

    @property
    def use_nms(self) -> bool:
        """Whether to apply non-maximum suppression."""
        return self.post_processing.use_nms

    def _load_model(self) -> UltralyticsYOLO:
        """Load the YOLOE model from ultralytics.

        Raises:
            ImportError: If ultralytics is not installed.
        """
        try:
            from ultralytics import YOLO  # noqa: PLC0415

            logging.getLogger("ultralytics").setLevel(logging.WARNING)
        except ImportError as e:
            msg = (
                "ultralytics is required for YOLOE. "
                "Install it with: uv pip install ultralytics"
            )
            raise ImportError(msg) from e

        from instantlearn.models.yoloe.weights import get_weights_path  # noqa: PLC0415

        model_file = YOLOE_MODELS[self.model_name]
        model_path = get_weights_path(model_file)
        logger.info("Loading YOLOE model: %s", model_path)
        model = YOLO(str(model_path))
        model.to(self.device_name)

        return model

    def _get_predictor_cls(self) -> type[BasePredictor]:
        """Return the correct YOLOE visual-prompt predictor class."""
        from ultralytics.models.yolo.yoloe.predict import (  # noqa: PLC0415
            YOLOEVPDetectPredictor,
            YOLOEVPSegPredictor,
        )

        if "seg" in self.model_name:
            return YOLOEVPSegPredictor
        return YOLOEVPDetectPredictor

    @staticmethod
    def _image_to_numpy(image: torch.Tensor) -> np.ndarray:
        """Convert a CHW torch image tensor to an HWC uint8 numpy array."""
        img_np = image.permute(1, 2, 0).cpu().numpy()
        if img_np.dtype != np.uint8:
            img_np = (
                (img_np * 255).astype(np.uint8)
                if img_np.max() <= 1.0
                else img_np.astype(np.uint8)
            )
        return img_np

    # ------------------------------------------------------------------
    # fit() dispatch
    # ------------------------------------------------------------------
    def fit(self, reference: Sample | list[Sample] | Batch) -> None:
        """Learn from reference images.

        Dispatches to ``_fit_text`` or ``_fit_visual_exemplar`` depending
        on ``prompt_mode``.

        Args:
            reference: Reference data to learn from.
        """
        reference_batch = Batch.collate(reference)

        if self.prompt_mode == YoloePromptMode.TEXT:
            self._fit_text(reference_batch)
        else:
            self._fit_visual_exemplar(reference_batch)

    def _fit_text(self, reference_batch: Batch) -> None:
        """Extract category names and fuse text embeddings into model.

        Uses ultralytics ``get_text_pe()`` + ``set_classes()`` to bake
        CLIP text embeddings into the classification head weights.
        """
        # Collect unique category names from reference
        categories: list[str] = []
        category_id_to_name: dict[int, str] = {}

        for sample in reference_batch.samples:
            if sample.categories is None:
                continue
            cat_ids = sample.category_ids
            if cat_ids is None:
                cat_ids = list(range(len(sample.categories)))
            for cat_name, cat_id in zip(sample.categories, cat_ids, strict=True):
                cid = int(cat_id)
                if cid not in category_id_to_name:
                    category_id_to_name[cid] = cat_name

        # Sort by ID for deterministic ordering
        sorted_ids = sorted(category_id_to_name.keys())
        categories = [category_id_to_name[cid] for cid in sorted_ids]
        self._category_mapping = {name: idx for idx, name in enumerate(categories)}

        if not categories:
            logger.warning("No categories found in reference data for text prompting.")
            return

        # Get CLIP text embeddings and fuse into model
        inner = self._model.model
        text_pe = inner.get_text_pe(categories)
        inner.set_classes(categories, text_pe)

        self._text_prompts_set = True
        logger.info(
            "Text prompts set with %d categories: %s",
            len(categories),
            categories,
        )

    def _fit_visual_exemplar(self, reference_batch: Batch) -> None:
        """Extract bounding boxes from reference masks for visual prompting."""
        self._prepare_visual_prompts(
            images=reference_batch.images,
            masks=reference_batch.masks,
            category_ids=reference_batch.category_ids,
        )

    def _prepare_visual_prompts(
        self,
        images: list[torch.Tensor],
        masks: list[torch.Tensor | None],
        category_ids: list[torch.Tensor],
    ) -> None:
        """Prepare and store visual prompts from reference data.

        Extracts bounding boxes from masks and builds the
        ``visual_prompts`` dict expected by the ultralytics YOLOE API.
        """
        all_bboxes: list[list[float]] = []
        all_cls: list[int] = []

        for mask_set, cat_ids in zip(masks, category_ids, strict=True):
            if mask_set is None:
                continue

            if mask_set.dim() == 2:
                mask_set = mask_set.unsqueeze(0)  # noqa: PLW2901

            cat_ids_tensor = (
                torch.atleast_1d(cat_ids)
                if isinstance(cat_ids, torch.Tensor)
                else torch.atleast_1d(torch.tensor(cat_ids))
            )

            for mask, cat_id in zip(mask_set, cat_ids_tensor, strict=True):
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

    # ------------------------------------------------------------------
    # predict() dispatch
    # ------------------------------------------------------------------
    def predict(self, target: Collatable) -> list[dict[str, torch.Tensor]]:
        """Detect objects in target images.

        Dispatches to ``_predict_text`` or ``_predict_visual_exemplar``
        depending on ``prompt_mode``.

        Args:
            target: Target data to infer.

        Returns:
            List of predictions per image, each containing:
                "pred_masks": [num_masks, H, W]
                "pred_boxes": [num_boxes, 5] with [x1, y1, x2, y2, score]
                "pred_labels": [num_masks] - category IDs

        """
        if self.prompt_mode == YoloePromptMode.TEXT:
            return self._predict_text(target)
        return self._predict_visual_exemplar(target)

    def _predict_text(self, target: Collatable) -> list[dict[str, torch.Tensor]]:
        """Predict using text-prompted model (classes fused into weights).

        Raises:
            RuntimeError: If fit() has not been called before predict().
        """
        if not self._text_prompts_set:
            msg = "No text prompts set. Call fit() first."
            raise RuntimeError(msg)

        target_batch = Batch.collate(target)
        all_results = []

        for image in target_batch.images:
            img_np = self._image_to_numpy(image)

            results = self._model.predict(
                source=img_np,
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

    def _predict_visual_exemplar(self, target: Collatable) -> list[dict[str, torch.Tensor]]:
        """Predict using visual prompts (bounding boxes from reference).

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

    @staticmethod
    def _parse_result(
        result: UltralyticsResults | None,
        original_size: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        """Parse ultralytics result into the standard prediction format."""
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

    def export(
        self,
        export_dir: str | Path = Path("./exports/yoloe"),
        backend: str | Backend = Backend.OPENVINO,
        **kwargs: object,
    ) -> Path:
        """Export is not supported directly on the model class.

        Use the export scripts in ``instantlearn.scripts.yoloe`` instead.

        Raises:
            NotImplementedError: Always raised.
        """
        msg = (
            "Direct export is not supported. Use the export scripts: "
            "instantlearn.scripts.yoloe.export_yoloe_openvino"
        )
        raise NotImplementedError(msg)
