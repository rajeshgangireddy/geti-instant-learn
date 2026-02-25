# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""EfficientSAM3 model for text and visual prompting.

A student distilled variant of SAM3 using lightweight backbones (EfficientViT,
RepViT, TinyViT) and MobileCLIP text encoder for efficient inference.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from itertools import zip_longest
from typing import TYPE_CHECKING

import numpy as np
import torch
from transformers import CLIPTokenizerFast

from instantlearn.data.base.batch import Batch
from instantlearn.models.base import Model
from instantlearn.models.sam3.post_processing import PostProcessingConfig
from instantlearn.models.sam3.sam3 import Sam3PromptMode

if TYPE_CHECKING:
    from instantlearn.data.base.sample import Sample

# Reuse SAM3 processing pipeline (identical image size, normalization, postprocessing)
from instantlearn.models.sam3.processing import (
    Sam3Postprocessor as EfficientSam3Postprocessor,
)
from instantlearn.models.sam3.processing import (
    Sam3Preprocessor as EfficientSam3Preprocessor,
)
from instantlearn.models.sam3.processing import (
    Sam3PromptPreprocessor as EfficientSam3PromptPreprocessor,
)
from instantlearn.utils import precision_to_torch_dtype

from .constants import BACKBONE_CONFIG, STUDENT_CONTEXT_LENGTH
from .model import EfficientSam3Model

logger = logging.getLogger(__name__)


class EfficientSAM3(Model):
    """EfficientSAM3 model for text and visual prompting.

    Uses lightweight student backbones (EfficientViT, RepViT, TinyViT) distilled
    from SAM3, with MobileCLIP-S1 text encoder. Supports the same prompting
    patterns as SAM3 (text prompts and/or bounding boxes).

    Prompt Modes:
        **CLASSIC** (default): Original behavior. Text/box prompts are
        provided per target image. ``fit()`` only stores category names.

        **VISUAL_EXEMPLAR**: Cross-image visual query detection. During ``fit()``,
        box/point prompts on reference images are encoded into geometry features and
        cached. During ``predict()``, these cached features are reused for each
        target image — no boxes/points needed on targets.

    Examples:
        >>> from instantlearn.models import EfficientSAM3
        >>> from instantlearn.models.sam3.sam3 import Sam3PromptMode
        >>> from instantlearn.data.base.sample import Sample
        >>> from instantlearn.data.base import Batch
        >>> import torch

        >>> model = EfficientSAM3(backbone_type="efficientvit", variant="b2")

        >>> # Classic text prompting
        >>> ref = Sample(categories=["cat", "dog"], category_ids=[0, 1])
        >>> model.fit(ref)
        >>> results = model.predict(Sample(image=torch.zeros(3, 640, 480)))

        >>> # Visual exemplar mode
        >>> model_ve = EfficientSAM3(
        ...     backbone_type="efficientvit",
        ...     variant="b2",
        ...     prompt_mode=Sam3PromptMode.VISUAL_EXEMPLAR,
        ... )
        >>> ref = Sample(
        ...     image=torch.zeros(3, 640, 480),
        ...     bboxes=[[100, 100, 200, 200]],
        ...     category_ids=[0],
        ...     categories=["cat"],
        ... )
        >>> model_ve.fit(ref)
        >>> results = model_ve.predict(Sample(image=torch.zeros(3, 640, 480)))
    """

    def __init__(
        self,
        backbone_type: str = "efficientvit",
        variant: str = "b2",
        device: str = "cuda",
        confidence_threshold: float = 0.4,
        resolution: int = 1008,
        precision: str = "fp32",
        post_processing: PostProcessingConfig | None = None,
        prompt_mode: Sam3PromptMode | str = Sam3PromptMode.CLASSIC,
        drop_spatial_bias: bool = False,
    ) -> None:
        """Initialize the EfficientSAM3 model.

        Args:
            backbone_type: Vision backbone family. One of:
                'efficientvit' (variants: b0, b1, b2),
                'repvit' (variants: m0_9, m1_1, m2_3),
                'tinyvit' (variants: 5m, 11m, 21m).
            variant: Model size variant within the backbone family.
            device: Target device ('cuda', 'xpu', or 'cpu').
            confidence_threshold: Score threshold for filtering predictions.
                Default is 0.4, balancing precision and IoU across datasets.
            resolution: Input image resolution. Default: 1008.
            precision: Model precision ('fp32' or 'bf16').
            post_processing: Optional post-processing configuration for NMS,
                mask overlap removal, and non-overlapping pixel constraints.
                Default enables mask IoM suppression at 0.3.
            prompt_mode: Prompt mode for inference. 'classic' for original
                behavior, 'visual_exemplar' for cross-image visual query detection.
            drop_spatial_bias: When True and in VISUAL_EXEMPLAR mode, skip
                coordinate projection and position encoding in the geometry
                encoder, keeping only ROI-pooled visual features.

        Raises:
            ValueError: If backbone_type/variant is not supported.
        """
        super().__init__()

        key = (backbone_type, variant)
        if key not in BACKBONE_CONFIG:
            msg = f"Unsupported backbone: {backbone_type}/{variant}. Available: {list(BACKBONE_CONFIG.keys())}"
            raise ValueError(msg)

        # Default post-processing: mask IoM suppression at 0.3.
        if post_processing is None:
            post_processing = PostProcessingConfig(mask_iom_threshold=0.3)

        self.backbone_type = backbone_type
        self.variant = variant
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.resolution = resolution
        self.precision = precision
        self.prompt_mode = Sam3PromptMode(prompt_mode)
        self.drop_spatial_bias = drop_spatial_bias

        self.category_mapping: dict[str, int] | None = None

        # Visual exemplar cached features (set during fit in VISUAL_EXEMPLAR mode)
        self.exemplar_geometry_features: list[torch.Tensor] | None = None
        self.exemplar_geometry_mask: list[torch.Tensor] | None = None
        self.exemplar_text_features: list[torch.Tensor] | None = None
        self.exemplar_text_mask: list[torch.Tensor] | None = None
        self.exemplar_category_ids: list[int] | None = None

        # Reuse SAM3 preprocessors (same image pipeline)
        self.image_preprocessor = EfficientSam3Preprocessor(target_size=resolution).to(device)
        self.prompt_preprocessor = EfficientSam3PromptPreprocessor(target_size=resolution).to(device)
        self.postprocessor = EfficientSam3Postprocessor(
            target_size=resolution,
            threshold=confidence_threshold,
            mask_threshold=0.5,
            post_processing=post_processing,
        ).to(device)

        # Reuse SAM3 CLIP tokenizer (same BPE vocabulary)
        # Use pad_token_id=0 to match the original SimpleTokenizer's zero-padding
        # behavior used during EfficientSAM3's distillation training.
        self.tokenizer = CLIPTokenizerFast.from_pretrained("jetjodh/sam3")
        self.tokenizer.pad_token_id = 0

        self.model = (
            EfficientSam3Model.from_pretrained(
                backbone_type=backbone_type,
                variant=variant,
                torch_dtype=precision_to_torch_dtype(precision),
            )
            .to(device)
            .eval()
        )

    # -- Public API --

    @torch.inference_mode()
    def fit(self, reference: Sample | list[Sample] | Batch) -> None:
        """Learn from reference samples.

        In CLASSIC mode, stores category mapping only (no image processing).
        In VISUAL_EXEMPLAR mode, encodes box/point prompts on reference images into
        geometry features and caches them for reuse during predict().

        Args:
            reference: Reference data containing category names and IDs.
        """
        reference_batch = Batch.collate(reference)

        if self.prompt_mode == Sam3PromptMode.CLASSIC:
            self._fit_classic(reference_batch)
        else:
            self._fit_visual_exemplar(reference_batch)

    @torch.inference_mode()
    def predict(
        self,
        target: Sample | list[Sample] | Batch,
    ) -> list[dict[str, torch.Tensor]]:
        """Perform inference on target images.

        In CLASSIC mode, processes text/box prompts per target image.
        In VISUAL_EXEMPLAR mode, reuses cached exemplar features from fit().

        Args:
            target: Target data to infer on.

        Returns:
            List of prediction dicts, one per image, with pred_masks, pred_boxes,
            pred_labels.
        """
        if self.prompt_mode == Sam3PromptMode.VISUAL_EXEMPLAR:
            return self._predict_visual_exemplar(target)
        return self._predict_classic(target)

    # -- Fit internals --

    def _fit_classic(self, reference_batch: Batch) -> None:
        """Store category mapping from reference batch.

        Args:
            reference_batch: Batch of reference samples.
        """
        self.category_mapping = self._build_category_mapping(reference_batch)

    def _fit_visual_exemplar(self, reference_batch: Batch) -> None:
        """Encode visual exemplar features from reference images and boxes/points.

        Args:
            reference_batch: Batch of reference samples with images and bboxes/points.

        Raises:
            ValueError: If no reference samples contain bboxes or points.
        """
        encoded_by_category, category_text_map = self._encode_batch_prompts(reference_batch)

        if not encoded_by_category:
            msg = "VISUAL_EXEMPLAR mode requires at least one reference sample with bboxes or points."
            raise ValueError(msg)

        geometry_features, geometry_masks, category_ids, text_prompts = self._aggregate_category_features(
            encoded_by_category,
            category_text_map,
        )
        self.exemplar_geometry_features = geometry_features
        self.exemplar_geometry_mask = geometry_masks
        self.exemplar_category_ids = category_ids
        self.exemplar_text_features, self.exemplar_text_mask = self._cache_text_features(text_prompts)
        self.category_mapping = self._build_category_mapping(reference_batch)

        shot_info = {
            category_text_map[cat_id]: sum(f[0].shape[1] for f in encoded_by_category[cat_id])
            for cat_id in self.exemplar_category_ids
        }
        logger.info(
            "Cached %d category exemplar(s) with n-shot encoding: %s, category_ids=%s",
            len(self.exemplar_category_ids),
            shot_info,
            self.exemplar_category_ids,
        )

    def _encode_batch_prompts(
        self,
        reference_batch: Batch,
    ) -> tuple[dict[int, list[tuple[torch.Tensor, torch.Tensor]]], dict[int, str]]:
        """Encode all samples' box/point prompts into per-category geometry features.

        Args:
            reference_batch: Batch of reference samples with images and bboxes/points.

        Returns:
            Tuple of (encoded_by_category, category_text_map).
        """
        encoded_by_category: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = defaultdict(list)
        category_text_map: dict[int, str] = {}

        for sample in reference_batch.samples:
            self._encode_sample_prompts(sample, encoded_by_category, category_text_map)

        return encoded_by_category, category_text_map

    def _encode_sample_prompts(
        self,
        sample: Sample,
        encoded_by_category: dict[int, list[tuple[torch.Tensor, torch.Tensor]]],
        category_text_map: dict[int, str],
    ) -> None:
        """Encode one sample's box/point prompts into per-category geometry features.

        Args:
            sample: Reference sample with image and bboxes/points.
            encoded_by_category: Accumulator mapping cat_id to encoded features.
            category_text_map: Accumulator mapping cat_id to text name.

        Raises:
            ValueError: If the sample has prompts but no image.
        """
        bboxes = sample.bboxes
        points = sample.points
        has_bboxes = bboxes is not None and not (isinstance(bboxes, (np.ndarray, torch.Tensor)) and bboxes.size == 0)
        has_points = points is not None and not (isinstance(points, (np.ndarray, torch.Tensor)) and points.size == 0)

        if not has_bboxes and not has_points:
            return
        if sample.image is None:
            msg = "VISUAL_EXEMPLAR mode requires images in reference samples."
            raise ValueError(msg)

        # Extract vision features with autocast for student model
        image_tensor = sample.image.unsqueeze(0) if sample.image.ndim == 3 else sample.image
        with torch.autocast(device_type=self.device, dtype=precision_to_torch_dtype(self.precision)):
            pixel_values, original_sizes = self.image_preprocessor(image_tensor.to(self.device))
            vision_embeds = self.model.get_vision_features(pixel_values)
        fpn_hidden_states = vision_embeds["fpn_hidden_states"][:-1]
        fpn_position_encoding = vision_embeds["fpn_position_encoding"][:-1]

        num_prompts = max(len(bboxes) if has_bboxes else 0, len(points) if has_points else 0)
        categories = sample.categories if sample.categories is not None else ["visual"] * num_prompts
        category_ids = sample.category_ids if sample.category_ids is not None else [0] * num_prompts

        category_coords: dict[int, list[torch.Tensor]] = defaultdict(list)
        prompts = bboxes if has_bboxes else points

        for prompt, category, cat_id in zip(prompts, categories, category_ids, strict=True):
            if has_bboxes:
                input_boxes, _ = self.prompt_preprocessor(original_sizes, input_boxes=prompt)
                coord = input_boxes[..., :2]
            else:
                _, coord = self.prompt_preprocessor(original_sizes, input_points=prompt)
            cat_id_int = int(cat_id)
            category_coords[cat_id_int].append(coord)
            category_text_map[cat_id_int] = category

        for cat_id, coords_list in category_coords.items():
            all_coords = torch.cat(coords_list, dim=1)
            num_points = all_coords.shape[1]

            with torch.autocast(device_type=self.device, dtype=precision_to_torch_dtype(self.precision)):
                geometry_outputs = self.model.geometry_encoder(
                    point_embeddings=all_coords.to(dtype=fpn_hidden_states[0].dtype),
                    point_mask=torch.ones(1, num_points, dtype=torch.bool, device=self.device),
                    point_labels=torch.ones((1, num_points), dtype=torch.long, device=self.device),
                    img_feats=fpn_hidden_states,
                    img_pos_embeds=fpn_position_encoding,
                    drop_spatial_bias=self.drop_spatial_bias,
                )
            encoded_by_category[cat_id].append((
                geometry_outputs["last_hidden_state"],
                geometry_outputs["attention_mask"],
            ))

    @staticmethod
    def _aggregate_category_features(
        encoded_by_category: dict[int, list[tuple[torch.Tensor, torch.Tensor]]],
        category_text_map: dict[int, str],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[int], list[str]]:
        """Merge per-category geometry features across reference images.

        Args:
            encoded_by_category: Per-category encoded features.
            category_text_map: Mapping from cat_id to text name.

        Returns:
            Tuple of (geometry_features, geometry_masks, category_ids, text_prompts).
        """
        all_geometry_features: list[torch.Tensor] = []
        all_geometry_masks: list[torch.Tensor] = []
        all_category_ids: list[int] = []
        all_text_prompts: list[str] = []

        for cat_id in sorted(encoded_by_category.keys()):
            features_list = encoded_by_category[cat_id]
            if len(features_list) == 1:
                geo_feats, geo_mask = features_list[0]
            else:
                geo_feats = torch.cat([f[0] for f in features_list], dim=1)
                geo_mask = torch.cat([f[1] for f in features_list], dim=1)

            all_geometry_features.append(geo_feats)
            all_geometry_masks.append(geo_mask)
            all_category_ids.append(cat_id)
            all_text_prompts.append(category_text_map[cat_id])

        return all_geometry_features, all_geometry_masks, all_category_ids, all_text_prompts

    def _cache_text_features(
        self,
        text_prompts: list[str],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Tokenize and embed text prompts, cache unique results.

        Args:
            text_prompts: Text prompts aligned with exemplar categories.

        Returns:
            Tuple of (text_features, text_masks) per exemplar.
        """
        unique_prompts = list(dict.fromkeys(text_prompts))
        text_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for prompt in unique_prompts:
            text_inputs = self.tokenizer(
                [prompt],
                return_tensors="pt",
                padding="max_length",
                max_length=STUDENT_CONTEXT_LENGTH,
                truncation=True,
            )
            input_ids = text_inputs.input_ids.to(self.device)
            attention_mask = text_inputs.attention_mask.to(self.device)
            with torch.autocast(device_type=self.device, dtype=precision_to_torch_dtype(self.precision)):
                text_outputs = self.model.get_text_features(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
            text_cache[prompt] = (text_outputs.pooler_output, attention_mask.bool())

        return (
            [text_cache[p][0] for p in text_prompts],
            [text_cache[p][1] for p in text_prompts],
        )

    # -- Predict internals --

    def _predict_classic(
        self,
        target: Sample | list[Sample] | Batch,
    ) -> list[dict[str, torch.Tensor]]:
        """Classic prediction with per-image text/box/point prompts.

        Args:
            target: Target data to infer on.

        Returns:
            List of prediction dicts per image.
        """
        target_batch = Batch.collate(target)
        results = []
        use_fitted_categories = self.category_mapping is not None

        for sample in target_batch.samples:
            img_size = sample.image.shape[-2:]
            bboxes = sample.bboxes if sample.bboxes is not None else []
            points = sample.points if sample.points is not None else []

            image_tensor = sample.image.unsqueeze(0) if sample.image.ndim == 3 else sample.image
            with (
                torch.no_grad(),
                torch.autocast(device_type=self.device, dtype=precision_to_torch_dtype(self.precision)),
            ):
                pixel_values, original_sizes = self.image_preprocessor(
                    image_tensor.to(self.device),
                )
                vision_embeds = self.model.get_vision_features(pixel_values)

            if use_fitted_categories:
                texts = list(self.category_mapping.keys())
                category_ids = list(self.category_mapping.values())
            else:
                texts = sample.categories or []
                category_ids = sample.category_ids
                num_visual_prompts = max(len(bboxes), len(points))
                if num_visual_prompts and len(texts) != num_visual_prompts:
                    texts = ["visual"] * num_visual_prompts

            all_masks: list[torch.Tensor] = []
            all_boxes: list[torch.Tensor] = []
            all_labels: list[torch.Tensor] = []

            for text, bbox, point, cat_id in zip_longest(
                texts,
                bboxes,
                points,
                category_ids,
                fillvalue=None,
            ):
                text_inputs = self.tokenizer(
                    [text or "visual"],
                    return_tensors="pt",
                    padding="max_length",
                    max_length=STUDENT_CONTEXT_LENGTH,
                    truncation=True,
                )
                input_ids = text_inputs.input_ids.to(self.device)
                attention_mask = text_inputs.attention_mask.to(self.device)

                input_boxes = None
                input_boxes_labels = None
                if bbox is not None and len(bbox):
                    input_boxes, _ = self.prompt_preprocessor(original_sizes, input_boxes=bbox)
                    input_boxes_labels = torch.ones(
                        (1, 1),
                        dtype=torch.long,
                        device=self.device,
                    )

                input_points = None
                input_points_labels = None
                if point is not None and len(point):
                    _, input_points = self.prompt_preprocessor(original_sizes, input_points=point)
                    input_points_labels = torch.ones(
                        (1, 1),
                        dtype=torch.long,
                        device=self.device,
                    )

                with (
                    torch.no_grad(),
                    torch.autocast(device_type=self.device, dtype=precision_to_torch_dtype(self.precision)),
                ):
                    outputs = self.model(
                        vision_embeds=vision_embeds,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        input_boxes=input_boxes,
                        input_boxes_labels=input_boxes_labels,
                        input_points=input_points,
                        input_points_labels=input_points_labels,
                    )

                result = self.postprocessor(outputs, target_sizes=[img_size])
                boxes_with_scores = torch.cat(
                    [result[0]["boxes"], result[0]["scores"].unsqueeze(1)],
                    dim=1,
                )
                all_masks.append(result[0]["masks"].cpu())
                all_boxes.append(boxes_with_scores.cpu())
                all_labels.append(
                    torch.full(
                        (len(result[0]["boxes"]),),
                        cat_id,
                        dtype=torch.int64,
                    ),
                )

            results.append(
                self._aggregate_results(all_masks, all_boxes, all_labels, img_size),
            )

        return results

    def _predict_visual_exemplar(
        self,
        target: Sample | list[Sample] | Batch,
    ) -> list[dict[str, torch.Tensor]]:
        """Visual exemplar prediction using cached geometry features from fit().

        Args:
            target: Target data to infer on.

        Returns:
            List of prediction dicts per image.

        Raises:
            RuntimeError: If fit() has not been called with visual exemplar data.
        """
        if self.exemplar_geometry_features is None:
            msg = "No cached exemplar features. Call fit() with reference images and bboxes first."
            raise RuntimeError(msg)

        target_batch = Batch.collate(target)
        results = []

        for sample in target_batch.samples:
            img_size = sample.image.shape[-2:]

            image_tensor = sample.image.unsqueeze(0) if sample.image.ndim == 3 else sample.image
            with (
                torch.no_grad(),
                torch.autocast(device_type=self.device, dtype=precision_to_torch_dtype(self.precision)),
            ):
                pixel_values, _ = self.image_preprocessor(image_tensor.to(self.device))
                vision_embeds = self.model.get_vision_features(pixel_values)

            all_masks: list[torch.Tensor] = []
            all_boxes: list[torch.Tensor] = []
            all_labels: list[torch.Tensor] = []

            for geo_feats, geo_mask, text_feats, text_mask, cat_id in zip(
                self.exemplar_geometry_features,
                self.exemplar_geometry_mask,
                self.exemplar_text_features,
                self.exemplar_text_mask,
                self.exemplar_category_ids,
                strict=True,
            ):
                with (
                    torch.no_grad(),
                    torch.autocast(device_type=self.device, dtype=precision_to_torch_dtype(self.precision)),
                ):
                    outputs = self.model(
                        vision_embeds=vision_embeds,
                        text_embeds=text_feats,
                        attention_mask=text_mask.long(),
                        precomputed_geometry_features=geo_feats,
                        precomputed_geometry_mask=geo_mask,
                    )

                result = self.postprocessor(outputs, target_sizes=[img_size])
                boxes_with_scores = torch.cat(
                    [result[0]["boxes"], result[0]["scores"].unsqueeze(1)],
                    dim=1,
                )
                all_masks.append(result[0]["masks"].cpu())
                all_boxes.append(boxes_with_scores.cpu())
                all_labels.append(
                    torch.full(
                        (len(result[0]["boxes"]),),
                        cat_id,
                        dtype=torch.int64,
                    ),
                )

            results.append(
                self._aggregate_results(all_masks, all_boxes, all_labels, img_size),
            )

        return results

    # -- Utilities --

    @staticmethod
    def _build_category_mapping(reference_batch: Batch) -> dict[str, int]:
        """Build category name → id mapping from reference samples.

        Args:
            reference_batch: Batch of reference samples.

        Returns:
            Mapping from category name to category id.
        """
        mapping: dict[str, int] = {}
        for sample in reference_batch.samples:
            if sample.categories is None or sample.category_ids is None:
                continue
            for category_id, category in zip(sample.category_ids, sample.categories, strict=False):
                if category not in mapping:
                    mapping[category] = int(category_id)
        return mapping

    @staticmethod
    def _aggregate_results(
        all_masks: list[torch.Tensor],
        all_boxes: list[torch.Tensor],
        all_labels: list[torch.Tensor],
        img_size: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        """Aggregate results from multiple prompt predictions.

        Args:
            all_masks: List of mask tensors per prompt.
            all_boxes: List of box tensors per prompt.
            all_labels: List of label tensors per prompt.
            img_size: Original image size (height, width).

        Returns:
            Merged prediction dict with pred_masks, pred_boxes, pred_labels.
        """
        non_empty_masks = [m for m in all_masks if m.numel() > 0]
        non_empty_boxes = [b for b in all_boxes if b.numel() > 0]
        non_empty_labels = [label for label in all_labels if label.numel() > 0]

        if non_empty_masks:
            return {
                "pred_masks": torch.cat(non_empty_masks, dim=0),
                "pred_boxes": torch.cat(non_empty_boxes, dim=0),
                "pred_labels": torch.cat(non_empty_labels, dim=0),
            }

        return {
            "pred_masks": torch.empty(0, *img_size),
            "pred_boxes": torch.empty(0, 5),
            "pred_labels": torch.empty(0, dtype=torch.long),
        }

    @staticmethod
    def available_backbones() -> list[tuple[str, str]]:
        """List all supported backbone configurations.

        Returns:
            List of (backbone_type, variant) tuples.
        """
        return list(BACKBONE_CONFIG.keys())
