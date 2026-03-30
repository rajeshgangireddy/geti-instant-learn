# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PerDino model."""

import torch
from torch.nn import functional

from instantlearn.components import CosineSimilarity, NegativeMaskToPoints, SamDecoder
from instantlearn.components.encoders import ImageEncoder
from instantlearn.components.feature_extractors import MaskedFeatureExtractor, ReferenceFeatures
from instantlearn.components.postprocessing import (
    PostProcessor,
    default_postprocessor,
)
from instantlearn.components.sam import load_sam_model
from instantlearn.data.base.batch import Batch, Collatable
from instantlearn.data.base.sample import BACKGROUND_CATEGORY_ID, Sample
from instantlearn.models.base import Model
from instantlearn.utils.constants import Backend, SAMModelName

from .prompt_generators import GridPromptGenerator


class PerDino(Model):
    """PerDino algorithm model for one-shot segmentation.

    Matches reference objects to target images by comparing features extracted by DINOv2
    using cosine similarity. A grid prompt generator creates multi-object aware prompts.

    The pipeline is fully traceable (ONNX/TorchScript compatible):
    - Encoder → MaskedFeatureExtractor → CosineSimilarity → GridPromptGenerator → SamDecoder

    Examples:
        >>> from instantlearn.models import PerDino
        >>> from instantlearn.data.base import Batch
        >>> from instantlearn.data.base.sample import Sample
        >>> import torch
        >>> import numpy as np

        >>> perdino = PerDino()

        >>> # Create reference sample
        >>> ref_sample = Sample(
        ...     image=torch.zeros((3, 1024, 1024)),
        ...     masks=torch.ones(30, 30, dtype=torch.bool).unsqueeze(0),
        ...     category_ids=np.array([1]),
        ...     is_reference=[True],
        ...     categories=["object"],
        ... )
        >>> ref_batch = Batch.collate([ref_sample])

        >>> # Create target sample
        >>> target_sample = Sample(
        ...     image=torch.zeros((3, 1024, 1024)),
        ...     is_reference=[False],
        ...     categories=["object"],
        ... )
        >>> target_batch = Batch.collate([target_sample])

        >>> # Run fit and predict
        >>> perdino.fit(ref_batch)
        >>> predict_results = perdino.predict(target_batch)
    """

    def __init__(
        self,
        sam: SAMModelName = SAMModelName.SAM_HQ_TINY,
        encoder_model: str = "dinov3_large",
        num_foreground_points: int = 40,
        num_background_points: int = 2,
        num_negative_points: int = 5,
        num_grid_cells: int = 16,
        point_selection_threshold: float = 0.65,
        confidence_threshold: float | None = 0.42,
        precision: str = "bf16",
        compile_models: bool = False,
        device: str = "cuda",
        postprocessor: PostProcessor | None = None,
    ) -> None:
        """Initialize the PerDino model.

        Args:
            sam: SAM model variant to use.
            encoder_model: ImageEncoder model ID to use.
            num_foreground_points: Maximum foreground points per category.
            num_background_points: Background points per category.
            num_negative_points: Points to sample per negative mask. Default: 5.
            num_grid_cells: Number of grid cells for prompt generation.
            point_selection_threshold: Minimum feature similarity for a pixel to be
                selected as a foreground point prompt for SAM. Used during prompt
                generation to identify candidate object locations. Higher values =
                fewer, more confident point proposals.
            confidence_threshold: Minimum confidence score for keeping predicted masks
                in the final output. Computed as a weighted combination of SAM's IoU
                prediction and mean similarity within the mask region. Higher values =
                stricter filtering, fewer masks.
            precision: Model precision ("bf16", "fp32").
            compile_models: Whether to compile models with torch.compile.
            device: Device for inference.
            postprocessor: Post-processor applied after predict().
                Defaults to :func:`~instantlearn.components.postprocessing.default_postprocessor`
                (MaskIoMNMS + BoxIoMNMS).
        """
        if postprocessor is None:
            postprocessor = default_postprocessor()
        super().__init__(postprocessor=postprocessor)
        self.sam_predictor = load_sam_model(
            sam,
            device=device,
            precision=precision,
            compile_models=compile_models,
        )

        self.encoder = ImageEncoder(
            model_id=encoder_model,
            backend=Backend.TIMM,
            device=device,
            precision=precision,
            compile_models=compile_models,
        )

        self.masked_feature_extractor = MaskedFeatureExtractor(
            input_size=self.encoder.input_size,
            patch_size=self.encoder.patch_size,
            device=device,
        )

        self.similarity_matcher = CosineSimilarity(feature_size=self.encoder.feature_size)

        max_points = num_foreground_points + num_background_points
        self.prompt_generator = GridPromptGenerator(
            num_grid_cells=num_grid_cells,
            point_selection_threshold=point_selection_threshold,
            num_bg_points=num_background_points,
            num_foreground_points=num_foreground_points,
            max_points=max_points,
        )

        self.segmenter = SamDecoder(
            sam_predictor=self.sam_predictor,
            confidence_threshold=confidence_threshold,
        )

        # Negative mask handling
        self.negative_mask_converter = NegativeMaskToPoints(num_points_per_mask=num_negative_points)
        self._negative_points: torch.Tensor | None = None  # (M, 2) cached during fit (SAM3 compat)
        self._negative_embedding: torch.Tensor | None = None  # (1, embed_dim) cached during fit

        self.ref_features: ReferenceFeatures | None = None

    def fit(self, reference: Sample | list[Sample] | Batch) -> None:
        """Learn from reference images.

        Negative masks (category_id == BACKGROUND_CATEGORY_ID) are used to
        extract negative embeddings that suppress similar-looking regions
        in target images during prediction.

        Args:
            reference: Reference data to learn from. Accepts:
                - Sample: A single reference sample
                - list[Sample]: A list of reference samples
                - Batch: A batch of reference samples
        """
        reference_batch = Batch.collate(reference)

        reference_embeddings = self.encoder(reference_batch.images)

        # Extract negative embedding from background mask regions
        self._negative_embedding = self._extract_negative_embedding(reference_embeddings, reference_batch)

        self.ref_features = self.masked_feature_extractor(
            reference_embeddings,
            reference_batch.masks,
            reference_batch.category_ids,
        )

    def _extract_negative_embedding(
        self,
        embeddings: torch.Tensor,
        batch: Batch,
    ) -> torch.Tensor | None:
        """Extract averaged feature embedding from background mask regions.

        Pools background masks to the patch grid and averages the corresponding
        encoder features. The result represents "what NOT to match" in feature space.

        Args:
            embeddings: Encoder features [B, num_patches, embed_dim].
            batch: Reference batch potentially containing background masks.

        Returns:
            Normalized negative embedding (1, embed_dim), or None.
        """
        neg_embeds: list[torch.Tensor] = []
        for idx, sample in enumerate(batch.samples):
            if sample.masks is None or sample.category_ids is None:
                continue
            embed = embeddings[idx]  # [num_patches, embed_dim]
            for cid, mask in zip(sample.category_ids, sample.masks, strict=False):
                val = int(cid.item()) if isinstance(cid, torch.Tensor) else int(cid)
                if val != BACKGROUND_CATEGORY_ID:
                    continue
                pooled = self.masked_feature_extractor.transform(mask).to(embed.device)
                keep = pooled.flatten().bool()
                if keep.any():
                    neg_embeds.append(embed[keep])
        if not neg_embeds:
            return None
        combined = torch.cat(neg_embeds, dim=0)  # [N, embed_dim]
        avg = combined.mean(dim=0, keepdim=True)  # [1, embed_dim]
        return functional.normalize(avg, p=2, dim=-1)

    def _adjust_similarities(
        self,
        similarities: torch.Tensor,
        target_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Suppress target regions that resemble the negative reference.

        Computes a per-target "negative similarity map" and subtracts its
        mean-centered version from each category's similarity map.  Mean-centering
        removes the constant baseline similarity that every patch shares with the
        (often very general) negative embedding, so only patches that are *more*
        background-like than average get penalized.

        Args:
            similarities: [T, C, feat_size, feat_size] per-category similarity maps.
            target_embeddings: [T, num_patches, embed_dim] target features.

        Returns:
            Adjusted similarities with above-average negative contribution subtracted.
        """
        neg_embed = self._negative_embedding  # (1, embed_dim)
        if neg_embed is None:
            return similarities

        feat_size = similarities.shape[-1]
        neg_embed = neg_embed.to(device=similarities.device, dtype=similarities.dtype)

        for t in range(similarities.shape[0]):
            target_embed = target_embeddings[t]  # [num_patches, embed_dim]
            neg_sim = (neg_embed @ target_embed.T).reshape(feat_size, feat_size)
            # Only penalize patches with above-average negative similarity;
            # this keeps absolute scores stable for true positive regions.
            penalty = torch.relu(neg_sim - neg_sim.mean())
            for c in range(similarities.shape[1]):
                similarities[t, c] = similarities[t, c] - penalty

        return similarities

    def predict(self, target: Collatable) -> list[dict[str, torch.Tensor]]:
        """Predict masks for target images.

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
                "pred_scores": [num_masks]
                "pred_labels": [num_masks] - category IDs

        Raises:
            RuntimeError: If reference features are not available.
        """
        target_batch = Batch.collate(target)
        if self.ref_features is None:
            msg = "No reference features. Call fit() first."
            raise RuntimeError(msg)

        # Get original sizes [T, 2]
        original_sizes = torch.tensor(
            [image.size()[-2:] for image in target_batch.images],
            device=self.ref_features.device,
        )

        # Encode targets [T, num_patches, embed_dim]
        target_embeddings = self.encoder(target_batch.images)

        # Compute similarities [T, C, feat_size, feat_size]
        similarities = self.similarity_matcher(
            self.ref_features.masked_ref_embeddings,
            target_embeddings,
            self.ref_features.category_ids,
        )

        # Generate prompts [T, C, max_points, 4], [T, C]
        point_prompts = self.prompt_generator(
            similarities,
            self.ref_features.category_ids,
            original_sizes,
        )

        # Penalize target regions similar to negative reference (if any)
        if self._negative_embedding is not None:
            similarities = self._adjust_similarities(similarities, target_embeddings)

        # Decode masks
        predictions = self.segmenter(
            target_batch.images,
            self.ref_features.category_ids,
            point_prompts=point_prompts,
            similarities=similarities,
        )
        return self.apply_postprocessing(predictions)
