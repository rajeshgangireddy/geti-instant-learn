# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""INSID3: Training-free in-context segmentation with frozen DINOv3.

Based on: 'INSID3: Training-Free In-Context Segmentation with DINOv3' (CVPR 2026).
Uses only frozen DINOv3 features for segmentation via:
1. Positional debiasing (SVD on noise image features)
2. Agglomerative clustering of target features
3. Seed-cluster selection via cross-image similarity
4. Cluster aggregation using combined cross/intra similarity
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.nn import functional

from instantlearn.components.encoders import ImageEncoder
from instantlearn.data.base.batch import Batch, Collatable
from instantlearn.data.base.sample import Sample
from instantlearn.models.base import Model
from instantlearn.utils.constants import Backend

from .clustering import agglomerative_clustering, compute_cluster_prototypes
from .debiasing import compute_debiasing_basis, debias_features

if TYPE_CHECKING:
    from torchvision import tv_tensors

logger = logging.getLogger(__name__)


class INSID3(Model):
    """Training-free in-context segmentation using frozen DINOv3 features.

    INSID3 performs in-context segmentation by:
    1. Extracting dense DINOv3 features from reference and target images
    2. Removing positional bias via SVD-based debiasing
    3. Clustering target features into coherent regions
    4. Selecting and aggregating clusters matching the reference

    Args:
        encoder_model: DINOv3 encoder variant (e.g., "dinov3_large").
        image_size: Input image resolution. Default 518 (standard for DINOv3 patch16).
        svd_components: Number of SVD components for positional debiasing (rank s).
        tau: Clustering sensitivity threshold. Higher = finer clusters.
        merge_threshold: Aggregation threshold alpha for combining clusters.
        precision: Model precision ("fp32", "fp16", "bf16").
        compile_models: Whether to torch.compile the encoder.
        device: Device to use.
    """

    def __init__(
        self,
        encoder_model: str = "dinov3_large",
        image_size: int = 518,
        svd_components: int = 500,
        tau: float = 0.6,
        merge_threshold: float = 0.2,
        precision: str = "bf16",
        compile_models: bool = False,
        device: str = "cuda",
    ) -> None:
        """Initialize the INSID3 model."""
        super().__init__()

        self.device = device
        self.image_size = image_size
        self.svd_components = svd_components
        self.tau = tau
        self.merge_threshold = merge_threshold

        # Load encoder
        self.encoder = ImageEncoder(
            model_id=encoder_model,
            backend=Backend.TIMM,
            device=device,
            precision=precision,
            compile_models=compile_models,
            input_size=image_size,
        )

        # Compute and cache debiasing basis
        self._positional_basis = self._compute_basis()

        # Reference state
        self._ref_features: torch.Tensor | None = None
        self._ref_features_debiased: torch.Tensor | None = None
        self._ref_masks: torch.Tensor | None = None
        self._category_id: int = 0

    def _compute_basis(self) -> torch.Tensor:
        """Compute the positional debiasing basis from a noise image."""
        basis = compute_debiasing_basis(
            encoder=self.encoder._model.model,
            processor=self.encoder._model.processor,
            image_size=self.image_size,
            svd_components=self.svd_components,
            ignore_token_length=self.encoder._model.ignore_token_length,
            device=self.device,
        )
        return basis.to(self.device)

    def _extract_features(self, images: list[tv_tensors.Image]) -> torch.Tensor:
        """Extract normalized patch features from images.

        Args:
            images: List of images.

        Returns:
            Features of shape (B, P, D), L2-normalized.
        """
        return self.encoder(images=images)  # Already L2-normalized

    def _debias(self, features: torch.Tensor) -> torch.Tensor:
        """Apply positional debiasing to features.

        Args:
            features: Shape (B, P, D) or (P, D).

        Returns:
            Debiased features, same shape, L2-normalized.
        """
        return debias_features(features, self._positional_basis)

    def fit(self, reference: Sample | list[Sample] | Batch) -> None:
        """Learn from reference samples by extracting and caching features.

        Args:
            reference: Reference data with images and masks.
        """
        reference_batch = Batch.collate(reference)

        # Store category IDs for prediction labels
        cat_ids = reference_batch.category_ids
        if cat_ids and len(cat_ids[0]) > 0:
            self._category_id = cat_ids[0][0].item()
        else:
            self._category_id = 0

        # Extract features (B, P, D)
        ref_features = self._extract_features(images=reference_batch.images)
        self._ref_features = ref_features
        self._ref_features_debiased = self._debias(ref_features)

        # Downscale masks to patch resolution
        feature_size = self.encoder._model.feature_size
        masks_list = reference_batch.masks  # list of (N, H, W) tensors

        # Process masks: merge multi-instance masks and downsample to patch grid
        downsampled_masks = []
        for mask in masks_list:
            if mask is None:
                downsampled_masks.append(torch.zeros(feature_size, feature_size, device=self.device, dtype=torch.bool))
                continue
            if mask.dim() == 3:
                # Multi-instance: merge into single binary mask
                mask = mask.any(dim=0).float()  # (H, W)
            else:
                mask = mask.float()

            # Downsample to patch grid
            mask_down = functional.interpolate(
                mask.unsqueeze(0).unsqueeze(0),
                size=(feature_size, feature_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze()  # (h, w)
            downsampled_masks.append(mask_down > 0.5)

        self._ref_masks = torch.stack(downsampled_masks).to(self.device)  # (B, h, w)

    @torch.no_grad()
    def predict(self, target: Collatable) -> list[dict[str, torch.Tensor]]:
        """Segment target images using learned reference context.

        Args:
            target: Target images to segment.

        Returns:
            List of prediction dicts with "pred_masks" and "pred_labels".
        """
        target_batch = Batch.collate(target)
        predictions = []

        for i, image in enumerate(target_batch.images):
            pred_mask = self._segment_single(image)

            # Resize to original image resolution
            H, W = image.shape[-2:]
            if pred_mask.shape != (H, W):
                pred_mask = functional.interpolate(
                    pred_mask.unsqueeze(0).unsqueeze(0).float(),
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze() > 0.5

            predictions.append({
                "pred_masks": pred_mask.unsqueeze(0),  # (1, H, W)
                "pred_labels": torch.tensor([self._category_id], device=self.device),
            })

        return predictions

    def _segment_single(self, target_image: tv_tensors.Image) -> torch.Tensor:
        """Segment a single target image.

        Args:
            target_image: Target image tensor (C, H, W).

        Returns:
            Binary mask at patch resolution (h, w).
        """
        feature_size = self.encoder._model.feature_size
        h, w = feature_size, feature_size

        # Extract target features
        tgt_features = self._extract_features(images=[target_image])  # (1, P, D)
        tgt_features = tgt_features.squeeze(0)  # (P, D)
        tgt_features_2d = tgt_features.reshape(h, w, -1)  # (h, w, D)

        tgt_features_deb = self._debias(tgt_features)  # (P, D)
        tgt_features_deb_2d = tgt_features_deb.reshape(h, w, -1)  # (h, w, D)

        # Compute reference prototype (debiased space)
        S = self._ref_features_debiased.shape[0]  # number of reference images
        ref_prototype = self._compute_reference_prototype()  # (D,)

        # Candidate localization via backward correspondence
        candidate_mask = self._locate_candidates(tgt_features_deb_2d, ref_prototype, h, w)

        if candidate_mask.sum() == 0:
            return candidate_mask

        # Agglomerative clustering on original features
        cluster_labels = agglomerative_clustering(tgt_features, self.tau)
        cluster_labels = cluster_labels.reshape(h, w)
        K = int(cluster_labels.max().item()) + 1

        # Compute cluster prototypes in debiased space
        cluster_protos_deb = compute_cluster_prototypes(
            tgt_features_deb, cluster_labels.reshape(-1), K
        )

        # Seed selection and aggregation
        pred_mask = self._seed_and_aggregate(
            candidate_mask, cluster_labels, cluster_protos_deb, K,
            ref_prototype, tgt_features_2d, h, w,
        )

        return pred_mask

    def _compute_reference_prototype(self) -> torch.Tensor:
        """Compute the average debiased prototype from reference masks.

        Returns:
            Reference prototype of shape (D,), L2-normalized.
        """
        prototypes = []
        for m in range(self._ref_features_debiased.shape[0]):
            mask = self._ref_masks[m].reshape(-1)  # (P,)
            feats = self._ref_features_debiased[m]  # (P, D)
            if mask.any():
                proto = feats[mask].mean(dim=0)
                prototypes.append(proto)

        if not prototypes:
            return torch.zeros(self._ref_features_debiased.shape[-1], device=self.device)

        prototype = torch.stack(prototypes).mean(dim=0)
        return functional.normalize(prototype, p=2, dim=0)

    def _locate_candidates(
        self,
        tgt_features_deb_2d: torch.Tensor,
        ref_prototype: torch.Tensor,
        h: int,
        w: int,
    ) -> torch.Tensor:
        """Locate candidate target patches via forward and backward matching.

        Args:
            tgt_features_deb_2d: Target debiased features (h, w, D).
            ref_prototype: Reference prototype (D,).
            h: Feature height.
            w: Feature width.

        Returns:
            Boolean candidate mask of shape (h, w).
        """
        S = self._ref_features_debiased.shape[0]
        D = tgt_features_deb_2d.shape[-1]

        # Forward: positive similarity to reference prototype
        tgt_flat = tgt_features_deb_2d.reshape(-1, D)  # (P, D)
        sim_fwd = tgt_flat @ ref_prototype  # (P,)
        sim_fwd_2d = sim_fwd.reshape(h, w)
        forward_mask = sim_fwd_2d > 0

        if forward_mask.sum() == 0:
            forward_mask = sim_fwd_2d > sim_fwd_2d.quantile(0.9)

        # Backward: majority-vote nearest neighbor in reference
        votes = torch.zeros(h, w, dtype=torch.int32, device=tgt_flat.device)

        for m in range(S):
            ref_feats_m = self._ref_features_debiased[m]  # (P, D)
            ref_mask_m = self._ref_masks[m].reshape(-1)  # (P,)

            # For each target patch, find its NN in reference
            sim_t_to_r = tgt_flat @ ref_feats_m.T  # (P_tgt, P_ref)
            nn_indices = sim_t_to_r.argmax(dim=1)  # (P_tgt,)

            # Check if NN falls in reference mask
            nn_in_mask = ref_mask_m[nn_indices].reshape(h, w)
            votes += nn_in_mask.to(torch.int32)

        majority_thresh = math.ceil(S / 2)
        backward_mask = votes >= majority_thresh

        return forward_mask & backward_mask

    def _seed_and_aggregate(
        self,
        candidate_mask: torch.Tensor,
        cluster_labels: torch.Tensor,
        cluster_protos_deb: torch.Tensor,
        K: int,
        ref_prototype: torch.Tensor,
        tgt_features_2d: torch.Tensor,
        h: int,
        w: int,
    ) -> torch.Tensor:
        """Select seed cluster and aggregate others based on combined similarity.

        Args:
            candidate_mask: Boolean mask of candidate patches (h, w).
            cluster_labels: Cluster labels (h, w).
            cluster_protos_deb: Debiased cluster prototypes (K, D).
            K: Number of clusters.
            ref_prototype: Reference prototype (D,).
            tgt_features_2d: Original target features (h, w, D).
            h: Feature height.
            w: Feature width.

        Returns:
            Final binary mask (h, w).
        """
        # Find candidate clusters (those overlapping with candidate_mask)
        matched_mask = candidate_mask & (cluster_labels >= 0)
        if matched_mask.sum() == 0:
            return candidate_mask

        matched_labels = cluster_labels[matched_mask]
        matched_ids = matched_labels.unique()

        # Cross-image similarity for each candidate cluster (debiased space)
        protos_matched = cluster_protos_deb[matched_ids]  # (M, D)
        cross_sim_matched = protos_matched @ ref_prototype  # (M,)

        # Seed selection: cluster with highest cross-image similarity
        seed_local_idx = int(cross_sim_matched.argmax().item())
        seed_id = matched_ids[seed_local_idx].item()

        # Compute original-space prototypes for intra-image similarity
        tgt_flat = tgt_features_2d.reshape(-1, tgt_features_2d.shape[-1])
        labels_flat = cluster_labels.reshape(-1)
        orig_protos = compute_cluster_prototypes(tgt_flat, labels_flat, K)

        # Intra-image similarity to seed (original space)
        seed_proto = orig_protos[seed_id]  # (D,)
        intra_sim = orig_protos @ seed_proto  # (K,)

        # Cross-image similarity per cluster (debiased space)
        cross_sim = cluster_protos_deb @ ref_prototype  # (K,)

        # Combined score
        combined = cross_sim * intra_sim  # (K,)

        # Final mask: merge clusters exceeding threshold
        final_mask = torch.zeros(h, w, dtype=torch.bool, device=cluster_labels.device)
        for k in range(K):
            if combined[k] > self.merge_threshold:
                final_mask |= (cluster_labels == k)

        return final_mask

    def export(
        self,
        export_dir: str | Path,
        backend: str | Backend = Backend.OPENVINO,
        **kwargs: Any,
    ) -> Path:
        """Export the INSID3 encoder to ONNX/OpenVINO.

        Exports the DINOv3 encoder with debiasing as a single graph.
        The clustering and aggregation steps remain in Python since
        agglomerative clustering is not traceable.

        Args:
            export_dir: Directory to save exported model.
            backend: Export backend (ONNX or OpenVINO).
            **kwargs: Additional export arguments.

        Returns:
            Path to the exported model file.
        """
        export_path = Path(export_dir)
        export_path.mkdir(parents=True, exist_ok=True)

        # Build exportable encoder graph
        encoder_graph = INSID3EncoderGraph(
            encoder=self.encoder._model.model,
            ignore_token_length=self.encoder._model.ignore_token_length,
            positional_basis=self._positional_basis,
            input_size=self.image_size,
        )
        encoder_graph = encoder_graph.float().cpu().eval()

        # Dummy input for tracing
        dummy_input = torch.randn(1, 3, self.image_size, self.image_size)

        # ONNX export
        onnx_path = export_path / "insid3_encoder.onnx"
        torch.onnx.export(
            encoder_graph,
            args=(dummy_input,),
            f=str(onnx_path),
            input_names=["image"],
            output_names=["features", "features_debiased"],
            dynamic_axes={
                "image": {0: "batch"},
                "features": {0: "batch"},
                "features_debiased": {0: "batch"},
            },
            opset_version=17,
            dynamo=False,
        )
        msg = f"Exported ONNX model to {onnx_path}"
        logger.info(msg)

        if backend == Backend.OPENVINO:
            import openvino

            core = openvino.Core()
            ov_model = core.read_model(str(onnx_path))
            ov_path = export_path / "insid3_encoder.xml"
            openvino.save_model(ov_model, str(ov_path))
            msg = f"Exported OpenVINO model to {ov_path}"
            logger.info(msg)
            return ov_path

        return onnx_path


class INSID3EncoderGraph(nn.Module):
    """Exportable encoder graph that combines DINOv3 forward + debiasing.

    This module is designed to be ONNX/OpenVINO exportable.
    """

    def __init__(
        self,
        encoder: nn.Module,
        ignore_token_length: int,
        positional_basis: torch.Tensor,
        input_size: int = 518,
    ) -> None:
        """Initialize the exportable encoder graph.

        Args:
            encoder: The DINOv3 encoder model.
            ignore_token_length: Number of prefix tokens to strip.
            positional_basis: Positional basis for debiasing (D, s).
            input_size: Input image size.
        """
        super().__init__()
        self.encoder = encoder
        self.ignore_token_length = ignore_token_length
        self.input_size = input_size

        # Register buffers for export
        self.register_buffer("positional_basis", positional_basis)
        self.register_buffer(
            "imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        )
        self.register_buffer(
            "imagenet_std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: encode + debias.

        Args:
            x: Input image tensor (B, 3, H, W) in [0, 255] range.

        Returns:
            Tuple of (features, features_debiased), each (B, P, D) L2-normalized.
        """
        # Preprocess
        x = x.float() / 255.0
        x = functional.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear")
        x = (x - self.imagenet_mean) / self.imagenet_std

        # Extract features
        features = self.encoder.forward_features(x)
        features = features[:, self.ignore_token_length:, :]
        features = functional.normalize(features, p=2, dim=-1)

        # Debias
        D = features.shape[-1]
        basis = self.positional_basis  # (D, s)
        projection = torch.eye(D, device=features.device, dtype=features.dtype) - basis @ basis.T
        features_debiased = torch.matmul(features, projection)
        features_debiased = functional.normalize(features_debiased, p=2, dim=-1)

        return features, features_debiased
