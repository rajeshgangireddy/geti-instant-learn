# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Positional debiasing for DINOv3 features."""

import torch
from torch import nn
from torch.nn import functional


def compute_debiasing_basis(
    encoder: nn.Module,
    processor: nn.Module,
    image_size: int,
    svd_components: int,
    ignore_token_length: int,
    device: str = "cuda",
) -> torch.Tensor:
    """Compute the positional bias basis from a noise image.

    Passes a Gaussian noise image through the encoder and extracts the
    top-s right singular vectors that capture positional structure.

    Args:
        encoder: The DINOv3 encoder model.
        processor: The image preprocessing transform.
        image_size: Input image resolution.
        svd_components: Number of SVD components (rank s) to remove.
        ignore_token_length: Number of prefix tokens to strip (CLS + registers).
        device: Device to compute on.

    Returns:
        Positional basis matrix of shape (D, s) where D is embedding dimension.
    """
    # Generate noise image
    noise_image = torch.randn(3, image_size, image_size, device=device)
    noise_image = (noise_image - noise_image.min()) / (noise_image.max() - noise_image.min())
    noise_image = (noise_image * 255).to(torch.uint8).float()

    # Process and extract features
    with torch.no_grad():
        processed = processor(noise_image).unsqueeze(0).to(device)
        features = encoder.forward_features(processed)
        features = features[:, ignore_token_length:, :]  # (1, P, D)

    # SVD to find positional subspace
    F_noise = features.squeeze(0).float()  # (P, D)
    _, _, Vh = torch.linalg.svd(F_noise, full_matrices=False)

    # Top-s right singular vectors as basis for positional subspace
    basis = Vh[:svd_components, :].T  # (D, s)
    return basis


def debias_features(features: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Project features onto the orthogonal complement of the positional subspace.

    Args:
        features: Feature tensor of shape (..., D).
        basis: Positional basis of shape (D, s).

    Returns:
        Debiased features, same shape as input, L2-normalized.
    """
    # P_perp = I - B @ B^T
    D = basis.shape[0]
    basis = basis.to(device=features.device, dtype=features.dtype)
    projection = torch.eye(D, device=features.device, dtype=features.dtype) - basis @ basis.T

    # Project: F_debiased = F @ P_perp^T = F @ P_perp (symmetric)
    original_shape = features.shape
    flat = features.reshape(-1, D)
    debiased = flat @ projection
    debiased = debiased.reshape(original_shape)

    return functional.normalize(debiased, p=2, dim=-1)
