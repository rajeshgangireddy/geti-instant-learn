# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""EfficientSAM3 model for text and visual prompting.

A student distilled variant of SAM3 using lightweight backbones (EfficientViT,
RepViT, TinyViT) and MobileCLIP text encoder for efficient inference.

Inherits the full exemplar mode pipeline from SAM3 and overrides only the
autocast behavior, tokenizer configuration, and model initialization.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from contextlib import nullcontext
from transformers import CLIPTokenizerFast

from instantlearn.components.postprocessing import PostProcessor, default_postprocessor
from instantlearn.models.sam3.post_processing import PostProcessingConfig
from instantlearn.models.sam3.processing import (
    Sam3Postprocessor as EfficientSam3Postprocessor,
)
from instantlearn.models.sam3.processing import (
    Sam3Preprocessor as EfficientSam3Preprocessor,
)
from instantlearn.models.sam3.processing import (
    Sam3PromptPreprocessor as EfficientSam3PromptPreprocessor,
)
from instantlearn.models.sam3.sam3 import SAM3, CanvasConfig, Sam3PromptMode
from instantlearn.utils import precision_to_torch_dtype

from .constants import BACKBONE_CONFIG, STUDENT_CONTEXT_LENGTH
from .model import EfficientSam3Model

logger = logging.getLogger(__name__)


def _resolve_precision(precision: str, device: str) -> str:
    """Resolve a precision string, expanding 'auto' based on device.

    'auto' picks 'bf16' when the target device is CUDA (with CUDA available)
    or Intel XPU (with XPU available), and falls back to 'fp32' otherwise.
    Any other value is returned unchanged (validation happens downstream in
    ``precision_to_torch_dtype``).

    Args:
        precision: Requested precision ('auto', 'fp32', 'fp16', or 'bf16').
        device: Target device string ('cuda', 'xpu', or 'cpu').

    Returns:
        Resolved precision string.
    """
    if precision != "auto":
        return precision
    device_type = torch.device(device).type
    if device_type == "cuda" and torch.cuda.is_available():
        return "bf16"
    if device_type == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
        return "bf16"
    return "fp32"


class EfficientSAM3(SAM3):
    """EfficientSAM3 model for text and visual prompting.

    Uses lightweight student backbones (EfficientViT, RepViT, TinyViT) distilled
    from SAM3, with MobileCLIP-S1 text encoder. Supports the same prompting
    patterns as SAM3 (text prompts and/or bounding boxes).

    Inherits all exemplar mode logic (fit, predict, encoding, caching) from SAM3,
    overriding only model initialization, autocast behavior, and tokenizer
    configuration.

    Prompt Modes:
        **CLASSIC** (default): Original behavior. Text/box prompts are
        provided per target image. ``fit()`` only stores category names.

        **VISUAL_EXEMPLAR**: Cross-image visual query detection. During ``fit()``,
        box/point prompts on reference images are encoded into geometry features and
        cached. During ``predict()``, these cached features are reused for each
        target image — no boxes/points needed on targets.

        **CANVAS**: Cross-image detection via image-stitching. Reference image(s)
        with box prompts are stitched alongside the target into a single canvas at
        predict time; the model then runs in classic mode with the reference bbox
        remapped to canvas coordinates. Inherits the SAM3 canvas pipeline
        unchanged; behaviour on distilled student backbones is empirical.

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
        precision: str = "auto",
        post_processing: PostProcessingConfig | None = None,
        prompt_mode: Sam3PromptMode | str = Sam3PromptMode.CLASSIC,
        canvas_config: CanvasConfig | None = None,
        drop_spatial_bias: bool = False,
        postprocessor: PostProcessor | None = None,
        ft: bool = False,
        compile_model: bool = False,
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
            precision: Model precision. One of 'auto' (default), 'fp32',
                'fp16', or 'bf16'. 'auto' resolves to 'bf16' on CUDA/XPU
                devices and 'fp32' on CPU.
            post_processing: Optional post-processing configuration for NMS,
                mask overlap removal, and non-overlapping pixel constraints.
                Default enables mask IoM suppression at 0.3.
            prompt_mode: Prompt mode for inference. 'classic' for original
                behavior, 'visual_exemplar' for cross-image visual query detection,
                'canvas' for stitched-canvas cross-image detection.
            canvas_config: Configuration for canvas mode (split ratio, crop
                padding, text caching, vision sharing). See :class:`CanvasConfig`.
                Default: ``None`` (uses ``CanvasConfig(split_ratio=0.5,
                crop_padding=1.5)``, tuned for the distilled student backbone
                via :file:`tools/probe_efficientsam3_canvas_full_sweep.py`).
                The SAM3 teacher default ``split_ratio=0.3`` underweights the
                reference strip and the student backbone fails to extract a
                usable exemplar embedding; doubling the strip and tightening
                ``crop_padding`` recovers F1 from 0.0 to ~0.5 on PerSeg.
            drop_spatial_bias: When True and in VISUAL_EXEMPLAR mode, skip
                coordinate projection and position encoding in the geometry
                encoder, keeping only ROI-pooled visual features.
            postprocessor: Post-processor applied after predict().
                Defaults to :func:`~instantlearn.components.postprocessing.default_postprocessor`
                (MaskIoMNMS + BoxIoMNMS).
            ft: When True, load the fine-tuned checkpoint instead of the
                stage-1 distilled checkpoint. Only available for the medium
                variant of each backbone family: efficientvit/b1, repvit/m1_1,
                tinyvit/11m. Raises ValueError for other variants.
            compile_model: When True, wrap the underlying model with
                :func:`torch.compile` for potentially faster inference.
                Compilation happens lazily on the first forward call. If the
                ``torch.compile`` call itself raises, a warning is logged and
                the uncompiled model is kept. Runtime compilation failures on
                the first forward are not caught here. TinyViT backbones use
                a custom (non-timm) implementation and may not compile cleanly.

        Raises:
            ValueError: If backbone_type/variant is not supported.
        """
        # Skip SAM3.__init__ -- we initialize nn.Module and set attributes directly
        # because EfficientSAM3 uses different model, tokenizer, and defaults.
        if postprocessor is None:
            postprocessor = default_postprocessor()
        super(SAM3, self).__init__(postprocessor=postprocessor)

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
        self.precision = _resolve_precision(precision, device)
        self.prompt_mode = Sam3PromptMode(prompt_mode)
        # EfficientSAM3-specific canvas defaults: larger reference strip
        # (split_ratio=0.5) and tighter crop padding (crop_padding=1.5) to
        # compensate for the distilled student backbone's weaker handling of
        # the canvas distribution. See tools/probe_efficientsam3_canvas_full_sweep.py.
        self.canvas_config = canvas_config or CanvasConfig(
            split_ratio=0.5, crop_padding=1.5,
        )
        self.drop_spatial_bias = drop_spatial_bias

        self.category_mapping: dict[str, int] | None = None

        # Visual exemplar cached features (set during fit in VISUAL_EXEMPLAR mode)
        self.exemplar_geometry_features: list[torch.Tensor] | None = None
        self.exemplar_geometry_mask: list[torch.Tensor] | None = None
        self.exemplar_text_features: list[torch.Tensor] | None = None
        self.exemplar_text_mask: list[torch.Tensor] | None = None
        self.exemplar_category_ids: list[int] | None = None

        # Canvas cached state (set during fit in CANVAS mode)
        self._canvas_refs_by_category: dict[int, dict] | None = None
        self._canvas_text_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        # Reuse SAM3 preprocessors (same image pipeline)
        self.image_preprocessor = EfficientSam3Preprocessor(target_size=resolution).to(device)
        self.prompt_preprocessor = EfficientSam3PromptPreprocessor(target_size=resolution).to(device)
        self.sam3_postprocessor = EfficientSam3Postprocessor(
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
                torch_dtype=precision_to_torch_dtype(self.precision),
                ft=ft,
            )
            .to(device)
            .eval()
        )

        if compile_model:
            self._try_compile_model()

    def _try_compile_model(self) -> None:
        """Wrap ``self.model`` with :func:`torch.compile`, falling back on failure.

        Compilation is best-effort: any exception raised by ``torch.compile``
        itself is logged as a warning and the original (uncompiled) model is
        kept. Runtime compilation failures on the first forward pass are not
        caught here and will propagate normally.
        """
        if self.backbone_type == "tinyvit":
            logger.warning(
                "compile_model=True with TinyViT backbone: the SAM-HQ TinyViT "
                "implementation uses custom ops that may not compile cleanly. "
                "Compilation will be attempted but may raise on first forward.",
            )
        try:
            self.model = torch.compile(self.model)
        except Exception as exc:  # noqa: BLE001 -- torch.compile may raise many error types
            logger.warning(
                "torch.compile failed for EfficientSAM3 (%s/%s): %s. "
                "Falling back to uncompiled model.",
                self.backbone_type,
                self.variant,
                exc,
            )

    # -- Hook overrides --

    def _get_autocast_context(self) -> torch.autocast | nullcontext:
        """Return torch.autocast for mixed-precision student inference."""
        device_type = torch.device(self.device).type
        return torch.autocast(device_type=device_type, dtype=precision_to_torch_dtype(self.precision))

    def _tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        """Tokenize with student-specific settings (context length, truncation).

        Args:
            texts: List of text strings to tokenize.

        Returns:
            Tokenizer output dict with input_ids and attention_mask on device.
        """
        text_inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            max_length=STUDENT_CONTEXT_LENGTH,
            truncation=True,
        )
        return {
            "input_ids": text_inputs.input_ids.to(self.device),
            "attention_mask": text_inputs.attention_mask.to(self.device),
        }

    # -- Utilities --

    @staticmethod
    def available_backbones() -> list[tuple[str, str]]:
        """List all supported backbone configurations.

        Returns:
            List of (backbone_type, variant) tuples.
        """
        return list(BACKBONE_CONFIG.keys())
