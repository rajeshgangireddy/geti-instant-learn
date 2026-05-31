# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""EfficientSAM3 OpenVINO inference model.

Lightweight student variant of SAM3 — swaps the heavy ViT backbone for a timm
backbone (default EfficientViT-B1) and CLIP text encoder for MobileCLIP-S1.

Provides the same inference API as :class:`SAM3OpenVINO` for **CLASSIC** and
**VISUAL_EXEMPLAR** prompt modes. **CANVAS** mode is not supported because
the distilled student backbones were not trained on canvas inputs.

Differences from :class:`SAM3OpenVINO`:

* Tokenizer: ``jetjodh/sam3`` CLIP tokenizer with ``pad_token_id=0``.
* Prompt decoder takes two extra inputs — raw ``text_features`` and
  ``text_mask`` — because :meth:`Sam3Model._get_scoring_features` is
  overridden to use raw text features (not the DETR-encoder-fused ones).
* ``presence_logits`` is a fixed zero placeholder (``use_presence=False``)
  so scoring relies on ``pred_logits`` only.
* CANVAS mode raises :class:`ValueError` at ``__init__`` and ``fit()``.

See Also:
    - :class:`SAM3OpenVINO`: parent class for SAM3 OpenVINO runtime.
    - :class:`EfficientSAM3`: PyTorch-based EfficientSAM3 high-level model.
    - ``export_efficient_sam3``: CLI script for exporting and quantizing models.
"""

import logging
from enum import Enum
from pathlib import Path

import numpy as np
from transformers import CLIPTokenizerFast

from instantlearn.models.sam3.sam3 import Sam3PromptMode
from instantlearn.models.sam3.sam3_openvino import (
    SAM3OVVariant,
    SAM3OpenVINO,
    _get_model_file,
    _GEOMETRY_ENCODER,
    _GEOMETRY_ENCODER_EXEMPLAR,
    _PROMPT_DECODER,
    _TEXT_ENCODER,
    _VISION_ENCODER,
)
from instantlearn.utils import device_to_openvino_device

from .constants import STUDENT_CONTEXT_LENGTH

logger = logging.getLogger(__name__)

# Default HuggingFace repo for EfficientSAM3 OpenVINO models.
EFFICIENT_SAM3_OV_REPO = "rajeshgangireddy/EfficientSAM3_OpenVINO"

# Tokenizer source matches the PyTorch EfficientSAM3 model.
_TOKENIZER_REPO = "jetjodh/sam3"
_TOKENIZER_PAD_ID = 0


class EfficientSAM3OVVariant(str, Enum):
    """Available EfficientSAM3 OpenVINO model variants.

    Each variant maps to a subdirectory on HuggingFace Hub
    (``rajeshgangireddy/EfficientSAM3_OpenVINO``).

    Recommended variants:

    * **FP16** — Best Intel GPU (Arc/Xe) performance. Default.
    * **INT8_SYM** — W8A16 weight-only compression. ~50% smaller than FP16.
    * **INT8_PTQ** — W8A8 post-training quantization (best CPU variant).
    * **ONNX** — Original ONNX exports (auto-converted to IR by OV runtime).
    """

    FP16 = "openvino-fp16"
    INT8_SYM = "openvino-int8_sym"
    INT8_PTQ = "openvino-int8_ptq_gpu"
    ONNX = "onnx"

    FP32 = "openvino-fp32"
    INT8_ASYM = "openvino-int8_asym"
    INT4_SYM = "openvino-int4_sym"
    INT4_ASYM = "openvino-int4_asym"


class EfficientSAM3OpenVINO(SAM3OpenVINO):
    """EfficientSAM3 model using OpenVINO runtime for inference.

    Supports **CLASSIC** and **VISUAL_EXEMPLAR** prompt modes. CANVAS mode is
    intentionally not supported (distilled student backbones are not trained
    on canvas inputs).

    Examples:
        >>> from instantlearn.models.efficient_sam3 import (
        ...     EfficientSAM3OpenVINO, EfficientSAM3OVVariant, Sam3PromptMode,
        ... )
        >>> from instantlearn.data.base.sample import Sample

        >>> # Auto-download default FP16 variant (classic text-only prompting)
        >>> model = EfficientSAM3OpenVINO(device="GPU")
        >>> model.fit(Sample(categories=["elephant"], category_ids=[0]))
        >>> results = model.predict(
        ...     Sample(
        ...         image_path="examples/assets/coco/000000286874.jpg",
        ...         categories=["elephant"],
        ...     ),
        ... )

        >>> # Local model directory
        >>> model = EfficientSAM3OpenVINO(
        ...     model_dir="./efficient-sam3-openvino/efficientvit_b1/openvino-fp16",
        ...     device="GPU",
        ... )

        >>> # Visual-exemplar prompting
        >>> import numpy as np
        >>> model = EfficientSAM3OpenVINO(
        ...     prompt_mode=Sam3PromptMode.VISUAL_EXEMPLAR, device="GPU",
        ... )
        >>> model.fit(Sample(
        ...     image_path="ref.jpg",
        ...     bboxes=np.array([[10, 10, 100, 100]]),
        ...     categories=["object"], category_ids=[0],
        ... ))
        >>> results = model.predict(Sample(image_path="target.jpg"))
    """

    def __init__(
        self,
        model_dir: str | Path | None = None,
        device: str = "AUTO",
        confidence_threshold: float = 0.5,
        resolution: int = 1008,
        prompt_mode: Sam3PromptMode = Sam3PromptMode.CLASSIC,
        tokenizer_path: str | Path | None = None,
        variant: EfficientSAM3OVVariant = EfficientSAM3OVVariant.FP16,
        repo_id: str = EFFICIENT_SAM3_OV_REPO,
        cache_dir: str | Path | None = None,
        decoder_device: str | None = None,
        gpu_inference_precision: str | None = None,
    ) -> None:
        """Initialise EfficientSAM3 OpenVINO model.

        Args:
            model_dir: Directory containing OpenVINO IR or ONNX sub-models.
                When ``None``, models are auto-downloaded from HuggingFace Hub.
            device: OpenVINO device (``"CPU"``, ``"GPU"``, ``"AUTO"``).
                PyTorch-style names (``"cuda"``, ``"cpu"``) are also accepted.
            confidence_threshold: Minimum confidence score for predictions.
            resolution: Input image resolution (must match exported model).
            prompt_mode: ``Sam3PromptMode.CLASSIC`` (default) or
                ``Sam3PromptMode.VISUAL_EXEMPLAR``. CANVAS is **not supported**.
            tokenizer_path: Explicit tokenizer path or HuggingFace model ID.
            variant: Model variant to download when *model_dir* is ``None``.
            repo_id: HuggingFace repository ID for auto-download.
            cache_dir: Directory for caching compiled OpenVINO models
                (see :class:`SAM3OpenVINO` for details).
            decoder_device: Optional override for the prompt-decoder device.
                Defaults to ``None`` (run on *device*). The decoder runs
                correctly on Intel GPU with ``INFERENCE_PRECISION_HINT=f16``;
                pass ``"CPU"`` only if you observe accuracy regressions on a
                specific driver/hardware combination (see
                ``agent-results/enable-operator/agent_report.md`` for the
                investigation of the underlying GPU plugin buffer-aliasing bug
                affecting the ``f32`` path).
            gpu_inference_precision: ``INFERENCE_PRECISION_HINT`` for GPU
                sub-models. Defaults to ``None`` (let the plugin choose, which
                is ``f16`` on Intel GPU). Pass ``"f32"`` when using the
                **EfficientViT-B1** backbone on Intel GPU to work around an
                FP16 overflow in linear-attention context modules that
                produces ``NaN`` features. RepViT-M1-1 does not need this
                override.

        Raises:
            ValueError: If ``prompt_mode`` is ``CANVAS``.
        """
        if prompt_mode == Sam3PromptMode.CANVAS:
            msg = (
                "EfficientSAM3OpenVINO does not support CANVAS prompt mode "
                "(distilled student backbones are not trained on canvas inputs). "
                "Use Sam3PromptMode.CLASSIC or Sam3PromptMode.VISUAL_EXEMPLAR."
            )
            raise ValueError(msg)

        # Map our variant enum onto the parent's enum machinery. The parent's
        # _resolve_model_dir() / __init__ both accept any object exposing a
        # `.value` matching the subdir name on HF, so the EfficientSAM3 enum
        # values plug in directly.
        super().__init__(
            model_dir=model_dir,
            device=device,
            confidence_threshold=confidence_threshold,
            resolution=resolution,
            prompt_mode=prompt_mode,
            drop_spatial_bias=True,  # No runtime effect; baked in at export time.
            tokenizer_path=tokenizer_path,
            variant=variant,  # type: ignore[arg-type]
            repo_id=repo_id,
            canvas_config=None,
            cache_dir=cache_dir,
        )

        # ------------------------------------------------------------------
        # Intel GPU configuration for EfficientSAM3
        # ------------------------------------------------------------------
        # MEAT/OMEGA investigation (see agent-results/enable-operator/) found
        # that running everything on GPU with the default ``f16`` precision
        # hint is both fast and numerically acceptable for the RepViT-M1-1
        # backbone (~91 ms end-to-end vs ~860 ms with the previous CPU-decoder
        # workaround). Two backbone-specific caveats remain, exposed as opt-in
        # overrides:
        #
        #   * EfficientViT-B1: the linear-attention ``Q @ context`` matmul in
        #     ``stages_2`` overflows f16 (values reach ~1.2e5, f16 max
        #     ~6.5e4) and produces NaN features. Pass
        #     ``gpu_inference_precision="f32"`` to force the slow but correct
        #     f32 path.
        #   * Decoder ``f32`` path: an Intel GPU plugin buffer-aliasing bug
        #     in ``prepare_buffer_fusing`` corrupts the unrolled DETR
        #     decoder's query state across layers, producing logit drift of
        #     ~0.4. The ``f16`` path is unaffected (50x more accurate) and
        #     is what we run by default. Pass ``decoder_device="CPU"`` to
        #     fall back to the previous workaround if needed.
        is_gpu_device = self.ov_device.startswith("GPU")

        # Encoders: only recompile with an explicit hint when the caller
        # asked for one (e.g. ``f32`` for EfficientViT). Otherwise leave the
        # default GPU FP16 path in place.
        if is_gpu_device and gpu_inference_precision is not None \
                and gpu_inference_precision != "f16":
            self._recompile_gpu_encoders_with_precision_hint(gpu_inference_precision)

        # Decoder: only override the device if the caller asked for it.
        if (
            decoder_device is not None
            and device_to_openvino_device(decoder_device) != self.ov_device
        ):
            decoder_ov_device = device_to_openvino_device(decoder_device)
            decoder_props = (
                {"PERFORMANCE_HINT": "LATENCY"} if decoder_ov_device != "CPU" else {}
            )
            decoder_path = _get_model_file(self.model_dir, _PROMPT_DECODER)
            logger.info(
                "Recompiling prompt-decoder on %s (user override).",
                decoder_ov_device,
            )
            self.decoder_model = self._ov_core.compile_model(
                decoder_path, decoder_ov_device, decoder_props
            )
            self._decoder_request = self.decoder_model.create_infer_request()
            self._decoder_ov_device = decoder_ov_device
        else:
            self._decoder_ov_device = self.ov_device
        logger.info(
            "EfficientSAM3 OpenVINO model loaded (mode=%s, vision_dev=%s, "
            "decoder_dev=%s, gpu_precision_hint=%s).",
            prompt_mode.value, self.ov_device, self._decoder_ov_device,
            gpu_inference_precision if is_gpu_device else "n/a",
        )

    def _recompile_gpu_encoders_with_precision_hint(self, hint: str) -> None:
        """Recompile vision/text/geometry sub-models on GPU with a precision hint.

        Args:
            hint: Value for the ``INFERENCE_PRECISION_HINT`` GPU property
                (e.g. ``"f32"`` or ``"f16"``).
        """
        props = {
            "PERFORMANCE_HINT": "LATENCY",
            "INFERENCE_PRECISION_HINT": hint,
        }
        core = self._ov_core
        logger.info(
            "Recompiling EfficientSAM3 GPU encoders with INFERENCE_PRECISION_HINT=%s "
            "(workaround for EfficientViT FP16 overflow on Intel GPU).",
            hint,
        )
        vision_path = _get_model_file(self.model_dir, _VISION_ENCODER)
        self.vision_model = core.compile_model(vision_path, self.ov_device, props)
        self._vision_request = self.vision_model.create_infer_request()

        text_path = _get_model_file(self.model_dir, _TEXT_ENCODER)
        self.text_model = core.compile_model(text_path, self.ov_device, props)
        self._text_request = self.text_model.create_infer_request()

        if self.geometry_model is not None:
            geo_path = _get_model_file(self.model_dir, _GEOMETRY_ENCODER, required=False)
            if geo_path is not None:
                self.geometry_model = core.compile_model(geo_path, self.ov_device, props)
                self._geometry_request = self.geometry_model.create_infer_request()

        if self.geometry_exemplar_model is not None:
            geo_ex_path = _get_model_file(self.model_dir, _GEOMETRY_ENCODER_EXEMPLAR)
            self.geometry_exemplar_model = core.compile_model(
                geo_ex_path, self.ov_device, props
            )
            self._geometry_exemplar_request = (
                self.geometry_exemplar_model.create_infer_request()
            )

    # ------------------------------------------------------------------ #
    # Overrides
    # ------------------------------------------------------------------ #

    def _load_tokenizer(self, tokenizer_path: str | Path | None) -> CLIPTokenizerFast:
        """Load the EfficientSAM3 CLIP tokenizer (jetjodh/sam3, pad_token_id=0).

        Args:
            tokenizer_path: Explicit path/repo, or ``None`` for auto-detection.

        Returns:
            Loaded ``CLIPTokenizerFast`` instance with ``pad_token_id=0``.
        """
        if tokenizer_path is not None:
            tokenizer = CLIPTokenizerFast.from_pretrained(str(tokenizer_path))
        elif (self.model_dir / "tokenizer.json").exists():
            tokenizer = CLIPTokenizerFast.from_pretrained(str(self.model_dir))
        else:
            tokenizer = CLIPTokenizerFast.from_pretrained(_TOKENIZER_REPO)
        tokenizer.pad_token_id = _TOKENIZER_PAD_ID
        return tokenizer

    @staticmethod
    def _resolve_model_dir(  # type: ignore[override]
        model_dir: str | Path | None,
        *,
        variant: EfficientSAM3OVVariant | SAM3OVVariant = EfficientSAM3OVVariant.FP16,
        repo_id: str = EFFICIENT_SAM3_OV_REPO,
    ) -> Path:
        """Resolve model directory, downloading from the EfficientSAM3 HF repo.

        Args:
            model_dir: Explicit local directory, or ``None`` for auto-download.
            variant: Model variant to download.
            repo_id: HuggingFace repository ID (defaults to the EfficientSAM3 repo).

        Returns:
            Local ``Path`` to the directory containing the sub-model files.

        Raises:
            FileNotFoundError: If *model_dir* is given but does not exist.
            ImportError: If ``huggingface_hub`` is not installed.
        """
        if model_dir is not None:
            path = Path(model_dir)
            if not path.is_dir():
                msg = f"Model directory not found: {path}"
                raise FileNotFoundError(msg)
            return path

        # Normalise to our enum (accepts string values too).
        variant = EfficientSAM3OVVariant(variant.value if isinstance(variant, Enum) else variant)
        subdir = variant.value

        try:
            from huggingface_hub import snapshot_download  # noqa: PLC0415
        except ImportError:
            msg = "huggingface_hub is required for auto-download. Install it with: uv pip install huggingface-hub"
            raise ImportError(msg)  # noqa: B904

        logger.info("Downloading EfficientSAM3 '%s' from HuggingFace: %s", variant.name, repo_id)
        cache_dir = snapshot_download(
            repo_id=repo_id,
            allow_patterns=[f"{subdir}/*", "tokenizer*", "special_tokens_map*", "vocab*", "merges*"],
        )
        return Path(cache_dir) / subdir

    def _run_prompt_decoder(
        self,
        vision_features: dict[str, np.ndarray],
        prompt_features: np.ndarray,
        prompt_mask: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Run the EfficientSAM3 prompt decoder.

        The exported decoder takes two extra inputs (``text_features`` and
        ``text_mask``) compared to the SAM3 decoder, because
        ``_get_scoring_features`` is overridden to score against raw text
        features (not the DETR-encoder-fused ones).

        The text portion is always the first :data:`STUDENT_CONTEXT_LENGTH`
        tokens of ``prompt_features`` / ``prompt_mask`` because all callers in
        the parent class concatenate as ``[text, geometry]``.

        Args:
            vision_features: FPN features from the vision encoder.
            prompt_features: ``[1, T, 256]`` concat of text + geometry features.
            prompt_mask: ``[1, T]`` bool concat of text + geometry masks.

        Returns:
            Dict with ``pred_masks``, ``pred_boxes``, ``pred_logits``,
            ``presence_logits`` (the last is a zero placeholder).
        """
        text_features = prompt_features[:, :STUDENT_CONTEXT_LENGTH, :]
        text_mask = prompt_mask[:, :STUDENT_CONTEXT_LENGTH]
        self._decoder_request.infer([
            vision_features["fpn_feat_0"],
            vision_features["fpn_feat_1"],
            vision_features["fpn_feat_2"],
            vision_features["fpn_pos_2"],
            prompt_features,
            prompt_mask,
            text_features,
            text_mask,
        ])
        pred_logits = np.array(self._decoder_request.get_tensor("pred_logits").data)
        # Match PyTorch EfficientSAM3 (``use_presence=False``): presence is not
        # used for scoring. The exported decoder emits a zero placeholder so the
        # ONNX graph has a fixed signature; we replace it with a large positive
        # constant so ``presence.sigmoid() ≈ 1.0`` is a no-op multiplier in
        # :class:`Sam3Postprocessor`.
        presence_noop = np.full((pred_logits.shape[0], 1), 10.0, dtype=np.float32)
        return {
            "pred_masks": np.array(self._decoder_request.get_tensor("pred_masks").data),
            "pred_boxes": np.array(self._decoder_request.get_tensor("pred_boxes").data),
            "pred_logits": pred_logits,
            "presence_logits": presence_noop,
        }
