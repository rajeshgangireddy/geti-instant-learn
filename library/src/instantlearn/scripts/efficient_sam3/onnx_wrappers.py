# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""ONNX-exportable ``nn.Module`` wrappers for EfficientSAM3 sub-components.

EfficientSAM3 inherits the full :class:`Sam3Model` pipeline (DETR encoder /
decoder, mask decoder, geometry encoder, dot-product scoring) and only swaps:

* ``vision_encoder`` → :class:`StudentVisionModel` (timm backbone + FPN).
  Returns the same ``fpn_hidden_states`` / ``fpn_position_encoding`` dict
  layout as SAM3's ``VisionModel``, so :class:`OnnxVisionEncoder` from
  :mod:`instantlearn.scripts.sam3.onnx_wrappers` is reused unchanged.
* ``text_encoder`` → :class:`MobileCLIPTextEncoder`. Returns a
  ``(all_tokens, input_embeds)`` tuple instead of a HuggingFace ``BaseModelOutput``,
  so a dedicated :class:`OnnxEfficientTextEncoder` wrapper is needed.
* ``_get_scoring_features`` → text-only scoring (geometry features are
  ignored for the dot-product scoring branch). This requires a dedicated
  :class:`OnnxEfficientPromptDecoder` wrapper that accepts the *raw* text
  features as an extra input (the DETR-encoded fused features that SAM3
  uses for scoring would dilute the text signal for the distilled student).

:class:`OnnxGeometryEncoder` is also reused from SAM3 unchanged.

See Also:
    :mod:`instantlearn.scripts.efficient_sam3.export_efficient_sam3` — CLI
    pipeline that uses these wrappers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

# Re-export the SAM3 wrappers that EfficientSAM3 reuses unchanged so callers
# can import everything from a single module.
from instantlearn.scripts.sam3.onnx_wrappers import (
    OnnxGeometryEncoder,
    OnnxVisionEncoder,
)

if TYPE_CHECKING:
    from instantlearn.models.efficient_sam3.model import EfficientSam3Model

__all__ = [
    "OnnxEfficientPromptDecoder",
    "OnnxEfficientTextEncoder",
    "OnnxGeometryEncoder",
    "OnnxVisionEncoder",
]


class OnnxEfficientTextEncoder(nn.Module):
    """ONNX-exportable wrapper around MobileCLIP text encoder + projection.

    MobileCLIP's ``forward()`` returns ``(all_tokens, input_embeds)`` instead
    of a HuggingFace ``BaseModelOutput``. ``attention_mask`` is accepted for
    API parity with SAM3 but is intentionally ignored (the distilled student
    was trained without key padding masks; passing one degrades detection
    quality — see :class:`MobileCLIPTextEncoder` for details).

    Inputs:
        input_ids:      ``[B, 32]`` int64
        attention_mask: ``[B, 32]`` int64 (forwarded for API parity; unused)

    Outputs (named):
        text_features: ``[B, 32, 256]`` float32 (projected token embeddings)
        text_mask:     ``[B, 32]`` bool (derived from attention_mask)
    """

    def __init__(self, model: EfficientSam3Model) -> None:
        """Store references to the MobileCLIP text encoder and projection."""
        super().__init__()
        self.text_encoder = model.text_encoder
        self.text_projection = model.text_projection

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode text tokens into projected features.

        Args:
            input_ids: Token IDs ``[B, seq_len]``.
            attention_mask: Attention mask ``[B, seq_len]`` (forwarded but
                ignored by the underlying MobileCLIP encoder).

        Returns:
            Tuple of ``(text_features, text_mask)``.
        """
        all_tokens, _ = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        text_features = self.text_projection(all_tokens)
        text_mask = attention_mask.bool()
        return text_features, text_mask


class OnnxEfficientPromptDecoder(nn.Module):
    """ONNX-exportable DETR pipeline + mask decoder for EfficientSAM3.

    Mirrors :class:`OnnxPromptDecoder` from :mod:`instantlearn.scripts.sam3.onnx_wrappers`
    but routes the *raw* text features (pre-DETR-encoder) into
    :class:`DotProductScoring` to match EfficientSAM3's overridden
    ``_get_scoring_features()``. SAM3 uses the DETR-encoder text output
    (mixed with geometry); the distilled student was trained with text-only
    scoring and degrades when geometry features are mixed in.

    Inputs:
        fpn_feat_0: ``[B, 256, 288, 288]``
        fpn_feat_1: ``[B, 256, 144, 144]``
        fpn_feat_2: ``[B, 256,  72,  72]``
        fpn_pos_2:  ``[B, 256,  72,  72]``
        prompt_features:  ``[B, T, 256]``  (text ± geometry, fed to DETR encoder/decoder)
        prompt_mask:      ``[B, T]``       bool
        text_features:    ``[B, 32, 256]`` raw text features used only for scoring
        text_mask:        ``[B, 32]``      bool — mask for scoring text features

    Outputs (named):
        pred_masks:       ``[B, 200, H_mask, W_mask]``
        pred_boxes:       ``[B, 200, 4]`` xyxy normalised
        pred_logits:      ``[B, 200]``
        presence_logits:  ``[B, 1]``
    """

    def __init__(self, model: EfficientSam3Model) -> None:
        """Store references to DETR encoder, decoder, mask decoder, and scoring."""
        super().__init__()
        self.detr_encoder = model.detr_encoder
        self.detr_decoder = model.detr_decoder
        self.mask_decoder = model.mask_decoder
        self.dot_product_scoring = model.dot_product_scoring

    def forward(
        self,
        fpn_feat_0: torch.Tensor,
        fpn_feat_1: torch.Tensor,
        fpn_feat_2: torch.Tensor,
        fpn_pos_2: torch.Tensor,
        prompt_features: torch.Tensor,
        prompt_mask: torch.Tensor,
        text_features: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run DETR pipeline and mask decoder with text-only scoring.

        Args:
            fpn_feat_0: FPN features at 4x scale.
            fpn_feat_1: FPN features at 2x scale.
            fpn_feat_2: FPN features at 1x scale.
            fpn_pos_2: Position encoding at 1x scale.
            prompt_features: Pre-concatenated prompt features fed to DETR encoder
                / decoder ``[B, T, 256]``. May be text-only (classic) or
                text + geometry (visual exemplar).
            prompt_mask: Prompt attention mask ``[B, T]``.
            text_features: Raw text features used only for dot-product scoring
                ``[B, 32, 256]``.
            text_mask: Mask for ``text_features`` ``[B, 32]``.

        Returns:
            Tuple of ``(pred_masks, pred_boxes, pred_logits, presence_logits)``.
        """
        from instantlearn.models.sam3.common import (  # noqa: PLC0415
            box_cxcywh_to_xyxy,
            inverse_sigmoid,
        )

        # DETR encoder -- uses only the 1x scale as the vision "level"
        encoder_outputs = self.detr_encoder(
            vision_features=[fpn_feat_2],
            text_features=prompt_features,
            vision_pos_embeds=[fpn_pos_2],
            text_mask=prompt_mask,
        )

        # DETR decoder
        decoder_outputs = self.detr_decoder(
            vision_features=encoder_outputs["last_hidden_state"],
            text_features=encoder_outputs["text_features"],
            vision_pos_encoding=encoder_outputs["pos_embeds_flattened"],
            text_mask=prompt_mask,
            spatial_shapes=encoder_outputs["spatial_shapes"],
        )

        # Box refinement (last layer)
        all_box_offsets = self.detr_decoder.box_head(
            decoder_outputs["intermediate_hidden_states"],
        )
        reference_boxes_inv_sig = inverse_sigmoid(decoder_outputs["reference_boxes"])
        all_pred_boxes_cxcywh = (reference_boxes_inv_sig + all_box_offsets).sigmoid()
        all_pred_boxes = box_cxcywh_to_xyxy(all_pred_boxes_cxcywh)

        # Dot-product scoring uses the RAW text features (not the DETR
        # encoder fused features) — matches EfficientSam3Model._get_scoring_features.
        all_pred_logits = self.dot_product_scoring(
            decoder_hidden_states=decoder_outputs["intermediate_hidden_states"],
            text_features=text_features,
            text_mask=text_mask,
        ).squeeze(-1)

        pred_logits = all_pred_logits[-1]
        pred_boxes = all_pred_boxes[-1]
        decoder_hidden_states = decoder_outputs["intermediate_hidden_states"][-1]

        # EfficientSAM3 disables the presence_token mechanism (use_presence=False)
        # because the distilled student produces unreliable presence scores.
        # Emit a zero placeholder so the ONNX graph keeps a fixed signature and
        # downstream scoring code (which weights by presence) effectively ignores
        # it after sigmoid (sigmoid(0) = 0.5 -> neutral). The runtime detects
        # this case and skips the presence multiplication.
        presence_logits_stack = decoder_outputs["presence_logits"]
        if presence_logits_stack is not None:
            presence_logits = presence_logits_stack[-1]
        else:
            presence_logits = torch.zeros(
                pred_logits.shape[0], 1, dtype=pred_logits.dtype, device=pred_logits.device,
            )

        # Mask decoder
        fpn_hidden_states = [fpn_feat_0, fpn_feat_1, fpn_feat_2]
        mask_outputs = self.mask_decoder(
            decoder_queries=decoder_hidden_states,
            backbone_features=fpn_hidden_states,
            encoder_hidden_states=encoder_outputs["last_hidden_state"],
            prompt_features=prompt_features,
            prompt_mask=prompt_mask,
        )

        return (
            mask_outputs["pred_masks"],
            pred_boxes,
            pred_logits,
            presence_logits,
        )
