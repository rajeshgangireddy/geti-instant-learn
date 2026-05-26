# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Export EfficientSAM3 PyTorch model to OpenVINO IR via ONNX.

Loads an :class:`EfficientSam3Model` (with a chosen timm vision backbone and
MobileCLIP-S1 text encoder) from HuggingFace or a local checkpoint, exports
each sub-component to ONNX, converts to OpenVINO IR, and optionally applies
NNCF weight compression (INT8/INT4).

Produces 5 OpenVINO IR models (same 5-model split as SAM3):

* ``vision-encoder``    — timm backbone + projection head + FPN
* ``text-encoder``      — MobileCLIP-S1 text encoder + projection
* ``geometry-encoder``  — Geometry encoder (classic, ``drop_spatial_bias=False``)
* ``geometry-encoder-exemplar`` — Geometry encoder (exemplar, ``drop_spatial_bias=True``)
* ``prompt-decoder``    — DETR encoder/decoder + box head + text-only scoring + mask decoder

Output layout:

    output-dir/
      onnx/
        vision-encoder.onnx
        text-encoder.onnx
        geometry-encoder.onnx
        geometry-encoder-exemplar.onnx
        prompt-decoder.onnx
        tokenizer.json, tokenizer_config.json, ...
      openvino-fp16/
        ... (.xml/.bin pairs + tokenizer)

Usage:
    python export_efficient_sam3.py --output-dir ./efficient-sam3-openvino/efficientvit_b1

    python export_efficient_sam3.py --backbone-type efficientvit --variant b1 \\
        --output-dir ./efficient-sam3-openvino/efficientvit_b1 --validate

    python export_efficient_sam3.py --quantize --compression-modes int8_sym \\
        --source-dir ./efficient-sam3-openvino/efficientvit_b1/openvino-fp16 \\
        --output-dir ./efficient-sam3-openvino/efficientvit_b1
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Canonical sub-model names (5-model split, identical to SAM3)
MODEL_NAMES = [
    "vision-encoder",
    "text-encoder",
    "geometry-encoder",
    "geometry-encoder-exemplar",
    "prompt-decoder",
]

_VISION_ENCODER = "vision-encoder"
_TEXT_ENCODER = "text-encoder"
_GEOMETRY_ENCODER = "geometry-encoder"
_GEOMETRY_ENCODER_EXEMPLAR = "geometry-encoder-exemplar"
_PROMPT_DECODER = "prompt-decoder"

# CLIP-style tokenizer files needed for inference (EfficientSAM3 reuses
# the SAM3 CLIP BPE vocab via the "jetjodh/sam3" tokenizer repository).
_TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
]
_TOKENIZER_REPO = "jetjodh/sam3"

# Context length used by the EfficientSAM3 student (matches SAM3).
_STUDENT_CONTEXT_LENGTH = 32


# ONNX export

def export_to_onnx(  # noqa: PLR0915
    backbone_type: str,
    variant: str,
    output_dir: Path,
    *,
    pretrained_model_name_or_path: str | None = None,
    ft: bool = False,
    resolution: int = 1008,
    opset_version: int = 17,
) -> tuple[Path, dict[str, Path]]:
    """Load ``EfficientSam3Model`` and export all sub-components to ONNX.

    Args:
        backbone_type: timm backbone family ('efficientvit', 'repvit', 'tinyvit').
        variant: Backbone size variant (e.g. 'b1', 'm1_1', '11m').
        output_dir: Base output directory. ONNX files go into ``output_dir/onnx``.
        pretrained_model_name_or_path: Optional HF repo ID or local ``.pth``.
            ``None`` uses the default ``Simon7108528/EfficientSAM3`` repo.
        ft: Load the fine-tuned checkpoint variant.
        resolution: Input image resolution.
        opset_version: ONNX opset version.

    Returns:
        Tuple of ``(onnx_dir, exported)`` — path to the ONNX directory and
        a mapping from model name to ONNX file path.
    """
    import torch  # noqa: PLC0415
    from transformers import CLIPTokenizerFast  # noqa: PLC0415

    from instantlearn.models.efficient_sam3.model import EfficientSam3Model  # noqa: PLC0415
    from instantlearn.scripts.efficient_sam3.onnx_wrappers import (  # noqa: PLC0415
        OnnxEfficientPromptDecoder,
        OnnxEfficientTextEncoder,
        OnnxGeometryEncoder,
        OnnxVisionEncoder,
    )

    logger.info(
        "Loading EfficientSam3Model: backbone=%s variant=%s ft=%s ...",
        backbone_type, variant, ft,
    )
    model = EfficientSam3Model.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        backbone_type=backbone_type,
        variant=variant,
        device="cpu",
        torch_dtype=torch.float32,
        ft=ft,
    )
    model.eval()
    logger.info("Model loaded successfully.")

    onnx_dir = output_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device

    exported: dict[str, Path] = {}

    # 1. Vision encoder (reuses SAM3 wrapper — StudentVisionModel returns
    #    the same fpn_hidden_states / fpn_position_encoding dict layout).
    logger.info("Exporting vision encoder (timm %s/%s)...", backbone_type, variant)
    vision_wrapper = OnnxVisionEncoder(model)
    vision_wrapper.eval()
    dummy_pixel = torch.randn(1, 3, resolution, resolution, device=device)
    vision_path = onnx_dir / f"{_VISION_ENCODER}.onnx"
    torch.onnx.export(
        vision_wrapper,
        (dummy_pixel,),
        str(vision_path),
        opset_version=opset_version,
        dynamo=False,
        input_names=["pixel_values"],
        output_names=["fpn_feat_0", "fpn_feat_1", "fpn_feat_2", "fpn_pos_2"],
        dynamic_axes={"pixel_values": {0: "batch"}},
    )
    exported[_VISION_ENCODER] = vision_path
    logger.info("  -> %s", vision_path)

    # 2. Text encoder (MobileCLIP — tuple return adapted in wrapper).
    logger.info("Exporting text encoder (MobileCLIP-S1)...")
    text_wrapper = OnnxEfficientTextEncoder(model)
    text_wrapper.eval()
    dummy_ids = torch.ones(1, _STUDENT_CONTEXT_LENGTH, dtype=torch.long, device=device)
    dummy_mask = torch.ones(1, _STUDENT_CONTEXT_LENGTH, dtype=torch.long, device=device)
    text_path = onnx_dir / f"{_TEXT_ENCODER}.onnx"
    torch.onnx.export(
        text_wrapper,
        (dummy_ids, dummy_mask),
        str(text_path),
        opset_version=opset_version,
        dynamo=False,
        input_names=["input_ids", "attention_mask"],
        output_names=["text_features", "text_mask"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "attention_mask": {0: "batch"},
        },
    )
    exported[_TEXT_ENCODER] = text_path
    logger.info("  -> %s", text_path)

    # FPN feature size at 1x scale. SAM3's ViT downsamples by 14x; EfficientSAM3's
    # StudentVisionModel uses ImageProjectionHead that bilinearly resamples the
    # backbone output to IMAGE_ENCODER_EMBED_SIZE (=72 for 1008px input), so
    # the 1x FPN level is always 72x72 regardless of backbone.
    feat_size = resolution // 14

    # 3. Geometry encoder (classic) — reuses SAM3 wrapper.
    logger.info("Exporting geometry encoder (classic)...")
    geo_wrapper = OnnxGeometryEncoder(model, drop_spatial_bias=False)
    geo_wrapper.eval()
    dummy_fpn = torch.randn(1, 256, feat_size, feat_size, device=device)
    dummy_pos = torch.randn(1, 256, feat_size, feat_size, device=device)
    dummy_boxes = torch.rand(1, 1, 4, device=device)
    dummy_box_labels = torch.ones(1, 1, dtype=torch.long, device=device)
    dummy_points = torch.rand(1, 1, 2, device=device)
    dummy_point_labels = torch.full((1, 1), -10, dtype=torch.long, device=device)

    geo_path = onnx_dir / f"{_GEOMETRY_ENCODER}.onnx"
    torch.onnx.export(
        geo_wrapper,
        (dummy_fpn, dummy_pos, dummy_boxes, dummy_box_labels, dummy_points, dummy_point_labels),
        str(geo_path),
        opset_version=opset_version,
        dynamo=False,
        input_names=[
            "fpn_feat_2", "fpn_pos_2",
            "input_boxes", "input_boxes_labels",
            "input_points", "input_points_labels",
        ],
        output_names=["geometry_features", "geometry_mask"],
        dynamic_axes={
            "input_boxes": {0: "batch", 1: "num_boxes"},
            "input_boxes_labels": {0: "batch", 1: "num_boxes"},
            "input_points": {0: "batch", 1: "num_points"},
            "input_points_labels": {0: "batch", 1: "num_points"},
        },
    )
    exported[_GEOMETRY_ENCODER] = geo_path
    logger.info("  -> %s", geo_path)

    # 4. Geometry encoder (exemplar, drop_spatial_bias=True) — reuses SAM3 wrapper.
    logger.info("Exporting geometry encoder (exemplar)...")
    geo_exemplar_wrapper = OnnxGeometryEncoder(model, drop_spatial_bias=True)
    geo_exemplar_wrapper.eval()
    dummy_boxes_ignore = torch.zeros(1, 1, 4, device=device)
    dummy_box_labels_ignore = torch.full((1, 1), -10, dtype=torch.long, device=device)
    dummy_ex_points = torch.rand(1, 1, 2, device=device)
    dummy_ex_point_labels = torch.ones(1, 1, dtype=torch.long, device=device)

    geo_exemplar_path = onnx_dir / f"{_GEOMETRY_ENCODER_EXEMPLAR}.onnx"
    torch.onnx.export(
        geo_exemplar_wrapper,
        (dummy_fpn, dummy_pos, dummy_boxes_ignore, dummy_box_labels_ignore,
         dummy_ex_points, dummy_ex_point_labels),
        str(geo_exemplar_path),
        opset_version=opset_version,
        dynamo=False,
        input_names=[
            "fpn_feat_2", "fpn_pos_2",
            "input_boxes", "input_boxes_labels",
            "input_points", "input_points_labels",
        ],
        output_names=["geometry_features", "geometry_mask"],
        dynamic_axes={
            "input_boxes": {0: "batch", 1: "num_boxes"},
            "input_boxes_labels": {0: "batch", 1: "num_boxes"},
            "input_points": {0: "batch", 1: "num_points"},
            "input_points_labels": {0: "batch", 1: "num_points"},
        },
    )
    exported[_GEOMETRY_ENCODER_EXEMPLAR] = geo_exemplar_path
    logger.info("  -> %s", geo_exemplar_path)

    # 5. Prompt decoder (EfficientSAM3 wrapper — text-only scoring branch).
    logger.info("Exporting prompt decoder (text-only scoring)...")
    decoder_wrapper = OnnxEfficientPromptDecoder(model)
    decoder_wrapper.eval()
    feat_4x = feat_size * 4
    feat_2x = feat_size * 2
    dummy_f0 = torch.randn(1, 256, feat_4x, feat_4x, device=device)
    dummy_f1 = torch.randn(1, 256, feat_2x, feat_2x, device=device)
    dummy_f2 = torch.randn(1, 256, feat_size, feat_size, device=device)
    dummy_p2 = torch.randn(1, 256, feat_size, feat_size, device=device)
    dummy_prompt = torch.randn(1, _STUDENT_CONTEXT_LENGTH, 256, device=device)
    dummy_pmask = torch.ones(1, _STUDENT_CONTEXT_LENGTH, dtype=torch.bool, device=device)
    dummy_text_feat = torch.randn(1, _STUDENT_CONTEXT_LENGTH, 256, device=device)
    dummy_text_mask = torch.ones(1, _STUDENT_CONTEXT_LENGTH, dtype=torch.bool, device=device)

    decoder_path = onnx_dir / f"{_PROMPT_DECODER}.onnx"
    torch.onnx.export(
        decoder_wrapper,
        (dummy_f0, dummy_f1, dummy_f2, dummy_p2, dummy_prompt, dummy_pmask,
         dummy_text_feat, dummy_text_mask),
        str(decoder_path),
        opset_version=opset_version,
        dynamo=False,
        input_names=[
            "fpn_feat_0", "fpn_feat_1", "fpn_feat_2", "fpn_pos_2",
            "prompt_features", "prompt_mask",
            "text_features", "text_mask",
        ],
        output_names=["pred_masks", "pred_boxes", "pred_logits", "presence_logits"],
        dynamic_axes={
            "prompt_features": {0: "batch", 1: "prompt_len"},
            "prompt_mask": {0: "batch", 1: "prompt_len"},
            "text_features": {0: "batch", 1: "text_len"},
            "text_mask": {0: "batch", 1: "text_len"},
        },
    )
    exported[_PROMPT_DECODER] = decoder_path
    logger.info("  -> %s", decoder_path)

    # Save tokenizer alongside the ONNX files (CLIP BPE vocab).
    logger.info("Saving tokenizer (%s)...", _TOKENIZER_REPO)
    tokenizer = CLIPTokenizerFast.from_pretrained(_TOKENIZER_REPO)
    tokenizer.pad_token_id = 0
    tokenizer.save_pretrained(str(onnx_dir))
    logger.info("  Tokenizer saved to %s", onnx_dir)

    logger.info("ONNX export complete. %d models written to %s", len(exported), onnx_dir)
    return onnx_dir, exported


# ONNX → OpenVINO IR conversion

def convert_to_openvino(
    onnx_dir: Path,
    output_dir: Path,
    *,
    precision: str = "fp16",
) -> dict[str, Path]:
    """Convert all EfficientSAM3 ONNX models to OpenVINO IR and copy tokenizer.

    Args:
        onnx_dir: Directory containing ONNX files from :func:`export_to_onnx`.
        output_dir: Directory to write OpenVINO IR files.
        precision: ``"fp32"`` or ``"fp16"``.

    Returns:
        Mapping from model name to ``.xml`` path.
    """
    import openvino as ov  # noqa: PLC0415

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    compress_to_fp16 = precision == "fp16"

    converted: dict[str, Path] = {}
    for name in MODEL_NAMES:
        onnx_path = onnx_dir / f"{name}.onnx"
        if not onnx_path.exists():
            logger.warning("Skipping %s — ONNX file not found.", name)
            continue

        logger.info("Converting %s to OpenVINO IR...", name)
        ov_model = ov.convert_model(str(onnx_path))
        ir_path = output_dir / f"{name}.xml"
        ov.save_model(ov_model, str(ir_path), compress_to_fp16=compress_to_fp16)
        converted[name] = ir_path
        logger.info("  -> %s", ir_path)

    # Copy tokenizer files from source
    for filename in _TOKENIZER_FILES:
        src = onnx_dir / filename
        dst = output_dir / filename
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            logger.info("Copied tokenizer file: %s", filename)

    logger.info("OpenVINO conversion complete. %d models written to %s", len(converted), output_dir)
    return converted


# Validation

def validate_openvino_models(  # noqa: PLR0915
    model_dir: Path,
    device: str = "CPU",
    resolution: int = 1008,
) -> None:
    """Validate exported EfficientSAM3 OpenVINO models with dummy inference.

    Args:
        model_dir: Directory containing OpenVINO IR files.
        device: OpenVINO device for validation.
        resolution: Model input resolution.
    """
    import openvino as ov  # noqa: PLC0415

    core = ov.Core()
    rng = np.random.default_rng(42)
    feat_size = resolution // 14

    logger.info("Validating EfficientSAM3 OpenVINO models in %s...", model_dir)

    # 1. Vision encoder
    vision_xml = model_dir / f"{_VISION_ENCODER}.xml"
    if vision_xml.exists():
        vision_model = core.compile_model(vision_xml, device)
        dummy_img = rng.standard_normal((1, 3, resolution, resolution)).astype(np.float32)
        vision_result = vision_model([dummy_img])
        logger.info(
            "  Vision encoder: OK — %s",
            {k: vision_result[k].shape for k in ["fpn_feat_0", "fpn_feat_1", "fpn_feat_2", "fpn_pos_2"]},
        )
    else:
        logger.warning("  Vision encoder: MISSING (%s)", vision_xml)

    # 2. Text encoder
    text_xml = model_dir / f"{_TEXT_ENCODER}.xml"
    if text_xml.exists():
        text_model = core.compile_model(text_xml, device)
        dummy_ids = np.ones((1, _STUDENT_CONTEXT_LENGTH), dtype=np.int64)
        dummy_mask = np.ones((1, _STUDENT_CONTEXT_LENGTH), dtype=np.int64)
        text_result = text_model([dummy_ids, dummy_mask])
        logger.info(
            "  Text encoder: OK — %s",
            {k: text_result[k].shape for k in ["text_features", "text_mask"]},
        )
    else:
        logger.warning("  Text encoder: MISSING (%s)", text_xml)

    # 3. Geometry encoder (classic)
    geo_xml = model_dir / f"{_GEOMETRY_ENCODER}.xml"
    if geo_xml.exists():
        geo_model = core.compile_model(geo_xml, device)
        dummy_fpn = rng.standard_normal((1, 256, feat_size, feat_size)).astype(np.float32)
        dummy_pos = rng.standard_normal((1, 256, feat_size, feat_size)).astype(np.float32)
        dummy_boxes = rng.random((1, 1, 4)).astype(np.float32)
        dummy_box_labels = np.ones((1, 1), dtype=np.int64)
        dummy_points = np.zeros((1, 1, 2), dtype=np.float32)
        dummy_point_labels = np.full((1, 1), -10, dtype=np.int64)
        geo_result = geo_model([dummy_fpn, dummy_pos, dummy_boxes, dummy_box_labels, dummy_points, dummy_point_labels])
        logger.info(
            "  Geometry encoder (classic): OK — %s",
            {k: geo_result[k].shape for k in ["geometry_features", "geometry_mask"]},
        )
    else:
        logger.warning("  Geometry encoder (classic): MISSING (%s)", geo_xml)

    # 4. Geometry encoder (exemplar)
    geo_ex_xml = model_dir / f"{_GEOMETRY_ENCODER_EXEMPLAR}.xml"
    if geo_ex_xml.exists():
        geo_ex_model = core.compile_model(geo_ex_xml, device)
        dummy_boxes_ign = np.zeros((1, 1, 4), dtype=np.float32)
        dummy_box_labels_ign = np.full((1, 1), -10, dtype=np.int64)
        dummy_pts = rng.random((1, 1, 2)).astype(np.float32)
        dummy_pt_labels = np.ones((1, 1), dtype=np.int64)
        dummy_fpn_ex = rng.standard_normal((1, 256, feat_size, feat_size)).astype(np.float32)
        dummy_pos_ex = rng.standard_normal((1, 256, feat_size, feat_size)).astype(np.float32)
        geo_ex_result = geo_ex_model([
            dummy_fpn_ex,
            dummy_pos_ex,
            dummy_boxes_ign,
            dummy_box_labels_ign,
            dummy_pts,
            dummy_pt_labels,
        ])
        logger.info(
            "  Geometry encoder (exemplar): OK — %s",
            {k: geo_ex_result[k].shape for k in ["geometry_features", "geometry_mask"]},
        )
    else:
        logger.warning("  Geometry encoder (exemplar): MISSING (%s)", geo_ex_xml)

    # 5. Prompt decoder
    dec_xml = model_dir / f"{_PROMPT_DECODER}.xml"
    if dec_xml.exists():
        dec_model = core.compile_model(dec_xml, device)
        dummy_f0 = rng.standard_normal((1, 256, feat_size * 4, feat_size * 4)).astype(np.float32)
        dummy_f1 = rng.standard_normal((1, 256, feat_size * 2, feat_size * 2)).astype(np.float32)
        dummy_f2 = rng.standard_normal((1, 256, feat_size, feat_size)).astype(np.float32)
        dummy_p2 = rng.standard_normal((1, 256, feat_size, feat_size)).astype(np.float32)
        dummy_prompt = rng.standard_normal((1, _STUDENT_CONTEXT_LENGTH, 256)).astype(np.float32)
        dummy_pmask = np.ones((1, _STUDENT_CONTEXT_LENGTH), dtype=bool)
        dummy_txt = rng.standard_normal((1, _STUDENT_CONTEXT_LENGTH, 256)).astype(np.float32)
        dummy_txt_mask = np.ones((1, _STUDENT_CONTEXT_LENGTH), dtype=bool)
        dec_result = dec_model([
            dummy_f0, dummy_f1, dummy_f2, dummy_p2,
            dummy_prompt, dummy_pmask,
            dummy_txt, dummy_txt_mask,
        ])
        logger.info(
            "  Prompt decoder: OK — %s",
            {k: dec_result[k].shape for k in ["pred_masks", "pred_boxes", "pred_logits", "presence_logits"]},
        )
    else:
        logger.warning("  Prompt decoder: MISSING (%s)", dec_xml)

    logger.info("Validation complete!")


# Weight compression (quantization)

def apply_weight_compression(
    source_dir: Path,
    output_dir: Path,
    mode: str = "int8_sym",
) -> Path:
    """Apply NNCF weight compression to EfficientSAM3 OpenVINO IR models.

    Args:
        source_dir: Directory containing FP16 OpenVINO IR models.
        output_dir: Base output directory.
        mode: Compression mode string (e.g. ``"int8_sym"``, ``"int4_sym"``).

    Returns:
        Path to directory containing compressed OpenVINO IR models.
    """
    import openvino as ov  # noqa: PLC0415

    from instantlearn.utils.compression import compress_model  # noqa: PLC0415
    from instantlearn.utils.constants import CompressionMode  # noqa: PLC0415

    compression_mode = CompressionMode(mode)

    logger.info("=" * 60)
    logger.info("Applying %s weight compression", mode.upper())
    logger.info("=" * 60)

    ir_dir = output_dir / f"openvino-{mode}"
    ir_dir.mkdir(parents=True, exist_ok=True)

    core = ov.Core()
    logger.info("Compressing %d models", len(MODEL_NAMES))

    for model_name in MODEL_NAMES:
        xml_path = source_dir / f"{model_name}.xml"
        if not xml_path.exists():
            logger.warning("Source model not found: %s", xml_path)
            continue

        logger.info("Compressing %s with %s...", model_name, mode.upper())

        ov_model = core.read_model(xml_path)

        try:
            compressed_model = compress_model(ov_model, mode=compression_mode, group_size=-1)
        except Exception:
            logger.exception("Failed to compress %s with %s", model_name, mode)
            continue

        out_xml = ir_dir / f"{model_name}.xml"
        ov.save_model(compressed_model, out_xml)

        bin_path = ir_dir / f"{model_name}.bin"
        size_mb = bin_path.stat().st_size / (1024 * 1024)
        logger.info("Saved: %s (%.1f MB)", out_xml, size_mb)

    # Copy tokenizer files
    for filename in _TOKENIZER_FILES:
        src = source_dir / filename
        dst = ir_dir / filename
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    return ir_dir


# CLI

def main() -> None:
    """CLI entry point for EfficientSAM3 PyTorch → ONNX → OpenVINO export."""
    parser = argparse.ArgumentParser(
        description="Export EfficientSAM3 PyTorch model to OpenVINO IR via ONNX.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--backbone-type",
        type=str,
        default="efficientvit",
        choices=["efficientvit", "repvit", "tinyvit"],
        help="timm backbone family. Default: efficientvit (MVP target).",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="b1",
        help="Backbone size variant (e.g. b1, m1_1, 11m). Default: b1 (MVP target).",
    )
    parser.add_argument(
        "--ft",
        action="store_true",
        help="Load fine-tuned checkpoint (only efficientvit/b1, repvit/m1_1, tinyvit/11m). "
             "Default: stage-1 distilled checkpoint (empirically better on classic mode).",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="HuggingFace repo ID or local path. Default: Simon7108528/EfficientSAM3.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./efficient-sam3-openvino"),
        help="Output directory. Default: ./efficient-sam3-openvino.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "fp16"],
        default="fp16",
        help="Target OpenVINO IR precision. Default: fp16.",
    )
    parser.add_argument(
        "--onnx-dir",
        type=Path,
        default=None,
        help="Existing ONNX directory (skip PyTorch export).",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip PyTorch → ONNX export and only convert existing ONNX files.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1008,
        help="Input image resolution. Default: 1008.",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=17,
        help="ONNX opset version. Default: 17.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate converted models with dummy inference.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="CPU",
        help="OpenVINO device for validation. Default: CPU.",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply NNCF weight compression or full PTQ to existing FP16 IR models.",
    )
    parser.add_argument(
        "--compression-modes",
        type=str,
        nargs="+",
        default=["int8_sym"],
        help=(
            "Compression modes: int8_sym, int8_asym, int4_sym, int4_asym, int8_ptq, "
            "or 'all'. int8_ptq requires --calibration-root."
        ),
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Directory with FP16 OpenVINO IR models (required for --quantize).",
    )
    parser.add_argument(
        "--calibration-root",
        type=Path,
        default=None,
        help=(
            "Root directory containing calibration images for int8_ptq. "
            "Recursively scanned for JPEG/PNG images. Default: None "
            "(required when int8_ptq is in --compression-modes)."
        ),
    )
    parser.add_argument(
        "--calibration-categories",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Text categories for text-encoder calibration in int8_ptq. "
            "Defaults to LVIS_DEFAULT_CATEGORIES from constants."
        ),
    )
    parser.add_argument(
        "--ptq-target-device",
        type=str,
        default="GPU",
        choices=["ANY", "CPU", "GPU", "NPU", "CPU_SPR"],
        help="NNCF target device for int8_ptq. Default: GPU.",
    )
    parser.add_argument(
        "--ptq-num-calibration",
        type=int,
        default=200,
        help="Number of calibration images for int8_ptq. Default: 200.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(name)s: %(message)s",
        stream=sys.stdout,
    )

    # Step 1: PyTorch → ONNX export
    if args.skip_export:
        if args.onnx_dir is None:
            parser.error("--onnx-dir is required when using --skip-export")
        onnx_dir = args.onnx_dir
        logger.info("Skipping PyTorch export; using ONNX models from %s", onnx_dir)
    else:
        onnx_dir, exported = export_to_onnx(
            backbone_type=args.backbone_type,
            variant=args.variant,
            output_dir=args.output_dir,
            pretrained_model_name_or_path=args.model_id,
            ft=args.ft,
            resolution=args.resolution,
            opset_version=args.opset_version,
        )
        logger.info("ONNX export complete: %s", list(exported.keys()))

    # Step 2: ONNX → OpenVINO IR
    ir_dir = args.output_dir / f"openvino-{args.precision}"
    logger.info("Converting ONNX → OpenVINO IR (precision=%s)...", args.precision)
    converted = convert_to_openvino(
        onnx_dir=onnx_dir,
        output_dir=ir_dir,
        precision=args.precision,
    )
    logger.info("OpenVINO conversion complete!")
    for name, path in converted.items():
        logger.info("  %s: %s", name, path)

    # Step 3: Validate
    if args.validate:
        validate_openvino_models(
            ir_dir,
            device=args.device,
            resolution=args.resolution,
        )

    # Step 4: Weight compression (quantization)
    if args.quantize:
        if args.source_dir is None:
            parser.error("--source-dir is required when using --quantize")

        weight_only_modes = ["int8_sym", "int8_asym", "int4_sym", "int4_asym"]
        all_modes = [*weight_only_modes, "int8_ptq"]
        modes = all_modes if "all" in args.compression_modes else args.compression_modes

        result_dirs: dict[str, Path] = {}

        # Weight-only modes (data-free)
        for mode in modes:
            if mode == "int8_ptq":
                continue
            try:
                result_dirs[mode] = apply_weight_compression(args.source_dir, args.output_dir, mode)
            except Exception:
                logger.exception("Failed compression: %s", mode)

        # Full PTQ (requires calibration data)
        if "int8_ptq" in modes:
            from instantlearn.scripts.efficient_sam3.ptq_efficient_sam3 import (  # noqa: PLC0415
                quantize_efficient_sam3_ptq,
            )
            if args.calibration_root is None:
                parser.error("--calibration-root is required for int8_ptq mode")
            try:
                result_dirs["int8_ptq"] = quantize_efficient_sam3_ptq(
                    source_dir=args.source_dir,
                    output_dir=args.output_dir,
                    calibration_dir=args.calibration_root,
                    target_device=args.ptq_target_device,
                    num_calibration=args.ptq_num_calibration,
                    variant_name="int8_ptq_gpu",
                )
            except Exception:
                logger.exception("Failed full PTQ (int8_ptq)")

        if args.validate:
            for name, result_dir in result_dirs.items():
                logger.info("-" * 60)
                logger.info("Validating: %s", name)
                validate_openvino_models(result_dir, device=args.device, resolution=args.resolution)

        logger.info("=" * 60)
        logger.info("Quantization complete!")

    logger.info("Done! Models saved to: %s", args.output_dir)


if __name__ == "__main__":
    main()
