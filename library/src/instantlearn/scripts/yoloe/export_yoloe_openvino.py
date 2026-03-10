# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Export a YOLOE model to OpenVINO IR with a fixed set of classes.

Supports both **text prompt** and **visual prompt** export.  In both
cases the class embeddings are permanently fused into the convolution
weights (via ``set_classes`` → ``_fuse_tp``), so the exported model is
locked to the classes established at export time.

Usage::

    # Text prompt export
    python -m instantlearn.scripts.yoloe.export_yoloe_openvino \\
        --model yoloe-26s-seg \\
        --classes person car dog \\
        --output-dir exports/yoloe_person_car_dog \\
        --imgsz 640 \\
        --half

    # Visual prompt export
    python -m instantlearn.scripts.yoloe.export_yoloe_openvino \\
        --model yoloe-26s-seg \\
        --classes elephant \\
        --refer-image ref.jpg \\
        --bboxes 100,50,400,350 \\
        --output-dir exports/yoloe_elephant_vp \\
        --imgsz 640
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def export_yoloe_openvino(
    model_name: str,
    classes: list[str],
    output_dir: str | Path,
    imgsz: int = 640,
    half: bool = False,
    *,
    refer_image: str | Path | None = None,
    bboxes: list[list[float]] | None = None,
    cls_ids: list[int] | None = None,
) -> Path:
    """Export a YOLOE model to OpenVINO IR for a fixed set of classes.

    Supports two prompt modes:

    * **Text prompt** (default): class names are embedded via the CLIP
      text encoder.
    * **Visual prompt**: a reference image with bounding boxes is used
      to compute visual embeddings via the SAVPE module.  Provide
      ``refer_image``, ``bboxes``, and optionally ``cls_ids``.

    In both cases the embeddings are permanently fused into the conv
    weights — the exported IR is structurally identical.

    Args:
        model_name: YOLOE model variant (e.g. ``"yoloe-26s-seg"``).
        classes: List of class names to bake into the model.
        output_dir: Directory to save the OpenVINO IR files.
        imgsz: Input image size.
        half: Whether to export with FP16 precision.
        refer_image: Path to the reference image for visual prompt
            export.  When ``None`` (default), text prompt is used.
        bboxes: Bounding boxes ``[[x1, y1, x2, y2], ...]`` on the
            reference image.  Required when ``refer_image`` is given.
        cls_ids: Integer class indices aligned with *bboxes*.  Defaults
            to ``[0, 1, 2, ...]`` if not provided.

    Returns:
        Path to the output directory containing the IR files.

    Raises:
        ImportError: If ultralytics is not installed.
        ValueError: If the model name is not recognised or arguments
            are inconsistent.
    """
    from ultralytics import YOLO

    from instantlearn.models.yoloe.yoloe import YOLOE_MODELS
    from instantlearn.models.yoloe.weights import get_weights_path

    if model_name not in YOLOE_MODELS:
        available = ", ".join(YOLOE_MODELS.keys())
        msg = f"Unknown YOLOE model '{model_name}'. Available: {available}"
        raise ValueError(msg)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load PyTorch model
    model_file = YOLOE_MODELS[model_name]
    model_path = get_weights_path(model_file)
    logger.info("Loading YOLOE model: %s", model_path)
    model = YOLO(str(model_path))
    inner = model.model

    prompt_mode: str

    if refer_image is not None:
        # ---- Visual prompt ------------------------------------------------
        if bboxes is None:
            msg = "bboxes are required when refer_image is provided."
            raise ValueError(msg)
        if cls_ids is None:
            cls_ids = list(range(len(bboxes)))

        prompt_mode = "visual"
        logger.info(
            "Computing visual prompt embeddings from %s (%d boxes)...",
            refer_image,
            len(bboxes),
        )

        from ultralytics.models.yolo.yoloe.predict import (
            YOLOEVPDetectPredictor,
            YOLOEVPSegPredictor,
        )
        predictor_cls = (
            YOLOEVPSegPredictor if "seg" in model_name
            else YOLOEVPDetectPredictor
        )

        visual_prompts = {"bboxes": bboxes, "cls": cls_ids}
        num_cls = len(set(cls_ids))
        inner.model[-1].nc = num_cls
        inner.names = classes[:num_cls]

        predictor = predictor_cls(
            overrides={
                "task": inner.task,
                "mode": "predict",
                "save": False,
                "verbose": False,
                "batch": 1,
                "imgsz": imgsz,
            },
        )
        predictor.set_prompts(visual_prompts)
        predictor.setup_model(model=inner)

        vpe = predictor.get_vpe(str(refer_image))
        inner.set_classes(classes[:num_cls], vpe)
        logger.info("Visual embeddings fused into model weights.")
    else:
        # ---- Text prompt --------------------------------------------------
        prompt_mode = "text"
        logger.info("Setting classes (text prompt): %s", classes)
        text_pe = inner.get_text_pe(classes)
        inner.set_classes(classes, text_pe)

    # Export to OpenVINO IR
    logger.info("Exporting to OpenVINO IR (imgsz=%d, half=%s)...", imgsz, half)
    ov_dir = model.export(
        format="openvino",
        imgsz=imgsz,
        half=half,
        dynamic=False,
    )
    ov_dir = Path(ov_dir)

    # Move files to the desired output directory
    if ov_dir.resolve() != output_path.resolve():
        for f in ov_dir.iterdir():
            dest = output_path / f.name
            shutil.move(str(f), str(dest))
        # Clean up the temporary directory
        if ov_dir.exists() and not any(ov_dir.iterdir()):
            ov_dir.rmdir()

    # Write additional metadata for the inference class
    meta_path = output_path / "export_config.json"
    meta = {
        "model_name": model_name,
        "classes": classes,
        "imgsz": imgsz,
        "half": half,
        "prompt_mode": prompt_mode,
    }
    if refer_image is not None:
        meta["refer_image"] = str(refer_image)
        meta["bboxes"] = bboxes
        meta["cls_ids"] = cls_ids
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Export complete (%s prompt). Files saved to %s", prompt_mode, output_path)
    return output_path


def main() -> None:
    """CLI entry point for YOLOE OpenVINO export."""
    parser = argparse.ArgumentParser(
        description="Export a YOLOE model to OpenVINO IR with a fixed set of classes.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yoloe-26s-seg",
        help="YOLOE model variant name.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        required=True,
        help="Class names to bake into the model.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="exports/yoloe_openvino",
        help="Output directory for the OpenVINO IR.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size.",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Export with FP16 precision.",
    )
    parser.add_argument(
        "--refer-image",
        type=str,
        default=None,
        help="Reference image for visual prompt export.",
    )
    parser.add_argument(
        "--bboxes",
        nargs="+",
        type=str,
        default=None,
        help="Bounding boxes as x1,y1,x2,y2 (comma-separated, one per box).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(levelname)s - %(name)s: %(message)s",
    )

    # Parse bounding boxes from CLI
    bboxes = None
    if args.bboxes:
        bboxes = [[float(v) for v in b.split(",")] for b in args.bboxes]

    try:
        export_yoloe_openvino(
            model_name=args.model,
            classes=args.classes,
            output_dir=args.output_dir,
            imgsz=args.imgsz,
            half=args.half,
            refer_image=args.refer_image,
            bboxes=bboxes,
        )
    except Exception:
        logger.exception("Export failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
