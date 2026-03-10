# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Quantize a YOLOE OpenVINO IR model using NNCF.

Supports:
  - **INT8** post-training quantization via ``nncf.quantize()``.
  - **INT4** weight compression via ``nncf.compress_weights()``.

The script takes an already-exported OpenVINO IR directory (from
``export_yoloe_openvino``) and produces a quantized model alongside it.

Usage::

    # INT8 quantization (requires a calibration dataset)
    python -m instantlearn.scripts.yoloe.quantize_yoloe_openvino \\
        --model-dir exports/yoloe_person_car_dog \\
        --mode int8 \\
        --calibration-dir /path/to/calibration/images \\
        --output-dir exports/yoloe_person_car_dog_int8

    # INT4 weight compression (no calibration data needed)
    python -m instantlearn.scripts.yoloe.quantize_yoloe_openvino \\
        --model-dir exports/yoloe_person_car_dog \\
        --mode int4 \\
        --output-dir exports/yoloe_person_car_dog_int4
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _load_calibration_images(
    calibration_dir: Path,
    imgsz: tuple[int, int],
    max_images: int = 300,
) -> list[np.ndarray]:
    """Load and preprocess calibration images for INT8 quantization.

    Args:
        calibration_dir: Directory containing calibration images.
        imgsz: Target model input size (H, W).
        max_images: Maximum number of images to use.

    Returns:
        List of preprocessed input tensors [1, 3, H, W] float32.
    """
    import cv2

    from instantlearn.models.yoloe.postprocessing import preprocess_image

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    image_paths = sorted(
        p for p in calibration_dir.iterdir()
        if p.suffix.lower() in image_extensions
    )[:max_images]

    if not image_paths:
        msg = f"No calibration images found in {calibration_dir}"
        raise FileNotFoundError(msg)

    logger.info("Loading %d calibration images from %s", len(image_paths), calibration_dir)

    inputs: list[np.ndarray] = []
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            logger.warning("Skipping unreadable image: %s", path)
            continue
        blob, _, _ = preprocess_image(img, imgsz)
        inputs.append(blob)

    return inputs


def quantize_int8(
    model_dir: str | Path,
    calibration_dir: str | Path,
    output_dir: str | Path,
    max_images: int = 300,
) -> Path:
    """Quantize a YOLOE OpenVINO model to INT8 using NNCF.

    Args:
        model_dir: Path to the exported OpenVINO model directory.
        calibration_dir: Path to directory with calibration images.
        output_dir: Output directory for the quantized model.
        max_images: Maximum calibration images to use.

    Returns:
        Path to the output directory with the quantized model.
    """
    import nncf
    import openvino as ov

    model_path = Path(model_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find the XML file
    xml_files = list(model_path.glob("*.xml"))
    if not xml_files:
        msg = f"No .xml file found in {model_path}"
        raise FileNotFoundError(msg)
    xml_path = xml_files[0]

    core = ov.Core()
    model = core.read_model(str(xml_path))

    # Parse imgsz from metadata
    imgsz = (640, 640)
    meta_path = model_path / "metadata.yaml"
    if meta_path.exists():
        import yaml

        with meta_path.open() as f:
            meta = yaml.safe_load(f) or {}
        sz = meta.get("imgsz", [640, 640])
        if isinstance(sz, int):
            sz = [sz, sz]
        imgsz = (sz[0], sz[1])

    # Load calibration data
    calibration_data = _load_calibration_images(
        Path(calibration_dir), imgsz, max_images
    )

    # Create NNCF dataset
    def transform_fn(input_tensor: np.ndarray) -> dict:
        return {0: input_tensor}

    calibration_dataset = nncf.Dataset(calibration_data, transform_fn)

    # Quantize
    logger.info("Running INT8 quantization with %d calibration images...", len(calibration_data))
    quantized_model = nncf.quantize(
        model,
        calibration_dataset,
        model_type=nncf.ModelType.TRANSFORMER,
        preset=nncf.QuantizationPreset.MIXED,
    )

    # Save quantized model
    output_xml = output_path / xml_path.name
    ov.save_model(quantized_model, str(output_xml))

    # Copy metadata files
    for meta_file in ["metadata.yaml", "export_config.json"]:
        src = model_path / meta_file
        if src.exists():
            shutil.copy2(str(src), str(output_path / meta_file))

    logger.info("INT8 quantized model saved to %s", output_path)
    return output_path


def quantize_int4(
    model_dir: str | Path,
    output_dir: str | Path,
) -> Path:
    """Compress YOLOE OpenVINO model weights to INT4 using NNCF.

    No calibration data is needed for weight-only compression.

    Args:
        model_dir: Path to the exported OpenVINO model directory.
        output_dir: Output directory for the compressed model.

    Returns:
        Path to the output directory with the compressed model.
    """
    import nncf
    import openvino as ov

    model_path = Path(model_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    xml_files = list(model_path.glob("*.xml"))
    if not xml_files:
        msg = f"No .xml file found in {model_path}"
        raise FileNotFoundError(msg)
    xml_path = xml_files[0]

    core = ov.Core()
    model = core.read_model(str(xml_path))

    logger.info("Running INT4 weight compression...")
    compressed_model = nncf.compress_weights(
        model,
        mode=nncf.CompressWeightsMode.INT4_SYM,
        ratio=0.8,
        group_size=128,
    )

    output_xml = output_path / xml_path.name
    ov.save_model(compressed_model, str(output_xml))

    # Copy metadata files
    for meta_file in ["metadata.yaml", "export_config.json"]:
        src = model_path / meta_file
        if src.exists():
            shutil.copy2(str(src), str(output_path / meta_file))

    logger.info("INT4 compressed model saved to %s", output_path)
    return output_path


def main() -> None:
    """CLI entry point for YOLOE OpenVINO quantization."""
    parser = argparse.ArgumentParser(
        description="Quantize a YOLOE OpenVINO IR model using NNCF.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to the exported OpenVINO model directory.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["int8", "int4"],
        default="int8",
        help="Quantization mode.",
    )
    parser.add_argument(
        "--calibration-dir",
        type=str,
        default=None,
        help="Path to calibration images (required for INT8).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for the quantized model.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=300,
        help="Maximum number of calibration images for INT8.",
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

    try:
        if args.mode == "int8":
            if not args.calibration_dir:
                logger.error("--calibration-dir is required for INT8 quantization.")
                sys.exit(1)
            quantize_int8(
                model_dir=args.model_dir,
                calibration_dir=args.calibration_dir,
                output_dir=args.output_dir,
                max_images=args.max_images,
            )
        elif args.mode == "int4":
            quantize_int4(
                model_dir=args.model_dir,
                output_dir=args.output_dir,
            )
    except Exception:
        logger.exception("Quantization failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
