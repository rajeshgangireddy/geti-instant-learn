# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive YOLOE benchmark: PyTorch vs OpenVINO across prompt modes.

Benchmarks four phases:
  1. PyTorch visual prompt (GPU) — fit with reference mask, predict with VP.
  2. PyTorch text prompt (GPU) — set_classes with text, standard predict.
  3. OpenVINO text prompt (CPU/GPU) — pre-exported IR in FP32/FP16/INT8/INT4.
  4. OpenVINO visual prompt (CPU/GPU) — exported using visual embeddings.

Tests all YOLOE-26 variants (N/S/M/L/X). Runs 20 iterations per configuration
(batch=1), drops the fastest and slowest, reports trimmed average. Results are
saved to a CSV file.

Usage::

    cd library

    # Full benchmark on CPU (default)
    python examples/yoloe_benchmark.py

    # OpenVINO on Intel GPU (use --ov-device GPU)
    python examples/yoloe_benchmark.py --ov-device GPU

    # Subset of models / formats
    python examples/yoloe_benchmark.py --models 26n 26s --formats fp32 fp16

    # Custom iteration count
    python examples/yoloe_benchmark.py --iters 50
"""

from __future__ import annotations

import argparse
import csv
import gc
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.tv_tensors import Image

from instantlearn.data.base.sample import Sample
from instantlearn.models.yoloe import YOLOE, YOLOEOpenVINO
from instantlearn.models.yoloe.yoloe import YOLOE_MODELS
from instantlearn.scripts.yoloe.export_yoloe_openvino import export_yoloe_openvino
from instantlearn.scripts.yoloe.quantize_yoloe_openvino import (
    quantize_int4,
    quantize_int8,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ASSETS = Path(__file__).parent / "assets" / "coco"
REF_IMAGE = ASSETS / "000000286874.jpg"
REF_MASK = ASSETS / "000000286874_mask.png"
TARGET_IMAGES = [
    ASSETS / "000000390341.jpg",
    ASSETS / "000000173279.jpg",
    ASSETS / "000000267704.jpg",
    ASSETS / "000000286874.jpg",
]
CLASSES = ["elephant"]
IMGSZ = 640

MODEL_VARIANTS: dict[str, str] = {
    "26n": "yoloe-26n-seg",
    "26s": "yoloe-26s-seg",
    "26m": "yoloe-26m-seg",
    "26l": "yoloe-26l-seg",
    "26x": "yoloe-26x-seg",
}

OV_FORMATS = ["fp32", "fp16", "int8", "int4"]
EXPORT_ROOT = Path("exports/yoloe_benchmark")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_image(path: Path) -> torch.Tensor:
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img).permute(2, 0, 1)


def load_mask(path: Path) -> torch.Tensor:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return torch.from_numpy(mask > 127).unsqueeze(0)


def trimmed_avg(times: list[float]) -> float:
    """Drop the fastest and slowest, return average of the rest."""
    s = sorted(times)
    trimmed = s[1:-1]
    return sum(trimmed) / len(trimmed) if trimmed else (sum(s) / len(s))


def benchmark_predict(
    model,
    target_samples: list[Sample],
    n_iters: int,
) -> list[float]:
    """Run n single-image predictions, return per-inference wall-clock times."""
    times: list[float] = []
    for i in range(n_iters):
        sample = target_samples[i % len(target_samples)]
        t0 = time.perf_counter()
        model.predict(sample)
        times.append(time.perf_counter() - t0)
    return times


def benchmark_predict_text_prompt(
    ul_model,
    target_images: list[np.ndarray],
    n_iters: int,
    imgsz: int,
    conf: float = 0.25,
) -> list[float]:
    """Run n text-prompt predictions via raw ultralytics predict (no VP)."""
    times: list[float] = []
    for i in range(n_iters):
        img = target_images[i % len(target_images)]
        t0 = time.perf_counter()
        ul_model.predict(source=img, imgsz=imgsz, conf=conf, verbose=False)
        times.append(time.perf_counter() - t0)
    return times


def make_samples() -> tuple[Sample, list[Sample]]:
    """Build reference and target Sample objects from COCO assets."""
    ref_img = load_image(REF_IMAGE)
    ref_mask = load_mask(REF_MASK)

    ref_sample = Sample(
        image=Image(ref_img),
        masks=ref_mask,
        category_ids=np.array([0]),
        is_reference=[True],
        categories=CLASSES,
    )
    target_samples = [
        Sample(image=Image(load_image(p)), is_reference=[False], categories=CLASSES)
        for p in TARGET_IMAGES
    ]
    return ref_sample, target_samples


# ---------------------------------------------------------------------------
# Export / quantise helpers
# ---------------------------------------------------------------------------
def export_ov_fp(model_name: str, half: bool, out_dir: Path, **kwargs) -> Path:
    """Export YOLOE to OpenVINO IR (FP32 or FP16)."""
    if out_dir.exists() and any(out_dir.glob("*.xml")):
        print(f"    Reusing existing export: {out_dir}")
        return out_dir
    return export_yoloe_openvino(
        model_name=model_name,
        classes=CLASSES,
        output_dir=out_dir,
        imgsz=IMGSZ,
        half=half,
        **kwargs,
    )


def export_ov_int8(fp32_dir: Path, out_dir: Path) -> Path:
    """Quantise an FP32 IR to INT8 (uses COCO assets as calibration)."""
    if out_dir.exists() and any(out_dir.glob("*.xml")):
        print(f"    Reusing existing quantised model: {out_dir}")
        return out_dir
    return quantize_int8(
        model_dir=fp32_dir,
        calibration_dir=ASSETS,
        output_dir=out_dir,
    )


def export_ov_int4(fp32_dir: Path, out_dir: Path) -> Path:
    """Compress an FP32 IR to INT4 weights."""
    if out_dir.exists() and any(out_dir.glob("*.xml")):
        print(f"    Reusing existing compressed model: {out_dir}")
        return out_dir
    return quantize_int4(model_dir=fp32_dir, output_dir=out_dir)


def get_ov_dir(variant_key: str, fmt: str, prompt_mode: str = "tp") -> Path:
    """Return the export directory path for a variant + format + mode combo."""
    return EXPORT_ROOT / f"{variant_key}_{fmt}_{prompt_mode}"


def get_ref_bboxes() -> list[list[float]]:
    """Extract bounding boxes from the reference mask."""
    mask = cv2.imread(str(REF_MASK), cv2.IMREAD_GRAYSCALE)
    mask_bin = (mask > 127).astype(np.uint8)
    ys, xs = np.where(mask_bin > 0)
    if len(ys) == 0:
        msg = "Reference mask is empty."
        raise ValueError(msg)
    return [[float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]]


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------
def run_benchmark(
    models: list[str],
    formats: list[str],
    n_iters: int,
    ov_device: str,
    csv_path: Path,
) -> None:
    ref_sample, target_samples = make_samples()

    rows: list[dict[str, str | float]] = []

    # ---- Phase 1: PyTorch visual prompt ---------------------------------
    print("\n" + "=" * 70)
    print("Phase 1 — PyTorch Visual Prompt (GPU)")
    print("=" * 70)

    for vkey in models:
        model_name = MODEL_VARIANTS[vkey]
        print(f"\n  [{model_name}] Loading...")
        pt_model = YOLOE(
            model_name=model_name, device="cuda", imgsz=IMGSZ, precision="fp32",
        )
        pt_model.fit(ref_sample)

        # Warmup
        pt_model.predict(target_samples[0])

        print(f"  [{model_name}] Running {n_iters} predictions...")
        times = benchmark_predict(pt_model, target_samples, n_iters)
        avg_ms = trimmed_avg(times) * 1000.0

        rows.append({
            "model": model_name,
            "backend": "pytorch",
            "format": "fp32",
            "device": "cuda",
            "mode": "visual_prompt",
            "avg_ms": round(avg_ms, 2),
            "min_ms": round(min(times) * 1000, 2),
            "max_ms": round(max(times) * 1000, 2),
            "n_iters": n_iters,
        })
        print(f"  [{model_name}] avg={avg_ms:.1f} ms")

        # Free GPU memory
        del pt_model
        gc.collect()
        torch.cuda.empty_cache()

    # ---- Phase 2: PyTorch text prompt ------------------------------------
    print("\n" + "=" * 70)
    print("Phase 2 — PyTorch Text Prompt (GPU)")
    print("=" * 70)

    # Preload target images as numpy for raw ultralytics predict
    target_images_np = [
        cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        for p in TARGET_IMAGES
    ]

    for vkey in models:
        model_name = MODEL_VARIANTS[vkey]
        print(f"\n  [{model_name}] Loading...")

        from ultralytics import YOLO
        from instantlearn.utils.weights import get_weights_path
        ul_model = YOLO(str(get_weights_path(YOLOE_MODELS[model_name])))

        # Set text prompt: compute text embeddings and fuse into weights
        inner = ul_model.model
        text_pe = inner.get_text_pe(CLASSES)
        inner.set_classes(CLASSES, text_pe)

        # Move to GPU
        ul_model.to("cuda")

        # Warmup
        ul_model.predict(
            source=target_images_np[0], imgsz=IMGSZ, conf=0.25, verbose=False,
        )

        print(f"  [{model_name}] Running {n_iters} predictions...")
        times = benchmark_predict_text_prompt(
            ul_model, target_images_np, n_iters, IMGSZ,
        )
        avg_ms = trimmed_avg(times) * 1000.0

        rows.append({
            "model": model_name,
            "backend": "pytorch",
            "format": "fp32",
            "device": "cuda",
            "mode": "text_prompt",
            "avg_ms": round(avg_ms, 2),
            "min_ms": round(min(times) * 1000, 2),
            "max_ms": round(max(times) * 1000, 2),
            "n_iters": n_iters,
        })
        print(f"  [{model_name}] avg={avg_ms:.1f} ms")

        del ul_model, inner
        gc.collect()
        torch.cuda.empty_cache()

    # ---- Phase 3: OpenVINO text prompt ----------------------------------
    print("\n" + "=" * 70)
    print(f"Phase 3 — OpenVINO Text Prompt (device={ov_device})")
    print("=" * 70)

    for vkey in models:
        model_name = MODEL_VARIANTS[vkey]

        # --- Export FP32 first (needed as base for INT8/INT4) ---
        fp32_dir = get_ov_dir(vkey, "fp32", "tp")
        print(f"\n  [{model_name}] Exporting FP32 (text prompt)...")
        export_ov_fp(model_name, half=False, out_dir=fp32_dir)

        for fmt in formats:
            print(f"\n  [{model_name} / {fmt.upper()}] Preparing...")
            ov_dir: Path

            if fmt == "fp32":
                ov_dir = fp32_dir
            elif fmt == "fp16":
                ov_dir = get_ov_dir(vkey, "fp16", "tp")
                export_ov_fp(model_name, half=True, out_dir=ov_dir)
            elif fmt == "int8":
                ov_dir = get_ov_dir(vkey, "int8", "tp")
                export_ov_int8(fp32_dir, ov_dir)
            elif fmt == "int4":
                ov_dir = get_ov_dir(vkey, "int4", "tp")
                export_ov_int4(fp32_dir, ov_dir)
            else:
                print(f"    Unknown format '{fmt}', skipping.")
                continue

            print(f"  [{model_name} / {fmt.upper()}] Loading on {ov_device}...")
            ov_model = YOLOEOpenVINO(
                model_dir=ov_dir,
                device=ov_device,
                confidence_threshold=0.5,
            )
            ov_model.fit(ref_sample)

            # Warmup
            ov_model.predict(target_samples[0])

            print(f"  [{model_name} / {fmt.upper()}] Running {n_iters} predictions...")
            times = benchmark_predict(ov_model, target_samples, n_iters)
            avg_ms = trimmed_avg(times) * 1000.0

            rows.append({
                "model": model_name,
                "backend": "openvino",
                "format": fmt,
                "device": ov_device,
                "mode": "text_prompt",
                "avg_ms": round(avg_ms, 2),
                "min_ms": round(min(times) * 1000, 2),
                "max_ms": round(max(times) * 1000, 2),
                "n_iters": n_iters,
            })
            print(f"  [{model_name} / {fmt.upper()}] avg={avg_ms:.1f} ms")

            del ov_model
            gc.collect()

    # ---- Phase 4: OpenVINO visual prompt --------------------------------
    print("\n" + "=" * 70)
    print(f"Phase 4 — OpenVINO Visual Prompt (device={ov_device})")
    print("=" * 70)

    ref_bboxes = get_ref_bboxes()

    for vkey in models:
        model_name = MODEL_VARIANTS[vkey]

        # --- Export FP32 with visual prompt first ---
        vp_fp32_dir = get_ov_dir(vkey, "fp32", "vp")
        print(f"\n  [{model_name}] Exporting FP32 (visual prompt)...")
        export_ov_fp(
            model_name, half=False, out_dir=vp_fp32_dir,
            refer_image=str(REF_IMAGE), bboxes=ref_bboxes, cls_ids=[0],
        )

        for fmt in formats:
            print(f"\n  [{model_name} / VP {fmt.upper()}] Preparing...")
            ov_dir: Path

            if fmt == "fp32":
                ov_dir = vp_fp32_dir
            elif fmt == "fp16":
                ov_dir = get_ov_dir(vkey, "fp16", "vp")
                export_ov_fp(
                    model_name, half=True, out_dir=ov_dir,
                    refer_image=str(REF_IMAGE), bboxes=ref_bboxes, cls_ids=[0],
                )
            elif fmt == "int8":
                ov_dir = get_ov_dir(vkey, "int8", "vp")
                export_ov_int8(vp_fp32_dir, ov_dir)
            elif fmt == "int4":
                ov_dir = get_ov_dir(vkey, "int4", "vp")
                export_ov_int4(vp_fp32_dir, ov_dir)
            else:
                print(f"    Unknown format '{fmt}', skipping.")
                continue

            print(f"  [{model_name} / VP {fmt.upper()}] Loading on {ov_device}...")
            ov_model = YOLOEOpenVINO(
                model_dir=ov_dir,
                device=ov_device,
                confidence_threshold=0.5,
            )
            ov_model.fit(ref_sample)

            # Warmup
            ov_model.predict(target_samples[0])

            print(f"  [{model_name} / VP {fmt.upper()}] Running {n_iters} predictions...")
            times = benchmark_predict(ov_model, target_samples, n_iters)
            avg_ms = trimmed_avg(times) * 1000.0

            rows.append({
                "model": model_name,
                "backend": "openvino",
                "format": fmt,
                "device": ov_device,
                "mode": "visual_prompt",
                "avg_ms": round(avg_ms, 2),
                "min_ms": round(min(times) * 1000, 2),
                "max_ms": round(max(times) * 1000, 2),
                "n_iters": n_iters,
            })
            print(f"  [{model_name} / VP {fmt.upper()}] avg={avg_ms:.1f} ms")

            del ov_model
            gc.collect()

    # ---- Write CSV -------------------------------------------------------
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model", "backend", "format", "device", "mode",
        "avg_ms", "min_ms", "max_ms", "n_iters",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to {csv_path}")

    # ---- Print summary table --------------------------------------------
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    hdr = (
        f"  {'Model':<20s} {'Backend':<12s} {'Format':<8s} "
        f"{'Device':<8s} {'Mode':<16s} {'Avg (ms)':>10s}"
    )
    print(hdr)
    print("  " + "-" * len(hdr.strip()))
    for r in rows:
        print(
            f"  {r['model']:<20s} {r['backend']:<12s} {r['format']:<8s} "
            f"{r['device']:<8s} {r['mode']:<16s} {r['avg_ms']:>10.1f}"
        )
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLOE benchmark: PyTorch visual prompt vs OpenVINO text prompt",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_VARIANTS.keys()),
        default=list(MODEL_VARIANTS.keys()),
        help="Model variant keys to benchmark (default: all).",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=OV_FORMATS,
        default=OV_FORMATS,
        help="OpenVINO formats to benchmark (default: all).",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=20,
        help="Number of inference iterations per config (default: 20).",
    )
    parser.add_argument(
        "--ov-device",
        type=str,
        default="CPU",
        choices=["CPU", "GPU", "AUTO"],
        help="OpenVINO device for inference (default: CPU). Use GPU for Intel GPU.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="exports/yoloe_benchmark/results.csv",
        help="Path to save the CSV results file.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s - %(name)s: %(message)s",
    )
    args = parse_args()
    run_benchmark(
        models=args.models,
        formats=args.formats,
        n_iters=args.iters,
        ov_device=args.ov_device,
        csv_path=Path(args.csv),
    )


if __name__ == "__main__":
    main()
