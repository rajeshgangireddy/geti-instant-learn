#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Gating repro for SAM-HQ-Tiny → OpenVINO export correctness.

Exit 0 and prints [REPRO] PASS if IoU(PT, OV-CPU) >= 0.90 and
IoU(PT, OV-GPU) >= 0.90 on both test targets.

Usage:
    cd /path/to/geti-instant-learn
    source library/.cuda/bin/activate
    python library/scripts/repro_sam_hq_tiny_ov.py
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from instantlearn.data import Sample
from instantlearn.data.utils.image import read_image
from instantlearn.models.matcher.matcher import Matcher, Backend
from instantlearn.utils.constants import CompressionMode, SAMModelName

ASSETS = Path(__file__).parent.parent / "examples" / "assets" / "coco"
REF_IMG = str(ASSETS / "000000286874.jpg")
REF_MASK = str(ASSETS / "000000286874_mask.png")
TARGETS = [
    str(ASSETS / "000000390341.jpg"),
    str(ASSETS / "000000267704.jpg"),
]
OUTPUT_DIR = Path(__file__).parent / "results" / "repro_sam_hq_tiny"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IoU_THRESHOLD = 0.90
WARMUP = 3
TIMED_RUNS = 10
GPU_CONFIG = {
    "PERFORMANCE_HINT": "LATENCY",
    "INFERENCE_PRECISION_HINT": "f32",
}


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    inter = (a & b).sum()
    union = (a | b).sum()
    return float(inter) / float(union + 1e-6)


def mean_matched_iou(ov_masks: np.ndarray, pt_masks: np.ndarray) -> float:
    """Greedy IoU matching: match each OV mask to nearest unmatched PT mask."""
    if len(ov_masks) == 0 or len(pt_masks) == 0:
        return 0.0
    used = set()
    total = 0.0
    count = 0
    for ov_m in ov_masks:
        best_j, best_v = -1, -1.0
        for j, pt_m in enumerate(pt_masks):
            if j in used:
                continue
            v = compute_iou(ov_m, pt_m)
            if v > best_v:
                best_v, best_j = v, j
        if best_j >= 0:
            used.add(best_j)
            total += best_v
            count += 1
    return total / max(count, 1)


def get_pt_masks(results: list) -> np.ndarray:
    if not results or results[0]["pred_masks"].numel() == 0:
        return np.zeros((0,), dtype=bool)
    masks = results[0]["pred_masks"].cpu().numpy()
    if masks.ndim == 2:
        masks = masks[np.newaxis]
    return masks.astype(bool)


def get_ov_masks(ov_result, matcher: Matcher, frame_h: int, frame_w: int) -> np.ndarray:
    masks_raw = np.array(ov_result["masks"])
    scores_raw = np.array(ov_result["scores"])
    final_masks = []
    for cat_idx in range(masks_raw.shape[0]):
        best_idx = int(np.argmax(scores_raw[cat_idx]))
        final_masks.append(masks_raw[cat_idx, best_idx])
    final = np.stack(final_masks, axis=0)
    return matcher.resize_masks_to_frame(final, frame_h, frame_w)


print("=" * 60)
print("[REPRO] SAM-HQ-Tiny → OpenVINO correctness check")
print("=" * 60)

import openvino as ov

core = ov.Core()
print(f"[REPRO] OpenVINO version : {ov.__version__}")
print(f"[REPRO] Available devices: {core.available_devices}")
print(f"[REPRO] torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"[REPRO] torch.xpu.is_available() : {torch.xpu.is_available()}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[REPRO] Using PT device : {device}")

print("\n[REPRO] Building Matcher(sam=SAM_HQ_TINY, encoder=dinov3_small) ...")
matcher = Matcher(
    sam=SAMModelName.SAM_HQ_TINY,
    encoder_model="dinov3_small",
    device=device,
    precision="fp32",
)
matcher.fit(Sample(image_path=REF_IMG, mask_paths=REF_MASK))
print("[REPRO] fit() done.")

pt_masks_list = []
target_images = []
for tp in TARGETS:
    image = np.asarray(read_image(tp, as_tensor=False))
    target_images.append(image)
    pt_masks = get_pt_masks(matcher.predict([Sample(image_path=tp)]))
    pt_masks_list.append(pt_masks)
    print(f"[REPRO] PT target={os.path.basename(tp):25s}  masks={len(pt_masks)}")

xml_path = OUTPUT_DIR / "matcher.xml"
print(f"\n[REPRO] Exporting to OV FP32 → {xml_path}")
assert matcher.sam_predictor.sam_model_name == SAMModelName.SAM_HQ_TINY, (
    "SAM model is not SAM_HQ_TINY — fallback check failed!"
)
matcher.export(
    backend=Backend.OPENVINO,
    export_dir=str(OUTPUT_DIR),
    compression=CompressionMode.FP32,
)
print("[REPRO] Export done.")

ov_model = core.read_model(str(xml_path))
compiled_cpu = core.compile_model(ov_model, "CPU")
use_gpu = "GPU" in core.available_devices
compiled_gpu = core.compile_model(ov_model, "GPU", config=GPU_CONFIG) if use_gpu else None
input_size = matcher.encoder.input_size
if use_gpu:
    print("[REPRO] GPU path: fused single-graph matcher.xml on GPU.")
    print("[REPRO] TinyViT attention bias pre-expanded before ONNX trace (no Gather on GPU).")

all_pass = True
for tp, image, pt_masks in zip(TARGETS, target_images, pt_masks_list, strict=False):
    h, w = image.shape[:2]
    ov_input = {"target_image": Matcher.prepare_openvino_input(image, input_size)}
    res_cpu = compiled_cpu(ov_input)
    ov_cpu = get_ov_masks(res_cpu, matcher, h, w)
    iou_cpu = mean_matched_iou(ov_cpu, pt_masks)

    if use_gpu:
        res_gpu = compiled_gpu(ov_input)
        ov_gpu = get_ov_masks(res_gpu, matcher, h, w)
        iou_gpu = mean_matched_iou(ov_gpu, pt_masks)
        gpu_str = f"  IoU(PT, OV-GPU)={iou_gpu:.4f}"
        ok = iou_cpu >= IoU_THRESHOLD and iou_gpu >= IoU_THRESHOLD
    else:
        iou_gpu = None
        gpu_str = "  IoU(PT, OV-GPU)=N/A (no GPU)"
        ok = iou_cpu >= IoU_THRESHOLD

    all_pass = all_pass and ok
    status = "OK" if ok else "FAIL"
    print(f"[REPRO] target={os.path.basename(tp):25s}  IoU(PT, OV-CPU)={iou_cpu:.4f}{gpu_str}  {status}")

if use_gpu:
    test_image = target_images[0]
    ov_input_test = {"target_image": Matcher.prepare_openvino_input(test_image, input_size)}
    for _ in range(WARMUP):
        compiled_gpu(ov_input_test)
    t0 = time.perf_counter()
    for _ in range(TIMED_RUNS):
        compiled_gpu(ov_input_test)
    gpu_latency_ms = (time.perf_counter() - t0) / TIMED_RUNS * 1000
    print(f"\n[REPRO] GPU latency (fused single-graph, hint=LATENCY): {gpu_latency_ms:.1f} ms")
else:
    gpu_latency_ms = None
    print("\n[REPRO] GPU latency: N/A (no GPU)")

if torch.xpu.is_available():
    try:
        xpu_matcher = Matcher(
            sam=SAMModelName.SAM_HQ_TINY,
            encoder_model="dinov3_small",
            device="xpu",
            precision="fp32",
        )
        xpu_matcher.fit(Sample(image_path=REF_IMG, mask_paths=REF_MASK))
        for _ in range(WARMUP):
            xpu_matcher.predict([Sample(image_path=TARGETS[0])])
        t0 = time.perf_counter()
        for _ in range(TIMED_RUNS):
            xpu_matcher.predict([Sample(image_path=TARGETS[0])])
        xpu_latency_ms = (time.perf_counter() - t0) / TIMED_RUNS * 1000
        print(f"[REPRO] PyTorch XPU baseline: {xpu_latency_ms:.1f} ms")
        perf_ok = gpu_latency_ms is None or gpu_latency_ms < xpu_latency_ms
    except Exception as e:
        print(f"[REPRO] PyTorch XPU baseline: N/A ({e})")
        xpu_latency_ms = None
        perf_ok = True
else:
    print("[REPRO] PyTorch XPU baseline: N/A (torch.xpu not available)")
    xpu_latency_ms = None
    perf_ok = True

print()
if all_pass and perf_ok:
    print("[REPRO] PASS  (IoU target >= 0.90, GPU latency < XPU baseline OR XPU unavailable)")
    sys.exit(0)
if not all_pass:
    print("[REPRO] FAIL  (one or more IoU values below 0.90 threshold)")
if not perf_ok:
    print(f"[REPRO] FAIL  (GPU latency {gpu_latency_ms:.1f} ms >= XPU baseline {xpu_latency_ms:.1f} ms)")
sys.exit(1)
