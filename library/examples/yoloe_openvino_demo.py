# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Demo: YOLOE PyTorch vs OpenVINO side-by-side comparison.

Uses the elephant example from the library README — one reference
image with mask, three target images.  Runs both PyTorch and
OpenVINO inference, measures timings, and saves a comparison grid.

Usage::

    cd library
    python examples/yoloe_openvino_demo.py
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.tv_tensors import Image

from instantlearn.data.base.sample import Sample
from instantlearn.models.yoloe import YOLOE, YOLOEOpenVINO
from instantlearn.scripts.yoloe.export_yoloe_openvino import export_yoloe_openvino

# ---------------------------------------------------------------------------
# Config — elephant example from library README
# ---------------------------------------------------------------------------
ASSETS = Path(__file__).parent / "assets" / "coco"
REF_IMAGE = ASSETS / "000000286874.jpg"
REF_MASK = ASSETS / "000000286874_mask.png"
TARGET_IMAGES = [
    ASSETS / "000000390341.jpg",
    ASSETS / "000000173279.jpg",
    ASSETS / "000000267704.jpg",
]

MODEL_NAME = "yoloe-26n-seg"
CLASSES = ["elephant"]
EXPORT_DIR = Path("exports/yoloe_demo_ov")
OUTPUT_IMAGE = Path("yoloe_pytorch_vs_openvino.png")
IMGSZ = 640


def load_image(path: Path) -> torch.Tensor:
    """Load an image as a CHW uint8 tensor."""
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img).permute(2, 0, 1)  # HWC → CHW


def load_mask(path: Path) -> torch.Tensor:
    """Load a mask as a boolean [1, H, W] tensor."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return torch.from_numpy(mask > 127).unsqueeze(0)  # [1, H, W]


def draw_results(image: np.ndarray, result: dict, label: str) -> np.ndarray:
    """Draw masks and boxes on an image copy."""
    vis = image.copy()
    masks = result["pred_masks"]
    boxes = result["pred_boxes"]
    scores = boxes[:, 4] if len(boxes) > 0 else []

    # Overlay masks
    overlay = vis.copy()
    for i in range(len(masks)):
        mask_np = masks[i].cpu().numpy() if isinstance(masks[i], torch.Tensor) else masks[i]
        color = (0, 200, 0) if i % 2 == 0 else (200, 0, 200)
        overlay[mask_np > 0] = color
    vis = cv2.addWeighted(vis, 0.5, overlay, 0.5, 0)

    # Draw boxes
    for i in range(len(boxes)):
        box = boxes[i].cpu().numpy() if isinstance(boxes[i], torch.Tensor) else boxes[i]
        x1, y1, x2, y2 = map(int, box[:4])
        score = float(box[4])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, f"{score:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Title
    cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return vis


def resize_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    """Resize image to a target height, preserving aspect ratio."""
    h, w = img.shape[:2]
    if h == target_h:
        return img
    scale = target_h / h
    return cv2.resize(img, (int(w * scale), target_h))


def main() -> None:
    # Load data
    ref_img = load_image(REF_IMAGE)
    ref_mask = load_mask(REF_MASK)
    target_imgs = [load_image(p) for p in TARGET_IMAGES]

    ref_sample = Sample(
        image=Image(ref_img),
        masks=ref_mask,
        category_ids=np.array([0]),
        is_reference=[True],
        categories=CLASSES,
    )
    target_samples = [
        Sample(image=Image(img), is_reference=[False], categories=CLASSES)
        for img in target_imgs
    ]

    # ------------------------------------------------------------------
    # 1. YOLOE PyTorch
    # ------------------------------------------------------------------
    print("=" * 60)
    print("YOLOE PyTorch")
    print("=" * 60)
    pt_model = YOLOE(model_name=MODEL_NAME, device="cpu", imgsz=IMGSZ, precision="fp32")

    t0 = time.perf_counter()
    pt_model.fit(ref_sample)
    pt_fit_time = time.perf_counter() - t0
    print(f"  fit():     {pt_fit_time:.3f}s")

    pt_results = []
    t0 = time.perf_counter()
    for ts in target_samples:
        pt_results.extend(pt_model.predict(ts))
    pt_pred_time = time.perf_counter() - t0
    print(f"  predict(): {pt_pred_time:.3f}s  ({len(target_samples)} images)")
    for i, r in enumerate(pt_results):
        print(f"    image {i}: {len(r['pred_boxes'])} detections")

    # ------------------------------------------------------------------
    # 2. Export to OpenVINO
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Exporting to OpenVINO...")
    print("=" * 60)
    t0 = time.perf_counter()
    ov_dir = export_yoloe_openvino(
        model_name=MODEL_NAME,
        classes=CLASSES,
        output_dir=EXPORT_DIR,
        imgsz=IMGSZ,
        half=False,
    )
    export_time = time.perf_counter() - t0
    print(f"  export:    {export_time:.3f}s → {ov_dir}")

    # ------------------------------------------------------------------
    # 3. YOLOE OpenVINO
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("YOLOE OpenVINO")
    print("=" * 60)
    # The OV model does class-level detection (not visual-prompt matching),
    # so it tends to find more candidates.  A higher threshold helps.
    ov_model = YOLOEOpenVINO(model_dir=ov_dir, device="cpu", confidence_threshold=0.5)

    t0 = time.perf_counter()
    ov_model.fit(ref_sample)
    ov_fit_time = time.perf_counter() - t0
    print(f"  fit():     {ov_fit_time:.3f}s")

    ov_results = []
    t0 = time.perf_counter()
    for ts in target_samples:
        ov_results.extend(ov_model.predict(ts))
    ov_pred_time = time.perf_counter() - t0
    print(f"  predict(): {ov_pred_time:.3f}s  ({len(target_samples)} images)")
    for i, r in enumerate(ov_results):
        print(f"    image {i}: {len(r['pred_boxes'])} detections")

    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Timing Summary")
    print("=" * 60)
    print(f"  {'':20s} {'PyTorch':>10s}  {'OpenVINO':>10s}")
    print(f"  {'fit()':20s} {pt_fit_time:>10.3f}s {ov_fit_time:>10.3f}s")
    print(f"  {'predict() x3':20s} {pt_pred_time:>10.3f}s {ov_pred_time:>10.3f}s")
    print(f"  {'total':20s} {pt_fit_time+pt_pred_time:>10.3f}s {ov_fit_time+ov_pred_time:>10.3f}s")

    # ------------------------------------------------------------------
    # 5. Comparison grid: 2 rows (PyTorch / OpenVINO) × 3 target images
    # ------------------------------------------------------------------
    row_pt, row_ov = [], []
    for i, timg in enumerate(target_imgs):
        bgr = cv2.cvtColor(timg.permute(1, 2, 0).numpy().copy(), cv2.COLOR_RGB2BGR)
        name = TARGET_IMAGES[i].stem
        row_pt.append(draw_results(bgr, pt_results[i], f"PT {name}"))
        row_ov.append(draw_results(bgr, ov_results[i], f"OV {name}"))

    # Normalise heights within each row
    h = max(img.shape[0] for img in row_pt + row_ov)
    row_pt = [resize_to_height(img, h) for img in row_pt]
    row_ov = [resize_to_height(img, h) for img in row_ov]

    grid = np.vstack([np.hstack(row_pt), np.hstack(row_ov)])
    cv2.imwrite(str(OUTPUT_IMAGE), grid)
    print(f"\nComparison image saved to: {OUTPUT_IMAGE}")


if __name__ == "__main__":
    main()
