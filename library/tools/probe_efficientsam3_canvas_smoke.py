# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Smoke test: run EfficientSAM3 in CANVAS mode on the EDSA vehicles assets.

Mirrors ``library/examples/sam3_canvas_example.ipynb`` — same reference frame,
same target frames, same bboxes — only the model is EfficientSAM3 with
``prompt_mode=CANVAS`` instead of SAM3 (full).

Goals:
1. Confirm CANVAS now runs end-to-end on EfficientSAM3 (no shape/dtype errors)
   after the rejection guard was removed.
2. Produce a side-by-side PNG to eyeball whether detections look plausible.

Outputs:
    library/examples/output/efficientsam3_canvas_smoke/cell_{single,multi}_category.png
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from instantlearn.data import Sample
from instantlearn.models import EfficientSAM3
from instantlearn.models.sam3.sam3 import CanvasConfig, Sam3PromptMode
from instantlearn.visualizer import render_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)
logger = logging.getLogger("canvas_smoke")
logger.setLevel(logging.INFO)

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS = REPO_ROOT / "examples" / "assets" / "edsa_vehicles"
OUT_DIR = REPO_ROOT / "examples" / "output" / "efficientsam3_canvas_smoke"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REF_PATH = ASSETS / "EDSA-Orense-2_mp4-0077.jpg"
TARGET_PATHS = [
    ASSETS / "EDSA-Orense-2_mp4-0151.jpg",
    ASSETS / "EDSA-Orense-2_mp4-0164.jpg",
]

ANN_W, ANN_H = 1920, 1080
IMG_W, IMG_H = 800, 600
SX, SY = IMG_W / ANN_W, IMG_H / ANN_H


def _norm_to_xyxy(cx: float, cy: float, bw: float, bh: float) -> list[float]:
    return [
        (cx - bw / 2) * SX,
        (cy - bh / 2) * SY,
        (cx + bw / 2) * SX,
        (cy + bh / 2) * SY,
    ]


SUV_BBOX = np.array([_norm_to_xyxy(1176.55, 495.96, 451.33, 551.48)], dtype=np.float32)
MOTO_BBOX = np.array([_norm_to_xyxy(578.85, 504.60, 194.74, 352.75)], dtype=np.float32)

COLOR_MAP_SINGLE = {0: [30, 144, 255]}
COLOR_MAP_MULTI = {0: [30, 144, 255], 1: [255, 99, 71]}


def _load_rgb(path: Path) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def _per_class_counts(pred: dict, num_classes: int) -> dict[int, int]:
    labels = pred.get("pred_labels")
    if labels is None or len(labels) == 0:
        return dict.fromkeys(range(num_classes), 0)
    labels_np = labels.detach().cpu().numpy()
    return {i: int((labels_np == i).sum()) for i in range(num_classes)}


def save_grid(
    rows: list[tuple[str, list[dict]]],
    color_map: dict[int, list[int]],
    title: str,
    out_name: str,
) -> None:
    n_rows = len(rows)
    n_cols = 1 + len(TARGET_PATHS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for r, (cell_title, preds) in enumerate(rows):
        axes[r, 0].text(0.5, 0.5, cell_title, ha="center", va="center", wrap=True, fontsize=11)
        axes[r, 0].axis("off")
        for c, (path, pred) in enumerate(zip(TARGET_PATHS, preds, strict=False), start=1):
            img_rgb = _load_rgb(path)
            vis = render_predictions(img_rgb, pred, color_map, show_scores=True)
            n = int(pred["pred_boxes"].shape[0])
            axes[r, c].imshow(vis)
            axes[r, c].set_title(f"{path.name} — {n} det(s)", fontsize=9)
            axes[r, c].axis("off")

    fig.suptitle(title, fontsize=13, y=1.005)
    plt.tight_layout()
    out = OUT_DIR / out_name
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out)


def make_targets() -> list[Sample]:
    return [Sample(image_path=str(p)) for p in TARGET_PATHS]


def run_single_category(model: EfficientSAM3) -> tuple[list[dict], list[dict]]:
    ref_no_text = Sample(image_path=str(REF_PATH), bboxes=SUV_BBOX, category_ids=np.array([0]))
    model.fit(ref_no_text)
    preds_no_text = model.predict(make_targets())

    ref_with_text = Sample(
        image_path=str(REF_PATH),
        bboxes=SUV_BBOX,
        categories=["SUV"],
        category_ids=np.array([0]),
    )
    model.fit(ref_with_text)
    preds_with_text = model.predict(make_targets())

    return preds_no_text, preds_with_text


def run_multi_category(model: EfficientSAM3) -> tuple[list[dict], list[dict]]:
    ref_suv = Sample(image_path=str(REF_PATH), bboxes=SUV_BBOX, category_ids=np.array([0]))
    ref_moto = Sample(image_path=str(REF_PATH), bboxes=MOTO_BBOX, category_ids=np.array([1]))
    model.fit([ref_suv, ref_moto])
    preds_no_text = model.predict(make_targets())

    ref_suv_t = Sample(
        image_path=str(REF_PATH),
        bboxes=SUV_BBOX,
        categories=["SUV"],
        category_ids=np.array([0]),
    )
    ref_moto_t = Sample(
        image_path=str(REF_PATH),
        bboxes=MOTO_BBOX,
        categories=["Motorcycle"],
        category_ids=np.array([1]),
    )
    model.fit([ref_suv_t, ref_moto_t])
    preds_with_text = model.predict(make_targets())

    return preds_no_text, preds_with_text


def main() -> None:
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = "xpu"
    else:
        device = "cpu"
    logger.info("Using device: %s", device)

    model = EfficientSAM3(
        backbone_type="efficientvit",
        variant="b1",
        device=device,
        prompt_mode=Sam3PromptMode.CANVAS,
        canvas_config=CanvasConfig(split_ratio=0.3, crop_padding=2.0),
        confidence_threshold=0.4,
    )

    logger.info("=" * 70)
    logger.info("CELL 1+2: Single category (SUV)")
    suv_no_text, suv_with_text = run_single_category(model)
    for path, p_nt, p_wt in zip(TARGET_PATHS, suv_no_text, suv_with_text, strict=False):
        cnt_nt = _per_class_counts(p_nt, num_classes=1)
        cnt_wt = _per_class_counts(p_wt, num_classes=1)
        logger.info("%s | no-text=%d | with-text(SUV)=%d", path.name, cnt_nt[0], cnt_wt[0])

    save_grid(
        rows=[
            ("SUV only\nno text", suv_no_text),
            ('SUV only\nwith text\ncategory="SUV"', suv_with_text),
        ],
        color_map=COLOR_MAP_SINGLE,
        title="EfficientSAM3 CANVAS — single category, text vs no text",
        out_name="cell_single_category.png",
    )

    logger.info("=" * 70)
    logger.info("CELL 3+4: Multi category (SUV + Motorcycle)")
    multi_no_text, multi_with_text = run_multi_category(model)
    for path, p_nt, p_wt in zip(TARGET_PATHS, multi_no_text, multi_with_text, strict=False):
        cnt_nt = _per_class_counts(p_nt, num_classes=2)
        cnt_wt = _per_class_counts(p_wt, num_classes=2)
        logger.info(
            "%s | no-text SUV=%d Moto=%d | with-text SUV=%d Moto=%d",
            path.name, cnt_nt[0], cnt_nt[1], cnt_wt[0], cnt_wt[1],
        )

    save_grid(
        rows=[
            ("SUV + Moto\nno text", multi_no_text),
            ("SUV + Moto\nwith text", multi_with_text),
        ],
        color_map=COLOR_MAP_MULTI,
        title="EfficientSAM3 CANVAS — multi category, text vs no text",
        out_name="cell_multi_category.png",
    )

    logger.info("Done. Outputs in %s", OUT_DIR)


if __name__ == "__main__":
    main()
