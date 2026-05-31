# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Probe EfficientSAM3 VISUAL_EXEMPLAR for the two known SAM3 weaknesses.

Mirrors `library/examples/sam3_canvas_example.ipynb` so the comparison is
apples-to-apples. The same reference frame, target frames, and bboxes are
reused; only the model + prompt mode change.

The two questions:

1. **Category-name sensitivity.** Does omitting a real category name (the
   default "visual" placeholder) hurt accuracy vs. providing the true name
   ("SUV")?
2. **Multi-category degradation.** Does adding a second category
   (SUV + Motorcycle) confuse the model in either text/no-text setting?

Each cell prints per-target detection counts and per-image visualizations are
saved to ``library/examples/output/efficientsam3_visual_exemplar_probe/``.

Run from the library/ dir, e.g. ``.cuda/bin/python tools/probe_efficientsam3_visual_exemplar.py``.
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
from instantlearn.models.sam3.sam3 import Sam3PromptMode
from instantlearn.visualizer import render_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)
logger = logging.getLogger("ve_probe")
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Constants — copied verbatim from sam3_canvas_example.ipynb
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]  # library/
ASSETS = REPO_ROOT / "examples" / "assets" / "edsa_vehicles"
OUT_DIR = REPO_ROOT / "examples" / "output" / "efficientsam3_visual_exemplar_probe"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REF_PATH = ASSETS / "EDSA-Orense-2_mp4-0077.jpg"
TARGET_PATHS = [
    ASSETS / "EDSA-Orense-2_mp4-0151.jpg",
    ASSETS / "EDSA-Orense-2_mp4-0164.jpg",
]

# Roboflow annotations are at 1920x1080; the image files are 800x600.
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


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

COLOR_MAP_SINGLE = {0: [30, 144, 255]}
COLOR_MAP_MULTI = {0: [30, 144, 255], 1: [255, 99, 71]}  # SUV blue, moto red


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
    """Save a grid: rows = list of (cell_title, list_of_preds_per_target)."""
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


# ---------------------------------------------------------------------------
# Experiment cells
# ---------------------------------------------------------------------------


def make_targets() -> list[Sample]:
    """Build a fresh list of target Samples (one per target image)."""
    return [Sample(image_path=str(p)) for p in TARGET_PATHS]


def run_single_category(model: EfficientSAM3) -> tuple[list[dict], list[dict]]:
    """SUV only: no text ("visual" placeholder) vs with text ("SUV")."""
    # No text — let SAM3 default the placeholder to "visual"
    ref_no_text = Sample(image_path=str(REF_PATH), bboxes=SUV_BBOX, category_ids=np.array([0]))
    model.fit(ref_no_text)
    preds_no_text = model.predict(make_targets())

    # With text
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
    """SUV + Motorcycle: no text vs with text."""
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the four-cell sweep and print + save results."""
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = "xpu"
    else:
        device = "cpu"
    logger.info("Using device: %s", device)

    # Use the efficientvit/b1 backbone — same one we benchmarked. Default
    # confidence threshold and prompt mode locked to VISUAL_EXEMPLAR.
    model = EfficientSAM3(
        backbone_type="efficientvit",
        variant="b1",
        device=device,
        prompt_mode=Sam3PromptMode.VISUAL_EXEMPLAR,
        confidence_threshold=0.4,
    )

    # ------ Single category ------
    logger.info("=" * 70)
    logger.info("CELL 1+2: Single category (SUV)")
    suv_no_text, suv_with_text = run_single_category(model)
    for path, p_nt, p_wt in zip(TARGET_PATHS, suv_no_text, suv_with_text, strict=False):
        cnt_nt = _per_class_counts(p_nt, num_classes=1)
        cnt_wt = _per_class_counts(p_wt, num_classes=1)
        logger.info(
            "%s | no-text(SUV=visual)=%d | with-text(SUV)=%d",
            path.name, cnt_nt[0], cnt_wt[0],
        )

    save_grid(
        rows=[
            ('SUV only\nno text\n(placeholder "visual")', suv_no_text),
            ('SUV only\nwith text\ncategory="SUV"', suv_with_text),
        ],
        color_map=COLOR_MAP_SINGLE,
        title="EfficientSAM3 VISUAL_EXEMPLAR — single category, text vs no text",
        out_name="cell_single_category.png",
    )

    # ------ Multi category ------
    logger.info("=" * 70)
    logger.info("CELL 3+4: Multi category (SUV + Motorcycle)")
    multi_no_text, multi_with_text = run_multi_category(model)
    for path, p_nt, p_wt in zip(TARGET_PATHS, multi_no_text, multi_with_text, strict=False):
        cnt_nt = _per_class_counts(p_nt, num_classes=2)
        cnt_wt = _per_class_counts(p_wt, num_classes=2)
        logger.info(
            "%s | no-text  SUV=%d Moto=%d | with-text  SUV=%d Moto=%d",
            path.name, cnt_nt[0], cnt_nt[1], cnt_wt[0], cnt_wt[1],
        )

    save_grid(
        rows=[
            ('SUV + Moto\nno text\n(both = "visual")', multi_no_text),
            ('SUV + Moto\nwith text\n("SUV" + "Motorcycle")', multi_with_text),
        ],
        color_map=COLOR_MAP_MULTI,
        title="EfficientSAM3 VISUAL_EXEMPLAR — multi category, text vs no text",
        out_name="cell_multi_category.png",
    )

    logger.info("Done. Outputs in %s", OUT_DIR)


if __name__ == "__main__":
    main()
