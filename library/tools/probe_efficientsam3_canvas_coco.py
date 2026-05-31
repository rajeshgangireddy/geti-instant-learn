# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""COCO probe: EfficientSAM3 CANVAS mode on Potatoes (1 cat) + Nuts (2 cats).

Counterpart to ``probe_efficientsam3_ve_coco.py``. Same dataset loader, same
reference selection, same scorer (P/R/F1 at IoU 0.5). Only the model's
``prompt_mode`` and ``canvas_config`` change.

Sweeps:
  - canvas split_ratio ∈ {0.3, 0.5}
  - canvas share_vision ∈ {'auto', False}        (only meaningful for >1 category)
  - reference text         ∈ {no-text, with-text}

Prints one row per (dataset, split_ratio, share_vision, text) configuration.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch

from benchmark_efficient_sam3 import _load_image_tensor, match_and_score  # noqa: PLC2701
from probe_efficientsam3_ve_coco import (  # noqa: PLC2701
    load_coco,
    pick_reference_image,
)
from pycocotools.coco import COCO

from instantlearn.data import Sample
from instantlearn.models import EfficientSAM3
from instantlearn.models.sam3.sam3 import CanvasConfig, Sam3PromptMode

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)
logger = logging.getLogger("canvas_probe_coco")
logger.setLevel(logging.INFO)

DATA_ROOT = Path("/home/rgangire/workspace/data/prompt/geti_datasets/COCO")
DATASETS = {
    "potatoes": DATA_ROOT / "Potatoes",
    "nuts": DATA_ROOT / "Nuts",
}

SPLIT_RATIOS = [0.3, 0.5]
SHARE_VISION_OPTIONS: list[str | bool] = ["auto", False]
TEXT_OPTIONS = [("no-text", False), ("with-text", True)]
MAX_REFS_PER_CAT = 1  # CANVAS is heavy: one ref bbox per category (1-shot FSS setup).


def build_canvas_reference_samples(
    coco: COCO,
    image_root: Path,
    ref_img_id: int,
    cat_id_to_index: dict[int, int],
    with_text: bool,
    cat_names: list[str],
) -> list[Sample]:
    """Build ONE Sample per reference bbox, capped at MAX_REFS_PER_CAT per category.

    SAM3 CANVAS groups references by category, but `group_references_by_category`
    reads only ``bboxes[0]`` / ``category_ids[0]`` / ``categories[0]`` per Sample,
    so multi-category fits require a *list* of single-bbox Samples (one per box).
    """
    img_info = coco.loadImgs(ref_img_id)[0]
    img_path = image_root / img_info["file_name"]
    ann_ids = coco.getAnnIds(imgIds=ref_img_id)
    anns = coco.loadAnns(ann_ids)

    samples: list[Sample] = []
    per_cat_count: dict[int, int] = {}
    for a in anns:
        idx = cat_id_to_index[a["category_id"]]
        if per_cat_count.get(idx, 0) >= MAX_REFS_PER_CAT:
            continue
        per_cat_count[idx] = per_cat_count.get(idx, 0) + 1
        x, y, w, h = a["bbox"]
        name = cat_names[idx] if with_text else "visual"
        samples.append(
            Sample(
                image_path=str(img_path),
                bboxes=np.array([[x, y, x + w, y + h]], dtype=np.float32),
                category_ids=np.array([idx], dtype=np.int64),
                categories=[name],
            ),
        )
    return samples


def evaluate(
    model: EfficientSAM3,
    coco: COCO,
    image_root: Path,
    target_img_ids: list[int],
    cat_id_to_index: dict[int, int],
) -> dict[str, float | int]:
    """Run predict on each target image, accumulate per-class TP/FP/FN + latency."""
    totals = {"tp": 0, "fp": 0, "fn": 0, "n_pred": 0, "n_gt": 0}
    t0 = time.perf_counter()
    for img_id in target_img_ids:
        img_info = coco.loadImgs(img_id)[0]
        img_path = image_root / img_info["file_name"]
        img_t = _load_image_tensor(img_path)
        pred = model.predict(Sample(image=img_t))[0]

        boxes_full = pred["pred_boxes"].detach().cpu().numpy().astype(np.float32)
        pred_boxes = boxes_full[:, :4] if boxes_full.size > 0 else np.zeros((0, 4), dtype=np.float32)
        pred_labels = pred["pred_labels"].detach().cpu().numpy().astype(np.int64)

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        gt_boxes = np.array(
            [[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in anns],
            dtype=np.float32,
        ) if anns else np.zeros((0, 4), dtype=np.float32)
        gt_labels = np.array([cat_id_to_index[a["category_id"]] for a in anns], dtype=np.int64)

        tp, fp, fn = match_and_score(pred_boxes, pred_labels, gt_boxes, gt_labels, iou_threshold=0.5)
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
        totals["n_pred"] += len(pred_boxes)
        totals["n_gt"] += len(gt_boxes)

    elapsed = time.perf_counter() - t0
    n = max(1, len(target_img_ids))
    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {
        **totals,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "ms_per_img": round(elapsed / n * 1000, 1),
    }


def run_dataset(name: str, dataset_dir: Path, model: EfficientSAM3) -> list[dict]:
    """Sweep all configs on one dataset; return a list of result rows."""
    logger.info("=" * 80)
    logger.info("Dataset: %s", name)

    coco, image_root, image_ids, cat_names, cat_id_to_index = load_coco(dataset_dir)
    coco_cat_ids = sorted(coco.getCatIds())
    logger.info("Categories (%d): %s", len(cat_names), cat_names)

    ref_img_id = pick_reference_image(coco, image_ids, coco_cat_ids)
    target_img_ids = [i for i in image_ids if i != ref_img_id]
    ref_info = coco.loadImgs(ref_img_id)[0]
    n_ref_boxes = len(coco.getAnnIds(imgIds=ref_img_id))
    logger.info(
        "Reference: %s (id=%d, %d boxes); %d target images",
        ref_info["file_name"], ref_img_id, n_ref_boxes, len(target_img_ids),
    )

    rows: list[dict] = []
    is_multi_cat = len(cat_names) > 1
    for split_ratio in SPLIT_RATIOS:
        share_options = SHARE_VISION_OPTIONS if is_multi_cat else ["auto"]  # share_vision is a no-op for 1 cat
        for share_vision in share_options:
            # Mutate canvas_config in place; SAM3 reads it inside _predict_canvas.
            model.canvas_config = CanvasConfig(
                split_ratio=split_ratio,
                crop_padding=2.0,
                share_vision=share_vision,
            )
            for text_label, with_text in TEXT_OPTIONS:
                ref_samples = build_canvas_reference_samples(
                    coco, image_root, ref_img_id, cat_id_to_index, with_text, cat_names,
                )
                model.fit(ref_samples)
                metrics = evaluate(model, coco, image_root, target_img_ids, cat_id_to_index)
                row = {
                    "dataset": name,
                    "split_ratio": split_ratio,
                    "share_vision": share_vision,
                    "text": text_label,
                    **metrics,
                }
                rows.append(row)
                logger.info(
                    "  ratio=%.1f share=%-5s %s | tp=%3d fp=%3d fn=%3d | P=%.3f R=%.3f F1=%.3f | %sms/img",
                    split_ratio, str(share_vision), text_label,
                    metrics["tp"], metrics["fp"], metrics["fn"],
                    metrics["precision"], metrics["recall"], metrics["f1"],
                    metrics["ms_per_img"],
                )
    return rows


def print_summary(rows: list[dict]) -> None:
    """Print a compact table of all results across datasets."""
    logger.info("=" * 80)
    logger.info("SUMMARY (best F1 per dataset highlighted)")
    logger.info(
        "  %-10s %-6s %-6s %-10s %-7s %-7s %-7s",
        "dataset", "ratio", "share", "text", "P", "R", "F1",
    )
    by_ds: dict[str, list[dict]] = {}
    for r in rows:
        by_ds.setdefault(r["dataset"], []).append(r)
    for ds_name, ds_rows in by_ds.items():
        best_f1 = max(r["f1"] for r in ds_rows)
        for r in ds_rows:
            star = "  *" if r["f1"] == best_f1 else "   "
            logger.info(
                "%s %-10s %-6.1f %-6s %-10s %-7.3f %-7.3f %-7.3f",
                star, r["dataset"], r["split_ratio"], str(r["share_vision"]),
                r["text"], r["precision"], r["recall"], r["f1"],
            )


def main() -> None:
    """Run the CANVAS sweep on both COCO datasets."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    model = EfficientSAM3(
        backbone_type="efficientvit",
        variant="b1",
        device=device,
        prompt_mode=Sam3PromptMode.CANVAS,
        canvas_config=CanvasConfig(split_ratio=0.3, crop_padding=2.0),
        confidence_threshold=0.4,
    )

    all_rows: list[dict] = []
    for name, dataset_dir in DATASETS.items():
        all_rows.extend(run_dataset(name, dataset_dir, model))

    print_summary(all_rows)
    logger.info("Done.")


if __name__ == "__main__":
    main()
