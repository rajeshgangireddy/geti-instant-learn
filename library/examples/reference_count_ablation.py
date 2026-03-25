# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Reference count ablation experiment.

Sweeps the number of instance masks provided as reference (k = 1, 2, ..., N)
and measures how detection quality scales. Repeats the experiment across
multiple reference images (NUM_RUNS) to ensure the pattern is robust.

For each run (reference image) × k (mask count), fits the model, predicts on
all remaining (GT-annotated) target images, and computes pixel-level metrics
(IoU, F1, Precision, Recall) plus detection statistics.

Outputs:
    - Per-run visualization folders: ``outputs/<name>/run{r}_<image>/k{k}_of_{N}/``
    - Per-run summary CSV:           ``outputs/<name>/run{r}_<image>/summary.csv``
    - Combined summary CSV:          ``outputs/<name>/all_runs_summary.csv``
    - Averaged summary CSV:          ``outputs/<name>/averaged_summary.csv``

Usage:
    cd library
    python examples/reference_count_ablation.py
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import ndimage

from instantlearn.data import Sample
from instantlearn.data.utils import read_image
from instantlearn.models import Matcher
from instantlearn.utils.metrics import SegmentationMetrics
from instantlearn.visualizer import render_predictions, setup_colors

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration (edit these) ────────────────────────────────────────────────
DATASET_ROOT = Path("/home/rgangire/workspace/data/prompt/geti_datasets/COCO")
DEVICE = "cuda"  # "cuda", "xpu", "cpu"
OUTPUT_DIR = Path("outputs/reference_count_ablation")
MODEL = Matcher(device=DEVICE)  # Matcher, SoftMatcher, PerDino, or SAM3
MAX_TARGETS = None  # Limit target images, or None for all
NUM_RUNS = 5  # Number of reference images to rotate through
# ──────────────────────────────────────────────────────────────────────────────


def load_coco_dataset(dataset_root: Path) -> tuple[list[dict], list[dict], dict[int, str]]:
    """Load COCO annotations and return (images, annotations, category_map)."""
    ann_file = dataset_root / "annotations" / "instances_default.json"
    with open(ann_file) as f:
        coco = json.load(f)

    cat_map = {c["id"]: c["name"] for c in coco["categories"]}
    return coco["images"], coco["annotations"], cat_map


def coco_anns_to_mask(
    anns: list[dict],
    img_h: int,
    img_w: int,
) -> tuple[np.ndarray, list[int]]:
    """Convert COCO polygon annotations to binary masks per instance.

    Returns:
        masks: (N, H, W) uint8 array with 1 for foreground
        cat_ids: list of category IDs per mask
    """
    masks = []
    cat_ids = []
    for ann in anns:
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        for seg in ann["segmentation"]:
            pts = np.array(seg, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(mask, [pts], 1)
        masks.append(mask)
        cat_ids.append(ann["category_id"])
    return np.stack(masks) if masks else np.zeros((0, img_h, img_w), dtype=np.uint8), cat_ids


def sort_masks_by_area(masks: np.ndarray) -> np.ndarray:
    """Return indices that sort masks by area, largest first."""
    areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
    return np.argsort(-areas)


def compute_detection_stats(pred: dict[str, torch.Tensor]) -> dict[str, float]:
    """Compute detection statistics from a single prediction."""
    masks = pred["pred_masks"]
    scores = pred["pred_scores"]
    n_det = masks.shape[0]

    if n_det == 0:
        return {"n_detections": 0, "connected_components": 0, "mask_area": 0, "avg_score": 0.0}

    # Merge all predicted masks into one binary mask for CC counting
    merged = masks.any(dim=0).cpu().numpy().astype(np.uint8)
    _, n_cc = ndimage.label(merged)
    area = int(merged.sum())
    avg_score = scores.mean().item()

    return {"n_detections": n_det, "connected_components": n_cc, "mask_area": area, "avg_score": avg_score}


def build_gt_one_hot(
    masks: np.ndarray,
    cat_ids: list[int],
    cat_id_to_label: dict[int, int],
    num_classes: int,
    h: int,
    w: int,
) -> torch.Tensor:
    """Build ground truth one-hot tensor (1, C, H, W) from instance masks."""
    gt = torch.zeros(1, num_classes, h, w, dtype=torch.bool)
    for mask, cid in zip(masks, cat_ids, strict=False):
        if cid in cat_id_to_label:
            label = cat_id_to_label[cid]
            gt[0, label] |= torch.from_numpy(mask).bool()
    return gt


def build_pred_one_hot(
    pred: dict[str, torch.Tensor],
    num_classes: int,
    h: int,
    w: int,
) -> torch.Tensor:
    """Build prediction one-hot tensor (1, C, H, W) from model output."""
    one_hot = torch.zeros(1, num_classes, h, w, dtype=torch.bool)
    for mask, label in zip(pred["pred_masks"], pred["pred_labels"], strict=False):
        label_id = label.item()
        if 0 <= label_id < num_classes:
            one_hot[0, label_id] |= mask.bool().cpu()
    return one_hot


def compute_per_image_metrics(
    pred_one_hot: torch.Tensor,
    gt_one_hot: torch.Tensor,
    num_classes: int,
) -> dict[str, float]:
    """Compute IoU, F1, Precision, Recall for a single image."""
    metrics = SegmentationMetrics(num_classes=num_classes, device=torch.device("cpu"))
    metrics.update(pred_one_hot, gt_one_hot)
    computed = metrics.compute()
    return {
        "iou": computed["iou"].mean().item(),
        "f1": computed["f1"].mean().item(),
        "precision": computed["precision"].mean().item(),
        "recall": computed["recall"].mean().item(),
    }


def run_single_ablation(
    run_idx: int,
    ref_info: dict,
    target_img_infos: list[dict],
    images_dir: Path,
    img_id_to_anns: dict[int, list[dict]],
    cat_map: dict[int, str],
    model: object,
    output_dir: Path,
) -> list[dict]:
    """Run ablation for a single reference image, return summary rows."""
    ref_name = Path(ref_info["file_name"]).stem
    run_dir = output_dir / f"run{run_idx}_{ref_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    ref_path = images_dir / ref_info["file_name"]
    ref_h, ref_w = ref_info["height"], ref_info["width"]
    ref_anns = img_id_to_anns.get(ref_info["id"], [])
    all_ref_masks, all_ref_cat_ids = coco_anns_to_mask(ref_anns, ref_h, ref_w)
    n_total = all_ref_masks.shape[0]

    if n_total == 0:
        logger.warning("Run %d: %s has no annotations, skipping", run_idx, ref_info["file_name"])
        return []

    # Sort by area (largest first)
    order = sort_masks_by_area(all_ref_masks)
    all_ref_masks = all_ref_masks[order]
    all_ref_cat_ids = [all_ref_cat_ids[i] for i in order]

    logger.info(
        "Run %d: reference=%s (%dx%d) with %d instance masks",
        run_idx,
        ref_info["file_name"],
        ref_w,
        ref_h,
        n_total,
    )

    # Category mapping
    unique_cats = sorted(set(all_ref_cat_ids))
    cat_id_to_label = {cid: i for i, cid in enumerate(unique_cats)}
    label_to_name = {cat_id_to_label[cid]: cat_map[cid] for cid in unique_cats}
    num_classes = len(unique_cats)
    color_map = setup_colors(label_to_name)

    # Precompute target GT one-hot tensors (excluding the reference image)
    target_gts: list[dict] = []
    for tgt_info in target_img_infos:
        if tgt_info["id"] == ref_info["id"]:
            continue
        tgt_h, tgt_w = tgt_info["height"], tgt_info["width"]
        tgt_anns = img_id_to_anns.get(tgt_info["id"], [])
        tgt_masks, tgt_cat_ids = coco_anns_to_mask(tgt_anns, tgt_h, tgt_w)
        gt_one_hot = build_gt_one_hot(tgt_masks, tgt_cat_ids, cat_id_to_label, num_classes, tgt_h, tgt_w)
        target_gts.append({"info": tgt_info, "gt_one_hot": gt_one_hot})

    summary_rows: list[dict] = []
    detail_rows: list[dict] = []

    for k in range(1, n_total + 1):
        logger.info("  k=%d/%d", k, n_total)

        exp_dir = run_dir / f"k{k}_of_{n_total}"
        exp_dir.mkdir(parents=True, exist_ok=True)

        ref_masks_k = all_ref_masks[:k]
        ref_cat_ids_k = all_ref_cat_ids[:k]
        ref_labels_k = [cat_id_to_label[cid] for cid in ref_cat_ids_k]

        sample = Sample(
            image_path=str(ref_path),
            masks=torch.from_numpy(ref_masks_k),
            categories=[cat_map[cid] for cid in ref_cat_ids_k],
            category_ids=ref_labels_k,
        )
        model.fit(sample)

        # Save reference visualization
        ref_image_rgb = read_image(ref_path, as_tensor=False)
        ref_pred = {
            "pred_masks": torch.from_numpy(ref_masks_k),
            "pred_labels": torch.tensor(ref_labels_k),
        }
        ref_vis = render_predictions(ref_image_rgb, ref_pred, color_map, show_scores=False)
        cv2.imwrite(str(exp_dir / f"00_ref_{ref_info['file_name']}"), cv2.cvtColor(ref_vis, cv2.COLOR_RGB2BGR))

        agg_metrics = SegmentationMetrics(num_classes=num_classes, device=torch.device("cpu"))
        agg_stats = {"n_detections": [], "connected_components": [], "mask_area": [], "avg_score": []}

        for i, tgt_data in enumerate(target_gts, start=1):
            tgt_info = tgt_data["info"]
            tgt_path = images_dir / tgt_info["file_name"]
            tgt_h, tgt_w = tgt_info["height"], tgt_info["width"]

            predictions = model.predict(Sample(image_path=str(tgt_path)))
            pred = predictions[0]

            stats = compute_detection_stats(pred)
            for key in agg_stats:
                agg_stats[key].append(stats[key])

            pred_one_hot = build_pred_one_hot(pred, num_classes, tgt_h, tgt_w)
            gt_one_hot = tgt_data["gt_one_hot"]
            agg_metrics.update(pred_one_hot, gt_one_hot)

            img_metrics = compute_per_image_metrics(pred_one_hot, gt_one_hot, num_classes)

            detail_rows.append({
                "run": run_idx,
                "ref_image": ref_info["file_name"],
                "k": k,
                "image": tgt_info["file_name"],
                "n_detections": stats["n_detections"],
                "connected_components": stats["connected_components"],
                "mask_area": stats["mask_area"],
                "avg_score": round(stats["avg_score"], 4),
                "iou": round(img_metrics["iou"], 4),
                "f1": round(img_metrics["f1"], 4),
                "precision": round(img_metrics["precision"], 4),
                "recall": round(img_metrics["recall"], 4),
            })

            tgt_image_rgb = read_image(tgt_path, as_tensor=False)
            vis = render_predictions(tgt_image_rgb, pred, color_map)
            cv2.imwrite(str(exp_dir / f"{i:02d}_{tgt_info['file_name']}"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

        computed = agg_metrics.compute()
        n_targets = len(target_gts)
        row = {
            "run": run_idx,
            "ref_image": ref_info["file_name"],
            "n_masks_available": n_total,
            "k": k,
            "iou": round(computed["iou"].mean().item(), 4),
            "f1": round(computed["f1"].mean().item(), 4),
            "precision": round(computed["precision"].mean().item(), 4),
            "recall": round(computed["recall"].mean().item(), 4),
            "avg_connected_components": round(sum(agg_stats["connected_components"]) / n_targets, 1),
            "avg_mask_area": round(sum(agg_stats["mask_area"]) / n_targets, 0),
            "avg_score": round(sum(agg_stats["avg_score"]) / n_targets, 4),
            "avg_detections": round(sum(agg_stats["n_detections"]) / n_targets, 1),
        }

        if summary_rows:
            prev = summary_rows[-1]
            row["delta_iou"] = round(row["iou"] - prev["iou"], 4)
            row["delta_f1"] = round(row["f1"] - prev["f1"], 4)
        else:
            row["delta_iou"] = 0.0
            row["delta_f1"] = 0.0

        summary_rows.append(row)
        logger.info(
            "    IoU=%.4f (Δ%+.4f) | F1=%.4f (Δ%+.4f) | CC=%.1f | Det=%.1f",
            row["iou"],
            row["delta_iou"],
            row["f1"],
            row["delta_f1"],
            row["avg_connected_components"],
            row["avg_detections"],
        )

    # Write per-run summary
    if summary_rows:
        run_summary_path = run_dir / "summary.csv"
        with open(run_summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)

    # Write per-run detail
    if detail_rows:
        run_detail_path = run_dir / "per_image_details.csv"
        with open(run_detail_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=detail_rows[0].keys())
            writer.writeheader()
            writer.writerows(detail_rows)

    return summary_rows


def main() -> None:
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    images, annotations, cat_map = load_coco_dataset(DATASET_ROOT)
    logger.info("Dataset: %d images, %d annotations, categories: %s", len(images), len(annotations), cat_map)

    img_id_to_anns: dict[int, list[dict]] = {}
    for ann in annotations:
        img_id_to_anns.setdefault(ann["image_id"], []).append(ann)

    images_dir = DATASET_ROOT / "images"
    if (images_dir / "default").is_dir():
        images_dir = images_dir / "default"

    num_runs = min(NUM_RUNS, len(images))
    logger.info("Running %d ablation runs (one per reference image)", num_runs)

    model = MODEL
    all_summary_rows: list[dict] = []

    for run_idx in range(1, num_runs + 1):
        logger.info("=" * 70)
        logger.info("RUN %d/%d", run_idx, num_runs)
        logger.info("=" * 70)

        ref_info = images[run_idx - 1]
        # All other images are targets
        target_img_infos = [img for img in images if img["id"] != ref_info["id"]]
        if MAX_TARGETS:
            target_img_infos = target_img_infos[:MAX_TARGETS]

        rows = run_single_ablation(
            run_idx=run_idx,
            ref_info=ref_info,
            target_img_infos=target_img_infos,
            images_dir=images_dir,
            img_id_to_anns=img_id_to_anns,
            cat_map=cat_map,
            model=model,
            output_dir=output_dir,
        )
        all_summary_rows.extend(rows)

    # ── Write combined summary CSV ────────────────────────────────────────────
    if all_summary_rows:
        combined_path = output_dir / "all_runs_summary.csv"
        with open(combined_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_summary_rows)
        logger.info("Combined summary saved to %s", combined_path)

    # ── Compute averaged summary across runs ──────────────────────────────────
    # Group by k, average numeric columns across runs
    from collections import defaultdict

    k_groups: dict[int, list[dict]] = defaultdict(list)
    for row in all_summary_rows:
        k_groups[row["k"]].append(row)

    avg_fields = [
        "iou",
        "f1",
        "precision",
        "recall",
        "avg_connected_components",
        "avg_mask_area",
        "avg_score",
        "avg_detections",
    ]

    averaged_rows: list[dict] = []
    for k in sorted(k_groups.keys()):
        group = k_groups[k]
        avg_row: dict = {"k": k, "n_runs": len(group)}
        for field in avg_fields:
            values = [r[field] for r in group]
            avg_row[field] = round(sum(values) / len(values), 4)
            avg_row[f"{field}_std"] = round(
                (sum((v - avg_row[field]) ** 2 for v in values) / len(values)) ** 0.5,
                4,
            )
        if averaged_rows:
            prev = averaged_rows[-1]
            avg_row["delta_iou"] = round(avg_row["iou"] - prev["iou"], 4)
            avg_row["delta_f1"] = round(avg_row["f1"] - prev["f1"], 4)
        else:
            avg_row["delta_iou"] = 0.0
            avg_row["delta_f1"] = 0.0
        averaged_rows.append(avg_row)

    if averaged_rows:
        avg_path = output_dir / "averaged_summary.csv"
        with open(avg_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=averaged_rows[0].keys())
            writer.writeheader()
            writer.writerows(averaged_rows)
        logger.info("Averaged summary saved to %s", avg_path)

    # ── Print averaged summary table ──────────────────────────────────────────
    logger.info("")
    logger.info("=" * 100)
    logger.info("AVERAGED SUMMARY (across %d reference images)", num_runs)
    logger.info("=" * 100)
    header = (
        f"{'k':>3} {'runs':>4} | {'IoU':>7} {'±std':>7} {'ΔIoU':>7} | "
        f"{'F1':>7} {'±std':>7} {'ΔF1':>7} | "
        f"{'Prec':>7} {'Rec':>7} | {'CC':>6} {'Det':>5}"
    )
    logger.info(header)
    logger.info("-" * 100)
    for r in averaged_rows:
        logger.info(
            "%3d %4d | %7.4f %7.4f %+7.4f | %7.4f %7.4f %+7.4f | %7.4f %7.4f | %6.1f %5.1f",
            r["k"],
            r["n_runs"],
            r["iou"],
            r["iou_std"],
            r["delta_iou"],
            r["f1"],
            r["f1_std"],
            r["delta_f1"],
            r["precision"],
            r["recall"],
            r["avg_connected_components"],
            r["avg_detections"],
        )
    logger.info("=" * 100)

    # ── Print per-run summary ─────────────────────────────────────────────────
    logger.info("")
    logger.info("PER-RUN RESULTS:")
    for run_idx in range(1, num_runs + 1):
        run_rows = [r for r in all_summary_rows if r["run"] == run_idx]
        if not run_rows:
            continue
        ref_name = run_rows[0]["ref_image"]
        n_masks = run_rows[0]["n_masks_available"]
        first_iou = run_rows[0]["iou"]
        last_iou = run_rows[-1]["iou"]
        logger.info(
            "  Run %d (%s, %d masks): IoU %.4f → %.4f (Δ%+.4f)",
            run_idx,
            ref_name,
            n_masks,
            first_iou,
            last_iou,
            last_iou - first_iou,
        )

    logger.info("All results saved to %s", output_dir)


if __name__ == "__main__":
    main()
