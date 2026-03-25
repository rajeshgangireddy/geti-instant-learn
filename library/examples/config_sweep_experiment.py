# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Model configuration sweep experiment.

Tests how different model parameters affect detection count and quality
on a COCO potato dataset. Sweeps confidence threshold, NMS thresholds,
foreground points, SAM backends, encoder sizes, and model variants.

Uses all instance masks from the reference image (k=max) and evaluates
on all remaining target images with GT annotations.

Outputs:
    - Per-config visualization folders
    - Combined results CSV: ``outputs/<name>/config_sweep_results.csv``

Usage:
    cd library
    python examples/config_sweep_experiment.py
"""

from __future__ import annotations

import csv
import gc
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import ndimage

from instantlearn.components.postprocessing import (
    BoxIoMNMS,
    MaskIoMNMS,
    PostProcessorPipeline,
    ScoreFilter,
)
from instantlearn.data import Sample
from instantlearn.data.utils import read_image
from instantlearn.models import Matcher, SoftMatcher
from instantlearn.utils.constants import SAMModelName
from instantlearn.utils.metrics import SegmentationMetrics
from instantlearn.visualizer import render_predictions, setup_colors

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
DATASET_ROOT = Path("/home/rgangire/workspace/data/prompt/geti_datasets/COCO")
DEVICE = "cuda"
OUTPUT_DIR = Path("outputs/config_sweep")
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
    """Convert COCO polygon annotations to binary masks per instance."""
    masks, cat_ids = [], []
    for ann in anns:
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        for seg in ann["segmentation"]:
            pts = np.array(seg, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(mask, [pts], 1)
        masks.append(mask)
        cat_ids.append(ann["category_id"])
    return np.stack(masks) if masks else np.zeros((0, img_h, img_w), dtype=np.uint8), cat_ids


def compute_detection_stats(pred: dict[str, torch.Tensor]) -> dict[str, float]:
    """Compute detection statistics from a single prediction."""
    masks = pred["pred_masks"]
    scores = pred["pred_scores"]
    n_det = masks.shape[0]
    if n_det == 0:
        return {"n_detections": 0, "connected_components": 0, "mask_area": 0, "avg_score": 0.0}
    merged = masks.any(dim=0).cpu().numpy().astype(np.uint8)
    _, n_cc = ndimage.label(merged)
    return {
        "n_detections": n_det,
        "connected_components": n_cc,
        "mask_area": int(merged.sum()),
        "avg_score": scores.mean().item(),
    }


def build_one_hot(
    masks: torch.Tensor | np.ndarray,
    labels: list[int] | torch.Tensor,
    num_classes: int,
    h: int,
    w: int,
) -> torch.Tensor:
    """Build one-hot tensor (1, C, H, W) from masks + labels."""
    one_hot = torch.zeros(1, num_classes, h, w, dtype=torch.bool)
    for mask, label in zip(masks, labels, strict=False):
        lid = label.item() if isinstance(label, torch.Tensor) else int(label)
        if 0 <= lid < num_classes:
            m = torch.from_numpy(mask).bool() if isinstance(mask, np.ndarray) else mask.bool().cpu()
            one_hot[0, lid] |= m
    return one_hot


# ── Experiment configurations ─────────────────────────────────────────────────


def build_experiments() -> list[dict]:
    """Build the list of experiment configurations."""
    experiments = []

    # --- Phase A: Threshold sensitivity ---

    # A1: Confidence threshold sweep
    for ct in [0.1, 0.2, 0.3, 0.38, 0.5]:
        experiments.append({
            "name": f"A1_conf_{ct}",
            "group": "A1_confidence_threshold",
            "model_cls": Matcher,
            "model_kwargs": {"confidence_threshold": ct, "device": DEVICE},
        })

    # A2: NMS IoM threshold sweep
    for iom in [0.3, 0.5, 0.7, 0.9]:
        pp = PostProcessorPipeline([
            ScoreFilter(min_score=0.0),
            MaskIoMNMS(iom_threshold=iom),
            BoxIoMNMS(iom_threshold=iom),
        ])
        experiments.append({
            "name": f"A2_nms_iom_{iom}",
            "group": "A2_nms_iom_threshold",
            "model_cls": Matcher,
            "model_kwargs": {"device": DEVICE, "postprocessor": pp},
        })

    # A3: No NMS at all (only score filter)
    pp_no_nms = PostProcessorPipeline([ScoreFilter(min_score=0.0)])
    experiments.append({
        "name": "A3_no_nms",
        "group": "A3_no_nms",
        "model_cls": Matcher,
        "model_kwargs": {"device": DEVICE, "postprocessor": pp_no_nms},
    })

    # --- Phase B: Model capacity ---

    # B1: Encoder size sweep
    for enc in ["dinov3_small", "dinov3_large"]:
        experiments.append({
            "name": f"B1_encoder_{enc}",
            "group": "B1_encoder_size",
            "model_cls": Matcher,
            "model_kwargs": {"encoder_model": enc, "device": DEVICE},
        })

    # B2: SAM backend sweep
    for sam in [SAMModelName.SAM_HQ_TINY, SAMModelName.SAM_HQ]:
        experiments.append({
            "name": f"B2_sam_{sam.value}",
            "group": "B2_sam_backend",
            "model_cls": Matcher,
            "model_kwargs": {"sam": sam, "device": DEVICE},
        })

    # B3: Foreground points sweep
    for nfg in [20, 40, 80, 120]:
        experiments.append({
            "name": f"B3_fg_points_{nfg}",
            "group": "B3_foreground_points",
            "model_cls": Matcher,
            "model_kwargs": {"num_foreground_points": nfg, "device": DEVICE},
        })

    # --- Phase C: Mask refinement ---

    # C1: Mask refinement toggle
    for refine in [True, False]:
        experiments.append({
            "name": f"C1_refinement_{refine}",
            "group": "C1_mask_refinement",
            "model_cls": Matcher,
            "model_kwargs": {"use_mask_refinement": refine, "device": DEVICE},
        })

    # --- Phase D: Alternative model ---

    # D1: SoftMatcher comparison
    experiments.append({
        "name": "D1_softmatcher",
        "group": "D1_softmatcher",
        "model_cls": SoftMatcher,
        "model_kwargs": {"device": DEVICE},
    })

    return experiments


def run_experiment(
    exp: dict,
    ref_sample: Sample,
    ref_info: dict,
    target_data: list[dict],
    images_dir: Path,
    cat_map: dict[int, str],
    cat_id_to_label: dict[int, int],
    num_classes: int,
    color_map: dict,
    output_dir: Path,
) -> dict:
    """Run a single experiment configuration and return metrics."""
    name = exp["name"]
    logger.info("─" * 60)
    logger.info("Experiment: %s", name)

    exp_dir = output_dir / name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Create model
    model = exp["model_cls"](**exp["model_kwargs"])
    model.fit(ref_sample)

    # Save reference visualization
    ref_image_rgb = read_image(ref_sample.image_path, as_tensor=False)
    ref_pred = {
        "pred_masks": ref_sample.masks
        if isinstance(ref_sample.masks, torch.Tensor)
        else torch.from_numpy(ref_sample.masks),
        "pred_labels": torch.tensor(ref_sample.category_ids),
    }
    ref_vis = render_predictions(ref_image_rgb, ref_pred, color_map, show_scores=False)
    cv2.imwrite(str(exp_dir / f"00_ref_{ref_info['file_name']}"), cv2.cvtColor(ref_vis, cv2.COLOR_RGB2BGR))

    # Predict on targets
    agg_metrics = SegmentationMetrics(num_classes=num_classes, device=torch.device("cpu"))
    stats_lists = {"n_detections": [], "connected_components": [], "mask_area": [], "avg_score": []}
    per_image_rows = []

    for i, tgt in enumerate(target_data, start=1):
        tgt_info = tgt["info"]
        tgt_path = images_dir / tgt_info["file_name"]
        tgt_h, tgt_w = tgt_info["height"], tgt_info["width"]

        predictions = model.predict(Sample(image_path=str(tgt_path)))
        pred = predictions[0]

        stats = compute_detection_stats(pred)
        for key in stats_lists:
            stats_lists[key].append(stats[key])

        pred_oh = build_one_hot(pred["pred_masks"], pred["pred_labels"], num_classes, tgt_h, tgt_w)
        agg_metrics.update(pred_oh, tgt["gt_one_hot"])

        # Per-image metrics
        img_met = SegmentationMetrics(num_classes=num_classes, device=torch.device("cpu"))
        img_met.update(pred_oh, tgt["gt_one_hot"])
        img_comp = img_met.compute()

        per_image_rows.append({
            "experiment": name,
            "image": tgt_info["file_name"],
            "n_detections": stats["n_detections"],
            "connected_components": stats["connected_components"],
            "mask_area": stats["mask_area"],
            "avg_score": round(stats["avg_score"], 4),
            "iou": round(img_comp["iou"].mean().item(), 4),
            "f1": round(img_comp["f1"].mean().item(), 4),
        })

        # Save visualization
        tgt_image_rgb = read_image(tgt_path, as_tensor=False)
        vis = render_predictions(tgt_image_rgb, pred, color_map)
        cv2.imwrite(str(exp_dir / f"{i:02d}_{tgt_info['file_name']}"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    computed = agg_metrics.compute()
    n_targets = len(target_data)

    result = {
        "experiment": name,
        "group": exp["group"],
        "iou": round(computed["iou"].mean().item(), 4),
        "f1": round(computed["f1"].mean().item(), 4),
        "precision": round(computed["precision"].mean().item(), 4),
        "recall": round(computed["recall"].mean().item(), 4),
        "avg_detections": round(sum(stats_lists["n_detections"]) / n_targets, 2),
        "avg_cc": round(sum(stats_lists["connected_components"]) / n_targets, 2),
        "avg_mask_area": round(sum(stats_lists["mask_area"]) / n_targets, 0),
        "avg_score": round(sum(stats_lists["avg_score"]) / n_targets, 4),
    }

    logger.info(
        "  IoU=%.4f | F1=%.4f | Prec=%.4f | Rec=%.4f | Det=%.2f | CC=%.2f",
        result["iou"],
        result["f1"],
        result["precision"],
        result["recall"],
        result["avg_detections"],
        result["avg_cc"],
    )

    # Write per-image detail
    if per_image_rows:
        detail_path = exp_dir / "per_image.csv"
        with open(detail_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=per_image_rows[0].keys())
            writer.writeheader()
            writer.writerows(per_image_rows)

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return result


def main() -> None:
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    images, annotations, cat_map = load_coco_dataset(DATASET_ROOT)
    logger.info("Dataset: %d images, %d annotations, categories: %s", len(images), len(annotations), cat_map)

    img_id_to_anns: dict[int, list[dict]] = {}
    for ann in annotations:
        img_id_to_anns.setdefault(ann["image_id"], []).append(ann)

    images_dir = DATASET_ROOT / "images"
    if (images_dir / "default").is_dir():
        images_dir = images_dir / "default"

    # Use image 0 as reference (same as previous experiments for comparison)
    ref_info = images[0]
    ref_h, ref_w = ref_info["height"], ref_info["width"]
    ref_anns = img_id_to_anns.get(ref_info["id"], [])
    ref_masks, ref_cat_ids = coco_anns_to_mask(ref_anns, ref_h, ref_w)
    ref_path = images_dir / ref_info["file_name"]

    # Sort masks by area (largest first)
    areas = ref_masks.reshape(ref_masks.shape[0], -1).sum(axis=1)
    order = np.argsort(-areas)
    ref_masks = ref_masks[order]
    ref_cat_ids = [ref_cat_ids[i] for i in order]

    # Category mapping
    unique_cats = sorted(set(ref_cat_ids))
    cat_id_to_label = {cid: i for i, cid in enumerate(unique_cats)}
    label_to_name = {cat_id_to_label[cid]: cat_map[cid] for cid in unique_cats}
    num_classes = len(unique_cats)
    color_map = setup_colors(label_to_name)

    ref_labels = [cat_id_to_label[cid] for cid in ref_cat_ids]
    ref_sample = Sample(
        image_path=str(ref_path),
        masks=torch.from_numpy(ref_masks),
        categories=[cat_map[cid] for cid in ref_cat_ids],
        category_ids=ref_labels,
    )

    logger.info("Reference: %s with %d masks (using all)", ref_info["file_name"], ref_masks.shape[0])

    # Precompute target GT
    target_data = []
    for tgt_info in images[1:]:
        tgt_h, tgt_w = tgt_info["height"], tgt_info["width"]
        tgt_anns = img_id_to_anns.get(tgt_info["id"], [])
        tgt_masks, tgt_cat_ids = coco_anns_to_mask(tgt_anns, tgt_h, tgt_w)
        tgt_labels = [cat_id_to_label[cid] for cid in tgt_cat_ids]
        gt_one_hot = build_one_hot(tgt_masks, tgt_labels, num_classes, tgt_h, tgt_w)
        target_data.append({"info": tgt_info, "gt_one_hot": gt_one_hot})

    # Build and run experiments
    experiments = build_experiments()
    logger.info("Running %d experiments ...", len(experiments))

    all_results: list[dict] = []
    for exp in experiments:
        try:
            result = run_experiment(
                exp=exp,
                ref_sample=ref_sample,
                ref_info=ref_info,
                target_data=target_data,
                images_dir=images_dir,
                cat_map=cat_map,
                cat_id_to_label=cat_id_to_label,
                num_classes=num_classes,
                color_map=color_map,
                output_dir=output_dir,
            )
            all_results.append(result)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            logger.error("Experiment %s FAILED: %s", exp["name"], e)
            gc.collect()
            torch.cuda.empty_cache()

    # Write combined results
    if not all_results:
        logger.error("No experiments completed successfully.")
        return
    results_path = output_dir / "config_sweep_results.csv"
    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    logger.info("Results saved to %s", results_path)

    # Print summary table
    logger.info("")
    logger.info("=" * 110)
    logger.info("CONFIG SWEEP RESULTS")
    logger.info("=" * 110)
    header = f"{'Experiment':<25} {'Group':<25} | {'IoU':>7} {'F1':>7} {'Prec':>7} {'Rec':>7} | {'Det':>6} {'CC':>6}"
    logger.info(header)
    logger.info("-" * 110)

    current_group = None
    for r in all_results:
        if r["group"] != current_group:
            if current_group is not None:
                logger.info("-" * 110)
            current_group = r["group"]
        logger.info(
            "%-25s %-25s | %7.4f %7.4f %7.4f %7.4f | %6.2f %6.2f",
            r["experiment"],
            r["group"],
            r["iou"],
            r["f1"],
            r["precision"],
            r["recall"],
            r["avg_detections"],
            r["avg_cc"],
        )
    logger.info("=" * 110)

    # Print best configs
    best_iou = max(all_results, key=lambda r: r["iou"])
    best_det = max(all_results, key=lambda r: r["avg_detections"])
    best_cc = max(all_results, key=lambda r: r["avg_cc"])
    logger.info("")
    logger.info("BEST CONFIGS:")
    logger.info("  Best IoU:        %s (%.4f)", best_iou["experiment"], best_iou["iou"])
    logger.info("  Most detections: %s (%.2f)", best_det["experiment"], best_det["avg_detections"])
    logger.info("  Most CC:         %s (%.2f)", best_cc["experiment"], best_cc["avg_cc"])
    logger.info("All results saved to %s", output_dir)


if __name__ == "__main__":
    main()
