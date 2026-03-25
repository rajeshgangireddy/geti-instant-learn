# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Multi-reference and advanced model experiments.

Tests how multiple reference images, SAM2 backends, SoftMatcher combos,
and per-CC NMS affect detection count and quality on a COCO potato dataset.

Builds on findings from config_sweep_experiment.py:
  - Precision ~0.99, recall/detection count is the bottleneck
  - Bottleneck is in the prompt generator (matching stage)
  - SoftMatcher was the only config that increased detections (5.44 vs 5.0)
  - No-NMS revealed ~10 connected components (close to 8 real objects)

Experiments:
  E1: Multi-reference Matcher (1, 2, 3, 5 ref images)
  E2: Multi-reference SoftMatcher (1, 2, 3, 5 ref images)
  E3: SAM2 backends (SAM2-tiny, SAM2-small)
  E4: Per-CC NMS (keep best mask per connected component region)
  E5: Best combos (multi-ref + SoftMatcher + per-CC NMS)

Usage:
    cd library
    python examples/multi_ref_experiment.py
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
from instantlearn.components.postprocessing.base import PostProcessor
from instantlearn.data import Sample
from instantlearn.data.utils import read_image
from instantlearn.models import Matcher, SoftMatcher
from instantlearn.utils.metrics import SegmentationMetrics
from instantlearn.visualizer import render_predictions, setup_colors

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
DATASET_ROOT = Path("/home/rgangire/workspace/data/prompt/geti_datasets/COCO")
DEVICE = "cuda"
OUTPUT_DIR = Path("outputs/multi_ref_sweep")
# ──────────────────────────────────────────────────────────────────────────────


# ── Data loading utilities ────────────────────────────────────────────────────


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


# ── Custom per-CC NMS ─────────────────────────────────────────────────────────


class PerComponentNMS(PostProcessor):
    """Keep only the highest-scoring mask per connected component region.

    Instead of global IoM-based NMS which suppresses by pairwise overlap,
    this groups masks by the connected component they most overlap with
    and keeps the best-scoring one per region.
    """

    def __init__(self, min_overlap: float = 0.3) -> None:
        super().__init__()
        self.min_overlap = min_overlap

    def forward(
        self,
        masks: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if masks.shape[0] <= 1:
            return masks, scores, labels

        # Merge all masks to find connected components
        merged = masks.any(dim=0).cpu().numpy().astype(np.uint8)
        cc_map, n_cc = ndimage.label(merged)
        cc_map_t = torch.from_numpy(cc_map).to(masks.device)

        # For each mask, find which CC it overlaps most with
        # Then keep best-scoring mask per CC
        best_per_cc: dict[int, int] = {}  # cc_id -> mask_index
        masks_cpu = masks.cpu()
        scores_cpu = scores.cpu()

        for i in range(masks.shape[0]):
            mask_i = masks_cpu[i].bool()
            # Count overlap with each CC
            masked_cc = cc_map_t.cpu()[mask_i]
            if masked_cc.numel() == 0:
                continue
            # Find dominant CC for this mask
            cc_ids, counts = masked_cc.unique(return_counts=True)
            # Skip background (CC=0)
            nonzero = cc_ids > 0
            if not nonzero.any():
                continue
            cc_ids = cc_ids[nonzero]
            counts = counts[nonzero]
            # Check minimum overlap with dominant CC
            dominant_cc = cc_ids[counts.argmax()].item()
            overlap_frac = counts.max().item() / mask_i.sum().item()
            if overlap_frac < self.min_overlap:
                continue
            # Keep if best score for this CC
            if dominant_cc not in best_per_cc or scores_cpu[i] > scores_cpu[best_per_cc[dominant_cc]]:
                best_per_cc[dominant_cc] = i

        if not best_per_cc:
            return masks, scores, labels

        keep = sorted(best_per_cc.values())
        keep_t = torch.tensor(keep, device=masks.device)
        return masks[keep_t], scores[keep_t], labels[keep_t]


# ── Experiment definitions ────────────────────────────────────────────────────


def build_experiments(n_images: int) -> list[dict]:
    """Build experiment list. n_images is the total dataset size."""
    experiments = []
    max_ref = min(5, n_images - 1)  # Leave at least 1 target

    # E1: Multi-reference Matcher
    for n_ref in [1, 2, 3, max_ref]:
        experiments.append({
            "name": f"E1_matcher_ref_{n_ref}",
            "group": "E1_multi_ref_matcher",
            "model_cls": Matcher,
            "model_kwargs": {"device": DEVICE},
            "n_ref_images": n_ref,
        })

    # E2: Multi-reference SoftMatcher
    for n_ref in [1, 2, 3, max_ref]:
        experiments.append({
            "name": f"E2_softmatcher_ref_{n_ref}",
            "group": "E2_multi_ref_softmatcher",
            "model_cls": SoftMatcher,
            "model_kwargs": {"device": DEVICE},
            "n_ref_images": n_ref,
        })

    # E3: SAM2 backends — SKIPPED: SAM2ImagePredictor has incompatible API
    # (no set_torch_image method). Requires library-level fix in predictor.py.
    # for sam_name in [SAMModelName.SAM2_TINY, SAMModelName.SAM2_SMALL]:
    #     experiments.append({
    #         "name": f"E3_sam_{sam_name.value}",
    #         "group": "E3_sam2_backend",
    #         "model_cls": Matcher,
    #         "model_kwargs": {"sam": sam_name, "device": DEVICE},
    #         "n_ref_images": 1,
    #     })

    # E4: Per-CC NMS variants
    # 4a: Per-CC NMS only (no IoM NMS)
    pp_cc_only = PostProcessorPipeline([ScoreFilter(min_score=0.0), PerComponentNMS()])
    experiments.append({
        "name": "E4a_per_cc_nms",
        "group": "E4_per_cc_nms",
        "model_cls": Matcher,
        "model_kwargs": {"device": DEVICE, "postprocessor": pp_cc_only},
        "n_ref_images": 1,
    })
    # 4b: Per-CC NMS then standard NMS (clean up any remaining overlaps)
    pp_cc_then_nms = PostProcessorPipeline([
        ScoreFilter(min_score=0.0),
        PerComponentNMS(),
        MaskIoMNMS(iom_threshold=0.5),
        BoxIoMNMS(iom_threshold=0.5),
    ])
    experiments.append({
        "name": "E4b_per_cc_then_nms",
        "group": "E4_per_cc_nms",
        "model_cls": Matcher,
        "model_kwargs": {"device": DEVICE, "postprocessor": pp_cc_then_nms},
        "n_ref_images": 1,
    })

    # E5: Best combos — multi-ref + SoftMatcher + per-CC NMS
    pp_cc = PostProcessorPipeline([ScoreFilter(min_score=0.0), PerComponentNMS()])
    for n_ref in [3, max_ref]:
        # 5a: Multi-ref Matcher + per-CC NMS
        experiments.append({
            "name": f"E5a_matcher_ref_{n_ref}_cc_nms",
            "group": "E5_best_combos",
            "model_cls": Matcher,
            "model_kwargs": {"device": DEVICE, "postprocessor": pp_cc},
            "n_ref_images": n_ref,
        })
        # 5b: Multi-ref SoftMatcher + per-CC NMS
        experiments.append({
            "name": f"E5b_softmatcher_ref_{n_ref}_cc_nms",
            "group": "E5_best_combos",
            "model_cls": SoftMatcher,
            "model_kwargs": {"device": DEVICE, "postprocessor": pp_cc},
            "n_ref_images": n_ref,
        })

    return experiments


# ── Experiment runner ─────────────────────────────────────────────────────────


def run_experiment(
    exp: dict,
    all_image_data: list[dict],
    images_dir: Path,
    cat_id_to_label: dict[int, int],
    num_classes: int,
    color_map: dict,
    cat_map: dict[int, str],
    output_dir: Path,
) -> dict:
    """Run a single experiment and return metrics."""
    name = exp["name"]
    n_ref = exp["n_ref_images"]
    logger.info("─" * 60)
    logger.info("Experiment: %s (n_ref=%d)", name, n_ref)

    exp_dir = output_dir / name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Split into reference and target images
    ref_data = all_image_data[:n_ref]
    target_data = all_image_data[n_ref:]

    if not target_data:
        logger.warning("  No target images left for n_ref=%d, skipping", n_ref)
        return {"experiment": name, "group": exp["group"], "skipped": True}

    # Build reference Samples (one per image)
    ref_samples = []
    for rd in ref_data:
        ref_samples.append(
            Sample(
                image_path=str(images_dir / rd["info"]["file_name"]),
                masks=torch.from_numpy(rd["masks"]),
                categories=[cat_map[cid] for cid in rd["cat_ids"]],
                category_ids=rd["labels"],
            ),
        )

    # Create and fit model
    model = exp["model_cls"](**exp["model_kwargs"])
    if len(ref_samples) == 1:
        model.fit(ref_samples[0])
    else:
        model.fit(ref_samples)

    logger.info(
        "  Fitted with %d reference images (%d total masks)",
        n_ref,
        sum(rd["masks"].shape[0] for rd in ref_data),
    )

    # Save reference visualizations
    for ri, rd in enumerate(ref_data):
        ref_image_rgb = read_image(images_dir / rd["info"]["file_name"], as_tensor=False)
        ref_pred = {
            "pred_masks": torch.from_numpy(rd["masks"]),
            "pred_labels": torch.tensor(rd["labels"]),
        }
        ref_vis = render_predictions(ref_image_rgb, ref_pred, color_map, show_scores=False)
        cv2.imwrite(
            str(exp_dir / f"ref_{ri:02d}_{rd['info']['file_name']}"),
            cv2.cvtColor(ref_vis, cv2.COLOR_RGB2BGR),
        )

    # Predict on targets
    agg_metrics = SegmentationMetrics(num_classes=num_classes, device=torch.device("cpu"))
    stats_lists = {"n_detections": [], "connected_components": [], "mask_area": [], "avg_score": []}
    per_image_rows = []

    for i, td in enumerate(target_data):
        tgt_info = td["info"]
        tgt_path = images_dir / tgt_info["file_name"]
        tgt_h, tgt_w = tgt_info["height"], tgt_info["width"]

        predictions = model.predict(Sample(image_path=str(tgt_path)))
        pred = predictions[0]

        stats = compute_detection_stats(pred)
        for key in stats_lists:
            stats_lists[key].append(stats[key])

        pred_oh = build_one_hot(pred["pred_masks"], pred["pred_labels"], num_classes, tgt_h, tgt_w)
        agg_metrics.update(pred_oh, td["gt_one_hot"])

        # Per-image metrics
        img_met = SegmentationMetrics(num_classes=num_classes, device=torch.device("cpu"))
        img_met.update(pred_oh, td["gt_one_hot"])
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
        cv2.imwrite(
            str(exp_dir / f"tgt_{i:02d}_{tgt_info['file_name']}"),
            cv2.cvtColor(vis, cv2.COLOR_RGB2BGR),
        )

    computed = agg_metrics.compute()
    n_targets = len(target_data)

    result = {
        "experiment": name,
        "group": exp["group"],
        "n_ref_images": n_ref,
        "n_target_images": n_targets,
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
        with open(exp_dir / "per_image.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=per_image_rows[0].keys())
            writer.writeheader()
            writer.writerows(per_image_rows)

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return result


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    images, annotations, cat_map = load_coco_dataset(DATASET_ROOT)
    img_id_to_anns = {}
    for ann in annotations:
        img_id_to_anns.setdefault(ann["image_id"], []).append(ann)

    images_dir = DATASET_ROOT / "images"
    if (images_dir / "default").is_dir():
        images_dir = images_dir / "default"

    # Category mapping
    ref_info = images[0]
    ref_anns = img_id_to_anns.get(ref_info["id"], [])
    _, ref_cat_ids = coco_anns_to_mask(ref_anns, ref_info["height"], ref_info["width"])
    all_cat_ids = set()
    all_cat_ids.update(ann["category_id"] for ann in annotations)
    unique_cats = sorted(all_cat_ids)
    cat_id_to_label = {cid: i for i, cid in enumerate(unique_cats)}
    label_to_name = {cat_id_to_label[cid]: cat_map[cid] for cid in unique_cats}
    num_classes = len(unique_cats)
    color_map = setup_colors(label_to_name)

    logger.info("Dataset: %d images, %d categories", len(images), num_classes)

    # Precompute all image data (masks, labels, GT)
    all_image_data = []
    for img_info in images:
        h, w = img_info["height"], img_info["width"]
        anns = img_id_to_anns.get(img_info["id"], [])
        masks, cat_ids = coco_anns_to_mask(anns, h, w)
        labels = [cat_id_to_label[cid] for cid in cat_ids]
        # Sort masks by area (largest first)
        if masks.shape[0] > 0:
            areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
            order = np.argsort(-areas)
            masks = masks[order]
            labels = [labels[i] for i in order]
            cat_ids = [cat_ids[i] for i in order]
        gt_one_hot = build_one_hot(masks, labels, num_classes, h, w)
        all_image_data.append({
            "info": img_info,
            "masks": masks,
            "cat_ids": cat_ids,
            "labels": labels,
            "gt_one_hot": gt_one_hot,
        })

    # Build and run experiments
    experiments = build_experiments(len(images))
    logger.info("Running %d experiments ...", len(experiments))

    all_results: list[dict] = []
    for exp in experiments:
        # Skip completed experiments (resume support)
        exp_csv = output_dir / exp["name"] / "per_image.csv"
        if exp_csv.exists():
            logger.info("Skipping %s (already completed)", exp["name"])
            continue
        try:
            result = run_experiment(
                exp=exp,
                all_image_data=all_image_data,
                images_dir=images_dir,
                cat_id_to_label=cat_id_to_label,
                num_classes=num_classes,
                color_map=color_map,
                cat_map=cat_map,
                output_dir=output_dir,
            )
            if not result.get("skipped"):
                all_results.append(result)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            logger.error("Experiment %s FAILED: %s", exp["name"], e)
            gc.collect()
            torch.cuda.empty_cache()

    # Reload results for previously-completed experiments (resume support)
    for exp in experiments:
        if any(r["experiment"] == exp["name"] for r in all_results):
            continue  # Already in results from this run
        exp_csv = output_dir / exp["name"] / "per_image.csv"
        if not exp_csv.exists():
            continue
        with open(exp_csv) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        n = len(rows)
        all_results.append({
            "experiment": exp["name"],
            "group": exp["group"],
            "n_ref_images": exp["n_ref_images"],
            "n_target_images": n,
            "iou": round(sum(float(r["iou"]) for r in rows) / n, 4),
            "f1": round(sum(float(r["f1"]) for r in rows) / n, 4),
            "precision": 0.0,  # Not stored in per_image — will compute from aggregate
            "recall": 0.0,
            "avg_detections": round(sum(int(r["n_detections"]) for r in rows) / n, 2),
            "avg_cc": round(sum(int(r["connected_components"]) for r in rows) / n, 2),
            "avg_mask_area": round(sum(int(r["mask_area"]) for r in rows) / n, 0),
            "avg_score": round(sum(float(r["avg_score"]) for r in rows) / n, 4),
        })
    # Sort by experiment order
    exp_order = {exp["name"]: i for i, exp in enumerate(experiments)}
    all_results.sort(key=lambda r: exp_order.get(r["experiment"], 999))

    if not all_results:
        logger.error("No experiments completed successfully.")
        return

    # Write combined results
    results_path = output_dir / "multi_ref_results.csv"
    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    logger.info("Results saved to %s", results_path)

    # Print summary table
    logger.info("")
    logger.info("=" * 120)
    logger.info("MULTI-REFERENCE EXPERIMENT RESULTS")
    logger.info("=" * 120)
    header = (
        f"{'Experiment':<35} {'Group':<25} {'Ref':>3} {'Tgt':>3} |"
        f" {'IoU':>7} {'F1':>7} {'Prec':>7} {'Rec':>7} | {'Det':>6} {'CC':>6}"
    )
    logger.info(header)
    logger.info("-" * 120)

    current_group = None
    for r in all_results:
        if r["group"] != current_group:
            if current_group is not None:
                logger.info("-" * 120)
            current_group = r["group"]
        logger.info(
            "%-35s %-25s %3d %3d | %7.4f %7.4f %7.4f %7.4f | %6.2f %6.2f",
            r["experiment"],
            r["group"],
            r["n_ref_images"],
            r["n_target_images"],
            r["iou"],
            r["f1"],
            r["precision"],
            r["recall"],
            r["avg_detections"],
            r["avg_cc"],
        )
    logger.info("=" * 120)

    # Find best configs
    best_iou = max(all_results, key=lambda r: r["iou"])
    best_det = max(all_results, key=lambda r: r["avg_detections"])
    best_cc = max(all_results, key=lambda r: r["avg_cc"])
    best_f1 = max(all_results, key=lambda r: r["f1"])
    logger.info("")
    logger.info("BEST CONFIGS:")
    logger.info("  Best IoU:        %s (%.4f)", best_iou["experiment"], best_iou["iou"])
    logger.info("  Best F1:         %s (%.4f)", best_f1["experiment"], best_f1["f1"])
    logger.info("  Most detections: %s (%.2f)", best_det["experiment"], best_det["avg_detections"])
    logger.info("  Most CC:         %s (%.2f)", best_cc["experiment"], best_cc["avg_cc"])


if __name__ == "__main__":
    main()
