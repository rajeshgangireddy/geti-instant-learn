# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Few-shot visual prompting demo with Matcher on a COCO dataset.

Uses the first image + its masks as the reference prompt,
then predicts on the remaining images and saves visualizations.

Usage:
    cd library
    python examples/coco_matcher_demo.py
    python examples/coco_matcher_demo.py --device cpu
    python examples/coco_matcher_demo.py --output-dir /tmp/matcher_results
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import torch

from instantlearn.data import Sample
from instantlearn.data.utils import read_image
from instantlearn.models import PerDino
from instantlearn.visualizer import render_predictions, setup_colors

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DATASET_ROOT = Path("/home/rgangire/workspace/data/prompt/geti_datasets/COCO")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Matcher few-shot demo on COCO dataset")
    parser.add_argument("--dataset", type=Path, default=DATASET_ROOT)
    parser.add_argument("--device", default="cuda", choices=["auto", "cuda", "xpu", "cpu"])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/coco_matcher_demo"))
    parser.add_argument("--max-targets", type=int, default=None, help="Limit number of target images")
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load COCO dataset
    images, annotations, cat_map = load_coco_dataset(dataset_root)
    logger.info("Dataset: %d images, %d annotations, categories: %s", len(images), len(annotations), cat_map)

    # Build image_id -> annotations lookup
    img_id_to_anns: dict[int, list[dict]] = {}
    for ann in annotations:
        img_id_to_anns.setdefault(ann["image_id"], []).append(ann)

    # Find images subdirectory (handles 'default' subfolder)
    images_dir = dataset_root / "images"
    if (images_dir / "default").is_dir():
        images_dir = images_dir / "default"

    # Use first image as reference, rest as targets
    ref_img_info = images[0]
    target_img_infos = images[1:]
    if args.max_targets:
        target_img_infos = target_img_infos[: args.max_targets]

    # Load reference image and create masks
    ref_path = images_dir / ref_img_info["file_name"]
    ref_h, ref_w = ref_img_info["height"], ref_img_info["width"]

    ref_anns = img_id_to_anns.get(ref_img_info["id"], [])
    ref_masks, ref_cat_ids = coco_anns_to_mask(ref_anns, ref_h, ref_w)

    logger.info(
        "Reference: %s (%dx%d) with %d instance masks",
        ref_img_info["file_name"],
        ref_w,
        ref_h,
        len(ref_anns),
    )

    # Map category IDs to 0-indexed labels
    unique_cats = sorted(set(ref_cat_ids))
    cat_id_to_label = {cid: i for i, cid in enumerate(unique_cats)}
    ref_labels = [cat_id_to_label[cid] for cid in ref_cat_ids]
    label_to_name = {cat_id_to_label[cid]: cat_map[cid] for cid in unique_cats}

    # Create reference sample
    ref_sample = Sample(
        image_path=str(ref_path),
        masks=torch.from_numpy(ref_masks),
        categories=[cat_map[cid] for cid in ref_cat_ids],
        category_ids=ref_labels,
    )

    # Initialize Matcher and fit on reference
    logger.info("Loading Matcher on device=%s ...", args.device)
    model = PerDino(device=args.device)
    logger.info("Fitting on reference image ...")
    model.fit(ref_sample)

    # Save reference visualization
    color_map = setup_colors(label_to_name)
    ref_image_rgb = read_image(ref_path, as_tensor=False)  # HWC numpy for visualization
    ref_pred = {
        "pred_masks": torch.from_numpy(ref_masks),
        "pred_labels": torch.tensor(ref_labels),
    }
    ref_vis = render_predictions(ref_image_rgb, ref_pred, color_map, show_scores=False)
    ref_out = output_dir / f"00_reference_{ref_img_info['file_name']}"
    cv2.imwrite(str(ref_out), cv2.cvtColor(ref_vis, cv2.COLOR_RGB2BGR))
    logger.info("Saved reference visualization: %s", ref_out)

    # Predict on target images
    logger.info("Predicting on %d target images ...", len(target_img_infos))
    for i, tgt_info in enumerate(target_img_infos, start=1):
        tgt_path = images_dir / tgt_info["file_name"]

        predictions = model.predict(Sample(image_path=str(tgt_path)))
        pred = predictions[0]

        n_det = pred["pred_masks"].shape[0]
        scores_str = ""
        if n_det > 0:
            lo = pred["pred_scores"].min().item()
            hi = pred["pred_scores"].max().item()
            scores_str = f" scores=[{lo:.2f}..{hi:.2f}]"

        logger.info(
            "[%d/%d] %s -> %d detections%s",
            i,
            len(target_img_infos),
            tgt_info["file_name"],
            n_det,
            scores_str,
        )

        tgt_image_rgb = read_image(tgt_path, as_tensor=False)  # HWC numpy for visualization
        vis = render_predictions(tgt_image_rgb, pred, color_map)
        out_path = output_dir / f"{i:02d}_{tgt_info['file_name']}"
        cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    logger.info("All results saved to %s", output_dir)


if __name__ == "__main__":
    main()
