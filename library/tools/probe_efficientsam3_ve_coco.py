# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""COCO probe: VISUAL_EXEMPLAR no-text vs with-text on Potatoes (1 cat) + Nuts (2 cats).

Picks one annotated image per dataset as the reference (containing all category
bboxes), then runs prediction on the remaining images twice:
 (A) no-text: ``Sample(bboxes=..., category_ids=...)`` — defaults to "visual"
 (B) with-text: same but ``categories=[<real names>]``

Reports per-image detection counts and aggregate P/R/F1 at IoU 0.5 against COCO
ground truth. Companion to ``probe_efficientsam3_visual_exemplar.py`` which
uses the EDSA vehicle assets.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

# Reuse the IoU + matching helpers from the benchmark script
from benchmark_efficient_sam3 import _load_image_tensor, match_and_score  # noqa: PLC2701
from pycocotools.coco import COCO

from instantlearn.data import Sample
from instantlearn.models import EfficientSAM3
from instantlearn.models.sam3.sam3 import Sam3PromptMode

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)
logger = logging.getLogger("ve_probe_coco")
logger.setLevel(logging.INFO)

DATA_ROOT = Path("/home/rgangire/workspace/data/prompt/geti_datasets/COCO")
DATASETS = {
    "potatoes": DATA_ROOT / "Potatoes",
    "nuts": DATA_ROOT / "Nuts",
}


def load_coco(dataset_dir: Path) -> tuple[COCO, Path, list[int], list[str], dict[int, int]]:
    """Return (coco, image_root, image_ids, category_names_sorted_by_id, cat_id_to_index)."""
    coco = COCO(str(dataset_dir / "annotations" / "instances_default.json"))
    image_root = dataset_dir / "images" / "default"
    cat_ids = sorted(coco.getCatIds())
    cat_names = [coco.loadCats(cid)[0]["name"] for cid in cat_ids]
    cat_id_to_index = {cid: i for i, cid in enumerate(cat_ids)}
    return coco, image_root, sorted(coco.getImgIds()), cat_names, cat_id_to_index


def pick_reference_image(coco: COCO, image_ids: list[int], cat_ids: list[int]) -> int:
    """Pick the first image that contains at least one annotation for *every* category."""
    for img_id in image_ids:
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        present = {a["category_id"] for a in anns}
        if all(c in present for c in cat_ids):
            return img_id
    # Fallback: image with the most distinct categories
    best_id, best_cnt = image_ids[0], -1
    for img_id in image_ids:
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        cnt = len({a["category_id"] for a in anns})
        if cnt > best_cnt:
            best_id, best_cnt = img_id, cnt
    return best_id


def build_reference_sample(
    coco: COCO,
    image_root: Path,
    ref_img_id: int,
    cat_id_to_index: dict[int, int],
    with_text: bool,
    cat_names: list[str],
) -> Sample:
    """Build a Sample with all GT boxes from the reference image, one slot per box."""
    img_info = coco.loadImgs(ref_img_id)[0]
    img_path = image_root / img_info["file_name"]
    ann_ids = coco.getAnnIds(imgIds=ref_img_id)
    anns = coco.loadAnns(ann_ids)

    bboxes_xyxy: list[list[float]] = []
    cat_ids_per_box: list[int] = []
    cat_names_per_box: list[str] = []
    for a in anns:
        x, y, w, h = a["bbox"]
        bboxes_xyxy.append([x, y, x + w, y + h])
        idx = cat_id_to_index[a["category_id"]]
        cat_ids_per_box.append(idx)
        cat_names_per_box.append(cat_names[idx])

    kwargs = {
        "image_path": str(img_path),
        "bboxes": np.array(bboxes_xyxy, dtype=np.float32),
        "category_ids": np.array(cat_ids_per_box, dtype=np.int64),
    }
    # No-text path: explicitly pass length-N placeholders to bypass Sample's
    # auto-default to ['object'] (length 1) which would mismatch num_prompts.
    kwargs["categories"] = cat_names_per_box if with_text else ["visual"] * len(bboxes_xyxy)
    return Sample(**kwargs)


def evaluate(
    model: EfficientSAM3,
    coco: COCO,
    image_root: Path,
    target_img_ids: list[int],
    cat_id_to_index: dict[int, int],
) -> dict[str, float | int]:
    """Run predict on each target image, accumulate per-class TP/FP/FN."""
    totals = {"tp": 0, "fp": 0, "fn": 0, "n_pred": 0, "n_gt": 0}
    for img_id in target_img_ids:
        img_info = coco.loadImgs(img_id)[0]
        img_path = image_root / img_info["file_name"]
        img_t = _load_image_tensor(img_path)
        pred = model.predict(Sample(image=img_t))[0]

        # Predicted boxes: (N, 5) -> (N, 4)
        boxes_full = pred["pred_boxes"].detach().cpu().numpy().astype(np.float32)
        pred_boxes = boxes_full[:, :4] if boxes_full.size > 0 else np.zeros((0, 4), dtype=np.float32)
        pred_labels = pred["pred_labels"].detach().cpu().numpy().astype(np.int64)

        # GT
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

    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {**totals, "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}


def run_dataset(name: str, dataset_dir: Path, model: EfficientSAM3) -> None:
    """Run no-text vs with-text on one dataset and print a side-by-side report."""
    logger.info("=" * 72)
    logger.info("Dataset: %s", name)

    coco, image_root, image_ids, cat_names, cat_id_to_index = load_coco(dataset_dir)
    coco_cat_ids = sorted(coco.getCatIds())
    logger.info("Categories (%d): %s", len(cat_names), cat_names)

    ref_img_id = pick_reference_image(coco, image_ids, coco_cat_ids)
    target_img_ids = [i for i in image_ids if i != ref_img_id]
    ref_info = coco.loadImgs(ref_img_id)[0]
    n_ref_boxes = len(coco.getAnnIds(imgIds=ref_img_id))
    logger.info(
        "Reference image: %s (id=%d, %d boxes)", ref_info["file_name"], ref_img_id, n_ref_boxes,
    )
    logger.info("Evaluating on %d remaining images", len(target_img_ids))

    for label, with_text in [("NO  TEXT (default 'visual')", False), ("WITH TEXT (real names)    ", True)]:
        ref_sample = build_reference_sample(coco, image_root, ref_img_id, cat_id_to_index, with_text, cat_names)
        model.fit(ref_sample)
        metrics = evaluate(model, coco, image_root, target_img_ids, cat_id_to_index)
        logger.info(
            "  %s | tp=%3d fp=%3d fn=%3d | P=%.3f R=%.3f F1=%.3f | n_gt=%d n_pred=%d",
            label, metrics["tp"], metrics["fp"], metrics["fn"],
            metrics["precision"], metrics["recall"], metrics["f1"],
            metrics["n_gt"], metrics["n_pred"],
        )


def main() -> None:
    """Run the probe on both COCO datasets."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    model = EfficientSAM3(
        backbone_type="efficientvit",
        variant="b1",
        device=device,
        prompt_mode=Sam3PromptMode.VISUAL_EXEMPLAR,
        confidence_threshold=0.4,
    )

    for name, dataset_dir in DATASETS.items():
        run_dataset(name, dataset_dir, model)

    logger.info("Done.")


if __name__ == "__main__":
    main()
