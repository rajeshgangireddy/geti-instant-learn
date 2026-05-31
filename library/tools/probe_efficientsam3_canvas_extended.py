# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Extended CANVAS validation sweep for EfficientSAM3.

Tests CANVAS mode across:
  - Datasets: Potatoes (1 cat), Nuts (2 cat), Candies (2 cat), LVIS (multi-cat), PerSeg (1 cat)
  - Backbones: efficientvit/b1 (default), tinyvit/11m, repvit/m1_1
  - FT checkpoints: ft=False (stage1), ft=True (fine-tuned, medium variants only)
  - Text: no-text vs with-text
  - Canvas config: split_ratio=0.3 (SAM3 default)

Compares CANVAS vs VISUAL_EXEMPLAR (VE) on the same reference/target splits.

Outputs a table to stdout summarizing F1/precision/recall per configuration.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from pycocotools.coco import COCO

from benchmark_efficient_sam3 import _load_image_tensor, match_and_score  # noqa: PLC2701
from probe_efficientsam3_ve_coco import load_coco, pick_reference_image  # noqa: PLC2701

from instantlearn.data import Sample
from instantlearn.models import EfficientSAM3
from instantlearn.models.sam3.sam3 import CanvasConfig, Sam3PromptMode

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)
logger = logging.getLogger("canvas_extended")
logger.setLevel(logging.INFO)

# ─────────────────────────────────────────────────────────────────────────────
# Dataset paths
# ─────────────────────────────────────────────────────────────────────────────

COCO_ROOT = Path("/home/rgangire/workspace/data/prompt/geti_datasets/COCO")
LVIS_ROOT = Path("/home/rgangire/workspace/data/prompt/lvis")
PERSEG_ROOT = Path("/home/rgangire/workspace/data/prompt/PerSeg")

# LVIS categories to test (diverse: small objects, vehicles, food)
LVIS_CATEGORIES = ["avocado", "fire_engine", "chandelier"]
# PerSeg categories (single-instance, diverse objects)
PERSEG_CATEGORIES = ["dog", "backpack", "teapot", "elephant", "clock"]


# ─────────────────────────────────────────────────────────────────────────────
# COCO dataset helpers (reusing existing probe code)
# ─────────────────────────────────────────────────────────────────────────────


def build_canvas_refs(
    coco: COCO,
    image_root: Path,
    ref_img_id: int,
    cat_id_to_index: dict[int, int],
    cat_names: list[str],
    with_text: bool,
    max_per_cat: int = 1,
) -> list[Sample]:
    """Build one Sample per reference bbox, 1-shot per category."""
    img_info = coco.loadImgs(ref_img_id)[0]
    img_path = image_root / img_info["file_name"]
    ann_ids = coco.getAnnIds(imgIds=ref_img_id)
    anns = coco.loadAnns(ann_ids)

    samples: list[Sample] = []
    per_cat_count: dict[int, int] = {}
    for a in anns:
        idx = cat_id_to_index[a["category_id"]]
        if per_cat_count.get(idx, 0) >= max_per_cat:
            continue
        per_cat_count[idx] = per_cat_count.get(idx, 0) + 1
        x, y, w, h = a["bbox"]
        name = cat_names[idx] if with_text else "visual"
        samples.append(Sample(
            image_path=str(img_path),
            bboxes=np.array([[x, y, x + w, y + h]], dtype=np.float32),
            category_ids=np.array([idx], dtype=np.int64),
            categories=[name],
        ))
    return samples


def build_ve_ref(
    coco: COCO,
    image_root: Path,
    ref_img_id: int,
    cat_id_to_index: dict[int, int],
    cat_names: list[str],
    with_text: bool,
) -> Sample:
    """Build a single VE Sample with ALL reference bboxes (works for VE mode)."""
    img_info = coco.loadImgs(ref_img_id)[0]
    img_path = image_root / img_info["file_name"]
    ann_ids = coco.getAnnIds(imgIds=ref_img_id)
    anns = coco.loadAnns(ann_ids)

    bboxes: list[list[float]] = []
    cat_ids_per_box: list[int] = []
    cat_names_per_box: list[str] = []
    for a in anns:
        x, y, w, h = a["bbox"]
        bboxes.append([x, y, x + w, y + h])
        idx = cat_id_to_index[a["category_id"]]
        cat_ids_per_box.append(idx)
        cat_names_per_box.append(cat_names[idx])

    return Sample(
        image_path=str(img_path),
        bboxes=np.array(bboxes, dtype=np.float32),
        category_ids=np.array(cat_ids_per_box, dtype=np.int64),
        categories=cat_names_per_box if with_text else ["visual"] * len(bboxes),
    )


def evaluate_coco(
    model: EfficientSAM3,
    coco: COCO,
    image_root: Path,
    target_img_ids: list[int],
    cat_id_to_index: dict[int, int],
) -> dict[str, float | int]:
    """Run predict on COCO targets, compute P/R/F1 at IoU 0.5."""
    tp_tot, fp_tot, fn_tot = 0, 0, 0
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
        tp_tot += tp
        fp_tot += fp
        fn_tot += fn

    elapsed = time.perf_counter() - t0
    n = max(1, len(target_img_ids))
    prec = tp_tot / (tp_tot + fp_tot) if (tp_tot + fp_tot) > 0 else 0.0
    rec = tp_tot / (tp_tot + fn_tot) if (tp_tot + fn_tot) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"tp": tp_tot, "fp": fp_tot, "fn": fn_tot, "P": round(prec, 3), "R": round(rec, 3),
            "F1": round(f1, 3), "ms/img": round(elapsed / n * 1000, 1)}


# ─────────────────────────────────────────────────────────────────────────────
# LVIS helpers
# ─────────────────────────────────────────────────────────────────────────────


def load_lvis(categories: Sequence[str], max_targets: int = 15):
    """Load LVIS val split for given categories, return (api, image_root, ref_id, targets, cat_names, cat_id_map)."""
    from lvis import LVIS as LVISApi

    lvis_api = LVISApi(str(LVIS_ROOT / "lvis_v1_val.json"))
    all_cats = lvis_api.load_cats(lvis_api.get_cat_ids())
    name_to_id = {c["name"]: c["id"] for c in all_cats}
    valid_ids = [name_to_id[n] for n in categories if n in name_to_id]
    cat_id_to_index = {cid: i for i, cid in enumerate(sorted(valid_ids))}
    cat_names = [next(c["name"] for c in all_cats if c["id"] == cid) for cid in sorted(valid_ids)]

    # Find images containing at least one annotation for our categories
    ann_ids = lvis_api.get_ann_ids(cat_ids=valid_ids)
    anns = lvis_api.load_anns(ann_ids)
    img_ids_with_anns = sorted({a["image_id"] for a in anns})

    # Pick reference: first image with all categories present
    ref_img_id = None
    for img_id in img_ids_with_anns:
        img_anns = [a for a in anns if a["image_id"] == img_id]
        present = {a["category_id"] for a in img_anns}
        if all(c in present for c in valid_ids):
            ref_img_id = img_id
            break
    if ref_img_id is None:
        ref_img_id = img_ids_with_anns[0]

    target_ids = [i for i in img_ids_with_anns if i != ref_img_id][:max_targets]

    # LVIS images come from COCO val2017 or train2017
    image_root = LVIS_ROOT  # images in val2017/ and train2017/ subdirs

    return lvis_api, image_root, ref_img_id, target_ids, cat_names, cat_id_to_index, valid_ids


def build_lvis_canvas_refs(lvis_api, image_root: Path, ref_img_id: int,
                           cat_id_to_index: dict[int, int], cat_names: list[str],
                           with_text: bool, valid_cat_ids: list[int]) -> list[Sample]:
    """Build canvas-style per-bbox Samples for LVIS ref."""
    img_info = lvis_api.imgs[ref_img_id]
    # Resolve image path from coco_url
    from pathlib import PurePosixPath
    coco_url = PurePosixPath(img_info["coco_url"])
    img_path = image_root / coco_url.parent.name / coco_url.name

    ann_ids = lvis_api.get_ann_ids(img_ids=[ref_img_id], cat_ids=valid_cat_ids)
    anns = lvis_api.load_anns(ann_ids)

    samples: list[Sample] = []
    per_cat: dict[int, int] = {}
    for a in anns:
        cid = a["category_id"]
        if cid not in cat_id_to_index:
            continue
        idx = cat_id_to_index[cid]
        if per_cat.get(idx, 0) >= 1:
            continue
        per_cat[idx] = per_cat.get(idx, 0) + 1
        x, y, w, h = a["bbox"]
        name = cat_names[idx] if with_text else "visual"
        samples.append(Sample(
            image_path=str(img_path),
            bboxes=np.array([[x, y, x + w, y + h]], dtype=np.float32),
            category_ids=np.array([idx], dtype=np.int64),
            categories=[name],
        ))
    return samples


def build_lvis_ve_ref(lvis_api, image_root: Path, ref_img_id: int,
                      cat_id_to_index: dict[int, int], cat_names: list[str],
                      with_text: bool, valid_cat_ids: list[int]) -> Sample:
    """Build VE-style multi-bbox Sample for LVIS ref."""
    img_info = lvis_api.imgs[ref_img_id]
    from pathlib import PurePosixPath
    coco_url = PurePosixPath(img_info["coco_url"])
    img_path = image_root / coco_url.parent.name / coco_url.name

    ann_ids = lvis_api.get_ann_ids(img_ids=[ref_img_id], cat_ids=valid_cat_ids)
    anns = lvis_api.load_anns(ann_ids)

    bboxes, cat_ids_per_box, names_per_box = [], [], []
    for a in anns:
        cid = a["category_id"]
        if cid not in cat_id_to_index:
            continue
        idx = cat_id_to_index[cid]
        x, y, w, h = a["bbox"]
        bboxes.append([x, y, x + w, y + h])
        cat_ids_per_box.append(idx)
        names_per_box.append(cat_names[idx])

    return Sample(
        image_path=str(img_path),
        bboxes=np.array(bboxes, dtype=np.float32),
        category_ids=np.array(cat_ids_per_box, dtype=np.int64),
        categories=names_per_box if with_text else ["visual"] * len(bboxes),
    )


def evaluate_lvis(model, lvis_api, image_root: Path, target_ids: list[int],
                  cat_id_to_index: dict[int, int], valid_cat_ids: list[int]) -> dict:
    """Evaluate on LVIS targets."""
    from pathlib import PurePosixPath
    tp_tot, fp_tot, fn_tot = 0, 0, 0
    t0 = time.perf_counter()
    for img_id in target_ids:
        img_info = lvis_api.imgs[img_id]
        coco_url = PurePosixPath(img_info["coco_url"])
        img_path = image_root / coco_url.parent.name / coco_url.name
        img_t = _load_image_tensor(img_path)
        pred = model.predict(Sample(image=img_t))[0]

        boxes_full = pred["pred_boxes"].detach().cpu().numpy().astype(np.float32)
        pred_boxes = boxes_full[:, :4] if boxes_full.size > 0 else np.zeros((0, 4), dtype=np.float32)
        pred_labels = pred["pred_labels"].detach().cpu().numpy().astype(np.int64)

        ann_ids = lvis_api.get_ann_ids(img_ids=[img_id], cat_ids=valid_cat_ids)
        anns = lvis_api.load_anns(ann_ids)
        gt_boxes = np.array(
            [[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in anns],
            dtype=np.float32,
        ) if anns else np.zeros((0, 4), dtype=np.float32)
        gt_labels = np.array([cat_id_to_index[a["category_id"]] for a in anns if a["category_id"] in cat_id_to_index],
                             dtype=np.int64)

        tp, fp, fn = match_and_score(pred_boxes, pred_labels, gt_boxes, gt_labels, iou_threshold=0.5)
        tp_tot += tp
        fp_tot += fp
        fn_tot += fn

    elapsed = time.perf_counter() - t0
    n = max(1, len(target_ids))
    prec = tp_tot / (tp_tot + fp_tot) if (tp_tot + fp_tot) > 0 else 0.0
    rec = tp_tot / (tp_tot + fn_tot) if (tp_tot + fn_tot) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"tp": tp_tot, "fp": fp_tot, "fn": fn_tot, "P": round(prec, 3), "R": round(rec, 3),
            "F1": round(f1, 3), "ms/img": round(elapsed / n * 1000, 1)}


# ─────────────────────────────────────────────────────────────────────────────
# PerSeg helpers
# ─────────────────────────────────────────────────────────────────────────────


def load_perseg_category(category: str) -> tuple[Path, list[Path], list[Path]]:
    """Return (ref_image, target_images, target_masks) for one PerSeg category."""
    img_dir = PERSEG_ROOT / "Images" / category
    mask_dir = PERSEG_ROOT / "Annotations" / category

    imgs = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    masks = sorted(mask_dir.glob("*.png"))
    # First = reference, rest = targets
    return imgs[0], imgs[1:], masks[1:]


def bbox_from_mask(mask_path: Path) -> np.ndarray:
    """Compute tight bbox from a binary mask image.

    Treats any non-zero pixel as foreground (matches ``read_mask`` in the
    codebase). PerSeg masks use low gray values (e.g. {0, 38}), not {0, 255}.
    """
    import cv2
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.array([[0, 0, 1, 1]], dtype=np.float32)
    return np.array([[xs.min(), ys.min(), xs.max(), ys.max()]], dtype=np.float32)


def evaluate_perseg_category(
    model: EfficientSAM3,
    category: str,
    mode: str,
    with_text: bool,
) -> dict:
    """Run one PerSeg category. Returns metrics dict."""
    ref_img, target_imgs, target_masks = load_perseg_category(category)
    ref_mask_path = PERSEG_ROOT / "Annotations" / category / f"{ref_img.stem}.png"
    ref_bbox = bbox_from_mask(ref_mask_path)

    cat_name = category if with_text else "visual"

    if mode == "canvas":
        ref_sample = Sample(
            image_path=str(ref_img),
            bboxes=ref_bbox,
            category_ids=np.array([0], dtype=np.int64),
            categories=[cat_name],
        )
    else:  # VE
        ref_sample = Sample(
            image_path=str(ref_img),
            bboxes=ref_bbox,
            category_ids=np.array([0], dtype=np.int64),
            categories=[cat_name],
        )

    model.fit(ref_sample)

    # PerSeg uses IoU on masks for scoring, but for detection we use bbox IoU.
    # Compute GT bbox from each target mask.
    tp_tot, fp_tot, fn_tot = 0, 0, 0
    t0 = time.perf_counter()
    for tgt_img_path, tgt_mask_path in zip(target_imgs, target_masks, strict=False):
        img_t = _load_image_tensor(tgt_img_path)
        pred = model.predict(Sample(image=img_t))[0]

        boxes_full = pred["pred_boxes"].detach().cpu().numpy().astype(np.float32)
        pred_boxes = boxes_full[:, :4] if boxes_full.size > 0 else np.zeros((0, 4), dtype=np.float32)
        pred_labels = pred["pred_labels"].detach().cpu().numpy().astype(np.int64)

        gt_bbox = bbox_from_mask(tgt_mask_path)
        gt_labels = np.array([0], dtype=np.int64)

        tp, fp, fn = match_and_score(pred_boxes, pred_labels, gt_bbox, gt_labels, iou_threshold=0.5)
        tp_tot += tp
        fp_tot += fp
        fn_tot += fn

    elapsed = time.perf_counter() - t0
    n = max(1, len(target_imgs))
    prec = tp_tot / (tp_tot + fp_tot) if (tp_tot + fp_tot) > 0 else 0.0
    rec = tp_tot / (tp_tot + fn_tot) if (tp_tot + fn_tot) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"tp": tp_tot, "fp": fp_tot, "fn": fn_tot, "P": round(prec, 3), "R": round(rec, 3),
            "F1": round(f1, 3), "ms/img": round(elapsed / n * 1000, 1), "n_targets": n}


# ─────────────────────────────────────────────────────────────────────────────
# Main sweep
# ─────────────────────────────────────────────────────────────────────────────


BACKBONE_CONFIGS = [
    # (backbone_type, variant, ft)
    ("efficientvit", "b1", False),
    ("efficientvit", "b1", True),
    ("tinyvit", "11m", False),
    ("tinyvit", "11m", True),
    ("repvit", "m1_1", False),
    ("repvit", "m1_1", True),
]

COCO_DATASETS = {
    "Potatoes": COCO_ROOT / "Potatoes",
    "Nuts": COCO_ROOT / "Nuts",
    "Candies": COCO_ROOT / "Candies",
}


def run_coco_sweep(device: str) -> list[dict]:
    """Run CANVAS vs VE across COCO datasets × backbones × ft × text."""
    rows: list[dict] = []
    for bb_type, variant, ft in BACKBONE_CONFIGS:
        bb_label = f"{bb_type}/{variant}" + ("+ft" if ft else "")
        logger.info("─" * 80)
        logger.info("Backbone: %s", bb_label)

        # Create both models (CANVAS and VE)
        try:
            model_canvas = EfficientSAM3(
                backbone_type=bb_type, variant=variant, device=device, ft=ft,
                prompt_mode=Sam3PromptMode.CANVAS,
                canvas_config=CanvasConfig(split_ratio=0.3, crop_padding=2.0),
                confidence_threshold=0.4,
            )
            model_ve = EfficientSAM3(
                backbone_type=bb_type, variant=variant, device=device, ft=ft,
                prompt_mode=Sam3PromptMode.VISUAL_EXEMPLAR,
                confidence_threshold=0.4,
            )
        except (ValueError, FileNotFoundError) as e:
            logger.warning("  SKIP %s: %s", bb_label, e)
            continue

        for ds_name, ds_dir in COCO_DATASETS.items():
            coco, image_root, image_ids, cat_names, cat_id_to_index = load_coco(ds_dir)
            coco_cat_ids = sorted(coco.getCatIds())
            ref_img_id = pick_reference_image(coco, image_ids, coco_cat_ids)
            target_ids = [i for i in image_ids if i != ref_img_id]

            for text_label, with_text in [("no-text", False), ("with-text", True)]:
                # CANVAS
                refs_c = build_canvas_refs(coco, image_root, ref_img_id, cat_id_to_index, cat_names, with_text)
                model_canvas.canvas_config = CanvasConfig(split_ratio=0.3, crop_padding=2.0)
                model_canvas.fit(refs_c)
                m_c = evaluate_coco(model_canvas, coco, image_root, target_ids, cat_id_to_index)

                # VE
                ref_ve = build_ve_ref(coco, image_root, ref_img_id, cat_id_to_index, cat_names, with_text)
                model_ve.fit(ref_ve)
                m_ve = evaluate_coco(model_ve, coco, image_root, target_ids, cat_id_to_index)

                row_base = {"backbone": bb_label, "dataset": ds_name, "text": text_label}
                rows.append({**row_base, "mode": "CANVAS", **m_c})
                rows.append({**row_base, "mode": "VE", **m_ve})
                logger.info(
                    "  %s %s | CANVAS F1=%.3f | VE F1=%.3f | %s",
                    ds_name, text_label, m_c["F1"], m_ve["F1"],
                    "CANVAS+" if m_c["F1"] > m_ve["F1"] else ("VE+" if m_ve["F1"] > m_c["F1"] else "tie"),
                )

        del model_canvas, model_ve
        torch.cuda.empty_cache()

    return rows


def run_lvis_sweep(device: str) -> list[dict]:
    """Run CANVAS vs VE on LVIS multi-category split."""
    rows: list[dict] = []
    logger.info("═" * 80)
    logger.info("LVIS SWEEP — categories: %s", LVIS_CATEGORIES)

    lvis_api, image_root, ref_img_id, target_ids, cat_names, cat_id_to_index, valid_cat_ids = load_lvis(
        LVIS_CATEGORIES, max_targets=15,
    )
    logger.info("  Reference img_id=%d, %d targets", ref_img_id, len(target_ids))

    for bb_type, variant, ft in [("efficientvit", "b1", False), ("efficientvit", "b1", True)]:
        bb_label = f"{bb_type}/{variant}" + ("+ft" if ft else "")
        try:
            model_canvas = EfficientSAM3(
                backbone_type=bb_type, variant=variant, device=device, ft=ft,
                prompt_mode=Sam3PromptMode.CANVAS,
                canvas_config=CanvasConfig(split_ratio=0.3, crop_padding=2.0),
                confidence_threshold=0.4,
            )
            model_ve = EfficientSAM3(
                backbone_type=bb_type, variant=variant, device=device, ft=ft,
                prompt_mode=Sam3PromptMode.VISUAL_EXEMPLAR,
                confidence_threshold=0.4,
            )
        except (ValueError, FileNotFoundError) as e:
            logger.warning("  SKIP %s: %s", bb_label, e)
            continue

        for text_label, with_text in [("no-text", False), ("with-text", True)]:
            refs_c = build_lvis_canvas_refs(
                lvis_api, image_root, ref_img_id, cat_id_to_index, cat_names, with_text, valid_cat_ids,
            )
            model_canvas.fit(refs_c)
            m_c = evaluate_lvis(model_canvas, lvis_api, image_root, target_ids, cat_id_to_index, valid_cat_ids)

            ref_ve = build_lvis_ve_ref(
                lvis_api, image_root, ref_img_id, cat_id_to_index, cat_names, with_text, valid_cat_ids,
            )
            model_ve.fit(ref_ve)
            m_ve = evaluate_lvis(model_ve, lvis_api, image_root, target_ids, cat_id_to_index, valid_cat_ids)

            row_base = {"backbone": bb_label, "dataset": "LVIS(3cat)", "text": text_label}
            rows.append({**row_base, "mode": "CANVAS", **m_c})
            rows.append({**row_base, "mode": "VE", **m_ve})
            logger.info(
                "  LVIS %s %s | CANVAS F1=%.3f | VE F1=%.3f",
                bb_label, text_label, m_c["F1"], m_ve["F1"],
            )

        del model_canvas, model_ve
        torch.cuda.empty_cache()

    return rows


def run_perseg_sweep(device: str) -> list[dict]:
    """Run CANVAS vs VE on PerSeg categories (single-instance)."""
    rows: list[dict] = []
    logger.info("═" * 80)
    logger.info("PERSEG SWEEP — categories: %s", PERSEG_CATEGORIES)

    for bb_type, variant, ft in [("efficientvit", "b1", False), ("efficientvit", "b1", True)]:
        bb_label = f"{bb_type}/{variant}" + ("+ft" if ft else "")
        try:
            model_canvas = EfficientSAM3(
                backbone_type=bb_type, variant=variant, device=device, ft=ft,
                prompt_mode=Sam3PromptMode.CANVAS,
                canvas_config=CanvasConfig(split_ratio=0.3, crop_padding=2.0),
                confidence_threshold=0.4,
            )
            model_ve = EfficientSAM3(
                backbone_type=bb_type, variant=variant, device=device, ft=ft,
                prompt_mode=Sam3PromptMode.VISUAL_EXEMPLAR,
                confidence_threshold=0.4,
            )
        except (ValueError, FileNotFoundError) as e:
            logger.warning("  SKIP %s: %s", bb_label, e)
            continue

        for category in PERSEG_CATEGORIES:
            for text_label, with_text in [("no-text", False), ("with-text", True)]:
                m_c = evaluate_perseg_category(model_canvas, category, "canvas", with_text)
                m_ve = evaluate_perseg_category(model_ve, category, "ve", with_text)

                row_base = {"backbone": bb_label, "dataset": f"PerSeg/{category}", "text": text_label}
                rows.append({**row_base, "mode": "CANVAS", **m_c})
                rows.append({**row_base, "mode": "VE", **m_ve})
                logger.info(
                    "  PerSeg/%s %s | CANVAS F1=%.3f | VE F1=%.3f",
                    category, text_label, m_c["F1"], m_ve["F1"],
                )

        del model_canvas, model_ve
        torch.cuda.empty_cache()

    return rows


def print_final_summary(all_rows: list[dict]) -> None:
    """Print a final summary table."""
    logger.info("═" * 100)
    logger.info("FINAL SUMMARY — CANVAS vs VISUAL_EXEMPLAR")
    logger.info(
        "  %-22s %-14s %-9s %-7s %-6s %-6s %-6s %-8s",
        "backbone", "dataset", "text", "mode", "P", "R", "F1", "ms/img",
    )
    logger.info("─" * 100)

    # Group by (backbone, dataset, text) and show both modes
    from itertools import groupby
    key_fn = lambda r: (r["backbone"], r["dataset"], r["text"])
    sorted_rows = sorted(all_rows, key=key_fn)
    for _key, group in groupby(sorted_rows, key=key_fn):
        group_list = list(group)
        for r in group_list:
            logger.info(
                "  %-22s %-14s %-9s %-7s %-6.3f %-6.3f %-6.3f %-8s",
                r["backbone"], r["dataset"], r["text"], r["mode"],
                r["P"], r["R"], r["F1"], str(r["ms/img"]),
            )

    # Print wins summary
    canvas_wins, ve_wins, ties = 0, 0, 0
    for _key, group in groupby(sorted(all_rows, key=key_fn), key=key_fn):
        group_list = list(group)
        f1s = {r["mode"]: r["F1"] for r in group_list}
        if "CANVAS" in f1s and "VE" in f1s:
            if f1s["CANVAS"] > f1s["VE"]:
                canvas_wins += 1
            elif f1s["VE"] > f1s["CANVAS"]:
                ve_wins += 1
            else:
                ties += 1
    logger.info("─" * 100)
    logger.info("  CANVAS wins: %d | VE wins: %d | Ties: %d", canvas_wins, ve_wins, ties)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["coco", "lvis", "perseg"],
        default=["coco", "lvis", "perseg"],
        help="Which dataset sweeps to run.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s  Sweeps: %s", device, args.datasets)

    all_rows: list[dict] = []
    if "coco" in args.datasets:
        all_rows.extend(run_coco_sweep(device))
    if "lvis" in args.datasets:
        all_rows.extend(run_lvis_sweep(device))
    if "perseg" in args.datasets:
        all_rows.extend(run_perseg_sweep(device))

    print_final_summary(all_rows)
    logger.info("Extended sweep complete.")


if __name__ == "__main__":
    main()
