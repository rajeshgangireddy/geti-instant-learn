# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Benchmark EfficientSAM3 inference: speed + accuracy across configs and modes.

Sweeps four configurations on a fixed backbone (efficientvit/b1) to attribute
the contribution of recent changes:

- C1 baseline:     precision="fp32", ft=False, compile_model=False
- C2 bf16:         precision="bf16", ft=False, compile_model=False
- C3 bf16+compile: precision="bf16", ft=False, compile_model=True
- C4 bf16+ft:      precision="bf16", ft=True,  compile_model=False

For each config x mode (CLASSIC / VISUAL_EXEMPLAR) x dataset (Potatoes / Nuts /
Candies), measures:

- fit time (seconds)
- per-image predict latency: median, p90 (after warmup runs)
- detection precision / recall / F1 at IoU>=0.5 (greedy box matching)

Run once per device (uv runs from .cuda or .xpu venv). Intermediate results are
appended to a JSONL file after each cell to survive interruptions; a markdown
summary is regenerated each time.

Usage:
    cd library
    .cuda/bin/python tools/benchmark_efficient_sam3.py --device cuda
    .xpu/bin/python  tools/benchmark_efficient_sam3.py --device xpu
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import statistics
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO  # type: ignore[import-untyped]
from torchvision.transforms.functional import pil_to_tensor

from instantlearn.data import Sample
from instantlearn.models import EfficientSAM3
from instantlearn.models.sam3.sam3 import Sam3PromptMode

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("benchmark_efficient_sam3")

DATA_ROOT = Path("/home/rgangire/workspace/data/prompt/geti_datasets/COCO")
RESULTS_DIR = Path("tools/results")
WARMUP_RUNS = 3
BREATHING_ROOM_S = 10
BACKBONE = "efficientvit"
VARIANT = "b1"

DATASETS: dict[str, Path] = {
    "potatoes": DATA_ROOT / "Potatoes",
    "nuts": DATA_ROOT / "Nuts",
    "candies": DATA_ROOT / "Candies",
}

CONFIGS: list[tuple[str, str, bool, bool]] = [  # (name, precision, ft, compile_model)
    ("C1_baseline_fp32", "fp32", False, False),
    ("C2_bf16", "bf16", False, False),
    ("C3_bf16_compile", "bf16", False, True),
    ("C4_bf16_ft", "bf16", True, False),
]

MODES: list[str] = ["classic", "visual_exemplar"]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


@dataclass
class DatasetView:
    """Lightweight view of a COCO dataset for benchmarking."""

    name: str
    image_root: Path
    images: list[dict]         # [{id, file_name, path, gt_boxes_xyxy: list, gt_cat_ids: list}, ...]
    categories: list[str]      # category names ordered by id
    category_id_map: dict[int, str]  # coco_cat_id -> name


def load_dataset(name: str, root: Path) -> DatasetView:
    """Load a COCO-format dataset folder into a DatasetView."""
    ann_path = root / "annotations" / "instances_default.json"
    img_root = root / "images" / "default"
    coco = COCO(str(ann_path))
    cat_id_map = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
    categories = [cat_id_map[i] for i in sorted(cat_id_map)]
    images: list[dict] = []
    for img_info in coco.loadImgs(coco.getImgIds()):
        ann_ids = coco.getAnnIds(imgIds=img_info["id"])
        anns = coco.loadAnns(ann_ids)
        gt_boxes_xyxy = []
        gt_cat_ids = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            gt_boxes_xyxy.append([x, y, x + w, y + h])
            gt_cat_ids.append(ann["category_id"])
        images.append({
            "id": img_info["id"],
            "file_name": img_info["file_name"],
            "path": str(img_root / img_info["file_name"]),
            "gt_boxes_xyxy": gt_boxes_xyxy,
            "gt_cat_ids": gt_cat_ids,
        })
    return DatasetView(
        name=name,
        image_root=img_root,
        images=images,
        categories=categories,
        category_id_map=cat_id_map,
    )


# ---------------------------------------------------------------------------
# Box IoU + matching
# ---------------------------------------------------------------------------


def _box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between N boxes and M boxes (xyxy). Returns (N, M)."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    xa1, ya1, xa2, ya2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    xb1, yb1, xb2, yb2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    inter_w = np.clip(np.minimum(xa2, xb2) - np.maximum(xa1, xb1), 0, None)
    inter_h = np.clip(np.minimum(ya2, yb2) - np.maximum(ya1, yb1), 0, None)
    inter = inter_w * inter_h
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def match_and_score(
    pred_boxes: np.ndarray,
    pred_labels: np.ndarray,
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
    iou_threshold: float = 0.5,
) -> tuple[int, int, int]:
    """Greedy 1:1 matching per class. Returns (tp, fp, fn)."""
    tp = 0
    matched_gt = set()
    if pred_boxes.size == 0:
        return 0, 0, len(gt_boxes)
    if gt_boxes.size == 0:
        return 0, len(pred_boxes), 0
    for cls in np.unique(np.concatenate([pred_labels, gt_labels])):
        p_idx = np.where(pred_labels == cls)[0]
        g_idx = np.where(gt_labels == cls)[0]
        if p_idx.size == 0 or g_idx.size == 0:
            continue
        ious = _box_iou_xyxy(pred_boxes[p_idx], gt_boxes[g_idx])
        # Greedy: iterate predictions in input order
        used = set()
        for pi, _ in enumerate(p_idx):
            row = ious[pi].copy()
            for gi in used:
                row[gi] = -1.0
            best = int(np.argmax(row))
            if row[best] >= iou_threshold:
                used.add(best)
                matched_gt.add((int(cls), int(g_idx[best])))
                tp += 1
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def _load_image_tensor(path: str) -> torch.Tensor:
    """Load image as (C, H, W) uint8 tensor in [0, 255]."""
    return pil_to_tensor(Image.open(path).convert("RGB"))


def _device_sync(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.synchronize()


def _extract_pred_arrays(
    pred: dict[str, torch.Tensor],
    category_index_to_id: dict[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract pred_boxes (xyxy, Nx4) and pred_labels (N,) as COCO cat ids.

    EfficientSAM3 returns boxes as ``(N, 5)`` = ``[x1, y1, x2, y2, score]`` in
    absolute image coordinates; we slice off the score column here.
    """
    boxes_t = pred.get("pred_boxes")
    labels_t = pred.get("pred_labels")
    if boxes_t is None or labels_t is None or boxes_t.numel() == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    boxes_full = boxes_t.detach().cpu().numpy().astype(np.float32)
    boxes = boxes_full[:, :4] if boxes_full.shape[-1] >= 4 else boxes_full
    labels_idx = labels_t.detach().cpu().numpy().astype(np.int64)
    labels = np.array([category_index_to_id.get(int(i), -1) for i in labels_idx], dtype=np.int64)
    return boxes, labels


def run_classic_inference(
    model: EfficientSAM3,
    dataset: DatasetView,
    device: str,
) -> tuple[list[float], dict[str, int]]:
    """Predict on each image in CLASSIC mode using dataset categories as text prompts."""
    categories = dataset.categories
    cat_index_to_id = dict(enumerate(sorted(dataset.category_id_map.keys())))

    # Warmup using the first image
    first_img = _load_image_tensor(dataset.images[0]["path"])
    for _ in range(WARMUP_RUNS):
        _ = model.predict(Sample(image=first_img, categories=categories))
    _device_sync(device)

    latencies: list[float] = []
    totals = {"tp": 0, "fp": 0, "fn": 0, "n_gt": 0, "n_pred": 0}
    for img in dataset.images:
        img_tensor = _load_image_tensor(img["path"])
        sample = Sample(image=img_tensor, categories=categories)
        _device_sync(device)
        t0 = time.perf_counter()
        preds = model.predict(sample)
        _device_sync(device)
        latencies.append(time.perf_counter() - t0)

        pred = preds[0]
        pred_boxes, pred_labels = _extract_pred_arrays(pred, cat_index_to_id)
        gt_boxes = np.array(img["gt_boxes_xyxy"], dtype=np.float32) if img["gt_boxes_xyxy"] else np.zeros((0, 4))
        gt_labels = np.array(img["gt_cat_ids"], dtype=np.int64) if img["gt_cat_ids"] else np.zeros((0,), dtype=np.int64)
        tp, fp, fn = match_and_score(pred_boxes, pred_labels, gt_boxes, gt_labels)
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
        totals["n_gt"] += len(gt_boxes)
        totals["n_pred"] += len(pred_boxes)
    return latencies, totals


def run_visual_exemplar_inference(
    model: EfficientSAM3,
    dataset: DatasetView,
    device: str,
) -> tuple[float, list[float], dict[str, int]]:
    """fit() on one annotated image per category, then predict on the rest."""
    # Build one reference Sample per category: pick the first image containing that category
    coco_cat_ids = sorted(dataset.category_id_map.keys())
    cat_index_to_id = dict(enumerate(coco_cat_ids))
    id_to_index = {cid: i for i, cid in cat_index_to_id.items()}
    ref_samples: list[Sample] = []
    used_image_ids: set[int] = set()
    for cat_id in coco_cat_ids:
        for img in dataset.images:
            if cat_id in img["gt_cat_ids"]:
                # Aggregate all bboxes for this category from this image
                bboxes = [b for b, c in zip(img["gt_boxes_xyxy"], img["gt_cat_ids"], strict=True) if c == cat_id]
                if not bboxes:
                    continue
                img_t = _load_image_tensor(img["path"])
                ref_samples.append(Sample(
                    image=img_t,
                    bboxes=bboxes,
                    category_ids=[id_to_index[cat_id]] * len(bboxes),
                    categories=[dataset.category_id_map[cat_id]] * len(bboxes),
                ))
                used_image_ids.add(img["id"])
                break

    _device_sync(device)
    t0 = time.perf_counter()
    model.fit(ref_samples)
    _device_sync(device)
    fit_time = time.perf_counter() - t0

    # Build target list: exclude images used as references to avoid trivial matches
    target_imgs = [img for img in dataset.images if img["id"] not in used_image_ids]
    if not target_imgs:
        target_imgs = dataset.images  # tiny datasets: fall back to using all

    # Warmup on the first target
    first_t = _load_image_tensor(target_imgs[0]["path"])
    for _ in range(WARMUP_RUNS):
        _ = model.predict(Sample(image=first_t))
    _device_sync(device)

    latencies: list[float] = []
    totals = {"tp": 0, "fp": 0, "fn": 0, "n_gt": 0, "n_pred": 0}
    for img in target_imgs:
        img_t = _load_image_tensor(img["path"])
        _device_sync(device)
        t0 = time.perf_counter()
        preds = model.predict(Sample(image=img_t))
        _device_sync(device)
        latencies.append(time.perf_counter() - t0)

        pred = preds[0]
        pred_boxes, pred_labels = _extract_pred_arrays(pred, cat_index_to_id)
        gt_boxes = np.array(img["gt_boxes_xyxy"], dtype=np.float32) if img["gt_boxes_xyxy"] else np.zeros((0, 4))
        gt_labels = np.array(img["gt_cat_ids"], dtype=np.int64) if img["gt_cat_ids"] else np.zeros((0,), dtype=np.int64)
        tp, fp, fn = match_and_score(pred_boxes, pred_labels, gt_boxes, gt_labels)
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
        totals["n_gt"] += len(gt_boxes)
        totals["n_pred"] += len(pred_boxes)
    return fit_time, latencies, totals


# ---------------------------------------------------------------------------
# Result aggregation + IO
# ---------------------------------------------------------------------------


def summarize(latencies: list[float], totals: dict[str, int]) -> dict[str, Any]:
    """Compute median/p90 latency and P/R/F1 from totals."""
    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "n_images": len(latencies),
        "median_ms": round(statistics.median(latencies) * 1000, 2) if latencies else None,
        "p90_ms": round(np.percentile(latencies, 90) * 1000, 2) if latencies else None,
        "mean_ms": round(statistics.mean(latencies) * 1000, 2) if latencies else None,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_pred": totals["n_pred"],
        "n_gt": totals["n_gt"],
        "det_precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def append_result(path: Path, record: dict[str, Any]) -> None:
    """Append a single result record to the JSONL log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------


def run_cell(
    device: str,
    dataset: DatasetView,
    mode: str,
    config_name: str,
    precision: str,
    ft: bool,
    compile_model: bool,
) -> dict[str, Any]:
    """Build a model, run one (dataset, mode) cell, tear down."""
    logger.info("=" * 80)
    logger.info(
        "CELL device=%s | dataset=%s | mode=%s | config=%s (prec=%s ft=%s compile=%s)",
        device, dataset.name, mode, config_name, precision, ft, compile_model,
    )
    prompt_mode = (
        Sam3PromptMode.CLASSIC if mode == "classic" else Sam3PromptMode.VISUAL_EXEMPLAR
    )
    record: dict[str, Any] = {
        "device": device,
        "dataset": dataset.name,
        "mode": mode,
        "config": config_name,
        "precision": precision,
        "ft": ft,
        "compile_model": compile_model,
    }
    try:
        model = EfficientSAM3(
            backbone_type=BACKBONE,
            variant=VARIANT,
            device=device,
            precision=precision,
            ft=ft,
            compile_model=compile_model,
            prompt_mode=prompt_mode,
        )
    except Exception as exc:
        logger.exception("Model construction failed for config=%s mode=%s", config_name, mode)
        record["error"] = f"construction: {exc}"
        return record

    try:
        if mode == "classic":
            # For CLASSIC mode, fit just registers category names (very fast).
            t0 = time.perf_counter()
            # Build a Sample with just categories — no image needed for fit()
            categories = dataset.categories
            model.fit(Sample(image=_load_image_tensor(dataset.images[0]["path"]), categories=categories))
            record["fit_s"] = round(time.perf_counter() - t0, 3)
            latencies, totals = run_classic_inference(model, dataset, device)
            record.update(summarize(latencies, totals))
        else:
            fit_time, latencies, totals = run_visual_exemplar_inference(model, dataset, device)
            record["fit_s"] = round(fit_time, 3)
            record.update(summarize(latencies, totals))
    except Exception as exc:
        logger.exception("Inference failed for config=%s mode=%s", config_name, mode)
        record["error"] = f"inference: {exc}"
    finally:
        # Aggressively release model + GPU memory before sleep
        del model
        gc.collect()
        with suppress(Exception):
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif device == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.empty_cache()
    return record


def main() -> None:
    """Sweep configs x modes x datasets for a single device."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, choices=["cuda", "xpu"])
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory to write benchmark_efficient_sam3.{jsonl,md} into.",
    )
    args = parser.parse_args()
    device: str = args.device
    results_dir: Path = args.results_dir
    jsonl_path = results_dir / f"benchmark_efficient_sam3_{device}.jsonl"
    md_path = results_dir / "benchmark_efficient_sam3.md"

    logger.info("Device: %s | results: %s", device, jsonl_path)

    # Load all datasets up front (small files)
    datasets = {name: load_dataset(name, root) for name, root in DATASETS.items()}
    for ds in datasets.values():
        logger.info("Loaded %s: %d images, cats=%s", ds.name, len(ds.images), ds.categories)

    total_cells = len(datasets) * len(MODES) * len(CONFIGS)
    cell_idx = 0
    for config_name, precision, ft, compile_model in CONFIGS:
        for mode in MODES:
            for ds in datasets.values():
                cell_idx += 1
                logger.info("Progress: cell %d/%d", cell_idx, total_cells)
                record = run_cell(
                    device=device,
                    dataset=ds,
                    mode=mode,
                    config_name=config_name,
                    precision=precision,
                    ft=ft,
                    compile_model=compile_model,
                )
                append_result(jsonl_path, record)
                render_markdown_for_all(results_dir, md_path)
                # Breathing room between experiments (skip after the last cell)
                if cell_idx < total_cells:
                    logger.info("Sleeping %ds for breathing room...", BREATHING_ROOM_S)
                    time.sleep(BREATHING_ROOM_S)

    logger.info("Done. JSONL: %s | markdown: %s", jsonl_path, md_path)


def render_markdown_for_all(results_dir: Path, md_path: Path) -> None:
    """Merge all per-device JSONL files into one markdown report."""
    all_records: list[dict[str, Any]] = []
    for p in sorted(results_dir.glob("benchmark_efficient_sam3_*.jsonl")):
        all_records.extend(
            json.loads(line) for line in p.read_text().splitlines() if line.strip()
        )
    if not all_records:
        return
    lines: list[str] = [
        "# EfficientSAM3 benchmark — speed + accuracy attribution",
        "",
        f"Backbone: `{BACKBONE}/{VARIANT}` | warmup runs: {WARMUP_RUNS} | "
        f"IoU threshold: 0.5 | breathing room: {BREATHING_ROOM_S}s",
        "",
        "Configs: C1=fp32 baseline, C2=bf16 autocast, C3=bf16+torch.compile, C4=bf16+fine-tuned weights.",
        "",
        "| device | dataset | mode | config | precision | n_img | median ms | p90 ms | "
        "P | R | F1 | tp | fp | fn | fit s | error |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    lines.extend(
        f"| {r.get('device', '-')} | {r.get('dataset', '-')} | {r.get('mode', '-')} | "
        f"{r.get('config', '-')} | {r.get('precision', '-')} | {r.get('n_images', '-')} | "
        f"{r.get('median_ms', '-')} | {r.get('p90_ms', '-')} | "
        f"{r.get('det_precision', '-')} | {r.get('recall', '-')} | {r.get('f1', '-')} | "
        f"{r.get('tp', '-')} | {r.get('fp', '-')} | {r.get('fn', '-')} | "
        f"{r.get('fit_s', '-')} | {r.get('error', '')} |"
        for r in all_records
    )
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
