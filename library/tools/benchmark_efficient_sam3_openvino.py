# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Benchmark EfficientSAM3 **OpenVINO** inference: speed + accuracy.

Companion to ``benchmark_efficient_sam3.py``. Same datasets / modes / metrics
so the JSONL output can be compared row-by-row with the PyTorch baselines
(``benchmark_efficient_sam3_{cuda,xpu}.jsonl``).

Sweeps OpenVINO sub-model variants on CPU and Intel GPU (B60).

Run::

    cd library
    .cuda/bin/python tools/benchmark_efficient_sam3_openvino.py --device GPU
    .cuda/bin/python tools/benchmark_efficient_sam3_openvino.py --device CPU
    .cuda/bin/python tools/benchmark_efficient_sam3_openvino.py --device GPU --variants fp16 int8_sym
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
from instantlearn.models.efficient_sam3 import (
    EfficientSAM3OpenVINO,
    EfficientSAM3OVVariant,
)
from instantlearn.models.sam3.sam3 import Sam3PromptMode

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("benchmark_efficient_sam3_openvino")

DATA_ROOT = Path("/home/rgangire/workspace/data/prompt/geti_datasets/COCO")
RESULTS_DIR = Path("tools/results")
DEFAULT_MODEL_ROOT = Path("efficient-sam3-openvino/efficientvit_b1")
WARMUP_RUNS = 3
BREATHING_ROOM_S = 5
BACKBONE = "efficientvit"
VARIANT = "b1"

DATASETS: dict[str, Path] = {
    "potatoes": DATA_ROOT / "Potatoes",
    "nuts": DATA_ROOT / "Nuts",
    "candies": DATA_ROOT / "Candies",
}

# Map variant key -> (subdirectory name, label used in JSONL/markdown)
VARIANT_ALIASES: dict[str, EfficientSAM3OVVariant] = {
    "fp16": EfficientSAM3OVVariant.FP16,
    "fp32": EfficientSAM3OVVariant.FP32,
    "int8_sym": EfficientSAM3OVVariant.INT8_SYM,
    "int8_asym": EfficientSAM3OVVariant.INT8_ASYM,
    "int8_ptq": EfficientSAM3OVVariant.INT8_PTQ,
    "int4_sym": EfficientSAM3OVVariant.INT4_SYM,
    "int4_asym": EfficientSAM3OVVariant.INT4_ASYM,
}

MODES: list[str] = ["classic", "visual_exemplar"]


# ---------------------------------------------------------------------------
# Dataset loading (identical to PyTorch benchmark)
# ---------------------------------------------------------------------------


@dataclass
class DatasetView:
    """Lightweight view of a COCO dataset for benchmarking."""

    name: str
    image_root: Path
    images: list[dict]
    categories: list[str]
    category_id_map: dict[int, str]


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
        gt_boxes_xyxy: list[list[float]] = []
        gt_cat_ids: list[int] = []
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
# Box IoU + matching (identical to PyTorch benchmark)
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
    matched_gt: set[tuple[int, int]] = set()
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
        used: set[int] = set()
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


def _extract_pred_arrays(
    pred: dict[str, torch.Tensor],
    category_index_to_id: dict[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract pred_boxes (xyxy, Nx4) and pred_labels (N,) as COCO cat ids."""
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
    model: EfficientSAM3OpenVINO,
    dataset: DatasetView,
) -> tuple[list[float], dict[str, int]]:
    """Predict on each image in CLASSIC mode using dataset categories as text."""
    categories = dataset.categories
    cat_index_to_id = dict(enumerate(sorted(dataset.category_id_map.keys())))

    # Warmup using the first image (drives compile + caches)
    first_img = _load_image_tensor(dataset.images[0]["path"])
    for _ in range(WARMUP_RUNS):
        _ = model.predict(Sample(image=first_img, categories=categories))

    latencies: list[float] = []
    totals = {"tp": 0, "fp": 0, "fn": 0, "n_gt": 0, "n_pred": 0}
    for img in dataset.images:
        img_tensor = _load_image_tensor(img["path"])
        sample = Sample(image=img_tensor, categories=categories)
        t0 = time.perf_counter()
        preds = model.predict(sample)
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
    model: EfficientSAM3OpenVINO,
    dataset: DatasetView,
) -> tuple[float, list[float], dict[str, int]]:
    """fit() on one annotated image per category, then predict on the rest."""
    coco_cat_ids = sorted(dataset.category_id_map.keys())
    cat_index_to_id = dict(enumerate(coco_cat_ids))
    id_to_index = {cid: i for i, cid in cat_index_to_id.items()}
    ref_samples: list[Sample] = []
    used_image_ids: set[int] = set()
    for cat_id in coco_cat_ids:
        for img in dataset.images:
            if cat_id in img["gt_cat_ids"]:
                bboxes = [
                    b for b, c in zip(img["gt_boxes_xyxy"], img["gt_cat_ids"], strict=True) if c == cat_id
                ]
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

    t0 = time.perf_counter()
    model.fit(ref_samples)
    fit_time = time.perf_counter() - t0

    target_imgs = [img for img in dataset.images if img["id"] not in used_image_ids]
    if not target_imgs:
        target_imgs = dataset.images

    first_t = _load_image_tensor(target_imgs[0]["path"])
    for _ in range(WARMUP_RUNS):
        _ = model.predict(Sample(image=first_t))

    latencies: list[float] = []
    totals = {"tp": 0, "fp": 0, "fn": 0, "n_gt": 0, "n_pred": 0}
    for img in target_imgs:
        img_t = _load_image_tensor(img["path"])
        t0 = time.perf_counter()
        preds = model.predict(Sample(image=img_t))
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
    ov_device: str,
    dataset: DatasetView,
    mode: str,
    variant_key: str,
    model_dir: Path,
    decoder_device: str | None = None,
    gpu_inference_precision: str | None = None,
    backbone: str = "efficientvit_b1",
) -> dict[str, Any]:
    """Build a model, run one (dataset, mode) cell, tear down."""
    variant_enum = VARIANT_ALIASES[variant_key]
    dec_tag = f"_dec{decoder_device}" if decoder_device and decoder_device != ov_device else ""
    config_name = f"OV_{variant_key}_{ov_device}{dec_tag}"
    logger.info("=" * 80)
    logger.info(
        "CELL ov_device=%s | dataset=%s | mode=%s | variant=%s",
        ov_device, dataset.name, mode, variant_key,
    )

    prompt_mode = (
        Sam3PromptMode.CLASSIC if mode == "classic" else Sam3PromptMode.VISUAL_EXEMPLAR
    )
    record: dict[str, Any] = {
        "device": ov_device,
        "runtime": "openvino",
        "backbone": backbone,
        "dataset": dataset.name,
        "mode": mode,
        "config": config_name,
        "variant": variant_key,
        "precision": variant_key,
        "ft": False,
        "compile_model": False,
        "model_dir": str(model_dir),
        "decoder_device_requested": decoder_device or ov_device,
        "gpu_precision_hint": gpu_inference_precision or "default",
    }

    try:
        model = EfficientSAM3OpenVINO(
            model_dir=str(model_dir),
            device=ov_device,
            prompt_mode=prompt_mode,
            decoder_device=decoder_device,
            gpu_inference_precision=gpu_inference_precision,
        )
        # Record the effective decoder device after the smart defaults run.
        record["decoder_device"] = getattr(model, "_decoder_ov_device", ov_device)
    except Exception as exc:
        logger.exception("Model construction failed for variant=%s mode=%s", variant_key, mode)
        record["error"] = f"construction: {exc}"
        return record

    try:
        if mode == "classic":
            t0 = time.perf_counter()
            categories = dataset.categories
            model.fit(Sample(image=_load_image_tensor(dataset.images[0]["path"]), categories=categories))
            record["fit_s"] = round(time.perf_counter() - t0, 3)
            latencies, totals = run_classic_inference(model, dataset)
            record.update(summarize(latencies, totals))
        else:
            fit_time, latencies, totals = run_visual_exemplar_inference(model, dataset)
            record["fit_s"] = round(fit_time, 3)
            record.update(summarize(latencies, totals))
    except Exception as exc:
        logger.exception("Inference failed for variant=%s mode=%s", variant_key, mode)
        record["error"] = f"inference: {exc}"
    finally:
        del model
        gc.collect()
        with suppress(Exception):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return record


def main() -> None:
    """Sweep variants x modes x datasets on a single OpenVINO device."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, help="OpenVINO device: CPU, GPU, AUTO, GPU.0 ...")
    parser.add_argument(
        "--decoder-device",
        default=None,
        help=(
            "Optional override for the prompt-decoder device (e.g. CPU while "
            "the rest run on GPU). Workaround for Intel GPU plugin numerical "
            "bugs in the EfficientSAM3 decoder graph."
        ),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["fp16"],
        choices=sorted(VARIANT_ALIASES.keys()),
        help="OpenVINO variant subdirectories to sweep (matches subdir name).",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=DEFAULT_MODEL_ROOT,
        help="Root directory holding per-variant subdirectories (e.g. ./efficient-sam3-openvino/efficientvit_b1).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS.keys()),
        choices=sorted(DATASETS.keys()),
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=MODES,
        choices=MODES,
    )
    parser.add_argument(
        "--gpu-precision-hint",
        default=None,
        help=(
            "INFERENCE_PRECISION_HINT for GPU encoder sub-models "
            "(e.g. 'f32' or 'f16'). Default: let the runtime decide "
            "(f32 for EfficientViT to avoid NaN; f16 safe for TinyViT/RepViT)."
        ),
    )
    parser.add_argument(
        "--backbone",
        default="efficientvit_b1",
        help="Backbone label for JSONL records (e.g. 'tinyvit_11m', 'repvit_m1_1').",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory to write benchmark_efficient_sam3_openvino*.jsonl into.",
    )
    args = parser.parse_args()

    ov_device: str = args.device.upper() if args.device.lower() in {"cpu", "gpu", "auto"} else args.device
    decoder_device: str | None = (
        args.decoder_device.upper()
        if args.decoder_device and args.decoder_device.lower() in {"cpu", "gpu", "auto"}
        else args.decoder_device
    )
    results_dir: Path = args.results_dir
    dev_tag = ov_device.replace(".", "_")
    if decoder_device and decoder_device != ov_device:
        dev_tag += f"_dec{decoder_device.replace('.', '_')}"
    jsonl_path = results_dir / f"benchmark_efficient_sam3_openvino_{dev_tag}.jsonl"
    # Start each sweep from a clean slate.
    if jsonl_path.exists():
        jsonl_path.unlink()

    logger.info("OV device: %s | variants: %s | results: %s", ov_device, args.variants, jsonl_path)

    selected_datasets = {name: DATASETS[name] for name in args.datasets}
    datasets = {name: load_dataset(name, root) for name, root in selected_datasets.items()}
    for ds in datasets.values():
        logger.info("Loaded %s: %d images, cats=%s", ds.name, len(ds.images), ds.categories)

    # Validate variant subdirs exist
    variant_dirs: dict[str, Path] = {}
    for v in args.variants:
        subdir = args.model_root / VARIANT_ALIASES[v].value
        if not subdir.is_dir():
            logger.warning("Variant '%s' missing — expected %s. Skipping.", v, subdir)
            continue
        variant_dirs[v] = subdir
    if not variant_dirs:
        logger.error("No variant directories found under %s.", args.model_root)
        return

    total_cells = len(variant_dirs) * len(args.modes) * len(datasets)
    cell_idx = 0
    for variant_key, model_dir in variant_dirs.items():
        for mode in args.modes:
            for ds in datasets.values():
                cell_idx += 1
                logger.info("Progress: cell %d/%d", cell_idx, total_cells)
                record = run_cell(
                    ov_device=ov_device,
                    dataset=ds,
                    mode=mode,
                    variant_key=variant_key,
                    model_dir=model_dir,
                    decoder_device=decoder_device,
                    gpu_inference_precision=args.gpu_precision_hint,
                    backbone=args.backbone,
                )
                append_result(jsonl_path, record)
                if cell_idx < total_cells:
                    logger.info("Sleeping %ds for breathing room...", BREATHING_ROOM_S)
                    time.sleep(BREATHING_ROOM_S)

    logger.info("Done. JSONL: %s", jsonl_path)


if __name__ == "__main__":
    main()
