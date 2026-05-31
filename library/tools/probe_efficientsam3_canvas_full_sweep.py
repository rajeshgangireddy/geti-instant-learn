"""Full CanvasConfig parameter sweep for EfficientSAM3 across multiple scenarios.

Scenarios (each exercises a different canvas code path):

  S1: PerSeg 1-shot 1-cat  → _build_canvas_vertical
      Tunes: split_ratio × text
  S2: PerSeg 3-shot 1-cat  → _build_canvas_multishot (crop_padding active)
      Tunes: split_ratio × crop_padding × text
  S3: COCO Potatoes 1-shot 1-cat  → _build_canvas_vertical
      Tunes: split_ratio × text  (regression check)
  S4: COCO Nuts 1-shot 2-cat  → _build_canvas_shared_grouped / spaced / sequential
      Tunes: split_ratio × share_vision × text

Goal: pick the best DEFAULT CanvasConfig for EfficientSAM3.

Run:
  .cuda/bin/python -u tools/probe_efficientsam3_canvas_full_sweep.py 2>&1 | tee /tmp/canvas_full.log
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from pycocotools.coco import COCO

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_efficient_sam3 import _load_image_tensor, match_and_score  # noqa: E402
from probe_efficientsam3_ve_coco import load_coco, pick_reference_image  # noqa: E402

from instantlearn.data import Sample  # noqa: E402
from instantlearn.models import EfficientSAM3  # noqa: E402
from instantlearn.models.sam3.sam3 import CanvasConfig, Sam3PromptMode  # noqa: E402

logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO, force=True)
logger = logging.getLogger("canvas_full")


def say(msg: str) -> None:
    print(msg, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

PERSEG = Path("/home/rgangire/workspace/data/prompt/PerSeg")
COCO_ROOT = Path("/home/rgangire/workspace/data/prompt/geti_datasets/COCO")

SPLIT_RATIOS = [0.30, 0.40, 0.50]
CROP_PADDINGS = [1.5, 2.0, 3.0]
SHARE_VISIONS = ["grouped", "spaced", False]
TEXTS = [False, True]

BACKBONE = ("efficientvit", "b1", True)  # (type, variant, ft)
CONFIDENCE = 0.4
MAX_TARGETS_PERSEG = 6
MAX_TARGETS_COCO = 20

PERSEG_CATS_1SHOT = ["dog", "elephant", "teapot", "backpack", "clock"]
PERSEG_CATS_3SHOT = ["dog", "elephant"]  # multi-shot subset (slow)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def bbox_from_mask(mask_path: Path) -> np.ndarray:
    import cv2
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return np.array([[0, 0, 1, 1]], dtype=np.float32)
    return np.array([[xs.min(), ys.min(), xs.max(), ys.max()]], dtype=np.float32)


def perseg_paths(cat: str) -> tuple[list[Path], list[Path]]:
    img_dir = PERSEG / "Images" / cat
    ann_dir = PERSEG / "Annotations" / cat
    imgs = sorted(img_dir.glob("*.jpg"))
    masks = sorted(ann_dir.glob("*.png"))
    return imgs, masks


def make_model(canvas_cfg: CanvasConfig, device: str) -> EfficientSAM3:
    bt, var, ft = BACKBONE
    return EfficientSAM3(
        backbone_type=bt, variant=var, ft=ft, device=device,
        prompt_mode=Sam3PromptMode.CANVAS, confidence_threshold=CONFIDENCE,
        canvas_config=canvas_cfg,
    )


def fit_predict_score(
    model: EfficientSAM3,
    ref_samples: list[Sample],
    targets: list[tuple[Path, np.ndarray, np.ndarray]],
) -> dict:
    """Fit model on refs, predict on targets, return P/R/F1 over all targets."""
    model.fit(ref_samples if len(ref_samples) > 1 else ref_samples[0])
    tp = fp = fn = 0
    t0 = time.perf_counter()
    for img_path, gt_boxes, gt_labels in targets:
        pred = model.predict(Sample(image=_load_image_tensor(img_path)))[0]
        boxes = pred["pred_boxes"].cpu().numpy()
        if boxes.size > 0:
            pb = boxes[:, :4].astype(np.float32)
            pl = pred["pred_labels"].cpu().numpy().astype(np.int64)
        else:
            pb = np.empty((0, 4), dtype=np.float32)
            pl = np.empty((0,), dtype=np.int64)
        t, f, n = match_and_score(pb, pl, gt_boxes, gt_labels, 0.5)
        tp += t; fp += f; fn += n
    elapsed_ms = (time.perf_counter() - t0) * 1000 / max(1, len(targets))
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "P": p, "R": r, "F1": f1, "ms": elapsed_ms}


# ─────────────────────────────────────────────────────────────────────────────
# S1: PerSeg 1-shot 1-cat — split_ratio × text
# ─────────────────────────────────────────────────────────────────────────────


def s1_perseg_1shot(device: str) -> list[dict]:
    say("=" * 100)
    say("S1: PerSeg 1-shot 1-cat  (path: _build_canvas_vertical)  — tune split_ratio × text")
    say("=" * 100)
    rows = []
    for cat in PERSEG_CATS_1SHOT:
        imgs, masks = perseg_paths(cat)
        ref_img = imgs[0]
        ref_bbox = bbox_from_mask(masks[0])
        targets = []
        for ip, mp in list(zip(imgs[1:], masks[1:]))[:MAX_TARGETS_PERSEG]:
            targets.append((ip, bbox_from_mask(mp), np.array([0], dtype=np.int64)))

        for sr in SPLIT_RATIOS:
            for text_on in TEXTS:
                cfg = CanvasConfig(split_ratio=sr, crop_padding=2.0)
                model = make_model(cfg, device)
                cat_text = cat if text_on else "object"
                ref = Sample(image_path=str(ref_img), bboxes=ref_bbox,
                             category_ids=np.array([0]), categories=[cat_text])
                r = fit_predict_score(model, [ref], targets)
                row = {"scenario": "S1_perseg_1shot", "cat": cat, "sr": sr,
                       "cp": 2.0, "sv": "-", "text": text_on, **r}
                say(f"  {cat:<10s} sr={sr:.2f} text={'yes' if text_on else 'no ':<3s} | "
                    f"F1={r['F1']:.3f} P={r['P']:.2f} R={r['R']:.2f}  tp={r['tp']} fp={r['fp']} fn={r['fn']}")
                rows.append(row)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# S2: PerSeg 3-shot 1-cat — split_ratio × crop_padding × text
# ─────────────────────────────────────────────────────────────────────────────


def s2_perseg_3shot(device: str) -> list[dict]:
    say("=" * 100)
    say("S2: PerSeg 3-shot 1-cat  (path: _build_canvas_multishot)  — tune split_ratio × crop_padding × text")
    say("=" * 100)
    rows = []
    for cat in PERSEG_CATS_3SHOT:
        imgs, masks = perseg_paths(cat)
        # Use first 3 as refs (each must be Sample with its own bbox)
        ref_samples_base = []
        for ip, mp in list(zip(imgs[:3], masks[:3])):
            ref_samples_base.append((ip, bbox_from_mask(mp)))
        # Targets = next MAX_TARGETS_PERSEG images
        targets = []
        for ip, mp in list(zip(imgs[3:], masks[3:]))[:MAX_TARGETS_PERSEG]:
            targets.append((ip, bbox_from_mask(mp), np.array([0], dtype=np.int64)))

        for sr in SPLIT_RATIOS:
            for cp in CROP_PADDINGS:
                for text_on in TEXTS:
                    cfg = CanvasConfig(split_ratio=sr, crop_padding=cp)
                    model = make_model(cfg, device)
                    cat_text = cat if text_on else "object"
                    refs = [
                        Sample(image_path=str(ip), bboxes=bb,
                               category_ids=np.array([0]), categories=[cat_text])
                        for ip, bb in ref_samples_base
                    ]
                    r = fit_predict_score(model, refs, targets)
                    row = {"scenario": "S2_perseg_3shot", "cat": cat, "sr": sr,
                           "cp": cp, "sv": "-", "text": text_on, **r}
                    say(f"  {cat:<10s} sr={sr:.2f} cp={cp:.1f} text={'yes' if text_on else 'no ':<3s} | "
                        f"F1={r['F1']:.3f}  tp={r['tp']} fp={r['fp']} fn={r['fn']}")
                    rows.append(row)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# S3: COCO Potatoes 1-shot 1-cat — split_ratio × text  (regression check)
# ─────────────────────────────────────────────────────────────────────────────


def s3_coco_potatoes(device: str) -> list[dict]:
    say("=" * 100)
    say("S3: COCO Potatoes 1-shot 1-cat  (path: _build_canvas_vertical)  — tune split_ratio × text")
    say("=" * 100)
    coco, image_root, image_ids, cat_names, cat_id_to_index = load_coco(COCO_ROOT / "Potatoes")
    cat_ids = sorted(coco.getCatIds())
    ref_id = pick_reference_image(coco, image_ids, cat_ids)
    target_ids = [i for i in image_ids if i != ref_id][:MAX_TARGETS_COCO]

    # Build ref bbox (one per cat in image; here 1 cat)
    img_info = coco.loadImgs(ref_id)[0]
    ref_img_path = image_root / img_info["file_name"]
    anns = coco.loadAnns(coco.getAnnIds(imgIds=ref_id))
    a = anns[0]
    x, y, w, h = a["bbox"]
    ref_bbox = np.array([[x, y, x + w, y + h]], dtype=np.float32)

    # Pre-build targets list (img_path, gt_boxes, gt_labels)
    targets = []
    for tid in target_ids:
        info = coco.loadImgs(tid)[0]
        tp_path = image_root / info["file_name"]
        anns_t = coco.loadAnns(coco.getAnnIds(imgIds=tid))
        gt = np.array(
            [[a["bbox"][0], a["bbox"][1], a["bbox"][0]+a["bbox"][2], a["bbox"][1]+a["bbox"][3]] for a in anns_t],
            dtype=np.float32,
        ) if anns_t else np.zeros((0, 4), dtype=np.float32)
        gtl = np.array([cat_id_to_index[a["category_id"]] for a in anns_t], dtype=np.int64)
        targets.append((tp_path, gt, gtl))

    rows = []
    for sr in SPLIT_RATIOS:
        for text_on in TEXTS:
            cfg = CanvasConfig(split_ratio=sr, crop_padding=2.0)
            model = make_model(cfg, device)
            cat_text = cat_names[0] if text_on else "object"
            ref = Sample(image_path=str(ref_img_path), bboxes=ref_bbox,
                         category_ids=np.array([0]), categories=[cat_text])
            r = fit_predict_score(model, [ref], targets)
            rows.append({"scenario": "S3_potatoes", "cat": "potato", "sr": sr,
                         "cp": 2.0, "sv": "-", "text": text_on, **r})
            say(f"  potato     sr={sr:.2f} text={'yes' if text_on else 'no ':<3s} | "
                f"F1={r['F1']:.3f} P={r['P']:.2f} R={r['R']:.2f}  tp={r['tp']} fp={r['fp']} fn={r['fn']}")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# S4: COCO Nuts 1-shot 2-cat — split_ratio × share_vision × text
# ─────────────────────────────────────────────────────────────────────────────


def s4_coco_nuts(device: str) -> list[dict]:
    say("=" * 100)
    say("S4: COCO Nuts 1-shot 2-cat  (path: shared canvas)  — tune split_ratio × share_vision × text")
    say("=" * 100)
    coco, image_root, image_ids, cat_names, cat_id_to_index = load_coco(COCO_ROOT / "Nuts")
    cat_ids = sorted(coco.getCatIds())
    ref_id = pick_reference_image(coco, image_ids, cat_ids)
    target_ids = [i for i in image_ids if i != ref_id][:MAX_TARGETS_COCO]

    img_info = coco.loadImgs(ref_id)[0]
    ref_img_path = image_root / img_info["file_name"]
    anns = coco.loadAnns(coco.getAnnIds(imgIds=ref_id))
    # Build 1 sample per category (1-shot per cat)
    seen_cats: set[int] = set()

    ref_samples_data: list[tuple[np.ndarray, int]] = []
    for a in anns:
        idx = cat_id_to_index[a["category_id"]]
        if idx in seen_cats:
            continue
        seen_cats.add(idx)
        x, y, w, h = a["bbox"]
        ref_samples_data.append((np.array([[x, y, x+w, y+h]], dtype=np.float32), idx))

    targets = []
    for tid in target_ids:
        info = coco.loadImgs(tid)[0]
        tp_path = image_root / info["file_name"]
        anns_t = coco.loadAnns(coco.getAnnIds(imgIds=tid))
        gt = np.array(
            [[a["bbox"][0], a["bbox"][1], a["bbox"][0]+a["bbox"][2], a["bbox"][1]+a["bbox"][3]] for a in anns_t],
            dtype=np.float32,
        ) if anns_t else np.zeros((0, 4), dtype=np.float32)
        gtl = np.array([cat_id_to_index[a["category_id"]] for a in anns_t], dtype=np.int64)
        targets.append((tp_path, gt, gtl))

    rows = []
    for sr in SPLIT_RATIOS:
        for sv in SHARE_VISIONS:
            for text_on in TEXTS:
                cfg = CanvasConfig(split_ratio=sr, crop_padding=2.0, share_vision=sv)
                model = make_model(cfg, device)
                refs = [
                    Sample(image_path=str(ref_img_path), bboxes=bb,
                           category_ids=np.array([idx]),
                           categories=[cat_names[idx] if text_on else "visual"])
                    for bb, idx in ref_samples_data
                ]
                r = fit_predict_score(model, refs, targets)
                sv_label = sv if isinstance(sv, str) else "seq"
                rows.append({"scenario": "S4_nuts", "cat": "nuts", "sr": sr,
                             "cp": 2.0, "sv": sv_label, "text": text_on, **r})
                say(f"  nuts       sr={sr:.2f} sv={sv_label:<8s} text={'yes' if text_on else 'no ':<3s} | "
                    f"F1={r['F1']:.3f}  tp={r['tp']} fp={r['fp']} fn={r['fn']}")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────


def summarize(rows: list[dict]) -> None:
    import collections

    say("\n" + "=" * 100)
    say("MACRO-F1 by (split_ratio × text) — averaged over all 1-shot scenarios (S1+S3)")
    say("=" * 100)
    say(f"  {'sr':<6s} {'text=no':<10s} {'text=yes':<10s}")
    for sr in SPLIT_RATIOS:
        cells = []
        for t in (False, True):
            sel = [r for r in rows if r["scenario"] in {"S1_perseg_1shot", "S3_potatoes"}
                   and r["sr"] == sr and r["text"] == t]
            f1 = float(np.mean([r["F1"] for r in sel])) if sel else 0.0
            cells.append(f"{f1:.3f}")
        say(f"  {sr:.2f}   {cells[0]:<10s} {cells[1]:<10s}")

    say("\n" + "=" * 100)
    say("MACRO-F1 by (split_ratio × crop_padding) — multi-shot (S2)")
    say("=" * 100)
    say(f"  {'sr':<6s} " + " ".join(f"cp={cp:.1f}".ljust(10) for cp in CROP_PADDINGS))
    for sr in SPLIT_RATIOS:
        cells = []
        for cp in CROP_PADDINGS:
            sel = [r for r in rows if r["scenario"] == "S2_perseg_3shot"
                   and r["sr"] == sr and r["cp"] == cp]
            f1 = float(np.mean([r["F1"] for r in sel])) if sel else 0.0
            cells.append(f"{f1:.3f}")
        say(f"  {sr:.2f}   " + " ".join(c.ljust(10) for c in cells))

    say("\n" + "=" * 100)
    say("MACRO-F1 by (split_ratio × share_vision) — multi-cat (S4)")
    say("=" * 100)
    sv_labels = ["grouped", "spaced", "seq"]
    say(f"  {'sr':<6s} " + " ".join(f"sv={s}".ljust(14) for s in sv_labels))
    for sr in SPLIT_RATIOS:
        cells = []
        for s in sv_labels:
            sel = [r for r in rows if r["scenario"] == "S4_nuts"
                   and r["sr"] == sr and r["sv"] == s]
            f1 = float(np.mean([r["F1"] for r in sel])) if sel else 0.0
            cells.append(f"{f1:.3f}")
        say(f"  {sr:.2f}   " + " ".join(c.ljust(14) for c in cells))

    say("\n" + "=" * 100)
    say("TEXT EFFECT (no-text vs with-text) — macro-F1 per scenario at sr=0.50")
    say("=" * 100)
    for scen in ["S1_perseg_1shot", "S2_perseg_3shot", "S3_potatoes", "S4_nuts"]:
        no_t = [r["F1"] for r in rows if r["scenario"] == scen and r["sr"] == 0.50 and not r["text"]]
        ye_t = [r["F1"] for r in rows if r["scenario"] == scen and r["sr"] == 0.50 and r["text"]]
        a = float(np.mean(no_t)) if no_t else 0.0
        b = float(np.mean(ye_t)) if ye_t else 0.0
        say(f"  {scen:<22s} no-text={a:.3f}  with-text={b:.3f}  Δ={b-a:+.3f}")

    say("\n" + "=" * 100)
    say("OVERALL MACRO-F1 by split_ratio (all scenarios, all settings)")
    say("=" * 100)
    for sr in SPLIT_RATIOS:
        sel = [r for r in rows if r["sr"] == sr]
        f1 = float(np.mean([r["F1"] for r in sel])) if sel else 0.0
        say(f"  sr={sr:.2f}  F1={f1:.3f}  (n={len(sel)})")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    say(f"Device: {device}  Backbone: {BACKBONE}")
    all_rows: list[dict] = []
    all_rows += s1_perseg_1shot(device)
    all_rows += s2_perseg_3shot(device)
    all_rows += s3_coco_potatoes(device)
    all_rows += s4_coco_nuts(device)
    summarize(all_rows)


if __name__ == "__main__":
    main()
