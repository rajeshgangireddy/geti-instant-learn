"""Tune CanvasConfig for EfficientSAM3 on failing PerSeg categories.

Sweeps:
- split_ratio: 0.15, 0.20, 0.30, 0.40, 0.50
- precrop: False (raw ref image), True (manual crop around ref bbox)
  -> simulates the effect of crop_padding for the 1-shot vertical path,
     which would otherwise be ignored.
- categories: dog, elephant, teapot (CANVAS failures in extended sweep),
  + backpack, clock (CANVAS wins; sanity check we don't regress).
- backbone: efficientvit/b1 + ft (best PerSeg performer).
- text: with and without.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_efficient_sam3 import _load_image_tensor, match_and_score  # noqa: E402

from instantlearn.data import Sample  # noqa: E402
from instantlearn.models import EfficientSAM3  # noqa: E402
from instantlearn.models.sam3.sam3 import CanvasConfig, Sam3PromptMode  # noqa: E402

logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO, force=True)
logger = logging.getLogger("canvas_tune")


def say(msg: str) -> None:
    """Force-flush print so progress is visible under file redirection."""
    print(msg, flush=True)
    logger.info(msg)

PERSEG = Path("/home/rgangire/workspace/data/prompt/PerSeg")
CATS = ["dog", "elephant", "teapot", "backpack", "clock"]
SPLIT_RATIOS = [0.15, 0.20, 0.30, 0.40, 0.50]
PRECROP_PAD = 1.5  # bbox-relative crop padding when precrop=True
MAX_TARGETS = 8


def bbox_from_mask(mask_path: Path) -> np.ndarray:
    import cv2
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return np.array([[0, 0, 1, 1]], dtype=np.float32)
    return np.array([[xs.min(), ys.min(), xs.max(), ys.max()]], dtype=np.float32)


def precrop_image_and_bbox(
    img_tensor: torch.Tensor, bbox: np.ndarray, pad: float,
) -> tuple[torch.Tensor, np.ndarray]:
    """Crop around bbox with pad×size context; return cropped image + adjusted bbox."""
    _, h, w = img_tensor.shape
    x1, y1, x2, y2 = bbox[0]
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half_w = bw * pad / 2
    half_h = bh * pad / 2
    cx1 = int(max(0, cx - half_w)); cy1 = int(max(0, cy - half_h))
    cx2 = int(min(w, cx + half_w)); cy2 = int(min(h, cy + half_h))
    crop = img_tensor[:, cy1:cy2, cx1:cx2]
    adj = np.array([[x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1]], dtype=np.float32)
    return crop, adj


def load_category(cat: str) -> tuple[Path, list[Path], list[Path]]:
    img_dir = PERSEG / "Images" / cat
    ann_dir = PERSEG / "Annotations" / cat
    images = sorted(img_dir.glob("*.jpg"))
    masks = sorted(ann_dir.glob("*.png"))
    return images[0], images[1:1 + MAX_TARGETS], masks[1:1 + MAX_TARGETS]


def eval_run(
    cat: str,
    split_ratio: float,
    precrop: bool,
    with_text: bool,
    device: str,
) -> dict:
    ref_img_path, tgt_imgs, tgt_masks = load_category(cat)
    ref_bbox = bbox_from_mask(PERSEG / "Annotations" / cat / "00.png")
    text = cat if with_text else ""

    canvas_cfg = CanvasConfig(split_ratio=split_ratio, crop_padding=2.0)
    model = EfficientSAM3(
        backbone_type="efficientvit", variant="b1", ft=True, device=device,
        prompt_mode=Sam3PromptMode.CANVAS, confidence_threshold=0.4,
        canvas_config=canvas_cfg,
    )

    if precrop:
        ref_img_t = _load_image_tensor(ref_img_path)
        ref_img_t, ref_bbox_adj = precrop_image_and_bbox(ref_img_t, ref_bbox, PRECROP_PAD)
        ref_sample = Sample(
            image=ref_img_t, bboxes=ref_bbox_adj,
            category_ids=np.array([0]), categories=[text or "object"],
        )
    else:
        ref_sample = Sample(
            image_path=str(ref_img_path), bboxes=ref_bbox,
            category_ids=np.array([0]), categories=[text or "object"],
        )

    model.fit(ref_sample)

    tp = fp = fn = 0
    for img_p, mask_p in zip(tgt_imgs, tgt_masks):
        if not img_p.exists() or not mask_p.exists():
            continue
        gt = bbox_from_mask(mask_p)
        pred = model.predict(Sample(image=_load_image_tensor(img_p)))[0]
        boxes = pred["pred_boxes"].cpu().numpy()
        if boxes.size > 0:
            boxes = boxes[:, :4].astype(np.float32)
            labels = pred["pred_labels"].cpu().numpy().astype(np.int64)
        else:
            boxes = np.empty((0, 4), dtype=np.float32)
            labels = np.empty((0,), dtype=np.int64)
        t, f, n = match_and_score(boxes, labels, gt, np.array([0], dtype=np.int64), 0.5)
        tp += t; fp += f; fn += n

    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {
        "category": cat, "split_ratio": split_ratio, "precrop": precrop,
        "text": with_text, "tp": tp, "fp": fp, "fn": fn, "P": p, "R": r, "F1": f1,
    }


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    say(f"Device: {device}")
    rows: list[dict] = []
    for cat in CATS:
        for sr in SPLIT_RATIOS:
            for precrop in (False, True):
                for with_text in (False, True):
                    r = eval_run(cat, sr, precrop, with_text, device)
                    say(
                        f"  {r['category']:<9s} sr={sr:.2f} precrop={str(precrop):<5s} "
                        f"text={'yes' if with_text else 'no ':<3s} | F1={r['F1']:.3f} "
                        f"(P={r['P']:.2f} R={r['R']:.2f})  tp={r['tp']} fp={r['fp']} fn={r['fn']}",
                    )
                    rows.append(r)

    say("=" * 90)
    say("BEST PER CATEGORY")
    say("=" * 90)
    for cat in CATS:
        crows = [r for r in rows if r["category"] == cat]
        best = max(crows, key=lambda r: r["F1"])
        say(
            f"  {cat:<9s} : F1={best['F1']:.3f}  sr={best['split_ratio']:.2f} "
            f"precrop={str(best['precrop']):<5s} text={'yes' if best['text'] else 'no'}",
        )

    say("=" * 90)
    say("MACRO-AVERAGE F1 (no-text)  —  rows = split_ratio × precrop")
    say("=" * 90)
    header = "  sr     " + "  ".join(f"pc={pc}" for pc in (False, True))
    say(header)
    for sr in SPLIT_RATIOS:
        cells = []
        for pc in (False, True):
            sel = [r for r in rows if r["split_ratio"] == sr and r["precrop"] == pc and not r["text"]]
            f1 = float(np.mean([r["F1"] for r in sel])) if sel else 0.0
            cells.append(f"{f1:.3f}")
        say(f"  {sr:.2f}   " + "  ".join(cells))

    say("=" * 90)
    say("MACRO-AVERAGE F1 (with-text)")
    say("=" * 90)
    say(header)
    for sr in SPLIT_RATIOS:
        cells = []
        for pc in (False, True):
            sel = [r for r in rows if r["split_ratio"] == sr and r["precrop"] == pc and r["text"]]
            f1 = float(np.mean([r["F1"] for r in sel])) if sel else 0.0
            cells.append(f"{f1:.3f}")
        say(f"  {sr:.2f}   " + "  ".join(cells))


if __name__ == "__main__":
    main()
