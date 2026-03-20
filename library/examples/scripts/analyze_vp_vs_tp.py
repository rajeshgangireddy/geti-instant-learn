#!/usr/bin/env python3
"""Analyze why Visual Prompt (VP) detects fewer objects than Text Prompt (TP).

Investigates:
  1. Score distributions (VP vs TP) at the raw model output level
  2. Confidence threshold sweep: detection count at various thresholds
  3. Resolution (imgsz) impact: 640 vs 1024 vs 1280
  4. Bbox prompt quality: tight bbox vs padded bbox vs full-image bbox
  5. Multi-box prompts: all elephants in reference image

Run from the library directory:
    uv run python examples/scripts/analyze_vp_vs_tp.py
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import openvino as ov

# -- Constants ---------------------------------------------------------------
MODEL_NAME = "yoloe-26x-seg"
CLASSES = ["elephant"]

SCRIPT_DIR = Path(__file__).resolve().parent
ASSETS = SCRIPT_DIR.parent / "assets" / "coco"
EXPORT_ROOT = SCRIPT_DIR.parent.parent / "exports" / "vp_tp_analysis"
EXPORT_ROOT.mkdir(parents=True, exist_ok=True)

REF_IMAGE = ASSETS / "000000286874.jpg"
REF_MASK = ASSETS / "000000286874_mask.png"
TARGET_IMAGES = [
    ASSETS / "000000390341.jpg",
    ASSETS / "000000173279.jpg",
    ASSETS / "000000267704.jpg",
]


def bbox_from_mask(mask_path: Path) -> list[list[float]]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    ys, xs = np.where(mask > 127)
    return [[float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]]


def pad_bbox(bbox: list[float], pad_frac: float, img_shape: tuple[int, int]) -> list[float]:
    """Pad a bbox by `pad_frac` fraction of its size, clamped to image bounds."""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    px, py = w * pad_frac, h * pad_frac
    ih, iw = img_shape
    return [max(0, x1 - px), max(0, y1 - py), min(iw, x2 + px), min(ih, y2 + py)]


# -- Export helpers ----------------------------------------------------------

def export_model(tag: str, prompt_mode: str, imgsz: int = 640,
                 bboxes: list[list[float]] | None = None,
                 cls_ids: list[int] | None = None) -> Path:
    """Export a FP32 model and return its directory."""
    from instantlearn.scripts.yoloe.export_yoloe_openvino import export_yoloe_openvino

    out_dir = EXPORT_ROOT / tag
    if out_dir.exists() and any(out_dir.glob("*.xml")):
        return out_dir

    vp_kwargs = {}
    if prompt_mode == "visual" and bboxes is not None:
        vp_kwargs = dict(
            refer_image=str(REF_IMAGE),
            bboxes=bboxes,
            cls_ids=cls_ids or [0] * len(bboxes),
        )

    export_yoloe_openvino(
        model_name=MODEL_NAME,
        classes=CLASSES,
        output_dir=out_dir,
        imgsz=imgsz,
        half=False,
        **vp_kwargs,
    )
    return out_dir


def get_raw_scores(model_dir: Path, image_paths: list[Path]) -> list[np.ndarray]:
    """Run inference and return ALL 300 raw scores (no threshold)."""
    from instantlearn.models.yoloe.postprocessing import preprocess_image

    core = ov.Core()
    xml_files = list(model_dir.glob("*.xml"))
    model = core.read_model(str(xml_files[0]))
    compiled = core.compile_model(model, "CPU")
    infer_req = compiled.create_infer_request()

    # Read imgsz from metadata
    meta_path = model_dir / "metadata.yaml"
    import yaml
    with open(meta_path) as f:
        meta = yaml.safe_load(f)
    imgsz = meta.get("imgsz", [640, 640])
    if isinstance(imgsz, int):
        imgsz = [imgsz, imgsz]

    all_scores = []
    for p in image_paths:
        img = cv2.imread(str(p))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_tensor, _, _ = preprocess_image(img, tuple(imgsz))
        infer_req.infer({0: input_tensor})
        det_output = infer_req.get_output_tensor(0).data.copy()
        scores = det_output[0, :, 4]  # all 300 detection scores
        all_scores.append(scores)
    return all_scores


def count_detections_at_thresholds(
    model_dir: Path,
    image_paths: list[Path],
    thresholds: list[float],
) -> dict[float, list[int]]:
    """Return {threshold: [count_per_image]}."""
    raw = get_raw_scores(model_dir, image_paths)
    result = {}
    for thr in thresholds:
        result[thr] = [int((s >= thr).sum()) for s in raw]
    return result


# ============================================================================
# Experiment 1: Score Distribution (VP vs TP)
# ============================================================================
def experiment_score_distribution():
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Raw Score Distributions (VP vs TP)")
    print("=" * 70)

    tight_bbox = bbox_from_mask(REF_MASK)
    vp_dir = export_model("vp_fp32_640", "visual", imgsz=640, bboxes=tight_bbox)
    tp_dir = export_model("tp_fp32_640", "text", imgsz=640)

    vp_scores = get_raw_scores(vp_dir, TARGET_IMAGES)
    tp_scores = get_raw_scores(tp_dir, TARGET_IMAGES)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, (ax, path) in enumerate(zip(axes, TARGET_IMAGES)):
        vs = vp_scores[idx]
        ts = tp_scores[idx]

        # Only plot scores above a tiny threshold to see the distribution
        vs_pos = vs[vs > 0.01]
        ts_pos = ts[ts > 0.01]

        ax.hist(ts_pos, bins=50, alpha=0.6, label=f"TP ({len(ts_pos)} > 0.01)", color="blue")
        ax.hist(vs_pos, bins=50, alpha=0.6, label=f"VP ({len(vs_pos)} > 0.01)", color="red")
        ax.axvline(0.25, color="green", linestyle="--", label="threshold=0.25")
        ax.set_title(path.name)
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    fig.suptitle("Score Distributions: VP (red) vs TP (blue)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(EXPORT_ROOT / "exp1_score_distributions.png"), dpi=150)
    plt.show()

    print("\nScore statistics:")
    print(f"{'Image':<25} {'Mode':<6} {'Mean':>8} {'Max':>8} {'>#0.25':>8} {'>#0.10':>8} {'>#0.05':>8}")
    for idx, path in enumerate(TARGET_IMAGES):
        for name, scores in [("VP", vp_scores[idx]), ("TP", tp_scores[idx])]:
            s = scores
            print(f"  {path.name:<23} {name:<6} {s.mean():8.4f} {s.max():8.4f} "
                  f"{(s >= 0.25).sum():8d} {(s >= 0.10).sum():8d} {(s >= 0.05).sum():8d}")


# ============================================================================
# Experiment 2: Confidence Threshold Sweep
# ============================================================================
def experiment_threshold_sweep():
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Detection Count vs Confidence Threshold")
    print("=" * 70)

    tight_bbox = bbox_from_mask(REF_MASK)
    vp_dir = export_model("vp_fp32_640", "visual", imgsz=640, bboxes=tight_bbox)
    tp_dir = export_model("tp_fp32_640", "text", imgsz=640)

    thresholds = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

    vp_counts = count_detections_at_thresholds(vp_dir, TARGET_IMAGES, thresholds)
    tp_counts = count_detections_at_thresholds(tp_dir, TARGET_IMAGES, thresholds)

    # Aggregate counts (sum across images)
    vp_totals = [sum(vp_counts[t]) for t in thresholds]
    tp_totals = [sum(tp_counts[t]) for t in thresholds]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, tp_totals, "b-o", label="Text Prompt", linewidth=2)
    ax.plot(thresholds, vp_totals, "r-o", label="Visual Prompt", linewidth=2)
    ax.axvline(0.25, color="green", linestyle="--", alpha=0.5, label="current (0.25)")
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Total Detections (3 images)")
    ax.set_title("Detection Count vs Confidence Threshold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(EXPORT_ROOT / "exp2_threshold_sweep.png"), dpi=150)
    plt.show()

    print(f"\n{'Threshold':<12} {'VP Total':>10} {'TP Total':>10} {'Gap':>10}")
    for t, v, tp in zip(thresholds, vp_totals, tp_totals):
        print(f"  {t:<10.2f} {v:>10d} {tp:>10d} {tp - v:>10d}")


# ============================================================================
# Experiment 3: Resolution (imgsz) Impact
# ============================================================================
def experiment_resolution():
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Impact of Input Resolution (imgsz)")
    print("=" * 70)

    tight_bbox = bbox_from_mask(REF_MASK)
    resolutions = [640, 1024, 1280]
    threshold = 0.25

    results = {}
    for imgsz in resolutions:
        vp_tag = f"vp_fp32_{imgsz}"
        tp_tag = f"tp_fp32_{imgsz}"

        t0 = time.perf_counter()
        vp_dir = export_model(vp_tag, "visual", imgsz=imgsz, bboxes=tight_bbox)
        tp_dir = export_model(tp_tag, "text", imgsz=imgsz)
        export_time = time.perf_counter() - t0

        vp_counts = count_detections_at_thresholds(vp_dir, TARGET_IMAGES, [threshold])
        tp_counts = count_detections_at_thresholds(tp_dir, TARGET_IMAGES, [threshold])

        vp_total = sum(vp_counts[threshold])
        tp_total = sum(tp_counts[threshold])

        # Also measure inference speed
        t0 = time.perf_counter()
        get_raw_scores(vp_dir, TARGET_IMAGES)
        vp_time = (time.perf_counter() - t0) / len(TARGET_IMAGES)

        t0 = time.perf_counter()
        get_raw_scores(tp_dir, TARGET_IMAGES)
        tp_time = (time.perf_counter() - t0) / len(TARGET_IMAGES)

        results[imgsz] = {
            "vp_total": vp_total, "tp_total": tp_total,
            "vp_counts": vp_counts[threshold], "tp_counts": tp_counts[threshold],
            "vp_time_ms": vp_time * 1000, "tp_time_ms": tp_time * 1000,
        }

    print(f"\n{'imgsz':<8} {'VP Total':>10} {'TP Total':>10} {'Gap':>8} {'VP ms/img':>12} {'TP ms/img':>12}")
    for imgsz, r in results.items():
        gap = r["tp_total"] - r["vp_total"]
        print(f"  {imgsz:<6d} {r['vp_total']:>10d} {r['tp_total']:>10d} {gap:>8d} "
              f"{r['vp_time_ms']:>12.1f} {r['tp_time_ms']:>12.1f}")
    print(f"\n  Per-image breakdown (threshold={threshold}):")
    for imgsz, r in results.items():
        vc = r["vp_counts"]
        tc = r["tp_counts"]
        for i, p in enumerate(TARGET_IMAGES):
            print(f"    {imgsz}: {p.name}  VP={vc[i]}  TP={tc[i]}")

    # Bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(resolutions))
    width = 0.35
    vp_totals = [results[r]["vp_total"] for r in resolutions]
    tp_totals = [results[r]["tp_total"] for r in resolutions]
    ax.bar(x - width / 2, vp_totals, width, label="Visual Prompt", color="red", alpha=0.7)
    ax.bar(x + width / 2, tp_totals, width, label="Text Prompt", color="blue", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in resolutions])
    ax.set_xlabel("imgsz")
    ax.set_ylabel("Total Detections")
    ax.set_title("Impact of Resolution on Detection Count")
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(EXPORT_ROOT / "exp3_resolution.png"), dpi=150)
    plt.show()


# ============================================================================
# Experiment 4: Bbox Prompt Quality (tight, padded, full-image)
# ============================================================================
def experiment_bbox_quality():
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Bbox Prompt Quality (tight / padded / full-image)")
    print("=" * 70)

    ref_img = cv2.imread(str(REF_IMAGE))
    img_h, img_w = ref_img.shape[:2]
    tight_bbox = bbox_from_mask(REF_MASK)[0]  # [x1, y1, x2, y2]

    bbox_variants = {
        "tight": [tight_bbox],
        "pad_10pct": [pad_bbox(tight_bbox, 0.10, (img_h, img_w))],
        "pad_25pct": [pad_bbox(tight_bbox, 0.25, (img_h, img_w))],
        "pad_50pct": [pad_bbox(tight_bbox, 0.50, (img_h, img_w))],
        "full_image": [[0.0, 0.0, float(img_w), float(img_h)]],
    }

    threshold = 0.25
    results = {}
    for name, bboxes in bbox_variants.items():
        tag = f"vp_{name}_640"
        # Force re-export for each bbox variant
        out_dir = EXPORT_ROOT / tag
        if out_dir.exists():
            shutil.rmtree(out_dir)
        vp_dir = export_model(tag, "visual", imgsz=640, bboxes=bboxes)
        counts = count_detections_at_thresholds(vp_dir, TARGET_IMAGES, [threshold])
        raw = get_raw_scores(vp_dir, TARGET_IMAGES)
        max_scores = [float(s.max()) for s in raw]
        results[name] = {
            "counts": counts[threshold],
            "total": sum(counts[threshold]),
            "max_scores": max_scores,
            "bbox": bboxes[0],
        }

    # Also get TP baseline
    tp_dir = export_model("tp_fp32_640", "text", imgsz=640)
    tp_counts = count_detections_at_thresholds(tp_dir, TARGET_IMAGES, [threshold])
    tp_total = sum(tp_counts[threshold])

    print(f"\n{'Bbox Type':<14} {'Bbox':>35} {'Total':>8} {'Max Scores':>25}")
    for name, r in results.items():
        bbox_str = f"[{r['bbox'][0]:.0f},{r['bbox'][1]:.0f},{r['bbox'][2]:.0f},{r['bbox'][3]:.0f}]"
        max_str = ", ".join(f"{s:.3f}" for s in r["max_scores"])
        print(f"  {name:<12} {bbox_str:>35} {r['total']:>8d} {max_str:>25}")
    print(f"  {'TP baseline':<12} {'—':>35} {tp_total:>8d}")

    print(f"\n  Per-image breakdown:")
    for name, r in results.items():
        for i, p in enumerate(TARGET_IMAGES):
            print(f"    {name}: {p.name}  det={r['counts'][i]}  max_score={r['max_scores'][i]:.4f}")

    # Visualize reference image with bbox variants
    fig, axes = plt.subplots(1, len(bbox_variants), figsize=(4 * len(bbox_variants), 4))
    ref_img_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
    for ax, (name, bboxes) in zip(axes, bbox_variants.items()):
        vis = ref_img_rgb.copy()
        bb = bboxes[0]
        cv2.rectangle(vis, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0, 255, 0), 3)
        ax.imshow(vis)
        ax.set_title(f"{name}\ndet={results[name]['total']}")
        ax.axis("off")
    fig.suptitle("Reference Image Bbox Variants", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(EXPORT_ROOT / "exp4_bbox_variants.png"), dpi=150)
    plt.show()


# ============================================================================
# Experiment 5: Combined best settings
# ============================================================================
def experiment_combined():
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: VP with Best Settings vs TP Baseline")
    print("=" * 70)

    # Best VP: high resolution + padded bbox
    ref_img = cv2.imread(str(REF_IMAGE))
    img_h, img_w = ref_img.shape[:2]
    tight_bbox = bbox_from_mask(REF_MASK)[0]
    padded_bbox = pad_bbox(tight_bbox, 0.10, (img_h, img_w))

    configs = [
        ("VP 640 tight",   "visual", 640,  [bbox_from_mask(REF_MASK)[0]]),
        ("VP 640 padded",  "visual", 640,  [padded_bbox]),
        ("VP 1024 tight",  "visual", 1024, [bbox_from_mask(REF_MASK)[0]]),
        ("VP 1024 padded", "visual", 1024, [padded_bbox]),
        ("VP 1280 tight",  "visual", 1280, [bbox_from_mask(REF_MASK)[0]]),
        ("VP 1280 padded", "visual", 1280, [padded_bbox]),
        ("TP 640",         "text",   640,  None),
        ("TP 1024",        "text",   1024, None),
        ("TP 1280",        "text",   1280, None),
    ]

    thresholds = [0.10, 0.15, 0.20, 0.25]

    print(f"\n{'Config':<20}", end="")
    for t in thresholds:
        print(f"  thr={t:.2f}", end="")
    print()

    for label, mode, imgsz, bboxes in configs:
        tag = f"combined_{mode}_{imgsz}_{'padded' if bboxes and bboxes[0] != bbox_from_mask(REF_MASK)[0] else 'tight' if bboxes else 'text'}"
        if bboxes:
            model_dir = export_model(tag, mode, imgsz=imgsz, bboxes=bboxes)
        else:
            model_dir = export_model(tag, mode, imgsz=imgsz)

        counts = count_detections_at_thresholds(model_dir, TARGET_IMAGES, thresholds)

        print(f"  {label:<18}", end="")
        for t in thresholds:
            total = sum(counts[t])
            print(f"  {total:>8d}", end="")
        print()


# ============================================================================
# Main
# ============================================================================
def main():
    print("Visual Prompt vs Text Prompt Analysis")
    print(f"Model: {MODEL_NAME}")
    print(f"Reference: {REF_IMAGE}")
    print(f"Target images: {[p.name for p in TARGET_IMAGES]}")
    print(f"Exports: {EXPORT_ROOT}")

    experiment_score_distribution()
    experiment_threshold_sweep()
    experiment_resolution()
    experiment_bbox_quality()
    experiment_combined()

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print(f"Plots saved to: {EXPORT_ROOT}")
    print("=" * 70)


if __name__ == "__main__":
    main()
