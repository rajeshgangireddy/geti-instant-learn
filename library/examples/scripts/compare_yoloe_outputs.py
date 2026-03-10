"""Generate visual comparison grids: PyTorch vs OpenVINO YOLOE outputs.

For each of the 4 COCO target images, produces two grids
(text_prompt / visual_prompt) comparing N and X variants across backends:
  PyTorch (GPU/CPU), OpenVINO FP32, FP16, INT8, INT4 (CPU).

Each cell overlays predicted masks and bounding boxes on the target image.
Outputs 8 PNG files total (4 images × 2 prompt modes).

Usage::

    cd library
    python examples/compare_yoloe_outputs.py
    python examples/compare_yoloe_outputs.py --out-dir my_plots/
"""

from __future__ import annotations

import argparse
import gc
import logging
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from torchvision.tv_tensors import Image

from instantlearn.data.base.sample import Sample
from instantlearn.models.yoloe import YOLOE, YOLOEOpenVINO
from instantlearn.models.yoloe.yoloe import YOLOE_MODELS
from instantlearn.scripts.yoloe.export_yoloe_openvino import export_yoloe_openvino
from instantlearn.scripts.yoloe.quantize_yoloe_openvino import (
    quantize_int4,
    quantize_int8,
)

logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Constants ────────────────────────────────────────────────────────
ASSETS = Path(__file__).resolve().parent.parent / "assets" / "coco"
REF_IMAGE = ASSETS / "000000286874.jpg"
REF_MASK = ASSETS / "000000286874_mask.png"
TARGET_IMAGES = [
    ASSETS / "000000390341.jpg",
    ASSETS / "000000173279.jpg",
    ASSETS / "000000267704.jpg",
    ASSETS / "000000286874.jpg",
]
CLASSES = ["elephant"]
IMGSZ = 640

EXPORT_ROOT = Path("exports/yoloe_comparison")

VARIANTS = {
    "26n": "yoloe-26n-seg",
    "26x": "yoloe-26x-seg",
}

MASK_COLORS = [
    (0.12, 0.56, 1.0, 0.40),   # blue
    (1.0,  0.27, 0.0,  0.40),  # red-orange
    (0.2,  0.8,  0.2,  0.40),  # green
    (0.9,  0.1,  0.9,  0.40),  # magenta
    (1.0,  0.85, 0.0,  0.40),  # yellow
]

BOX_COLORS = [
    "#1E90FF",
    "#FF4500",
    "#33CC33",
    "#E600E6",
    "#DAA520",
]


# ── Helpers ──────────────────────────────────────────────────────────
def load_image_tensor(path: Path) -> torch.Tensor:
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img).permute(2, 0, 1)


def load_image_np(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(path: Path) -> torch.Tensor:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return torch.from_numpy(mask > 127).unsqueeze(0)


def make_ref_sample() -> Sample:
    return Sample(
        image=Image(load_image_tensor(REF_IMAGE)),
        masks=load_mask(REF_MASK),
        category_ids=np.array([0]),
        is_reference=[True],
        categories=CLASSES,
    )


def make_target_sample(path: Path) -> Sample:
    return Sample(
        image=Image(load_image_tensor(path)),
        is_reference=[False],
        categories=CLASSES,
    )


def get_ref_bboxes() -> list[list[float]]:
    mask = cv2.imread(str(REF_MASK), cv2.IMREAD_GRAYSCALE)
    mask_bin = (mask > 127).astype(np.uint8)
    ys, xs = np.where(mask_bin > 0)
    return [[float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]]


def get_ov_dir(vkey: str, fmt: str, prompt_mode: str) -> Path:
    return EXPORT_ROOT / f"{vkey}_{fmt}_{prompt_mode}"


def export_ov(model_name: str, vkey: str, fmt: str, prompt_mode: str) -> Path:
    """Export or quantize an OpenVINO model, return the directory."""
    vp_kwargs: dict = {}
    if prompt_mode == "vp":
        vp_kwargs = {
            "refer_image": str(REF_IMAGE),
            "bboxes": get_ref_bboxes(),
            "cls_ids": [0],
        }

    # Always need FP32 as base
    fp32_dir = get_ov_dir(vkey, "fp32", prompt_mode)
    if not fp32_dir.exists() or not any(fp32_dir.glob("*.xml")):
        export_yoloe_openvino(
            model_name=model_name,
            classes=CLASSES,
            output_dir=fp32_dir,
            imgsz=IMGSZ,
            half=False,
            **vp_kwargs,
        )

    if fmt == "fp32":
        return fp32_dir

    out_dir = get_ov_dir(vkey, fmt, prompt_mode)
    if out_dir.exists() and any(out_dir.glob("*.xml")):
        return out_dir

    if fmt == "fp16":
        export_yoloe_openvino(
            model_name=model_name,
            classes=CLASSES,
            output_dir=out_dir,
            imgsz=IMGSZ,
            half=True,
            **vp_kwargs,
        )
    elif fmt == "int8":
        quantize_int8(
            model_dir=fp32_dir,
            calibration_dir=ASSETS,
            output_dir=out_dir,
        )
    elif fmt == "int4":
        quantize_int4(model_dir=fp32_dir, output_dir=out_dir)

    return out_dir


# ── Inference runners ────────────────────────────────────────────────
def run_pytorch_vp(
    model_name: str,
    ref_sample: Sample,
    target_samples: list[Sample],
) -> list[dict]:
    """PyTorch visual prompt inference on multiple targets."""
    model = YOLOE(model_name=model_name, device=DEVICE, imgsz=IMGSZ, precision="fp32")
    model.fit(ref_sample)
    preds = [model.predict(ts)[0] for ts in target_samples]
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return preds


def _parse_ul_result(
    result, img_shape: tuple[int, int],
) -> dict[str, torch.Tensor]:
    """Parse a single ultralytics result into the standard format."""
    h, w = img_shape
    if result.boxes is not None and len(result.boxes) > 0:
        pred_boxes = torch.cat(
            [result.boxes.xyxy.cpu(), result.boxes.conf.unsqueeze(1).cpu()], dim=1,
        )
        pred_labels = result.boxes.cls.long().cpu()
        if result.masks is not None:
            pred_masks = (result.masks.data > 0.5).bool().cpu()
        else:
            pred_masks = torch.zeros(
                (len(pred_labels), h, w), dtype=torch.bool,
            )
        return {
            "pred_masks": pred_masks,
            "pred_boxes": pred_boxes,
            "pred_labels": pred_labels,
        }
    return {
        "pred_masks": torch.zeros((0, h, w), dtype=torch.bool),
        "pred_boxes": torch.zeros((0, 5)),
        "pred_labels": torch.zeros(0, dtype=torch.long),
    }


def run_pytorch_tp(
    model_name: str,
    target_images_np: list[np.ndarray],
) -> list[dict]:
    """PyTorch text prompt inference on multiple targets."""
    from ultralytics import YOLO
    from instantlearn.models.yoloe.weights import get_weights_path

    ul_model = YOLO(str(get_weights_path(YOLOE_MODELS[model_name])))
    inner = ul_model.model
    text_pe = inner.get_text_pe(CLASSES)
    inner.set_classes(CLASSES, text_pe)
    ul_model.to(DEVICE)

    preds = []
    for img_np in target_images_np:
        results = ul_model.predict(
            source=img_np, imgsz=IMGSZ, conf=0.25, verbose=False,
        )
        preds.append(_parse_ul_result(results[0], img_np.shape[:2]))

    del ul_model, inner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return preds


def run_openvino(
    model_name: str,
    vkey: str,
    fmt: str,
    prompt_mode: str,
    ref_sample: Sample,
    target_samples: list[Sample],
) -> list[dict]:
    """OpenVINO inference on multiple targets."""
    ov_dir = export_ov(model_name, vkey, fmt, prompt_mode)
    model = YOLOEOpenVINO(
        model_dir=ov_dir, device="CPU", confidence_threshold=0.25,
    )
    model.fit(ref_sample)
    preds = [model.predict(ts)[0] for ts in target_samples]
    del model
    gc.collect()
    return preds


# ── Visualisation ────────────────────────────────────────────────────
def draw_prediction(ax: plt.Axes, img: np.ndarray, pred: dict, title: str) -> None:
    """Draw image with overlaid masks and boxes on a matplotlib axes."""
    ax.imshow(img)

    masks = pred.get("pred_masks")
    boxes = pred.get("pred_boxes")
    labels = pred.get("pred_labels")
    n_det = len(labels) if labels is not None else 0

    # Overlay masks
    if masks is not None and len(masks) > 0:
        overlay = np.zeros((*img.shape[:2], 4), dtype=np.float32)
        for i, m in enumerate(masks):
            mask_np = m.cpu().numpy().astype(bool)
            # Resize mask if needed
            if mask_np.shape != img.shape[:2]:
                mask_np = cv2.resize(
                    mask_np.astype(np.uint8), (img.shape[1], img.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            color = MASK_COLORS[i % len(MASK_COLORS)]
            overlay[mask_np] = color
        ax.imshow(overlay)

    # Draw boxes
    if boxes is not None and len(boxes) > 0:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2, conf = box.cpu().numpy()
            color = BOX_COLORS[i % len(BOX_COLORS)]
            rect = Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor="none",
            )
            ax.add_patch(rect)
            label_str = f"elephant {conf:.2f}"
            ax.text(
                x1, y1 - 4, label_str,
                fontsize=7, fontweight="bold",
                color="white",
                bbox={"facecolor": color, "alpha": 0.7, "pad": 2, "edgecolor": "none"},
            )

    n_label = f"{title}\n{n_det} detection{'s' if n_det != 1 else ''}"
    ax.set_title(n_label, fontsize=9, fontweight="bold")
    ax.axis("off")


def _variant_label(vkey: str) -> str:
    """Human-readable variant name, e.g. '26n' -> 'YOLOE26-N'."""
    size = vkey.replace("26", "").upper()  # 'n' -> 'N'
    return f"YOLOE26-{size}"


def build_grid(
    prompt_mode: str,
    ref_sample: Sample,
    target_sample: Sample,
    target_img_np: np.ndarray,
    out_path: Path,
) -> None:
    """Build and save a comparison grid for one prompt mode and one image.

    Rows: model variants (N, X)
    Columns: PyTorch, OV FP32, OV FP16, OV INT8, OV INT4
    """
    ov_formats = ["fp32", "fp16", "int8", "int4"]
    device_label = "GPU" if DEVICE == "cuda" else "CPU"
    variant_keys = list(VARIANTS.keys())
    n_rows = len(variant_keys)
    n_cols = 1 + len(ov_formats)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.0 * n_cols, 3.6 * n_rows),
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    mode_label = "Visual Prompt" if prompt_mode == "vp" else "Text Prompt"

    for ri, vkey in enumerate(variant_keys):
        model_name = VARIANTS[vkey]
        vlabel = _variant_label(vkey)
        print(f"  {vlabel}: {model_name} ({mode_label})")

        # PyTorch
        print(f"    PyTorch {prompt_mode.upper()} ...")
        if prompt_mode == "vp":
            pt_pred = run_pytorch_vp(model_name, ref_sample, [target_sample])[0]
        else:
            pt_pred = run_pytorch_tp(model_name, [target_img_np])[0]
        draw_prediction(
            axes[ri, 0], target_img_np, pt_pred,
            f"{vlabel} — PyTorch ({device_label})",
        )

        # OpenVINO formats
        for ci, fmt in enumerate(ov_formats):
            print(f"    OV {fmt.upper()} ({prompt_mode.upper()}) ...")
            ov_pred = run_openvino(
                model_name, vkey, fmt, prompt_mode,
                ref_sample, [target_sample],
            )[0]
            draw_prediction(
                axes[ri, ci + 1], target_img_np, ov_pred,
                f"{vlabel} — OV {fmt.upper()} (CPU)",
            )

    fig.suptitle(
        f"YOLOE Inference Comparison — {mode_label}",
        fontsize=16, fontweight="bold", y=1.0,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Main ─────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate visual comparison grids for YOLOE backends.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent,  # library/
        help="Directory to save the output images (default: library/).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_sample = make_ref_sample()

    for img_path in TARGET_IMAGES:
        img_name = img_path.stem
        target_sample = make_target_sample(img_path)
        target_img_np = load_image_np(img_path)

        for prompt_mode, mode_suffix in [("tp", "text_prompt"), ("vp", "visual_prompt")]:
            mode_label = "Text Prompt" if prompt_mode == "tp" else "Visual Prompt"
            out_file = out_dir / f"yoloe_comparison_{img_name}_{mode_suffix}.png"

            print("=" * 70)
            print(f"{mode_label} — {img_name}")
            print("=" * 70)
            build_grid(
                prompt_mode=prompt_mode,
                ref_sample=ref_sample,
                target_sample=target_sample,
                target_img_np=target_img_np,
                out_path=out_file,
            )
            print()

    print("Done! Output images:")
    for img_path in TARGET_IMAGES:
        img_name = img_path.stem
        for suffix in ["text_prompt", "visual_prompt"]:
            print(f"  {out_dir / f'yoloe_comparison_{img_name}_{suffix}.png'}")



if __name__ == "__main__":
    main()
