# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Interactive Gradio playground for Matcher / SoftMatcher.

Workflow
-------
1. Select model variant (Matcher / SoftMatcher), SAM backbone, encoder, device, and precision.
2. Upload a reference image and provide a prompt (draw a mask *or* enter bounding boxes).
3. Upload one or more target images and run prediction.
4. View predicted masks, boxes, and scores overlaid on target images.

Launch
------
    cd library
    python examples/gradio_matcher.py
    python examples/gradio_matcher.py --device cpu
    python examples/gradio_matcher.py --share   # public Gradio link

Requires ``uv pip install gradio``.
"""

from __future__ import annotations

import argparse
import atexit
import colorsys
import logging
import shutil
import tempfile
from pathlib import Path
from time import perf_counter

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image as PILImage

from instantlearn.data.base.sample import Sample
from instantlearn.models import Matcher, SoftMatcher
from instantlearn.utils.constants import SAMModelName
from instantlearn.visualizer import render_predictions, setup_colors

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = SCRIPT_DIR / "assets" / "coco"
TMP_DIR = Path(tempfile.mkdtemp(prefix="gradio_matcher_"))
atexit.register(shutil.rmtree, TMP_DIR, ignore_errors=True)

# CLI ------------------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Matcher playground (Gradio)")
parser.add_argument("-d", "--device", default="auto", choices=["auto", "cuda", "xpu", "cpu"])
parser.add_argument("--host", default="0.0.0.0", type=str)
parser.add_argument("--port", default=7861, type=int)
parser.add_argument("--share", default=False, action="store_true")
args = parser.parse_args()

# Constants ------------------------------------------------------------------------------------------------------------

SAM_CHOICES = [e.value for e in SAMModelName]
ENCODER_CHOICES = ["dinov3_small", "dinov3_small_plus", "dinov3_base", "dinov3_large", "dinov3_huge"]
PRECISION_CHOICES = ["bf16", "fp16", "fp32"]
VARIANT_CHOICES = ["Matcher", "SoftMatcher"]

EXAMPLE_TARGETS = sorted(str(p) for p in EXAMPLES_DIR.glob("*.jpg")) if EXAMPLES_DIR.exists() else []


# State ----------------------------------------------------------------------------------------------------------------


class _AppState:
    """Mutable application state shared across callbacks."""

    def __init__(self) -> None:
        self.model: Matcher | SoftMatcher | None = None
        self.model_config: dict | None = None  # tracks current config to avoid reload
        self.raw_predictions: list[dict[str, torch.Tensor]] | None = None
        self.target_images: list[np.ndarray] | None = None
        self.ref_info: str = ""


_state = _AppState()


# Helpers --------------------------------------------------------------------------------------------------------------


def _resolve_device(preference: str = "auto") -> str:
    if preference != "auto":
        return preference
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _instance_color_map(n: int) -> dict[int, list[int]]:
    cmap: dict[int, list[int]] = {}
    for i in range(n):
        rgb = colorsys.hsv_to_rgb(i / max(n, 1), 0.85, 0.92)
        cmap[i] = [int(c * 255) for c in rgb]
    return cmap


def _visualize(
    image_rgb: np.ndarray,
    prediction: dict[str, torch.Tensor],
    *,
    instance_colors: bool = True,
    show_scores: bool = True,
) -> np.ndarray:
    pred = {**prediction}
    n_masks = pred["pred_masks"].shape[0]
    if instance_colors and n_masks > 0:
        pred["pred_labels"] = torch.arange(n_masks)
        cmap = _instance_color_map(n_masks)
    else:
        cmap = setup_colors({0: "object"})
    return render_predictions(image_rgb, pred, cmap, show_scores=show_scores)


def _parse_boxes(text: str) -> list[list[float]]:
    """Parse 'x1,y1,x2,y2; x1,y1,x2,y2; ...' into list of [x1,y1,x2,y2]."""
    boxes: list[list[float]] = []
    if not text or not text.strip():
        return boxes
    for box_str in text.strip().split(";"):
        box_str = box_str.strip()
        if not box_str:
            continue
        coords = [float(x.strip()) for x in box_str.split(",")]
        if len(coords) == 4:
            boxes.append(coords)
    return boxes


def _summary_pred(pred: dict[str, torch.Tensor]) -> str:
    n = pred["pred_masks"].shape[0]
    if n == 0:
        return "0 detections"
    scores = pred["pred_scores"]
    lo, hi = scores.min().item(), scores.max().item()
    return f"{n} detections — scores [{lo:.2f} .. {hi:.2f}]"


# Core callbacks -------------------------------------------------------------------------------------------------------


def load_and_fit(
    variant: str,
    sam_backbone: str,
    encoder: str,
    device: str,
    precision: str,
    confidence: float,
    num_fg_points: int,
    num_bg_points: int,
    use_mask_refinement: bool,
    prompt_mode: str,
    ref_mask_data: dict | None,
    box_text: str,
    category_name: str,
) -> str:
    """Load model (if config changed) and fit on reference image + prompt."""
    # Validate reference image
    if ref_mask_data is None:
        return "Upload a reference image first."

    bg = ref_mask_data.get("background")
    if bg is None:
        return "Upload a reference image into the editor."

    resolved_device = _resolve_device(device)
    sam_enum = SAMModelName(sam_backbone)

    # Build config dict to check if we need to reload
    new_config = {
        "variant": variant,
        "sam": sam_backbone,
        "encoder": encoder,
        "device": resolved_device,
        "precision": precision,
        "num_fg": num_fg_points,
        "num_bg": num_bg_points,
        "mask_refinement": use_mask_refinement,
    }

    # Load model if config changed
    if _state.model is None or _state.model_config != new_config:
        logger.info("Loading %s on %s/%s ...", variant, resolved_device, precision)
        cls = Matcher if variant == "Matcher" else SoftMatcher
        _state.model = cls(
            sam=sam_enum,
            encoder_model=encoder,
            device=resolved_device,
            precision=precision,
            confidence_threshold=confidence,
            num_foreground_points=num_fg_points,
            num_background_points=num_bg_points,
            use_mask_refinement=use_mask_refinement,
        )
        _state.model_config = new_config
        logger.info("Model loaded.")
    else:
        # Update threshold without reload
        _state.model.confidence_threshold = confidence

    # Build reference sample
    h, w = bg.shape[:2]
    ref_path = TMP_DIR / "ref_image.jpg"
    pil = PILImage.fromarray(bg)
    if pil.mode == "RGBA":
        pil = pil.convert("RGB")
    pil.save(str(ref_path))

    category = category_name.strip() or "object"

    if prompt_mode == "Mask (paint on image)":
        mask = None
        layers = ref_mask_data.get("layers", [])
        for layer in reversed(layers):
            if isinstance(layer, np.ndarray) and layer.ndim == 3:
                drawn = (layer[..., :3].sum(axis=-1) > 0).astype(np.uint8)
                if drawn.sum() > 0:
                    mask = drawn
                    break
        if mask is None:
            composite = ref_mask_data.get("composite")
            if composite is not None and isinstance(composite, np.ndarray):
                diff = np.abs(composite.astype(float) - bg.astype(float)).sum(axis=-1)
                mask = (diff > 30).astype(np.uint8)
        if mask is None or mask.sum() == 0:
            return "Draw a mask on the reference image (paint over the object)."

        mask_path = TMP_DIR / "ref_mask.png"
        cv2.imwrite(str(mask_path), mask * 255)
        ref_sample = Sample(
            image_path=str(ref_path),
            mask_paths=str(mask_path),
            categories=[category],
        )
        prompt_info = f"mask ({int(mask.sum()):,} px)"
    else:
        boxes = _parse_boxes(box_text)
        if not boxes:
            return "Enter at least one bounding box (x1,y1,x2,y2)."
        ref_sample = Sample(
            image_path=str(ref_path),
            bboxes=torch.tensor(boxes, dtype=torch.float32),
            categories=[category] * len(boxes),
            category_ids=torch.tensor([0] * len(boxes)),
        )
        prompt_info = f"{len(boxes)} box(es)"

    t0 = perf_counter()
    _state.model.fit(ref_sample)
    fit_ms = round(1000 * (perf_counter() - t0))

    _state.raw_predictions = None
    _state.target_images = None
    info = (
        f"Model fitted — {variant} | {sam_backbone} | {encoder} | {resolved_device}/{precision}\n"
        f"Reference: {w}x{h}, prompt: {prompt_info}, category: '{category}'\n"
        f"Fit time: {fit_ms} ms"
    )
    _state.ref_info = info
    return info


def run_prediction(
    target_images: list | None,
    instance_colors: bool,
    show_scores: bool,
    show_boxes: bool,
    confidence: float,
) -> tuple[list | None, str]:
    """Run prediction on uploaded target images."""
    if _state.model is None:
        return None, "Fit the model first (Step 1)."
    if not target_images:
        return None, "Upload at least one target image."

    # Update threshold
    _state.model.confidence_threshold = confidence

    results = []
    all_info: list[str] = []

    t0 = perf_counter()

    for i, tgt in enumerate(target_images):
        # tgt can be a filepath string or numpy array depending on Gradio version
        if isinstance(tgt, str):
            tgt_rgb = np.array(PILImage.open(tgt).convert("RGB"))
        elif isinstance(tgt, np.ndarray):
            if tgt.shape[-1] == 4:
                tgt_rgb = tgt[..., :3]
            else:
                tgt_rgb = tgt
        else:
            continue

        # Save to temp for prediction
        tgt_path = TMP_DIR / f"target_{i}.jpg"
        pil = PILImage.fromarray(tgt_rgb)
        pil.save(str(tgt_path))

        predictions = _state.model.predict(str(tgt_path))
        pred = predictions[0]

        vis = _visualize(
            tgt_rgb,
            pred,
            instance_colors=instance_colors,
            show_scores=show_scores,
        )

        # Draw bounding boxes if requested
        if show_boxes and pred["pred_masks"].shape[0] > 0:
            vis = _draw_boxes(vis, pred)

        results.append(vis)
        all_info.append(f"Image {i + 1}: {_summary_pred(pred)}")

    elapsed = round(1000 * (perf_counter() - t0))
    info = "\n".join(all_info) + f"\nTotal prediction time: {elapsed} ms"
    return results, info


def _draw_boxes(image: np.ndarray, pred: dict[str, torch.Tensor]) -> np.ndarray:
    """Draw bounding boxes derived from masks onto the image."""
    img = image.copy()
    masks = pred["pred_masks"].cpu().numpy()
    scores = pred["pred_scores"].cpu().numpy()
    n = masks.shape[0]
    for i in range(n):
        ys, xs = np.where(masks[i])
        if len(ys) == 0:
            continue
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        # Use instance color
        rgb = colorsys.hsv_to_rgb(i / max(n, 1), 0.85, 0.92)
        color = tuple(int(c * 255) for c in rgb)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{scores[i]:.2f}"
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img


def on_prompt_mode_change(mode: str) -> tuple:
    """Toggle visibility of mask editor brush hint vs box input."""
    is_mask = mode == "Mask (paint on image)"
    return gr.update(visible=is_mask), gr.update(visible=not is_mask)


# Build UI -------------------------------------------------------------------------------------------------------------


def build_app() -> gr.Blocks:
    """Construct the Gradio Blocks app."""
    default_device = _resolve_device(args.device)

    with gr.Blocks(title="Matcher Playground") as app:
        gr.Markdown("## Matcher Playground — InstantLearn")
        gr.Markdown(
            "Test **Matcher** and **SoftMatcher** with selectable SAM backbones, encoders, and devices.\n\n"
            "**Steps:** (1) Configure model & provide a reference image with prompt → "
            "(2) Upload target images → (3) View predictions.",
        )

        # --- Step 1: Model config + reference ---
        with gr.Accordion("Step 1 — Model Configuration & Reference", open=True):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Model Settings")
                    variant = gr.Dropdown(VARIANT_CHOICES, value="Matcher", label="Variant")
                    sam_backbone = gr.Dropdown(SAM_CHOICES, value=SAMModelName.SAM_HQ_TINY.value, label="SAM Backbone")
                    encoder = gr.Dropdown(ENCODER_CHOICES, value="dinov3_large", label="Image Encoder")
                    with gr.Row():
                        device = gr.Radio(
                            ["auto", "cuda", "xpu", "cpu"],
                            value=default_device,
                            label="Device",
                        )
                        precision = gr.Dropdown(PRECISION_CHOICES, value="bf16", label="Precision")

                    with gr.Accordion("Advanced", open=False):
                        confidence = gr.Slider(0.0, 1.0, value=0.38, step=0.01, label="Confidence Threshold")
                        num_fg = gr.Slider(1, 100, value=40, step=1, label="Foreground Points")
                        num_bg = gr.Slider(0, 20, value=2, step=1, label="Background Points")
                        mask_refinement = gr.Checkbox(value=True, label="Mask Refinement (2-stage SAM)")

                with gr.Column(scale=2):
                    gr.Markdown("### Reference Image & Prompt")
                    prompt_mode = gr.Radio(
                        ["Mask (paint on image)", "Bounding Box"],
                        value="Mask (paint on image)",
                        label="Prompt Type",
                    )
                    category_name = gr.Textbox(value="object", label="Category Name", max_lines=1)
                    ref_editor = gr.ImageEditor(
                        label="Reference Image (draw mask with brush)",
                        type="numpy",
                        brush=gr.Brush(colors=["#FF0000"], default_size=20),
                        eraser=gr.Eraser(default_size=20),
                        height=400,
                    )
                    mask_hint = gr.Markdown(
                        "*Paint over the object with the red brush to define the reference mask.*",
                        visible=True,
                    )
                    box_input = gr.Textbox(
                        label="Bounding Boxes (x1,y1,x2,y2; separate multiple with ;)",
                        placeholder="100,50,300,250; 400,100,600,350",
                        visible=False,
                    )

            fit_btn = gr.Button("Load Model & Fit Reference", variant="primary", size="lg")
            fit_status = gr.Textbox(label="Status", interactive=False, lines=3)

        # --- Step 2: Target images + prediction ---
        with gr.Accordion("Step 2 — Target Images & Prediction", open=True), gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Upload Targets")
                target_gallery = gr.File(
                    label="Target Images",
                    file_count="multiple",
                    file_types=["image"],
                )
                with gr.Row():
                    instance_colors = gr.Checkbox(value=True, label="Instance Colors")
                    show_scores = gr.Checkbox(value=True, label="Show Scores")
                    show_boxes = gr.Checkbox(value=True, label="Show Boxes")

                predict_btn = gr.Button("Run Prediction", variant="primary", size="lg")
                predict_status = gr.Textbox(label="Prediction Info", interactive=False, lines=4)

            with gr.Column(scale=2):
                gr.Markdown("### Results")
                result_gallery = gr.Gallery(
                    label="Predictions",
                    columns=2,
                    height=500,
                    object_fit="contain",
                )

        # --- Events ---
        prompt_mode.change(
            on_prompt_mode_change,
            inputs=[prompt_mode],
            outputs=[mask_hint, box_input],
        )

        fit_btn.click(
            load_and_fit,
            inputs=[
                variant,
                sam_backbone,
                encoder,
                device,
                precision,
                confidence,
                num_fg,
                num_bg,
                mask_refinement,
                prompt_mode,
                ref_editor,
                box_input,
                category_name,
            ],
            outputs=[fit_status],
        )

        predict_btn.click(
            run_prediction,
            inputs=[target_gallery, instance_colors, show_scores, show_boxes, confidence],
            outputs=[result_gallery, predict_status],
        )

    return app


# Main -----------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    demo = build_app()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
    )
