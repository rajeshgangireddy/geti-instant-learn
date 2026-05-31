# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Quick experiment: verify SAM-HQ-Tiny + OpenVINO + Intel GPU is still broken.

Procedure:
  1. Fit a Matcher (SAM-HQ-Tiny) on a single COCO reference pair.
  2. Run PyTorch baseline inference on a target image.
  3. Export to OpenVINO IR while BYPASSING the auto-fallback to SAM-HQ-base.
  4. Run OpenVINO inference on Intel GPU twice and compare:
       - mask IoU vs PyTorch baseline (quality)
       - mask IoU between GPU run-1 and run-2 (determinism)
  5. Report a short verdict.
"""

from __future__ import annotations

import logging
from pathlib import Path
from time import time

import numpy as np
import openvino
import torch

from instantlearn.data import Sample
from instantlearn.data.utils.image import read_image
from instantlearn.models import Matcher
from instantlearn.utils.constants import Backend, CompressionMode, SAMModelName

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("sam_hq_tiny_ov_experiment")

ASSETS = Path(__file__).resolve().parent.parent / "examples" / "assets" / "coco"
REF_IMAGE = ASSETS / "000000286874.jpg"
REF_MASK = ASSETS / "000000286874_mask.png"
TARGET_IMAGES = [
    ASSETS / "000000390341.jpg",
    ASSETS / "000000267704.jpg",
]
EXPORT_DIR = Path(__file__).resolve().parent / "results" / "sam_hq_tiny_ov_gpu_check"


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Mean per-mask IoU between two boolean mask stacks (matched by index, min count)."""
    if a.size == 0 or b.size == 0:
        return 0.0
    n = min(a.shape[0], b.shape[0])
    ious = []
    for i in range(n):
        ma, mb = a[i].astype(bool), b[i].astype(bool)
        inter = np.logical_and(ma, mb).sum()
        union = np.logical_or(ma, mb).sum()
        ious.append(inter / union if union else 0.0)
    return float(np.mean(ious))


def run_ov(model_xml: Path, device: str, image: np.ndarray, input_size: int) -> tuple[np.ndarray, float]:
    core = openvino.Core()
    compiled = core.compile_model(str(model_xml), device)
    inp = Matcher.prepare_openvino_input(image, input_size)
    t0 = time()
    out = compiled(inp)
    elapsed = time() - t0
    masks = out[compiled.output("masks")]
    masks = Matcher.resize_masks_to_frame(masks, image.shape[0], image.shape[1])
    return masks, elapsed


def main() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading Matcher with SAM-HQ-Tiny ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    matcher = Matcher(
        sam=SAMModelName.SAM_HQ_TINY,
        encoder_model="dinov3_small",
        device=device,
        precision="fp32",
    )

    log.info("Fitting on reference image ...")
    ref_sample = Sample(image_path=str(REF_IMAGE), mask_paths=str(REF_MASK))
    matcher.fit(ref_sample)

    log.info("PyTorch baseline inference ...")
    target_samples = [Sample(image_path=str(p)) for p in TARGET_IMAGES]
    target_imgs = [np.asarray(read_image(str(p), as_tensor=False)) for p in TARGET_IMAGES]
    torch_results = matcher.predict(target_samples)
    torch_masks_list = [r["pred_masks"].cpu().numpy().astype(bool) for r in torch_results]
    for i, m in enumerate(torch_masks_list):
        log.info("  PyTorch target %d: %d masks", i, m.shape[0])

    log.info("Exporting to OpenVINO (FP32, bypassing SAM-HQ-Tiny fallback) ...")
    # The exporter auto-falls-back to SAM-HQ-base when it sees SAM_HQ_TINY.
    # Trick it: relabel the actually-loaded TINY predictor as BASE so the
    # check passes — the real loaded weights remain SAM-HQ-Tiny.
    matcher.sam_predictor._sam_model_name = SAMModelName.SAM_HQ_BASE  # noqa: SLF001
    xml_path = matcher.export(
        export_dir=EXPORT_DIR,
        backend=Backend.OPENVINO,
        compression=CompressionMode.FP32,
    )
    log.info("  Exported to %s", xml_path)

    input_size = matcher.encoder.input_size

    def coverage(m: np.ndarray) -> float:
        return float(m.mean()) if m.size else 0.0

    print("\n========== SAM-HQ-Tiny + OpenVINO + Intel GPU =============")
    for idx, (img, torch_masks) in enumerate(zip(target_imgs, torch_masks_list, strict=False)):
        print(f"\n--- Target {idx}: {TARGET_IMAGES[idx].name} ---")
        cpu_masks, tc = run_ov(xml_path, "CPU", img, input_size)
        gpu_masks_1, t1 = run_ov(xml_path, "GPU", img, input_size)  # fresh compile
        gpu_masks_2, t2 = run_ov(xml_path, "GPU", img, input_size)  # fresh compile
        gpu_masks_3, t3 = run_ov(xml_path, "GPU", img, input_size)  # fresh compile

        print(
            f"  PyTorch  : {torch_masks.shape[0]:>3d} masks, cov={coverage(torch_masks):.4f}",
        )
        print(
            f"  OV-CPU   : {cpu_masks.shape[0]:>3d} masks, cov={coverage(cpu_masks):.4f}, "
            f"{tc * 1000:.0f} ms",
        )
        print(
            f"  OV-GPU#1 : {gpu_masks_1.shape[0]:>3d} masks, cov={coverage(gpu_masks_1):.4f}, "
            f"{t1 * 1000:.0f} ms",
        )
        print(
            f"  OV-GPU#2 : {gpu_masks_2.shape[0]:>3d} masks, cov={coverage(gpu_masks_2):.4f}, "
            f"{t2 * 1000:.0f} ms",
        )
        print(
            f"  OV-GPU#3 : {gpu_masks_3.shape[0]:>3d} masks, cov={coverage(gpu_masks_3):.4f}, "
            f"{t3 * 1000:.0f} ms",
        )
        print(f"  IoU PyTorch vs OV-CPU      : {iou(torch_masks, cpu_masks):.4f}")
        print(f"  IoU PyTorch vs OV-GPU#1    : {iou(torch_masks, gpu_masks_1):.4f}")
        print(f"  IoU OV-CPU  vs OV-GPU#1    : {iou(cpu_masks, gpu_masks_1):.4f}")
        print(f"  IoU OV-GPU#1 vs OV-GPU#2   : {iou(gpu_masks_1, gpu_masks_2):.4f}")
        print(f"  IoU OV-GPU#1 vs OV-GPU#3   : {iou(gpu_masks_1, gpu_masks_3):.4f}")
    print("\n===========================================================")


if __name__ == "__main__":
    main()
