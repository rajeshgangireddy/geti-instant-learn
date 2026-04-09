from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import torch

from instantlearn.data import Sample
from instantlearn.models import SAM3, SAM3OpenVINO, SAM3OVVariant

COLORS = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
]


def draw_predictions(
    image: np.ndarray,
    prediction: dict[str, torch.Tensor],
    categories: list[str] | None = None,
) -> np.ndarray:
    """Overlay masks and boxes on an image and return the result."""
    overlay = image.copy()
    for i in range(len(prediction["pred_masks"])):
        mask = prediction["pred_masks"][i].numpy()
        box = prediction["pred_boxes"][i][:4].int().tolist()
        label_id = prediction["pred_labels"][i].item()
        score = prediction["pred_boxes"][i][4].item() if prediction["pred_boxes"][i].shape[0] == 5 else 0.0
        color = COLORS[label_id % len(COLORS)]

        # Mask overlay
        if mask.shape[:2] == image.shape[:2]:
            mask_bool = mask.astype(bool)
            overlay[mask_bool] = (
                np.array(overlay[mask_bool], dtype=np.float32) * 0.5 + np.array(color, dtype=np.float32) * 0.5
            ).astype(np.uint8)

        # Bounding box
        cv2.rectangle(overlay, (box[0], box[1]), (box[2], box[3]), color, 2)

        # Label text
        label = categories[label_id] if categories and 0 <= label_id < len(categories) else str(label_id)
        text = f"{label}: {score:.2f}"
        cv2.putText(
            overlay,
            text,
            (box[0], max(box[1] - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
    return overlay


def save_side_by_side(
    image_path: str,
    pred_left: dict[str, torch.Tensor],
    pred_right: dict[str, torch.Tensor],
    title_left: str,
    title_right: str,
    output_path: Path,
    categories: list[str] | None = None,
) -> None:
    """Save a side-by-side comparison of two prediction sets."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return

    left = draw_predictions(image, pred_left, categories)
    right = draw_predictions(image, pred_right, categories)

    # Draw titles on each half
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 1
    for img, title in [(left, title_left), (right, title_right)]:
        (tw, th), _ = cv2.getTextSize(title, font, font_scale, thickness)
        # Background rectangle for readability
        cv2.rectangle(img, (5, 5), (tw + 15, th + 15), (0, 0, 0), -1)
        cv2.putText(img, title, (10, th + 10), font, font_scale, (255, 255, 255), thickness)

    # 2-pixel white separator
    sep = np.full((image.shape[0], 2, 3), 255, dtype=np.uint8)
    combined = np.hstack([left, sep, right])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), combined)
    print(f"Saved comparison: {output_path}")


def main() -> None:
    """Run SAM3 vs SAM3 OpenVINO comparison benchmark."""
    image_paths = [
        "examples/assets/coco/000000390341.jpg",
    ]
    categories = ["elephant"]
    device = "xpu"
    ov_variant = SAM3OVVariant.FP32

    # Initialize SAM3 (device: "xpu", "cuda", or "cpu")
    tic = time.time()
    model = SAM3(device=device)
    toc = time.time()
    sam3_init_time = toc - tic
    print(f"SAM3 initialization time: {sam3_init_time:.2f} seconds")

    # SAM3 is zero-shot — no fit() required. Just provide categories per sample.
    tic = time.time()
    predictions = model.predict([Sample(image_path=p, categories=categories) for p in image_paths])
    toc = time.time()
    sam3_infer_time = toc - tic
    print(f"SAM3 prediction time: {sam3_infer_time:.2f} seconds")

    tic = time.time()
    model_ov = SAM3OpenVINO(device=device, variant=ov_variant)
    toc = time.time()
    ov_init_time = toc - tic
    print(f"SAM3 OpenVINO ({ov_variant.name}) initialization time: {ov_init_time:.2f} seconds")

    tic = time.time()
    predictions_ov = model_ov.predict([Sample(image_path=p, categories=categories) for p in image_paths])
    toc = time.time()
    ov_infer_time = toc - tic
    print(f"SAM3 OpenVINO ({ov_variant.name}) prediction time: {ov_infer_time:.2f} seconds")

    for idx, img_path in enumerate(image_paths):
        save_side_by_side(
            image_path=img_path,
            pred_left=predictions[idx],
            pred_right=predictions_ov[idx],
            title_left=f"SAM3 (PyTorch/XPU) : {sam3_infer_time:.2f}s",
            title_right=f"SAM3 OpenVINO ({ov_variant.name}) : {ov_infer_time:.2f}s",
            output_path=Path(f"outputs/comparison_{Path(img_path).stem}_{ov_variant.name}.jpg"),
            categories=categories,
        )

    image_to_test = [
        "examples/assets/coco/000000286874.jpg",
        "examples/assets/coco/000000173279.jpg",
        "examples/assets/coco/000000390341.jpg",
        "examples/assets/coco/000000267704.jpg",
    ]

    num_infer = 10
    infer_time_sam3 = []
    infer_time_sam3_ov = []
    for i in range(num_infer):
        image_path = image_to_test[i % len(image_to_test)]
        tic = time.time()
        _ = model.predict([Sample(image_path=image_path, categories=categories)])
        toc = time.time()
        infer_time_sam3.append(toc - tic)

    print(f"SAM3 : {infer_time_sam3}")

    for i in range(num_infer):
        image_path = image_to_test[i % len(image_to_test)]
        tic = time.time()
        _ = model_ov.predict([Sample(image_path=image_path, categories=categories)])
        toc = time.time()
        infer_time_sam3_ov.append(toc - tic)

    print(f"SAM3 OpenVINO : {infer_time_sam3_ov}")

    avg_sam3 = (sum(infer_time_sam3) - max(infer_time_sam3) - min(infer_time_sam3)) / (num_infer - 2)
    avg_sam3_ov = (sum(infer_time_sam3_ov) - max(infer_time_sam3_ov) - min(infer_time_sam3_ov)) / (num_infer - 2)
    print(f"SAM3 average inference time (excluding outliers): {avg_sam3:.2f} seconds")
    print(f"SAM3 OpenVINO average inference time (excluding outliers): {avg_sam3_ov:.2f} seconds")


if __name__ == "__main__":
    main()
