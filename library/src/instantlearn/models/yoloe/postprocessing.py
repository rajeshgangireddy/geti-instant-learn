# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Postprocessing utilities for YOLOE OpenVINO inference.

Handles letterbox preprocessing, output parsing, and mask
generation for YOLOE models exported with end2end=True.

Output format (after set_classes + export):
  output0: [1, 300, 4+1+1+nm] — end2end detections (xyxy, score, cls, mask_coeffs)
  output1: [1, nm, proto_h, proto_w] — prototype masks (nm=32, proto = imgsz/4)
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def letterbox_image(
    image: np.ndarray,
    target_size: tuple[int, int] = (640, 640),
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize image with aspect-ratio-preserving letterbox padding.

    Args:
        image: Input image in HWC format (BGR or RGB), dtype uint8.
        target_size: Target (height, width).
        color: Padding fill color.

    Returns:
        Tuple of (padded_image, scale, (pad_w, pad_h)).
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size

    scale = min(target_w / w, target_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2

    padded = np.full((target_h, target_w, 3), color, dtype=np.uint8)
    padded[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized

    return padded, scale, (pad_w, pad_h)


def preprocess_image(
    image: np.ndarray,
    target_size: tuple[int, int] = (640, 640),
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Prepare an image for YOLOE OpenVINO inference.

    Applies letterbox, BGR→RGB conversion, normalises to [0, 1] float32,
    and transposes to NCHW layout.

    Args:
        image: Input image in HWC uint8 format (BGR or RGB).
        target_size: Model input (height, width).

    Returns:
        Tuple of (input_tensor [1,3,H,W], scale, (pad_w, pad_h)).
    """
    padded, scale, pad = letterbox_image(image, target_size)

    # HWC → CHW, uint8 → float32, normalise to [0, 1]
    blob = padded.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)  # CHW
    blob = np.expand_dims(blob, axis=0)  # NCHW
    return blob, scale, pad


def parse_detections(
    det_output: np.ndarray,
    confidence_threshold: float = 0.25,
    nm: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse end2end detection output tensor.

    Args:
        det_output: Detection tensor of shape [1, num_dets, 4+1+1+nm].
            Columns: x1, y1, x2, y2, score, class_id, mask_coeffs...
        confidence_threshold: Minimum score to keep a detection.
        nm: Number of mask coefficients per detection.

    Returns:
        Tuple of (boxes [N,4], scores [N], class_ids [N], mask_coeffs [N,nm]).
        All filtered by confidence_threshold.
    """
    dets = det_output[0]  # [num_dets, 4+1+1+nm]

    scores = dets[:, 4]
    keep = scores >= confidence_threshold
    dets = dets[keep]

    if len(dets) == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int64),
            np.empty((0, nm), dtype=np.float32),
        )

    boxes = dets[:, :4]  # xyxy
    scores = dets[:, 4]
    class_ids = dets[:, 5].astype(np.int64)
    mask_coeffs = dets[:, 6 : 6 + nm]

    return boxes, scores, class_ids, mask_coeffs


def process_mask_protos(
    mask_coeffs: np.ndarray,
    protos: np.ndarray,
    boxes: np.ndarray,
    original_shape: tuple[int, int],
    scale: float,
    pad: tuple[int, int],
    mask_threshold: float = 0.5,
) -> np.ndarray:
    """Generate instance masks from prototype masks and coefficients.

    Computes mask = sigmoid(coeffs @ protos), crops to bounding boxes,
    and rescales to the original image resolution.

    Args:
        mask_coeffs: Mask coefficients [N, nm].
        protos: Prototype mask tensor [1, nm, proto_h, proto_w].
        boxes: Detection boxes in letterboxed coords [N, 4] (xyxy).
        original_shape: Original image (H, W) before letterbox.
        scale: Letterbox scale factor.
        pad: Letterbox padding (pad_w, pad_h).
        mask_threshold: Threshold for binary mask.

    Returns:
        Binary masks of shape [N, orig_H, orig_W], dtype bool.
    """
    if len(mask_coeffs) == 0:
        return np.empty((0, *original_shape), dtype=bool)

    proto = protos[0]  # [nm, proto_h, proto_w]
    nm, proto_h, proto_w = proto.shape

    # coeffs @ proto → [N, proto_h * proto_w] → reshape
    proto_flat = proto.reshape(nm, -1)  # [nm, proto_h*proto_w]
    raw_masks = mask_coeffs @ proto_flat  # [N, proto_h*proto_w]
    raw_masks = raw_masks.reshape(-1, proto_h, proto_w)  # [N, proto_h, proto_w]

    # Sigmoid activation
    raw_masks = 1.0 / (1.0 + np.exp(-raw_masks))

    # Scale box coords from model input space to proto space
    # Proto is input_size / 4 (stride 4 for proto)
    input_h = proto_h * 4
    input_w = proto_w * 4
    scale_x = proto_w / input_w
    scale_y = proto_h / input_h

    orig_h, orig_w = original_shape
    pad_w, pad_h = pad

    # Crop each mask to its bounding box in proto space (like ultralytics crop_mask).
    # Without this, sigmoid activations outside the box bleed into the background.
    for i in range(len(raw_masks)):
        x1_p = max(0, int(boxes[i, 0] * scale_x))
        y1_p = max(0, int(boxes[i, 1] * scale_y))
        x2_p = min(proto_w, int(np.ceil(boxes[i, 2] * scale_x)))
        y2_p = min(proto_h, int(np.ceil(boxes[i, 3] * scale_y)))

        crop_mask = np.zeros((proto_h, proto_w), dtype=np.float32)
        crop_mask[y1_p:y2_p, x1_p:x2_p] = 1.0
        raw_masks[i] *= crop_mask

    # Resize all masks from proto space → original image space in one pass
    masks_out = np.zeros((len(raw_masks), orig_h, orig_w), dtype=bool)

    for i in range(len(raw_masks)):
        # Upscale to model input size
        full_mask = cv2.resize(
            raw_masks[i], (input_w, input_h), interpolation=cv2.INTER_LINEAR
        )

        # Remove letterbox padding
        unpad_h = int(round(orig_h * scale))
        unpad_w = int(round(orig_w * scale))
        full_mask = full_mask[pad_h : pad_h + unpad_h, pad_w : pad_w + unpad_w]

        # Resize to original image dimensions
        if full_mask.size > 0:
            full_mask = cv2.resize(
                full_mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
            )
            masks_out[i] = full_mask >= mask_threshold

    return masks_out


def scale_boxes_to_original(
    boxes: np.ndarray,
    scale: float,
    pad: tuple[int, int],
    original_shape: tuple[int, int],
) -> np.ndarray:
    """Rescale detection boxes from letterboxed coords to original image coords.

    Args:
        boxes: Bounding boxes [N, 4] in xyxy format (model input space).
        scale: Letterbox scale factor.
        pad: Letterbox padding (pad_w, pad_h).
        original_shape: Original image (H, W).

    Returns:
        Boxes [N, 4] in original image coordinates, clipped.
    """
    if len(boxes) == 0:
        return boxes.copy()

    scaled = boxes.copy().astype(np.float32)
    pad_w, pad_h = pad

    # Remove padding offset
    scaled[:, 0] -= pad_w
    scaled[:, 2] -= pad_w
    scaled[:, 1] -= pad_h
    scaled[:, 3] -= pad_h

    # Undo scale
    scaled /= scale

    # Clip to image bounds
    orig_h, orig_w = original_shape
    scaled[:, 0] = np.clip(scaled[:, 0], 0, orig_w)
    scaled[:, 2] = np.clip(scaled[:, 2], 0, orig_w)
    scaled[:, 1] = np.clip(scaled[:, 1], 0, orig_h)
    scaled[:, 3] = np.clip(scaled[:, 3], 0, orig_h)

    return scaled
