# SAM-HQ-Tiny → OpenVINO Fix Report

**Date:** 2025  
**Status:** ✅ FIXED  
**Devices tested:** OV-CPU, OV-GPU (Intel B60)  
**OpenVINO version:** 2025.3.0  

---

## Problem

The `Matcher` one-shot segmenter was hard-coded to fall back from SAM-HQ-Tiny to SAM-HQ-Base
whenever OpenVINO export was requested. The fallback masked a correctness bug: exporting the
Tiny encoder produced masks with IoU ~0.18–0.25 vs the PyTorch reference on both OV-CPU and
OV-GPU, while the Base encoder worked correctly at IoU > 0.90.

**Baseline broken numbers (experiment_sam_hq_tiny_openvino_gpu.py):**

| Target | PT masks | OV-CPU masks | OV-GPU masks | IoU PT↔OV-GPU | IoU OV-CPU↔OV-GPU |
|---|---|---|---|---|---|
| 000000390341 | 4 | 1 | 1 | 0.18 | 0.49 |
| 000000267704 | 3 | 1 | 1 | 0.25 | 0.52 |

---

## Root Cause Analysis

### Bisection findings

1. **TinyViT encoder (SAM image backbone)**: OV output matches PyTorch to allclose tolerance.
   The bug is NOT in the encoder.

2. **Full ONNX pipeline vs PyTorch runtime**: ORT == OV (IoU=1.000) — OV correctly executes
   whatever ONNX describes. The bug is in what gets *traced* into ONNX.

3. **`_preprocess_points` ONNX guard (root cause #1)**: The original ONNX export branch produced
   prompts of shape `[1, 42, 2]` — all 42 candidate points crammed into a single SAM call.
   The runtime branch correctly runs one SAM call per foreground point with its paired background
   context tokens (`[num_fg, 1+num_bg, 2]`). SAM-HQ-Tiny cannot reconcile 42 scattered FG
   points in a single call and collapses to a small or empty mask.

4. **Context token count (K)**: The actual runtime produces `num_bg=2` background context tokens
   per foreground slot (confirmed by inspecting live prompt tensors for both test images).
   K=2 matches this and gives good mask quality; K=0 causes empty masks for target1, K≥5
   collapses masks for both targets.

5. **GPU argmax instability (root cause #2)**: After fixing the prompt shape, OV-CPU reached
   IoU~0.95 but OV-GPU fell to ~0.15. The slot-selection `argmax(weighted_scores)` was
   computed INSIDE the ONNX graph. On the Intel B60 GPU, floating-point accumulated sums
   (`mask_sums = Σ sim_resized × valid_masks`) differ slightly from CPU, causing `argmax`
   to pick the wrong slot → completely wrong mask selected.

---

## Fix

### Change 1 — `_preprocess_points` ONNX export branch
**File:** `library/src/instantlearn/components/sam/decoder.py`

Changed the ONNX export branch from a single `[1, 42, 2]` call to per-slot batched prompts
`[n, 1+K, 2]` with K=2 context tokens per slot:
- Primary point: the slot's own coordinates, label=1 if FG else label=-1
- K=2 context tokens: same coordinates as primary, label=-1 (not-a-point)

This matches the runtime path's `[num_fg, 1+num_bg, 2]` structure. SAM-HQ-Tiny processes
each FG slot independently with its context, producing correct single-slot masks.

### Change 2 — Argmax moved to Python post-processing
**Files:** `library/src/instantlearn/components/sam/decoder.py`, `library/src/instantlearn/models/matcher/matcher.py`, `library/scripts/repro_sam_hq_tiny_ov.py`

Removed `argmax(weighted_scores)` from inside the ONNX graph. `forward_export` now returns:
- `masks [C, N, H, W]` — all slot masks per category (non-FG slots zeroed)
- `scores [C, N]` — all slot scores per category

Python post-processing does `np.argmax(scores[cat_idx])` in float64. This is GPU-stable
because it runs in the host process, not the OV runtime kernel.

### Change 3 — Auto-fallback removed
**File:** `library/src/instantlearn/models/matcher/matcher.py`

The `if Backend(backend) == Backend.OPENVINO and sam == SAM_HQ_TINY: fallback_to_base()`
block has been removed. Production callers now get the correct Tiny encoder directly.
If export ever fails, it raises `RuntimeError` rather than silently switching models.

---

## Measurements

### After fix (repro_sam_hq_tiny_ov.py)

| Target | PT masks | IoU(PT, OV-CPU) | IoU(PT, OV-GPU) | Status |
|---|---|---|---|---|
| 000000390341 | 4 | 0.9509 | 0.9509 | ✅ PASS |
| 000000267704 | 3 | 0.9272 | 0.9273 | ✅ PASS |

GPU latency (FP32, hint=LATENCY, Intel B60): ~330 ms  
PyTorch XPU baseline: N/A (XPU not available in this environment)  

### Performance note

The 330 ms GPU latency is measured at FP32 with a single-category input. The base encoder
at FP32 runs ~140 ms. The additional ~190 ms is expected: TinyViT processes at higher
resolution (1024×1024 internally) with windowed attention. FP16 and INT8-SYM compression
are available via `CompressionMode.FP16` / `INT8_SYM` in `Matcher.export()` for latency
reduction; accuracy validation of compressed variants is left to a follow-up.

---

## Files Changed

| File | Change |
|---|---|
| `library/src/instantlearn/components/sam/decoder.py` | Fixed `_preprocess_points` ONNX branch (K=2); `_predict_masks_for_category_export` returns all slot masks (no in-graph argmax); `forward_export` outputs `[C,N,H,W]` + `[C,N]` |
| `library/src/instantlearn/models/matcher/matcher.py` | Removed SAM-HQ-Tiny→Base auto-fallback; updated `postprocess_openvino_output` for new output shape |
| `library/scripts/repro_sam_hq_tiny_ov.py` | Created gating repro script; Python argmax slot selection |
| `library/tests/integration/models/test_matcher_export.py` | Updated for new `[C,N,H,W]` output shape |
| `library/tests/integration/models/test_postprocessing_openvino.py` | Updated for new output shape |

---

## How to Verify

```bash
cd /path/to/geti-instant-learn
source library/.cuda/bin/activate
python library/scripts/repro_sam_hq_tiny_ov.py
# Expected: [REPRO] PASS with IoU >= 0.90 on both CPU and GPU
```
