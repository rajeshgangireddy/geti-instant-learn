# SAM-HQ-Tiny OpenVINO Export Fix Summary

**Branch:** `fixes/matcher_tinyvit_quant`
**Date:** May 2026
**Devices:** OV-CPU + OV-GPU (Intel B60)
**OpenVINO:** 2025.3.0

---

## Problem Statement

SAM-HQ-Tiny Matcher export to OpenVINO was broken in two independent ways:

1. **Correctness bug (CPU + GPU):** ONNX export traced the wrong prompt structure (`[1, 42, 2]` instead of per-slot `[N, 1+K, 2]`) and embedded `argmax` inside the graph where GPU float rounding picked wrong slots.
2. **GPU divergence bug (GPU-only):** Even after fix #1, the TinyViT image encoder produced garbage on the Intel B60 GPU plugin due to kernel bugs in specific ONNX ops (residual Add in flattened token layout, Gather with 2D indices, LayerNormalization op).

The codebase hid both bugs behind an auto-fallback to SAM-HQ-Base.

---

## Fixes Applied

### Fix 1: ONNX Export Correctness (decoder.py, matcher.py)

- **`_preprocess_points` ONNX branch** now produces per-slot batched prompts `[N, 1+K, 2]` with K=2 context tokens, matching the runtime path.
- **Argmax moved to Python:** `forward_export` returns all slot masks `[C, N, H, W]` and scores `[C, N]`. Slot selection runs in host float64, eliminating GPU float instability.
- **Auto-fallback removed:** SAM-HQ-Tiny exports directly; no silent switch to Base.

### Fix 2: TinyViT GPU-Safe Export Patches (tinyvit_patches.py)

A monkey-patch module applied before `torch.onnx.export` rewrites problematic ops:

| Patch | What it does | Why |
|-------|-------------|-----|
| Attention bias pre-expansion | Materializes `ab` buffer as constant, no runtime Gather | B60 Gather with 2D index diverges |
| Manual arithmetic LayerNorm | Replaces ONNX `LayerNormalization` op with mean/var/rsqrt | B60 LayerNorm op diverges |
| Decomposed window partition | Avoids 6D view→transpose→reshape | Eliminates complex reshape fusion bugs |
| Conv2d_BN fusion | Folds BatchNorm into Conv2d weights | Reduces graph complexity |
| **4D spatial residual add** | Performs residual add in `[B,H,W,C]` not `[B,L,C]` | **Root cause**: B60 diverges on flattened Add `(1, 16384, 128)` by max_diff=33 |

### Fix 3: Integration Wiring (matcher.py, sam/__init__.py)

- Context manager `_tinyvit_gpu_safe_attention_bias_export_patch` wraps all three `torch.onnx.export` calls.
- `tinyvit_patches.py` exports `patch_tinyvit_for_export` / `unpatch_tinyvit`.

---

## Root Cause Deep Dive (GPU Divergence)

Bisection narrowed the divergence to `/image_encoder/layers.1/blocks.1/Add`:

- Inputs to the Add matched CPU↔GPU (max_diff < 0.0001)
- Output diverged by max_diff = 33.2, mean_diff = 2.6
- The Add operated on shape `(1, 16384, 128)` — flattened token layout

The Intel B60 OV-GPU plugin has a kernel bug in element-wise Add for this specific tensor shape/layout. Performing the same add in 4D spatial layout `(1, 128, 128, 128)` works correctly.

---

## Verification Results

| Target | IoU(PT, OV-CPU) | IoU(PT, OV-GPU) |
|--------|----------------:|----------------:|
| `000000390341.jpg` | 0.9509 | 0.9509 |
| `000000267704.jpg` | 0.9272 | 0.9272 |

- GPU latency (fused single-graph, FP32, LATENCY hint): **321.6 ms**
- Integration tests: **6 passed** (test_matcher_export + test_postprocessing_openvino)
- Gating script: `library/scripts/repro_sam_hq_tiny_ov.py` → `[REPRO] PASS`

---

## Files Changed

| File | Description |
|------|-------------|
| `library/src/instantlearn/components/sam/decoder.py` | Fixed ONNX export prompt shape (K=2), removed in-graph argmax |
| `library/src/instantlearn/components/sam/__init__.py` | Added tinyvit_patches exports |
| `library/src/instantlearn/components/sam/tinyvit_patches.py` | **NEW** — GPU-safe monkey-patches for TinyViT |
| `library/src/instantlearn/models/matcher/matcher.py` | Removed auto-fallback, added patch context manager, updated postprocessing |
| `library/tests/integration/models/test_matcher_export.py` | Updated for [C,N,H,W] output |
| `library/tests/integration/models/test_postprocessing_openvino.py` | Updated for new output shape |
| `library/scripts/repro_sam_hq_tiny_ov.py` | **NEW** — Gating repro script |
| `library/scripts/bisect_tinyvit_gpu.py` | **NEW** — Bisection diagnostic |
| `library/scripts/experiment_sam_hq_tiny_openvino_gpu.py` | **NEW** — Initial experiment script |

---

## Upstream Bug Report (Recommended)

File against [OpenVINO GPU Plugin](https://github.com/openvinotoolkit/openvino):

> **Title:** Intel B60 GPU: element-wise Add diverges for specific tensor shapes in TinyViT
>
> **Repro:** Export TinyViT-based SAM-HQ encoder to ONNX opset 17, compile on GPU.
> Residual Add with shape (1, 16384, 128) produces max_diff=33 vs CPU.
> Same Add in (1, 128, 128, 128) layout works correctly.
> Also affected: Gather with 2D index tensor, LayerNormalization op.
