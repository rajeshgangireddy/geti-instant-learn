# SAM-HQ-Tiny GPU Encoder Export Report

## Final status

The SAM-HQ-Tiny fused OpenVINO export now passes the GPU gating repro on Intel B60 without CPU offload of the image encoder.

## Bisect findings

A dedicated encoder-only bisect script was added at `library/scripts/bisect_tinyvit_gpu.py`. It exports the SAM-HQ-Tiny image encoder with explicit checkpoint outputs for one TinyViT block and compares OV-CPU vs OV-GPU.

### Target block

- TinyViT block: `image_encoder.layers.1.blocks.1`
- Attention/window shape seen in the bisect export:
  - `norm_pre_qkv`: `(361, 49, 128)`
  - `after_token_view`: `(1, 16384, 128)`
  - `encoder_features`: `(1, 256, 64, 64)`

### First bad checkpoint before the fix

Before moving the residual add, the first divergent checkpoint was the residual add itself:

- checkpoint: `after_residual_add`
- effective op: TinyViT residual `Add`
- shape: `(1, 16384, 128)`
- diff: `max_diff=33.218124`, `mean_diff=2.603927`

The immediately preceding checkpoints still matched:

- `proj_linear`: `max_diff=0.000074`, `mean_diff=0.000004`
- `after_window_reverse`: `max_diff=0.000074`, `mean_diff=0.000004`
- `after_spatial_slice`: `max_diff=0.000074`, `mean_diff=0.000004`
- `after_token_view`: `max_diff=0.000074`, `mean_diff=0.000004`
- `after_drop_path`: `max_diff=0.000074`, `mean_diff=0.000004`

So the failure was not in `proj`, not in the window-reverse transpose path, and not in `drop_path`; it started at the flattened token-layout residual add.

### ONNX ops between `proj` and the residual add

From the encoder bisect export for `image_encoder.layers.1.blocks.1`, the ops after `attn/proj` and before the residual add are:

1. `attn/proj/MatMul`
2. `attn/proj/Add`
3. `Reshape_5`
4. `Transpose_2`
5. `Reshape_6`
6. `Slice_1`
7. `Slice_2`
8. `Reshape_7`
9. `Add`

This confirms that `proj` **is** included before the bad add, and the window-reverse path is also included before the divergence point.

## Fix implemented

File changed: `library/src/instantlearn/components/sam/tinyvit_patches.py`

The patched `TinyViTBlock.forward()` now performs the attention residual add in 4D spatial layout:

- old unsafe layout: `res_x + drop_path(x.view(B, L, C))`
- fixed layout: `res_x.view(B, H, W, C) + drop_path(x_spatial)`
- flatten back to `[B, L, C]` only **after** the add

Why this works:

- The Intel B60 GPU diverged on the flattened residual `Add` with shape `(1, 16384, 128)`.
- Keeping the add in spatial layout avoids that buggy execution pattern.

The existing export patch stack remains in place as well:

- pre-expanded TinyViT attention bias buffer
- decomposed window partition / reverse path
- manual arithmetic LayerNorm in TinyViT attention / MLP
- fused `Conv2d_BN` modules in the SAM image encoder

## Post-fix bisect result

Running `library/.cuda/bin/python library/scripts/bisect_tinyvit_gpu.py` after the fix shows the block stays aligned across all checkpoints:

- `after_residual_add`: `max_diff=0.000229`, `mean_diff=0.000007`
- `encoder_features`: `max_diff=0.000004`, `mean_diff=0.000000`
- verdict: **no divergent checkpoint found**

## Final repro result

Command:

```bash
rm -f library/scripts/results/repro_sam_hq_tiny/matcher.onnx \
      library/scripts/results/repro_sam_hq_tiny/matcher.xml \
      library/scripts/results/repro_sam_hq_tiny/matcher.bin
library/.cuda/bin/python library/scripts/repro_sam_hq_tiny_ov.py
```

Observed result:

| Target | IoU(PT, OV-CPU) | IoU(PT, OV-GPU) |
| --- | ---: | ---: |
| `000000390341.jpg` | `0.9509` | `0.9509` |
| `000000267704.jpg` | `0.9272` | `0.9272` |

- GPU latency: `321.6 ms`
- Final status: `[REPRO] PASS`

## Integration tests

Command:

```bash
library/.cuda/bin/python -m pytest \
  library/tests/integration/models/test_matcher_export.py \
  library/tests/integration/models/test_postprocessing_openvino.py -x
```

Result:

- `6 passed, 182 warnings in 154.39s`

## Files changed

- `library/src/instantlearn/components/sam/tinyvit_patches.py`
- `library/scripts/bisect_tinyvit_gpu.py`
- `omega_results/sam_hq_tiny_gpu_encoder_report.md`
