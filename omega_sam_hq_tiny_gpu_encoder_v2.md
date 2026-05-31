# OMEGA Follow-Up #2: Monkey-Patch TinyViT Attention Ops for OV-GPU

## Mission

Fix the SAM-HQ-Tiny image encoder so it produces correct output on OV-GPU
directly (no CPU offload). The previous OMEGA run proved the divergence
lives in TinyViT's attention layers (Stage2 = patch_embed + layer0 + layer1)
and that NO combination of OV compile-time config hints fixes it.

**The fix approach is monkey-patching the TinyViT model BEFORE the ONNX
export trace** to replace the problematic ops with GPU-friendly equivalents.
Do NOT try more OV config hints — that was exhaustively tested and failed.

---

## The exact problematic code

File: `library/.cuda/lib/python3.13/site-packages/segment_anything_hq/modeling/tiny_vit_sam.py`

### Problem 1: Advanced indexing in Attention.forward (line ~278)

```python
# In Attention.forward():
attn = (
    (q @ k.transpose(-2, -1)) * self.scale
    +
    (self.attention_biases[:, self.attention_bias_idxs]
     if self.training else self.ab)
)
```

`self.attention_bias_idxs` is a `[N, N]` LongTensor (registered buffer).
`self.attention_biases` is `[num_heads, num_offsets]` Parameter.

The expression `self.attention_biases[:, self.attention_bias_idxs]` produces
shape `[num_heads, N, N]`. In ONNX this becomes a **Gather** op on axis=1
with a 2D index tensor. The Intel B60 GPU plugin **computes this Gather
incorrectly** — all f32 precision hints fail, the output diverges by ~50
absolute units.

**Fix approach**: Before the ONNX trace, materialize the gather result into
a fixed `[num_heads, N, N]` constant tensor (`self.ab`) and replace the
`forward` method to always use `self.ab` instead of the gather expression.
Since the model is in eval mode during export, `self.ab` already exists as a
buffer — but during tracing the `if self.training else self.ab` branch may
trace into the training path. The monkey-patch must force the eval path.

Alternatively: replace the Attention forward to do:
```python
# pre-compute: ab = attention_biases[:, attention_bias_idxs]  # [H, N, N]
# then in forward: attn = (q @ k^T) * scale + self.ab
```
where `self.ab` is a plain registered buffer (no indexing at runtime).

### Problem 2: Window partition/reverse reshapes (lines ~356-365)

```python
# window partition
x = x.view(B, nH, self.window_size, nW, self.window_size, C).transpose(2, 3).reshape(
    B * nH * nW, self.window_size * self.window_size, C)
# ... attention ...
# window reverse
x = x.view(B, nH, nW, self.window_size, self.window_size,
           C).transpose(2, 3).reshape(B, pH, pW, C)
```

These complex reshapes (6D view → transpose → reshape) may also confuse the
GPU plugin. If fixing Problem 1 alone doesn't solve it, try rewriting the
window partition as simpler operations (sequential unfold or reshape without
the 6D intermediate).

---

## Implementation strategy (PRESCRIPTIVE — follow this)

1. **Create a monkey-patch module** at:
   `library/src/instantlearn/components/sam/tinyvit_patches.py`

2. The module should export a function like:
   ```python
   def patch_tinyvit_for_export(model: nn.Module) -> None:
       """Monkey-patch TinyViT attention modules to eliminate GPU-unfriendly ops.
       
       Must be called BEFORE torch.onnx.export() while model is in eval mode.
       Modifies the model in-place.
       """
   ```

3. Inside this function:
   - Walk all `Attention` submodules in the TinyViT encoder.
   - For each: ensure `self.ab` buffer is materialized (call `.eval()` →
     check `hasattr(self, 'ab')`; if not, compute it from
     `attention_biases[:, attention_bias_idxs]` and register it).
   - Replace the module's `forward` method with a version that **always**
     uses `self.ab` (the pre-materialized constant), never the gather.
   - The replacement forward should be:
     ```python
     def patched_forward(self, x):
         B, N, _ = x.shape
         x = self.norm(x)
         qkv = self.qkv(x)
         q, k, v = qkv.view(B, N, self.num_heads, -1).split(
             [self.key_dim, self.key_dim, self.d], dim=3)
         q = q.permute(0, 2, 1, 3)
         k = k.permute(0, 2, 1, 3)
         v = v.permute(0, 2, 1, 3)
         attn = (q @ k.transpose(-2, -1)) * self.scale + self.ab
         attn = attn.softmax(dim=-1)
         x = (attn @ v).transpose(1, 2).reshape(B, N, self.dh)
         x = self.proj(x)
         return x
     ```
   - Bind this as a bound method on the module.

4. **Call the patch** inside `Matcher.export()` in `matcher.py`, right before
   `torch.onnx.export(...)`. It should be called on the
   `MatcherInferenceGraph`'s SAM encoder submodule.

5. **Test**: after patching, run the gating script with the GPU path set to
   fused single-graph (not split pipeline). Expect IoU ≥ 0.90.

6. If Problem 1 fix alone is insufficient, ALSO patch the
   `TinyViTBlock.forward` to replace the 6D window partition with:
   ```python
   # Simpler window partition using unfold:
   x = x.view(B, H, W, C)
   # ... pad if needed ...
   x = x.unfold(1, self.window_size, self.window_size).unfold(2, self.window_size, self.window_size)
   # x is [B, nH, nW, C, ws, ws] → reshape to [B*nH*nW, ws*ws, C]
   x = x.permute(0, 1, 2, 4, 5, 3).contiguous().reshape(B*nH*nW, self.window_size*self.window_size, C)
   ```

---

## Hard rules

- Working dir: `/home/rgangire/workspace/code/GetiPrompt/geti-instant-learn`
- Venv: `library/.cuda/bin/python` — do NOT recreate.
- Branch: `fixes/matcher_tinyvit_quant`. No commit, no push, no PR.
- Do NOT modify upstream `segment_anything_hq` site-packages files. All
  patches go in our `library/src/instantlearn/` tree.
- Do NOT reintroduce the SAM-HQ-Base fallback.
- INT4 out of scope. FP32 export only.
- OV devices: CPU + GPU only (Intel B60).

---

## Definition of done

1. `library/.cuda/bin/python library/scripts/repro_sam_hq_tiny_ov.py` exits 0
   with `[REPRO] PASS`.
2. The GPU path in the repro uses **fused single-graph** `matcher.xml` on
   `device="GPU"` — NO CPU offload of the image encoder.
3. IoU(PT, OV-GPU) ≥ 0.90 on both targets.
4. GPU latency ≤ 340 ms (should improve since there's no CPU↔GPU transfer).
5. Integration tests pass:
   `pytest library/tests/integration/models/test_matcher_export.py library/tests/integration/models/test_postprocessing_openvino.py -x -q`
6. Report at `omega_results/sam_hq_tiny_gpu_encoder_report.md`.
7. New file `library/src/instantlearn/components/sam/tinyvit_patches.py`
   exists with the monkey-patch.

---

## Files to read first

- `library/.cuda/lib/python3.13/site-packages/segment_anything_hq/modeling/tiny_vit_sam.py` (full TinyViT source)
- `library/src/instantlearn/models/matcher/matcher.py` (export function, ~lines 455-540)
- `library/scripts/repro_sam_hq_tiny_ov.py` (gating script — modify GPU section)
- `omega_results/sam_hq_tiny_report.md` (previous run findings)

---

## Previous investigation results (DO NOT REPEAT THESE)

The previous agent exhaustively tested these OV compile configs on the
already-exported IR and ALL produced max_diff ~49.4, mean_diff ~5.6:
- INFERENCE_PRECISION_HINT: f32, f16
- EXECUTION_MODE_HINT: ACCURACY
- PERFORMANCE_HINT: LATENCY, THROUGHPUT
- GPU_ENABLE_LOOP_UNROLLING: NO

These configure the compiled model but cannot fix an op that was traced
incorrectly. The fix must happen BEFORE the ONNX trace — by eliminating the
problematic op from the PyTorch graph that gets traced.

---

Go. Solve this.
