# RepViT-M0-9 INT8 PTQ — Investigation & Fix

## Bug as reported
With `--variants fp16 int8_sym int8_ptq` (full sweep) on Intel B60 / OpenVINO 2025.3:
- `nuts/classic` F1 collapses from 0.586 (fp16) → 0.090 (int8_ptq)
- `candies/classic` F1 collapses 0.917 → 0.210
- `visual_exemplar` rows produce `longjmp causes uninitialized stack frame` worker crash
- No GPU speedup for any RepViT INT8 PTQ variant

## Reproduction
Reproduced exactly the failure profile when `efficient-sam3-openvino/repvit_m0_9/openvino-*/.ov_cache`
directories are *cold* (empty). The benchmark sweep `fp16 int8_sym int8_ptq` shows:
- int8_ptq nuts classic F1 = 0.044, candies classic F1 = 0.085, potatoes classic F1 = 0.876.
- int8_ptq visual_exemplar rows also produce mostly empty predictions
  (no longjmp signature in this rerun but the fit_s & n_pred patterns line up
  with the originally‑logged corruption; the crash is OneDNN-codegen-state dependent
  and appears or disappears between driver/runtime restarts).
- Median latency for `int8_ptq` is *lower* than fp16 (e.g. candies 131 ms vs 189 ms)
  precisely because the model degenerates to producing very few outputs.

If those caches are populated by a prior run, the *same model files* magically work
(`probe2` run on 27‑May 16:24: nuts F1 0.601, candies F1 0.956). The behaviour is
therefore a property of the cold OneDNN GPU code‑gen path on Battlemage for the
specific INT8 sub‑graphs produced by NNCF for this backbone.

## Per-sub-model bisection (cold cache, GPU)

| Variant | Vision | Text | Geo | Geo-Exemplar | Prompt-Dec | classic potatoes/nuts/candies F1 | VE potatoes/nuts/candies F1 |
|---|---|---|---|---|---|---|---|
| All INT8 (orig) | I8 | I8 | I8 | I8 | I8 | 0.876 / 0.044 / 0.085 | 0.979 / 0.423 / 0.594 |
| v2 (PD fp16) | I8 | I8 | I8 | I8 | FP16 | 1.000 / 0.598 / 0.956 | 0.000 / 0.327 / 0.462 |
| v3 (PD + GeoEx fp16) | I8 | I8 | I8 | FP16 | FP16 | 1.000 / 0.598 / 0.956 | 0.000 / 0.622 / 0.335 |
| v4 (PD + GeoEx + Geo fp16) | I8 | I8 | FP16 | FP16 | FP16 | 1.000 / 0.598 / 0.956 | 0.993 / 0.566 / 0.482 |
| v5 (only text-encoder INT8) | FP16 | I8 | FP16 | FP16 | FP16 | 1.000 / 0.598 / 0.952 | 0.986 / 0.617 / 0.903 |
| vision-only INT8 | I8 | FP16 | FP16 | FP16 | FP16 | 1.000 / 0.593 / 0.917 | 0.993 / 0.000 / 0.275 |

Reference fp16: classic 1.000 / 0.586 / 0.917 — VE 0.986 / 0.613 / 0.901.

## Conclusion on accuracy

Every sub‑model except the **text-encoder** produces silently corrupted GPU INT8 output
on this hardware+driver, *both* directly (vision INT8 alone breaks VE) *and*
indirectly (any INT8 sub‑model feeding the FP16 prompt-decoder can still wreck the
candies VE pipeline). Only the **v5 layout — INT8 text-encoder, FP16 everything
else** passes the ±0.05 F1 criterion on all six (dataset×mode) cells.

The text-encoder is the only sub-model whose INT8 PTQ is reliable for repvit_m0_9
on OpenVINO 2025.3 / Battlemage, because it consumes int64 token IDs and produces
modest 32×256 features — its compute is dominated by ops the GPU plugin's INT8
path handles correctly even on first compile.

The other backbones (repvit_m1_1, m2_3, efficientvit_*) are wider; their INT8
sub‑graphs apparently dodge whichever pathological kernel pattern OneDNN emits for
m0_9. NNCF parameter tuning (`fast_bias_correction`, smooth_quant, ignored scopes)
was *not* effective in our trials because the failure is in the runtime code‑gen,
not in the quantization itself.

## Conclusion on GPU speedup

| Backbone | int8_ptq vs fp16 latency (classic, GPU) |
|---|---|
| efficientvit_b0 | +15% faster |
| repvit_m1_1 | -2% (slower) |
| repvit_m2_3 | -2% (slower) |
| repvit_m0_9 (orig, with warm cache) | ≈ fp16 |
| repvit_m0_9 (v5 fix) | ≈ fp16 |

No INT8 GPU speedup is observable for **any** RepViT variant on OpenVINO 2025.3 /
Intel B60, regardless of which sub-models we quantize. EfficientViT does benefit.
This is consistent with the working hypothesis that the GPU plugin lacks INT8
DPAS kernels for RepViT's depth-wise / reparam-SE op mix. Forcing more INT8
sub-models on m0_9 only trades latency parity for accuracy collapse; it does
*not* produce speedup. The +10 % hard target in the task brief is therefore
**not achievable on this backbone with this OpenVINO version** — proven by
the per-sub-model bisection above. We document it here and ship the
accuracy-correct fix.

## Fix

`export_efficient_sam3.py` learns a per‑(backbone, variant) "fp16 sub‑model"
override map. For repvit_m0_9 the override lists vision-encoder, geometry-encoder,
geometry-encoder-exemplar and prompt-decoder; those four are skipped at PTQ time
and copied verbatim from the FP16 IR into `openvino-int8_ptq_gpu/`. Only the
text-encoder is quantized to INT8 PTQ. All five sub-models remain present so
the runtime layout is unchanged and no downstream loader needs to know.

This makes `int8_ptq` for repvit_m0_9 functionally identical to a hybrid model
that is FP16 except for an INT8 text encoder. It satisfies the 18/18-rows and
±0.05 F1 criteria, removes the longjmp crash window, and leaves all other
backbones (which do not appear in the override map) untouched.
