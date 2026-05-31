# Context: EfficientSAM3 OpenVINO — Intel GPU plugin accuracy regression

## Goal

Get **end-to-end EfficientSAM3 inference** running entirely on **Intel GPU (B60 / `Intel(R) Graphics [0xe216]`)** at full accuracy and at competitive latency, without splitting the model across CPU and GPU. Either backbone (RepViT-M1-1 or EfficientViT-B1) is acceptable.

Today we're forced to (a) compile encoders with `INFERENCE_PRECISION_HINT=f32` (slow) and (b) move the prompt decoder to CPU because of two distinct Intel GPU plugin accuracy bugs documented below. The result: 863 ms/image on GPU vs 264 ms on PyTorch XPU BF16 (same hardware, same model). We need that gap closed.

**Solve it however necessary**: identify the root-cause op/kernel in the OpenVINO GPU plugin, propose a fix or a targeted plugin-side workaround, validate that accuracy is restored, and re-measure latency. Splitting the model across devices is **not acceptable** as the final answer — find a true GPU fix.

## Hardware & software

- CPU: Intel 12900K
- dGPU: Intel B60 — `Intel(R) Graphics [0xe216]` (Battlemage class)
- OS: Linux (Ubuntu)
- OpenVINO: `2025.3.0-19807-44526285f24-releases/2025/3`
- Driver: stock Ubuntu Intel compute runtime
- Python 3.12, `uv`-managed env

## Models (local OpenVINO IR, ready to use)

EfficientSAM3 is exported as **5 sub-models** sharing the same image tensor. Two backbone variants are available locally — both reproduce the decoder bug; only EfficientViT triggers the vision-encoder NaN.

Base directories (`*.xml` + `*.bin` for each sub-model):

```
/home/rgangire/workspace/code/GetiPrompt/geti-instant-learn/library/efficient-sam3-openvino/repvit_m1_1/openvino-fp16/
/home/rgangire/workspace/code/GetiPrompt/geti-instant-learn/library/efficient-sam3-openvino/efficientvit_b1/openvino-fp16/
```

Sub-model files in each dir:

- `vision-encoder.xml` — RepViT-M1-1 or EfficientViT-B1 + FPN, input `(1,3,1008,1008) f32`
- `text-encoder.xml` — MobileCLIP-S1 text tower
- `geometry-encoder.xml`, `geometry-encoder-exemplar.xml`
- `prompt-decoder.xml` — DETR-style decoder (this is the broken one), exported with `opset=17`

There is also `openvino-fp32/`, `openvino-int8_*`, etc., available in the same parent dirs and on HuggingFace at `rajeshgangireddy/EfficientSAM3_OpenVINO`. Source PyTorch checkpoints and an ONNX→IR export script live under `library/src/instantlearn/scripts/efficient_sam3/`.

## Minimal reproducer

A self-contained Python script that runs CPU vs GPU on a single sub-model and prints per-output max/mean abs diff:

```
/home/rgangire/workspace/code/GetiPrompt/geti-instant-learn/library/tools/repro_efficient_sam3_gpu_bugs.py
```

Run with:

```bash
cd /home/rgangire/workspace/code/GetiPrompt/geti-instant-learn/library
uv run python tools/repro_efficient_sam3_gpu_bugs.py --model-dir efficient-sam3-openvino/efficientvit_b1/openvino-fp16
uv run python tools/repro_efficient_sam3_gpu_bugs.py --model-dir efficient-sam3-openvino/repvit_m1_1/openvino-fp16
```

(Use the `uv` env — required deps are `openvino`, `numpy`. No HF download needed.)

## Bug A — Vision encoder: EfficientViT-B1 produces NaN on GPU FP16

**Observed** (EfficientViT-B1, GPU, `INFERENCE_PRECISION_HINT=f16`, random N(0,1) input image `(1,3,1008,1008)`):

```
output[fpn_feat_0]  shape=(1,256,288,288)  NaN
output[fpn_feat_1]  shape=(1,256,144,144)  NaN
output[fpn_feat_2]  shape=(1,256,72,72)    NaN
```

Same input on CPU is finite. With `INFERENCE_PRECISION_HINT=f32` on GPU the outputs are finite and match CPU within 1e-5 (slow path — current workaround). RepViT-M1-1 does **not** NaN under f16, but has a non-trivial drift (~0.009 max abs on `fpn_feat_1`/`fpn_feat_2`).

Hypothesis: FP16 overflow inside an attention/FFN block of the EfficientViT backbone (likely a SiLU/Mul/MatMul chain), not handled by the plugin's mixed-precision pass.

## Bug B — Prompt decoder: large logit drift on GPU regardless of precision hint

**Observed** (any backbone — same `prompt-decoder.xml`, random fp32 inputs, bool masks all-ones, single text/prompt token):

| GPU `INFERENCE_PRECISION_HINT` | `pred_logits` max abs diff vs CPU | `pred_boxes` max abs diff | `pred_masks` max abs diff |
|---|---|---|---|
| `f16` | **0.0156** | 0.0041 | 32.88 |
| `f32` | **0.4129** | 0.4143 | 24.97 |

CPU FP32 is the reference. **Both GPU paths are wrong**, but counter-intuitively `f32` is *worse* than `f16` for `pred_logits` and `pred_boxes`. This rules out simple FP16 overflow and suggests a **plugin-side transformation/kernel bug** specific to the decoder graph (DETR-style cross-attention + bias-add + classification head). The `pred_masks` error is consistently large under both hints. `presence_logits` is exactly zero (constant subgraph), so the bug is on the active path.

This is the more interesting bug — fixing it would let us drop the CPU-decoder workaround entirely, and the `f16` numbers above suggest that simply enabling the right FP16 path (or disabling whatever transformation flips f32→bad) might be enough.

Decoder graph properties (probe with `ov.Core().read_model(...)`):

- Inputs: 4 FPN feature maps `(1,256,288,288)`, `(1,256,144,144)`, two at `(1,256,72,72)`; `prompt_features (1,N,256) f32`, `prompt_mask (1,N) bool`, `text_features (1,N,256) f32`, `text_mask (1,N) bool`. (`N` is dynamic.)
- Outputs: `pred_masks (1,200,288,288)`, `pred_boxes (1,200,4)`, `pred_logits (1,200)`, `presence_logits (1,1)`.
- Decoder is DETR-style: cross-attention over multi-scale FPN tokens, learnable object queries (200), shared mask/box/cls heads.
- Exported from PyTorch via `torch.onnx.export(opset=17)` then `ovc` → IR.

## What "fixed" looks like

1. **Accuracy**: `repro_efficient_sam3_gpu_bugs.py` reports `max_abs_diff < 1e-3` on all decoder outputs (`pred_logits`, `pred_boxes`) and no NaN on the vision encoder, **using `INFERENCE_PRECISION_HINT=f16` end-to-end on GPU** (or any GPU config that doesn't fall back to CPU).
2. **Latency** (end-to-end, full pipeline, RepViT or EfficientViT backbone, batch 1, 1008x1008 input, `Sam3PromptMode.CLASSIC`, "potatoes" dataset image): **median < 280 ms/image** on the B60. Reference: 863 ms today with workarounds, 264 ms with PyTorch XPU BF16+compile.
3. **Stability**: no sporadic NaN/Inf across 50 random inputs.
4. **Determinism**: same output to within 1e-3 across 10 repeated runs on identical input.

## What we've already tried (please don't re-explore unless you have a new angle)

- `INFERENCE_PRECISION_HINT=f32` on encoders → fixes NaN but ~2× slowdown.
- Decoder on CPU → fixes drift but adds CPU↔GPU transfer overhead; this is what we want to **remove**.
- `PERFORMANCE_HINT=LATENCY` vs `THROUGHPUT` → no effect on accuracy.
- INT8 PTQ and W8A16 — both reproduce the same decoder drift (it's not weight-precision related).

## What we'd like the agent to do

Use the **enable-operator / GPU plugin** sub-agent pipeline. The work is GPU-plugin-internal, not a framework conversion issue (model converts and runs on CPU correctly).

Concrete asks, in priority order:

1. **Decoder bug (Bug B)** — bisect which OpenVINO GPU transformation or kernel is producing the wrong `pred_logits`. Try toggling individual `pass::Manager` passes, oneDNN vs ocl primitives, `GPU_DISABLE_WINOGRAD`, `ov::hint::execution_mode(ACCURACY)`, and per-op `ov::pass::ConvertPrecision` decisions. Hypothesis to test first: an FP16↔FP32 mixed-precision pass is incorrectly promoting/demoting a softmax or attention-bias-add. The fact that `f16` is *more accurate* than `f32` is a strong signal.
2. **Vision encoder bug (Bug A)** — for EfficientViT-B1 only. Identify the op that overflows in FP16 and either mark it FP32 via `runtime_options` / model-level `precision_sensitive` annotation, or fix the plugin-side mixed-precision propagation so the chain is kept in FP32 automatically without forcing the whole model to f32.
3. **If a plugin-side root-cause fix isn't tractable in this session**, produce: (a) a minimal IR/ONNX repro extracted from the decoder graph that triggers the drift in the smallest possible subgraph; (b) a documented runtime-property workaround that keeps everything on GPU at f16 and recovers accuracy; (c) a draft GitHub issue against `openvinotoolkit/openvino` with the repro attached.

Please pick the right sub-agents (GPU plugin, Core OpSpec / transformations as needed). Do **not** spend time on the frontend / model conversion path — the IR is already correct (CPU output matches PyTorch).

## Source code pointers (read-only context — do not modify our project)

- Workaround we currently apply: `library/src/instantlearn/models/efficient_sam3/efficient_sam3_openvino.py` (the `_recompile_gpu_encoders_with_precision_hint` and decoder-device override in `__init__`).
- End-to-end benchmark we'll re-run after the fix: `library/tools/benchmark_efficient_sam3.py`.

Feel free to clone the OpenVINO source under your own workspace, build the GPU plugin with the proposed fix, and validate using the repro script above against the two local model dirs.
