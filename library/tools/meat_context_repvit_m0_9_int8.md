# Context: EfficientSAM3 + RepViT-M0-9 INT8 quantization — accuracy collapse + GPU slowdown

> **Take as much time as you need.** This is an autonomous, end-to-end fix task.
> You may iterate, profile, re-export, re-benchmark, install OpenVINO/NNCF dev versions,
> rebuild submodules, etc. The only success criterion is the one in **"What 'fixed' looks like"**
> at the bottom of this file. Do not stop until you reach it or you have *proven* it is
> impossible on this hardware + this OpenVINO version.

## TL;DR

We just added INT8 quantization (weight-only `INT8_SYM` and full PTQ `INT8_PTQ`) to all 6
EfficientSAM3 backbones. **5 of 6 work correctly**. The smallest RepViT variant —
`repvit_m0_9` — fails on INT8_PTQ in two distinct ways:

1. **Accuracy collapse on `classic` (text-prompt) mode**, OOD categories:
   - `nuts`     F1 0.586 (fp16) → **0.090** (int8_ptq)  — ΔF1 = -0.496
   - `candies`  F1 0.917 (fp16) → **0.210** (int8_ptq)  — ΔF1 = -0.707
   - `potatoes` F1 1.000 (fp16) → 0.975 (int8_ptq)      — fine
2. **`visual_exemplar` mode CRASHES** on all 3 datasets with:
   ```
   *** longjmp causes uninitialized stack frame ***: terminated
   ```
   These are not counted as F1=0 — the worker process dies entirely.
3. **No GPU speedup** for RepViT INT8_PTQ even on the variants where accuracy is preserved
   (m1_1 and m2_3 are -2% slower than fp16). EfficientViT gets +15-23% from INT8_PTQ on the
   same hardware. The RepViT pattern is therefore both an **accuracy bug** (m0_9) and a
   **performance bug** (all RepViT variants).

We need the int8 variant of `repvit_m0_9` to:
- preserve `classic` F1 within ±0.05 of fp16 on all 3 benchmark datasets,
- not crash on `visual_exemplar` mode,
- be **faster** than fp16 on the Intel B60 GPU (target ≥ +10% latency reduction;
  EfficientViT-B0 reaches +15% on the same hardware, so this is achievable in principle).

The accuracy collapse is the priority — without it the speedup is moot.

## Hardware & software

- **CPU**: Intel 12900K
- **dGPU**: Intel B60 — `Intel(R) Graphics [0xe216]` (Battlemage, OpenCL `level_zero`)
- **OS**: Ubuntu, Linux kernel headers available
- **Python**: 3.13, `uv`-managed env at `library/.venv/` — use
  `library/.venv/bin/python` directly (avoid activating the venv from shell).
- **OpenVINO**: 2025.3.0-19807-44526285f24-releases/2025/3
- **NNCF**: installed in the same venv (use `pip show nncf` to check version)
- **Branch (do not switch)**: `feat/efficient-sam3-int8-quantization`
  in `/home/rgangire/workspace/code/GetiPrompt/geti-instant-learn`

## Known GPU plugin caveats (do NOT re-discover, just apply)

Two pre-existing bugs in OpenVINO 2025.3 GPU plugin, already worked around:

- **EfficientViT FP16 NaN** — workaround: pass `--gpu-precision-hint f32` to the
  benchmark tool. This does NOT apply to RepViT (RepViT runs fine without it).
- **TinyViT vision encoder numerically wrong on GPU** — excluded permanently, do not
  attempt to enable.

## Models on disk (everything already exported)

```
library/efficient-sam3-openvino/repvit_m0_9/
├── openvino-fp16/                # baseline, works
├── openvino-int8_sym/            # weight-only INT8 — works, no speedup
└── openvino-int8_ptq_gpu/        # FULL PTQ — BROKEN (this is the target)
```

Each subdir contains 5 `*.xml` + `*.bin` sub-models:
`vision-encoder`, `text-encoder`, `geometry-encoder`,
`geometry-encoder-exemplar`, `prompt-decoder`.

For comparison, the same layout exists for `efficientvit_b0`, `efficientvit_b1`,
`efficientvit_b2`, `repvit_m1_1`, `repvit_m2_3` — read those if you need a
working reference graph.

## Where the code lives

All paths relative to `/home/rgangire/workspace/code/GetiPrompt/geti-instant-learn`:

- **Export pipeline** (PyTorch → ONNX → OpenVINO IR):
  - `library/src/instantlearn/scripts/efficient_sam3/export_efficient_sam3.py`
- **INT8 PTQ pipeline** (calibration + per-sub-model `nncf.quantize`):
  - `library/src/instantlearn/scripts/efficient_sam3/ptq_efficient_sam3.py`
  - Existing fix: text-encoder is calibrated on `LVIS_92_BENCHMARK_CATEGORIES["fold_0"]`
    plus a supplement of nut/candy/potato categories. If `repvit_m0_9` needs different
    treatment (e.g. mixed precision, per-channel ignored scopes, more samples),
    make the change here.
- **Compression utilities** (NNCF wrappers):
  - `library/src/instantlearn/utils/compression.py` (has both
    `compress_model` weight-only and `quantize_model` full PTQ)
- **Constants** (CompressionMode enum, LVIS categories):
  - `library/src/instantlearn/utils/constants.py`
- **Inference / EfficientSAM3 runtime** (used by the benchmark):
  - `library/src/instantlearn/models/efficient_sam3/` — relevant for the
    `longjmp` crash on `visual_exemplar` mode (the geometry-encoder-exemplar sub-model
    is only loaded in that mode).
- **Benchmark tool**:
  - `library/tools/benchmark_efficient_sam3_openvino.py`
- **Calibration images** (LVIS train2017, 118k):
  - `/home/rgangire/workspace/data/prompt/lvis/train2017`
- **Eval datasets** (COCO-format, see benchmark tool):
  - `/home/rgangire/workspace/data/prompt/geti_datasets/COCO/{potatoes,nuts,candies}`

## Reproducing the bugs

From `library/`:

```bash
# Full benchmark (will reproduce all three failures in one go)
.venv/bin/python tools/benchmark_efficient_sam3_openvino.py \
    --device GPU \
    --backbone repvit_m0_9 \
    --model-root ./efficient-sam3-openvino/repvit_m0_9 \
    --variants fp16 int8_sym int8_ptq \
    --results-dir tools/results/repvit_m0_9
```

Expected (broken) output, JSONL at
`tools/results/repvit_m0_9/benchmark_efficient_sam3_openvino_GPU.jsonl`:
- 15/18 rows present (3 `int8_ptq` + `visual_exemplar` rows missing due to longjmp crash)
- `int8_ptq` + `nuts` + `classic`     → F1 = 0.090
- `int8_ptq` + `candies` + `classic`  → F1 = 0.210

To re-quantize only `repvit_m0_9` (after a change to `ptq_efficient_sam3.py`):

```bash
.venv/bin/python -m instantlearn.scripts.efficient_sam3.export_efficient_sam3 \
    --backbone-type repvit \
    --variant m0_9 \
    --quantize \
    --compression-modes int8_ptq \
    --source-dir ./efficient-sam3-openvino/repvit_m0_9 \
    --calibration-root /home/rgangire/workspace/data/prompt/lvis/train2017 \
    --ptq-target-device GPU \
    --ptq-num-calibration 300
```

(The script reuses ONNX files already in `--source-dir` if present — do not delete them.)

## Working references (look at these before changing anything)

- `efficientvit_b0` int8_ptq → +15% speedup, ΔF1 ≤ 0.061 on the worst case.
  Demonstrates that the INT8 path is fundamentally sound; the issue is
  RepViT-specific (and even more m0_9-specific).
- `repvit_m1_1` and `repvit_m2_3` int8_ptq → accuracy preserved (max |ΔF1| ≤ 0.04)
  but no speedup. The accuracy fix and the speed fix may be two separate problems.
- `repvit_m0_9` int8_sym (weight-only) → accuracy preserved, no crash, no speedup.
  So the activation quantization (`nncf.quantize` full PTQ) is what destroys
  m0_9 accuracy and triggers the longjmp.

## Hypotheses worth testing (use, discard, or extend)

1. **Per-channel weight quantization sensitivity** in RepViT's reparameterized 1×1
   convs. m0_9 has the narrowest channel counts → highest per-channel quantization
   error. Try `nncf.AdvancedQuantizationParameters` with smoothquant α tuning or
   bias correction enabled; consider mixed precision (`nncf.IgnoredScope`) for the
   first/last blocks or the depth-wise convs.
2. **Calibration set mismatch**. The text-encoder calibration was already patched
   for OOD nut/candy/potato categories. The *vision-encoder* of m0_9 may need
   a more diverse calibration set than the current 300-image LVIS sample — try
   500-1000 images, or specifically include nut/candy/potato images.
3. **`longjmp` crash on visual_exemplar**: the `geometry-encoder-exemplar`
   sub-model only loads in that mode. Likely an INT8 sub-graph the OpenVINO GPU
   plugin rejects at compile time. Check `model_dir/openvino-int8_ptq_gpu/`
   for that file; try compiling it standalone on GPU to isolate; consider
   leaving exemplar sub-models in FP16 even when others go to INT8 (per-sub-model
   precision policy).
4. **GPU performance** for RepViT INT8: the plugin may not have INT8 kernels for
   RepViT's specific op mix (depth-wise + reparam SE). Use `benchmark_app`
   per-sub-model on GPU to measure where the cycles go; consider `dynamic_quantization_group_size`
   adjustments or excluding ops that hurt more than they help.

These are starting points only — you are expected to validate / refute each
with measurements before acting on it.

## Rules

- **No `--no-verify` on git commits, no force-push.** Commit on the existing branch
  `feat/efficient-sam3-int8-quantization` only. Do **not** open a PR.
- **Don't change the public API surface** of `CompressionMode` or the CLI of
  `export_efficient_sam3.py` / `benchmark_efficient_sam3_openvino.py` unless
  necessary. If you must, document the change in the commit message.
- **Don't regress the 5 working backbones.** Re-run benchmarks for at least
  `efficientvit_b0` and `repvit_m1_1` after your final change to confirm no
  regression.
- **Use `uv`** for any Python package work. The venv is at `library/.venv`.
- **Save all intermediate results** under `library/tools/results/<backbone>/`
  (JSONL files). Don't overwrite the existing fp16/int8_sym/int8_ptq baselines —
  add `-v2`, `-v3` suffixes for new experiments and pick the winner at the end.
- **Log everything** you try (failed approaches included) into
  `library/tools/results/repvit_m0_9/INVESTIGATION.md` — both the agent report
  and the breadcrumbs are valuable.

## What "fixed" looks like

A single git commit on `feat/efficient-sam3-int8-quantization` that, when the
benchmark above is re-run, produces:

1. **18/18 rows** in `tools/results/repvit_m0_9/benchmark_efficient_sam3_openvino_GPU.jsonl`
   (no missing visual_exemplar rows, no longjmp crash, no segfault).
2. **`int8_ptq` accuracy** for each `(dataset, mode)` cell satisfies
   `|F1_int8_ptq - F1_fp16| ≤ 0.05` on potatoes/nuts/candies × classic/visual_exemplar.
3. **`int8_ptq` median latency ≤ 0.90 × fp16 median latency** on at least one of
   the three datasets, in `classic` mode. (Soft target: ≥ +10% speedup; stretch:
   match EfficientViT-B0's +15%.)
4. **No regression** on `efficientvit_b0` and `repvit_m1_1` benchmarks
   (re-run, compare against existing JSONL — F1 within ±0.02, latency within ±5%).
5. **Investigation report** at
   `library/tools/results/repvit_m0_9/INVESTIGATION.md` describing the root cause
   and the chosen fix in 1-2 pages.

Good luck. Take the time you need.
