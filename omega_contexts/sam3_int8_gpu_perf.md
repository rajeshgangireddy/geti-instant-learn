# SAM3 INT8 OpenVINO — Performance Regression on Intel Arc B60 (Xe2-HPG)

## Objective (single, hard requirement)

**Make `INT8_SYM` and/or `INT8_W8A16` faster than `FP16` on Intel Arc B60 dGPU
for the SAM3 OpenVINO pipeline.** Today they are *slower*. Once we have that,
we will iterate on `INT8_PTQ` separately — not in scope for this run.

Resources and time are NOT a constraint. You may run as many profiling
experiments, re-quantizations, and benchmarks as needed. You have full
permission to modify code in this user repository
(`/home/rgangire/workspace/code/GetiPrompt/geti-instant-learn`).

---

## The Problem (with measured data)

SAM3 has been exported to OpenVINO IR. On **Intel Arc B60 (Xe2-HPG, dGPU)**
the INT8 variants are SLOWER or equal to FP16, the opposite of what we expect.
Measurements come from
`library/tools/OpenVINO_SAM3_extended.csv` (averaged across datasets, `text`
prompting mode, GPU device):

| Variant           | Mean latency (ms) | vs FP16 |
|-------------------|------------------:|--------:|
| **openvino-fp16**       | ~443 | baseline |
| openvino-fp32           | ~446 |  +0.7%  |
| **openvino-int8_sym**   | ~471 |  **+6% slower**  |
| **openvino-int8_w8a16** | ~474 |  **+7% slower**  |
| openvino-int8_ptq_gpu   | ~647 |  +46% slower |
| openvino-int8_ve_only   | ~652 |  +47% slower |
| openvino-int8_ptq_E     | ~431 |  −3% but **F1 ≈ 0** (broken accuracy) |

For comparison, on the **same machine's CPU**, the same INT8 IRs ARE ~40%
faster than FP16 (FP16 ≈ 8600 ms → INT8_SYM ≈ 5200 ms), so the quantization
works on CPU; the regression is GPU-specific.

Full CSV with per-dataset numbers (Candies, Potatoes, LED, Nuts, LVIS-banana_book):
`library/tools/OpenVINO_SAM3_extended.csv`. Also includes a Xe3-LPG row
(PTL_Xe3) where the same pattern holds.

The **vision-encoder** sub-model dominates latency. Start there.

---

## In-scope: what counts as success

* `int8_sym` and/or `int8_w8a16` SAM3 vision-encoder running on Arc B60 GPU at
  **lower mean latency than `openvino-fp16`** on at least one representative
  dataset (Potatoes or Candies is fine), at full SAM3 input size
  (`1008 x 1008`).
* No more than ~1 pt mean-F1 degradation vs FP16 on that dataset
  (we can re-measure with the existing canvas eval script).
* End-to-end SAM3 inference (vision + text + geometry + prompt-decoder)
  should also be faster, but vision-encoder speedup alone is the gate.

## Out-of-scope for this run

* `INT8_PTQ` full activation quantization tuning — defer.
* Training-aware quantization, model retraining.
* OpenVINO source changes that require building OV from source
  (allowed only if a `nightly` / `pre-release` `openvino` wheel does NOT also
  solve the issue — please try the latest `openvino-nightly` first).

---

## Hardware

| Item | Value |
|---|---|
| GPU | Intel **Arc B60** discrete (Battlemage, **Xe2-HPG** arch) |
| Sub-group sizes available | 16, 32 (XMX / DPAS units, INT8/FP16 systolic) |
| Driver | check `clinfo | head -40` on the box |
| OS | Ubuntu Linux |
| CPU (for cross-reference) | check `lscpu \| head` |

Please run `clinfo` and `benchmark_app -d GPU -hint latency
--report_type detailed_counters` yourself early on so you have the real
device caps recorded in your report.

---

## Repositories and Environments

### This user repository (where you can edit freely)

`/home/rgangire/workspace/code/GetiPrompt/geti-instant-learn`

Relevant files:

| Path | Purpose |
|---|---|
| `library/src/instantlearn/models/sam3/sam3_openvino.py` | SAM3 OV runtime wrapper. Holds `CACHE_DIR`, `PERFORMANCE_HINT=LATENCY`. Add any new OV config props (`INFERENCE_PRECISION_HINT`, `GPU_HOST_TASK_PRIORITY`, dynamic quantization group size, etc.) here. |
| `library/src/instantlearn/scripts/sam3/export_sam3.py` | FP16/FP32 export + **weight-only compression** via `nncf.compress_weights` (this is where `int8_sym` / `int8_asym` / `int4_*` are produced). Default mode list at line ~659. Uses `group_size=-1` (per-channel). |
| `library/src/instantlearn/scripts/sam3/ptq_sam3.py` | Full PTQ (W8A8) via `nncf.quantize()`. Calibration builders per sub-model are in the same file. Out of scope for the *primary* objective but useful reference. |
| `library/tools/OpenVINO_SAM3_extended.csv` | All benchmark numbers. |
| `library/tools/benchmark_matcher_variants.py` | Benchmark harness pattern (sibling Matcher model). Adapt or write a new one for SAM3 sub-models if needed. |
| `library/examples/sam3_openvino.ipynb` | Notebook end-to-end example. |
| `library/examples/sam3_canvas_example.ipynb` | Canvas-mode example. |

### Already-exported IRs (no need to re-export FP16/INT8 unless changing the recipe)

HF cache, latest snapshot:
`~/.cache/huggingface/hub/models--rajeshgangireddy--SAM3_OpenVINO/snapshots/844e4143273676daa3402f69c866f8a71f65067f/`

Available variant subdirs (each contains `vision-encoder.{xml,bin}`,
`text-encoder.*`, `geometry-encoder.*`, `geometry-encoder-exemplar.*`,
`prompt-decoder.*`):

* `openvino-fp16/`         ← reference
* `openvino-fp32/`
* `openvino-int8_sym/`     ← **primary target**
* `openvino-int8_asym/`
* `openvino-int4_sym/`
* `openvino-int4_asym/`
* `openvino-int8_ptq_gpu/` (out of scope for primary)

Auto-download for other variants is done by `SAM3OpenVINO._resolve_model_dir`
in `sam3_openvino.py`; you can also load any local IR directly via
`SAM3OpenVINO(model_dir=...)`.

### Library Python environment

Activate with:
```
source /home/rgangire/workspace/code/GetiPrompt/geti-instant-learn/library/.cuda/bin/activate
```
Has `openvino`, `nncf`, `torch`, `transformers`, `huggingface_hub` already
installed. If a newer `openvino-nightly` is needed for the experiment, install
into a fresh venv (`uv venv .venv-ov-nightly && uv pip install
openvino-nightly nncf`) — do not break the existing `.cuda` env.

Project rule (`/.github/copilot-instructions.md`): use `uv` for Python env
management, not `pip` directly. Follow PEP8 + ruff. Type hints + Google-style
docstrings. Do not add unrequested files or refactors.

---

## What's likely wrong (hypotheses to investigate, in priority order)

You are free to disprove these and pursue your own. They are starting points
based on common Intel GPU INT8 anti-patterns.

1. **`INFERENCE_PRECISION_HINT` is silently overriding INT8 dispatch.**
   Today `sam3_openvino.py` sets only `PERFORMANCE_HINT=LATENCY`. On Arc the
   default GPU inference precision is `f16`. With weight-only INT8, OV may
   keep activations as F16 and perform on-the-fly dequant for *every* matmul,
   producing pure overhead instead of using INT8 XMX paths. Try setting
   `INFERENCE_PRECISION_HINT="i8"` (or `dynamic_quantization_group_size`)
   for the vision encoder compile.
2. **`compress_weights(group_size=-1, mode=INT8_SYM)` produces per-channel
   INT8 weights but no activation quantization.** XMX INT8 GEMM on Xe2-HPG
   wants **both** operands INT8. Without activation INT8, the kernel runs as
   `INT8 weights → F16 activations` (mixed precision) with a hidden dequant
   on every load — slower than pure F16. Two possible mitigations:
   (a) enable runtime *dynamic activation quantization* via OV property
       `DYNAMIC_QUANTIZATION_GROUP_SIZE` (try 32, 64, 128, or 0 for full
       per-token), or
   (b) move to W8A8 (`nncf.quantize`) — but accuracy/scope tuned, not the
       current broken `int8_ptq_gpu` recipe.
3. **Q/DQ pairs survive between Conv/MatMul nodes in the GPU plugin.**
   Inspect the compiled-model dump (`OV_GPU_DUMP_LAYERS_DST_ONLY=1`,
   `OV_GPU_DUMP_GRAPHS=1`) and IR `.xml`. Look for `FakeQuantize`, `Convert`,
   `Reorder`, or `Subtract`/`Multiply` chains between consecutive Convs in the
   ViT backbone. If fusion did not fire, that explains the regression.
4. **Vision encoder uses dynamic shapes that disable INT8 fast path.**
   Check `vision-encoder.xml` partial shapes. If anything is dynamic at this
   image size, reshape to static `[1, 3, 1008, 1008]` before saving — INT8
   GPU kernels often need static shapes to dispatch to XMX.
5. **`group_size=-1` vs `group_size=128` for weight-only.** Try per-group
   INT8 (128 / 64). Sometimes Arc GPU prefers grouped layout.
6. **Compile cache invalidation.** `.ov_cache` is per-variant. After changes,
   clear it (`rm -rf <variant_dir>/.ov_cache`) or measurements lie.

---

## Tools at your disposal

OMEGA gives you `gpu-kernel-device-timing` (clintercept wrapper), the **GPU
Plugin Agent**, the **Transformation Agent**, and the
**analyze-and-convert** agent (deep probing). For this perf task the most
useful skills are:

* `.workspace/openvino/.github/agents-prototype/skills/add-gpu-op/step3-profiling.md`
  — clintercept + `benchmark_app` device timing.
* `benchmark_app -m vision-encoder.xml -d GPU -hint latency -niter 200
  -report_type detailed_counters -report_folder ./perf/` per variant.
* `OV_GPU_DUMP_GRAPHS=1`, `OV_GPU_VERBOSE=1`, `ONEDNN_VERBOSE=1` env vars.
* OpenVINO Python API: `core.get_property("GPU", ...)` for device caps,
  `compiled.get_property("EXECUTION_DEVICES")` etc.
* `nncf` advanced quantization API (ignored_scope, advanced_parameters,
  per-group, model_type=Transformer).

Use `clintercept` per-OpenCL-kernel timing to compare FP16 vs INT8_SYM and
identify the exact kernel(s) that regress — that diagnosis is the key signal
of where to fix.

---

## Suggested workflow (you may diverge)

1. **Record device + driver info**: `clinfo`, `lscpu`, OV version,
   `python -c "import openvino; print(openvino.__version__)"`.
2. **Baseline `benchmark_app`** for `vision-encoder` only, on GPU, for
   `openvino-fp16` vs `openvino-int8_sym`. Note throughput, per-layer counters,
   exec graph.
3. **Hypothesis 1 first (cheapest)**: add `INFERENCE_PRECISION_HINT=i8` and/or
   `DYNAMIC_QUANTIZATION_GROUP_SIZE` to the compile call in
   `sam3_openvino.py` for the GPU path; re-measure with the same
   `benchmark_app` command. If this gets us under FP16, document and stop.
4. **Hypothesis 3**: dump GPU exec graph for both variants; diff. Find
   surviving Q/DQ / Reorder ops.
5. **Hypothesis 5**: re-run `compress_weights` with `group_size=128` (and 64)
   for vision-encoder only; produce new IR variants in
   `./scripts/results/sam3_int8_experiments/<name>/`; benchmark; record.
6. **Hypothesis 2b**: if weight-only stays stuck, try a small clean W8A8
   `nncf.quantize` on the vision-encoder only with
   `model_type=nncf.ModelType.TRANSFORMER`, MIXED preset, ignored scope
   excluding the patch-embed and the layer-norms — separately from the broken
   `int8_ptq_gpu` recipe. (Still INT8_W8A8 conceptually, but this is the
   variant the user is happy to use if it ships speedup.)
7. After each promising change, **re-run the canvas eval** to ensure
   F1/IoU does not collapse:
   `python library/tools/benchmark_matcher_variants.py --help`
   (or follow the SAM3 canvas eval pattern noted in repo memory
   `efficientsam3_canvas_eval.md`).
8. Produce a final report in `agent-results/sam3_int8_gpu/REPORT.md`
   containing:
   * device info,
   * a per-variant latency + accuracy table,
   * the change(s) that delivered speedup, with code diffs,
   * the diagnosis (which OV property / NNCF setting / fusion was the
     bottleneck), and
   * recommended permanent edits to `sam3_openvino.py` and/or
     `export_sam3.py` (do apply them in-repo as the user has approved this).

---

## Constraints from the user repo conventions

* Use `uv` for any Python env work (no bare `pip`).
* Don't introduce new top-level dependencies unless required; pin if you do.
* Don't add comments / docstrings to existing code you don't change.
* Don't create markdown summary files in the user repo unless explicitly
  needed (the OMEGA report under `agent-results/` is fine).
* Follow ruff defaults; type hints everywhere.

---

## Deliverables

1. Working code change in `library/src/instantlearn/models/sam3/sam3_openvino.py`
   (and/or `export_sam3.py`) that makes `int8_sym` or `int8_w8a16`
   measurably faster than `fp16` on Arc B60 for the SAM3 vision encoder.
2. A new IR variant subdir (if a re-quantization was needed) saved under
   `./scripts/results/sam3_int8_gpu/` with the recipe documented.
3. `agent-results/sam3_int8_gpu/REPORT.md` with the diagnosis, perf numbers,
   accuracy numbers, and what to do next for `INT8_PTQ` (since that's the
   follow-up).

Good hunting.
