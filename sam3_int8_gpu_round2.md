# SAM3 INT8 OpenVINO — Faster than FP16 on Intel Arc B60 (Round 2, after OV 2026.2)

## Objective (hard requirement)

Make **`INT8_SYM`** (W8A16 weight-only) and/or a freshly produced **`INT8_PTQ`**
(W8A8) SAM3 OpenVINO IR run with **lower mean latency than `openvino-fp16`**
on the **Intel Arc B60 dGPU (Battlemage, Xe2-HPG)** at SAM3's full input
resolution `1008 x 1008`, on at least one representative LVIS subset, with
≤ ~1 pt mean-F1 degradation vs FP16.

Time and resources are NOT a constraint. You may try **nightly OpenVINO/NNCF
wheels**, custom transformations, alternate compression recipes, and
selective `INT8_PTQ`. You have full permission to modify code under
`/home/rgangire/workspace/code/GetiPrompt/geti-instant-learn`.

If after thorough investigation it is **architecturally impossible** on the
current Arc B60 GPU plugin to make weight-only INT8 faster than FP16, that
conclusion (with evidence: kernel timings, dispatched kernel names, plugin
trace) is also an acceptable result — but only after you've genuinely tried
the W8A8 / selective-PTQ / nightly-plugin paths below.

---

## What has already been tried (don't repeat — extend from here)

### Round 1 (prior OMEGA run, session committed as 43dce712)

* Changed `nncf.compress_weights` recipe in
  `library/src/instantlearn/scripts/sam3/export_sam3.py` from
  `group_size=-1` (per-channel) to **`group_size=128, scale_estimation=True`**
  for `INT8_SYM`. This eliminated a 6–7 % regression on Arc dGPU; INT8 now
  runs **at parity** with FP16 on the committed recipe — but still NOT faster.
* Verified the `INFERENCE_PRECISION_HINT="i8"` route is **rejected** by the
  GPU plugin ("Supported values: { f16, f32, dynamic }"), so the weight-only
  INT8 path cannot be forced onto pure INT8 XMX via that hint on any tested
  OV version.

### OpenVINO version sweep (all on Arc B60 dGPU `[0xe216]`)

Measured at `[1, 3, 1008, 1008]`, 10 warmup + 50 iter,
`PERFORMANCE_HINT=LATENCY`, `NUM_STREAMS=1`, `INFERENCE_PRECISION_HINT=f16`,
vision-encoder only:

| OV version | FP16 (ms) | INT8_SYM (ms) | DQ group sizes tried | Result |
|---|---|---|---|---|
| 2025.2  | ~443 (baseline ~) | ~471 (+6 %) | n/a | slower |
| 2025.3.0 | similar | ~+4 % | rejected | slower |
| 2025.4.1 | similar | ~parity | 32/64/128 (no effect on weight-only) | parity |
| **2026.2.0** | **386.9** | **400.0 (+3.4 %)** | 32/64/128 (no effect) | still slower |

* `INFERENCE_PRECISION_HINT=i8` still rejected on 2026.2.
* `DYNAMIC_QUANTIZATION_GROUP_SIZE` (32/64/128) compiles cleanly on 2026.2
  but produces **no measurable change** on weight-only vision models —
  appears to only target LLM activation quantization paths.
* GPU `OPTIMIZATION_CAPABILITIES` reports `INT8` but this maps to the full
  PTQ (W8A8) path, not weight-only.
* Earlier full-PTQ attempt produced `openvino-int8_ptq_gpu` which is
  **46 % slower** than FP16 (~647 ms vs 443 ms) and a previous selective
  attempt (`int8_ptq_E`) broke accuracy (F1 ≈ 0).

So the obvious knobs in the **stable** plugin are exhausted. The next steps
need to be either (a) try **nightly** OV/NNCF wheels for newer GPU INT8
fast-paths or (b) a careful **selective W8A8 PTQ** recipe that quantizes
only the ops where the GPU plugin actually picks an INT8 kernel.

---

## In-scope success criteria

1. `vision-encoder` mean latency on Arc B60 dGPU strictly **< FP16** at
   `1008 x 1008` static shape, batch 1.
2. End-to-end SAM3 mean latency on at least one LVIS subset **≤ FP16**.
3. Mean F1 on that subset within ~1 pt of FP16.
4. A short `REPORT.md` (under `agent-results/sam3_int8_gpu_round2/`) with:
   * OV / NNCF / driver versions used,
   * Compression recipe (NNCF call signature) used for the winning model,
   * Latency table (FP16 vs winning INT8) per sub-model and end-to-end,
   * Accuracy table (mean F1) on the chosen LVIS subset,
   * Any code diffs applied to the user repository.

## Out-of-scope

* Training-aware quantization or fine-tuning.
* Building OpenVINO from source (only if a nightly wheel does **not** unlock
  the win, and even then strongly prefer a workaround).
* Touching the canvas / prompt-decoder logic for accuracy reasons unrelated
  to the INT8 perf question.

---

## Hardware

| Item | Value |
|---|---|
| GPU | Intel **Arc B60** discrete (Battlemage, **Xe2-HPG**, device id `0xe216`) |
| Sub-group sizes | 16, 32 (XMX / DPAS) |
| OPTIMIZATION_CAPABILITIES | FP32, BIN, FP16, INT8, GPU_HW_MATMUL, GPU_USM_MEMORY, EXPORT_IMPORT |
| OS | Ubuntu Linux |

Re-run `clinfo | head -50` and `benchmark_app -d GPU -hint latency -report_type detailed_counters`
yourself early on so you have the current device caps and a baseline.

---

## Repository and environments

### User repository (full edit permission)

`/home/rgangire/workspace/code/GetiPrompt/geti-instant-learn`

Branch is `fix/sam3_quant` (the committed Round-1 fix lives here). Make new
commits on top, or work on a fresh branch `fix/sam3_quant_round2`. Conventional
commits (`feat:`, `perf:`, `fix:`, etc.) per
`.github/copilot-instructions.md`. **uv** is mandatory (no raw `pip`),
PEP8 + ruff, Google-style docstrings, no unrequested refactors.

Key files:

| Path | Purpose |
|---|---|
| `library/src/instantlearn/models/sam3/sam3_openvino.py` | SAM3 OV runtime wrapper. Compile-properties block (~lines 256-285) is where to add/override GPU props. |
| `library/src/instantlearn/scripts/sam3/export_sam3.py` | FP16/FP32 export + `nncf.compress_weights` (weight-only INT8/INT4). Round-1 fix already uses `group_size=128, scale_estimation=True`. |
| `library/src/instantlearn/scripts/sam3/ptq_sam3.py` | Full PTQ (W8A8) via `nncf.quantize()`. Calibration builders for each sub-model. This is the recipe to refine for the W8A8 attempt. |
| `library/src/instantlearn/utils/compression.py` | Shared NNCF helper (`compress_model`). Round-1 commit lives here too. |
| `library/src/instantlearn/utils/constants.py` | `LVIS_DEFAULT_CATEGORIES` and `LVIS_92_BENCHMARK_CATEGORIES["fold_0"]` (use these as the prompt category lists). |
| `library/tools/OpenVINO_SAM3_extended.csv` | All historical SAM3 benchmark numbers (per dataset, per variant, CPU + GPU). |
| `library/tools/benchmark_matcher_variants.py` | Reference benchmark harness pattern; adapt for SAM3 sub-models if helpful. |
| `library/examples/sam3_openvino.ipynb` | Notebook end-to-end example. |
| `library/examples/sam3_canvas_example.ipynb` | Canvas-mode example used by the existing accuracy eval. |

### Already-exported IRs (no need to re-export FP16 unless changing the recipe)

`~/.cache/huggingface/hub/models--rajeshgangireddy--SAM3_OpenVINO/snapshots/844e4143273676daa3402f69c866f8a71f65067f/`

Subdirs: `openvino-fp16/`, `openvino-fp32/`, `openvino-int8_sym/`,
`openvino-int8_asym/`, `openvino-int4_sym/`, `openvino-int4_asym/`,
`openvino-int8_ptq_gpu/`, `onnx/`.

Each contains `vision-encoder.{xml,bin}`, `text-encoder.*`,
`geometry-encoder.*`, `geometry-encoder-exemplar.*`, `prompt-decoder.*`.

**Note**: the cached `openvino-int8_sym/` snapshot was produced **before** the
Round-1 recipe change (it is the old per-channel one). Re-export locally with
the current `export_sam3.py` recipe and **point benchmarks at the local
export**, not the HF cache, when measuring the committed recipe vs new
experiments.

### Python environment (already provisioned)

```bash
source /home/rgangire/workspace/code/GetiPrompt/geti-instant-learn/library/.cuda/bin/activate
```

Currently installed: `openvino==2026.2.0`, `nncf==3.2.0`, `torch`,
`transformers`, `huggingface_hub`. If you want to try a **nightly** OV /
NNCF wheel, build a sibling venv so you don't break `.cuda`:

```bash
cd /home/rgangire/workspace/code/GetiPrompt/geti-instant-learn/library
uv venv .venv-ov-nightly --python 3.13
uv pip install --python .venv-ov-nightly/bin/python \
    openvino-nightly nncf torch transformers huggingface_hub numpy
```

Then run benchmarks from that venv.

---

## Calibration / accuracy dataset

For both PTQ calibration and accuracy measurement, use the **LVIS subset**
already on disk:

| Resource | Path |
|---|---|
| LVIS annotations | `/home/rgangire/workspace/data/prompt/lvis/lvis_v1_val.json` |
| LVIS val images (COCO 2017 val) | `/home/rgangire/workspace/data/prompt/lvis/val2017/` (5000 images) |
| Alt image dirs | `/home/rgangire/workspace/data/prompt/lvis/val/`, `/home/rgangire/workspace/data/prompt/lvis/train2017/` |
| Default prompt categories | `LVIS_DEFAULT_CATEGORIES` in `constants.py` → `["cupcake", "sheep", "pastry", "doughnut"]` |
| 92-class benchmark categories (fold_0) | `LVIS_92_BENCHMARK_CATEGORIES["fold_0"]` in `constants.py` |

If you can not parse `lvis_v1_val.json`, fall back to the category lists in
`constants.py` and use `val2017/` images directly.

For NNCF PTQ calibration, **200–500 LVIS val2017 images** at the SAM3 input
size are sufficient. Pre-process exactly as `Sam3Preprocessor` (resolution
1008) so calibration matches deployment.

---

## Suggested investigation paths (in priority order — but use your judgment)

### Path A — Nightly OpenVINO + NNCF GPU INT8 fast-path

* Install `openvino-nightly` and the latest `nncf` (PyPI or from the
  `develop` branch).
* Re-benchmark the **existing committed `INT8_SYM`** export on Arc B60 GPU.
* Look for newer GPU runtime options (`OV_GPU_DUMP_*`, new compile props in
  `openvino/runtime/properties/intel_gpu/`) that enable weight-only INT8
  XMX dispatch.
* Goal: prove or disprove that nightly already wins without re-quantizing.

### Path B — Selective W8A8 PTQ (vision-encoder only)

* Use `ptq_sam3.py` as starting point. The current `int8_ptq_gpu` recipe is
  too aggressive (broke perf). Try a **conservative** recipe:
  * Quantize **only the vision encoder** (leave text / geometry / decoder FP16).
  * Inside vision-encoder, scope to **convolutions and large MatMuls**;
    exclude attention softmax / LayerNorm / GELU branches via
    `IgnoredScope`.
  * Calibration: 200–500 LVIS val2017 images at `1008 x 1008`.
  * Try `nncf.quantize(..., preset=PERFORMANCE, target_device=GPU)` and
    `target_device=ANY` to compare.
* Benchmark each variant on Arc B60. Take the fastest one that meets F1.

### Path C — IR-level inspection / fusion check

* Dump compiled graph: `OV_GPU_DUMP_GRAPHS=1`,
  `OV_GPU_DUMP_LAYERS_DST_ONLY=1`.
* In the committed `INT8_SYM` vision-encoder IR, check whether
  `FakeQuantize`/`Convert`/`Reorder` chains survive between consecutive Convs.
  If fusion did not fire, write the missing transformation or report it as a
  blocker.
* Check `vision-encoder.xml` for any dynamic partial shape; force static
  `[1, 3, 1008, 1008]` if anything is left dynamic.

### Path D — Compile-property combinations

* Even though `INFERENCE_PRECISION_HINT=i8` is rejected today, retry on
  nightly. Also try `dynamic` precision hint.
* Try `GPU_HOST_TASK_PRIORITY`, `CACHE_MODE`, `ALLOW_AUTO_BATCHING`,
  `GPU_QUEUE_PRIORITY` combinations — only those that meaningfully affect
  single-stream LATENCY.

### Path E — INT4 weight-only as a fallback

* If INT8 weight-only is structurally blocked, try `INT4_SYM` with
  `group_size=64` and `scale_estimation=True` and benchmark — sometimes the
  smaller weights win on bandwidth-bound vision encoders.
* Accuracy bar still applies (≤ ~1 pt F1 drop).

---

## Measurement protocol (use consistently)

For every variant you compare:

1. **Clear** any `.ov_cache` under the model dir (`rm -rf <model_dir>/.ov_cache`).
2. Use the **same** `benchmark_app` invocation:
   ```
   benchmark_app -m <vision-encoder.xml> -d GPU -hint latency \
       -niter 200 -nireq 1 -shape [1,3,1008,1008] -api sync
   ```
3. End-to-end: use `SAM3OpenVINO` with `prompt_mode=CANVAS`, `device="GPU"`,
   on 10–20 LVIS val2017 images, report mean latency over the full
   `fit()` + `predict()` cycle, excluding the first warm-up run.
4. Accuracy: re-use the existing canvas eval pipeline (notebook or script)
   on the same LVIS subset; report mean F1 per category and overall.

Each table in the final report must list OV+NNCF version, driver version,
device, model recipe (NNCF call signature), measurement script, and the
exact `.xml` file used.

---

## Final deliverables

1. A new commit (or PR) on `fix/sam3_quant_round2` containing:
   * Any updated `export_sam3.py` / `ptq_sam3.py` / `sam3_openvino.py`
     recipes.
   * The new INT8 IRs uploaded **only** if they meet the success criteria
     (otherwise leave them under `library/exports/`).
2. `agent-results/sam3_int8_gpu_round2/REPORT.md` with the tables, the
   winning recipe, and an honest verdict (win, parity, or hard blocker).
3. If a nightly OV / driver / NNCF version is required, document the exact
   versions in the report and in a top-level comment in `export_sam3.py`
   so a future user can reproduce.

Begin by reproducing the current FP16 vs `INT8_SYM` (committed recipe)
numbers on the existing `.cuda` env (OV 2026.2), then move to nightly and
selective PTQ. Good luck.
