# SAM3 INT8 GPU Regression Report

## Summary

I reproduced the regression on Intel Arc GPU and tested the requested hypotheses.

**Main finding:** this is **not** fixable with an OpenVINO runtime property on the current GPU plugin stack.

- `INFERENCE_PRECISION_HINT=i8/u8` is **not supported** on this GPU plugin for this model family; only `f16` and `f32` are accepted.
- `DYNAMIC_QUANTIZATION_GROUP_SIZE` on GPU fails with `Unsupported primitive type in program_builder: DynamicQuantize`.
- The current `openvino-int8_sym` / `openvino-int8_w8a16` export dequantizes weights back to **FP16 before MatMul**, so Arc runs FP16 GEMMs plus extra dequant overhead.

The best permanent fix I found is an **export recipe change**:

- `INT8_SYM`
- `group_size=128`
- `scale_estimation=True`

This cuts most of the GPU regression and slightly improves CPU too. On my Arc B570 test system it brings end-to-end SAM3 latency from **~570.6 ms** (current INT8) to **~539.3 ms**, essentially matching FP16 (**~539.8 ms**), but not beating it reliably in longer verification runs.

## Environment

- OpenVINO: `2025.2.0-19140-f6b04b38426-releases/2025/2`
- GPU: `Intel(R) Arc(TM) B570 Graphics [0xE20B]`
- Driver: `24.52.32224.5`
- CPU: `Intel(R) Core(TM) Ultra 9 285H`

Artifacts are stored in this directory:

- `runtime_config_bench.csv`
- `reexport_vision_encoder_bench.csv`
- `reexport_vision_encoder_awq_bench.csv`
- `sweep_results.csv`
- `ptq_bench.csv`
- `cpu_vision_encoder_bench.csv`
- `wrapper_bench.csv`
- `wrapper_candidate_compare.csv`
- `wrapper_confirm.csv`
- `wrapper_fp16_vs_sym_g128_scale_15iter.csv`

## Baseline reproduction

### benchmark_app (vision-encoder, GPU)

| Model | Avg latency (ms) |
|---|---:|
| `openvino-fp16` | 432.16 |
| `openvino-int8_sym` | 466.94 |

### Wrapper-level SAM3 run (GPU)

| Variant | Mean latency (ms) |
|---|---:|
| `openvino-fp16` | 539.75 |
| `openvino-int8_sym` | 570.55 |

## Hypotheses tested

### 1. Runtime property fix (`INFERENCE_PRECISION_HINT`, `DYNAMIC_QUANTIZATION_GROUP_SIZE`)

**Result: no viable runtime-only fix found.**

Findings:

- Default GPU `INFERENCE_PRECISION_HINT` is `float16`.
- Setting `INFERENCE_PRECISION_HINT` to `i8` / `u8` fails: supported values are only `f16` and `f32`.
- Setting `DYNAMIC_QUANTIZATION_GROUP_SIZE` to `32`, `64`, or `128` fails during GPU compile with:
  - `Unsupported primitive type in program_builder: DynamicQuantize`
- `DYNAMIC_QUANTIZATION_GROUP_SIZE=0` behaves like baseline and does not help.

### 2. No activation quantization / hidden dequant overhead

**Confirmed.**

Graph inspection showed:

- current FP16 vision encoder: `MatMul` runs with FP16 weights and FP16 activations
- current INT8 exports insert `Convert` / `Multiply` / `GroupConvolution` chains
- representative `MatMul` in `openvino-int8_sym` receives:
  - input 0: `float16`
  - input 1: `float16`

So the compressed INT8 weights are restored to FP16 before the GEMM. That explains why GPU sees the overhead without getting a true XMX-style INT8 kernel path.

### 3. Q/DQ / Convert chains survive

**Confirmed.**

`openvino-int8_sym` and `openvino-int8_w8a16` have many extra ops relative to FP16:

- `Convert`: 116
- `Multiply`: 58
- `GroupConvolution`: 58

The FP16 model does not have these extra dequantization-style nodes.

### 4. Dynamic shapes

**Not the issue.**

All relevant SAM3 OpenVINO submodels already use static shapes, including vision encoder input:

- `[1, 3, 1008, 1008]`

### 5. `group_size=-1` vs grouped compression

**Confirmed: grouped compression is much better on GPU.**

## Re-export results

### Vision encoder only (GPU)

| Recipe | Mean latency (ms) |
|---|---:|
| FP16 baseline | 432.16 |
| Current `INT8_SYM`, `group_size=-1` | 466.12 |
| `INT8_SYM`, `group_size=128` | 455.74 |
| `INT8_SYM`, `group_size=128`, `scale_estimation=True` | 444.46 |
| `INT8_SYM`, `group_size=128`, `awq=True`, `scale_estimation=True` | 440.65 |
| `INT8_ASYM`, `group_size=128`, `awq=True`, `scale_estimation=True` | 440.34 |

### Full SAM3 wrapper (GPU)

| Variant | Mean latency (ms) |
|---|---:|
| `openvino-fp16` | 539.75 |
| current `openvino-int8_sym` | 570.55 |
| candidate `INT8_SYM`, `group_size=128`, `scale_estimation=True` | 539.29 |
| candidate `INT8_ASYM`, `group_size=128`, `awq=True`, `scale_estimation=True` | 540.19 |

### Longer verification run

| Variant | Mean latency (ms) |
|---|---:|
| `openvino-fp16` | 536.74 |
| candidate `INT8_SYM`, `group_size=128`, `scale_estimation=True` | 538.57 |

Conclusion: the grouped+scale recipe **removes the large regression** and gets to **near parity**, but I could not prove a stable win over FP16 on this hardware/software stack.

## Additional findings

### `openvino-int8_w8a16` is not a distinct export today

The current exported binaries for these variants are identical:

- `openvino-int8_sym`
- `openvino-int8_w8a16`
- `openvino-int8_w8a8`

So the current `w8a16` and `w8a8` labels are misleading; they are effectively the same weight-only INT8 export in the existing snapshot.

### Full activation PTQ was worse

I also tested `nncf.quantize(...)` on the vision encoder. It produced a much slower GPU result (~541.76 ms), so full PTQ is **not** the immediate fix here.

## Permanent code changes

### 1. `library/src/instantlearn/scripts/sam3/export_sam3.py`

Implemented a recipe table and changed exports so that:

- `openvino-int8_sym` uses `group_size=128` + `scale_estimation=True`
- `openvino-int8_w8a16` uses `group_size=128` + `scale_estimation=True`

This is the main fix.

### 2. `library/src/instantlearn/models/sam3/sam3_openvino.py`

Refactored compile-property setup into a helper and made GPU defaults explicit:

- `NUM_STREAMS=1`
- `INFERENCE_PRECISION_HINT=f16`
- optional `compile_properties` override for future experiments

This does **not** create the GPU speedup by itself, but it makes the runtime configuration explicit and reproducible, and documents that the real fix is export-side rather than a hidden GPU runtime knob.

## Recommendation

Use the new export recipe as the default for `openvino-int8_sym` and `openvino-int8_w8a16`.

If the product requirement is **strictly faster than FP16 on Arc Xe2**, additional work is still needed upstream, likely one of:

1. a GPU path that keeps the compressed representation useful to MatMul instead of restoring FP16 before execution
2. a true mixed/activation INT8 path that maps to supported Arc kernels
3. selective quantization of only the layers that help on GPU instead of compressing every weight-bearing subgraph
4. upstream OpenVINO GPU-plugin changes for weight-only compressed transformer blocks

## Final answer to the key question

- **Which runtime property fixes it?** None found on this stack.
- **Which re-quantization recipe helps most?** `INT8_SYM` with **`group_size=128` + `scale_estimation=True`**.
- **Does it get below FP16?** Not reliably on my verified Arc B570 runs; it gets to near-parity and removes almost all of the regression.
