"""Minimal reproducer for two Intel GPU plugin bugs in EfficientSAM3 OpenVINO.

Bug A — Vision encoder FP16 -> NaN
    EfficientViT-B1 vision-encoder.xml, when compiled on GPU with the default
    FP16 inference precision, produces NaN feature maps for normal RGB inputs.
    Workaround: INFERENCE_PRECISION_HINT=f32 (slow).

Bug B — Prompt decoder logit drift on GPU
    The DETR-style prompt-decoder produces pred_logits that drift ~1.8 vs CPU
    on the same inputs, regardless of weight precision or precision hint.
    Workaround: run decoder on CPU.

Usage:
    cd library
    python tools/repro_efficient_sam3_gpu_bugs.py \\
        --model-dir efficient-sam3-openvino/repvit_m1_1/openvino-fp16
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import openvino as ov

RESOLUTION = 1008


def _norm(x: np.ndarray) -> np.ndarray:
    return x.reshape(-1).astype(np.float64)


def _diff(cpu: np.ndarray, gpu: np.ndarray) -> dict[str, float]:
    nan = bool(np.isnan(gpu).any())
    inf = bool(np.isinf(gpu).any())
    if nan or inf:
        return {"nan": nan, "inf": inf, "max_abs_diff": float("nan"), "mean_abs_diff": float("nan")}
    diff = np.abs(_norm(cpu) - _norm(gpu))
    return {
        "nan": False,
        "inf": False,
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
    }


def repro_vision_encoder(core: ov.Core, model_path: Path, hint: str) -> None:
    print(f"\n=== Bug A: vision-encoder ({model_path.name}) ===")
    print(f"GPU INFERENCE_PRECISION_HINT={hint!r}")
    rng = np.random.default_rng(0)
    image = rng.standard_normal((1, 3, RESOLUTION, RESOLUTION), dtype=np.float32)

    cpu_model = core.compile_model(model_path, "CPU")
    cpu_req = cpu_model.create_infer_request()
    cpu_req.infer({0: image})
    cpu_out = {k.any_name: cpu_req.get_tensor(k).data.copy() for k in cpu_model.outputs}

    gpu_props = {"INFERENCE_PRECISION_HINT": hint}
    gpu_model = core.compile_model(model_path, "GPU", gpu_props)
    gpu_req = gpu_model.create_infer_request()
    gpu_req.infer({0: image})
    gpu_out = {k.any_name: gpu_req.get_tensor(k).data.copy() for k in gpu_model.outputs}

    for name in cpu_out:
        d = _diff(cpu_out[name], gpu_out[name])
        print(f"  output[{name}] shape={cpu_out[name].shape}  {d}")


def repro_prompt_decoder(core: ov.Core, model_path: Path) -> None:
    print(f"\n=== Bug B: prompt-decoder ({model_path.name}) ===")
    cpu_model = core.compile_model(model_path, "CPU")
    rng = np.random.default_rng(1)
    feed: dict[str, np.ndarray] = {}
    for inp in cpu_model.inputs:
        name = inp.any_name
        # Use static shapes from the model's partial shape, filling dynamic dims with 1.
        shape = []
        for d in inp.get_partial_shape():
            shape.append(int(d.get_length()) if d.is_static else 1)
        dtype = inp.get_element_type().to_dtype()
        if np.issubdtype(dtype, np.floating):
            feed[name] = rng.standard_normal(shape, dtype=np.float32).astype(dtype)
        else:
            feed[name] = np.ones(shape, dtype=dtype)
        print(f"  input[{name}] shape={shape} dtype={dtype}")

    cpu_req = cpu_model.create_infer_request()
    cpu_req.infer(feed)
    cpu_out = {k.any_name: cpu_req.get_tensor(k).data.copy() for k in cpu_model.outputs}

    for hint in ("f16", "f32"):
        gpu_model = core.compile_model(model_path, "GPU", {"INFERENCE_PRECISION_HINT": hint})
        gpu_req = gpu_model.create_infer_request()
        gpu_req.infer(feed)
        gpu_out = {k.any_name: gpu_req.get_tensor(k).data.copy() for k in gpu_model.outputs}
        print(f"  -- GPU hint={hint!r} --")
        for name in cpu_out:
            d = _diff(cpu_out[name], gpu_out[name])
            print(f"    output[{name}] shape={cpu_out[name].shape}  {d}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-dir",
        type=Path,
        default=Path("efficient-sam3-openvino/repvit_m1_1/openvino-fp16"),
    )
    args = ap.parse_args()

    core = ov.Core()
    print(f"OpenVINO version: {ov.__version__}")
    print(f"Available devices: {core.available_devices}")
    if "GPU" not in core.available_devices:
        raise SystemExit("GPU not available — cannot reproduce GPU plugin bugs.")
    full_name = core.get_property("GPU", "FULL_DEVICE_NAME")
    print(f"GPU FULL_DEVICE_NAME: {full_name}")

    vision = args.model_dir / "vision-encoder.xml"
    decoder = args.model_dir / "prompt-decoder.xml"
    if not vision.is_file() or not decoder.is_file():
        raise SystemExit(f"Missing IR files in {args.model_dir}")

    repro_vision_encoder(core, vision, hint="f16")
    repro_vision_encoder(core, vision, hint="f32")
    repro_prompt_decoder(core, decoder)


if __name__ == "__main__":
    main()
