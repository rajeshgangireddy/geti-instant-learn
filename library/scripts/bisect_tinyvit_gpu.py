#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Export the SAM-HQ-Tiny image encoder with checkpoint outputs and compare OV CPU vs GPU.

This script focuses on a single TinyViT block and emits multiple intermediate
outputs so the first divergent checkpoint can be located quickly.
"""

from __future__ import annotations

import argparse
import types
from pathlib import Path

import numpy as np
import onnx
import openvino as ov
import torch
from torch import nn

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from instantlearn.components.sam.predictor import load_sam_model
from instantlearn.components.sam.tinyvit_patches import patch_tinyvit_for_export, unpatch_tinyvit
from instantlearn.data.utils.image import read_image
from instantlearn.utils.constants import SAMModelName

ASSETS = Path(__file__).resolve().parent.parent / "examples" / "assets" / "coco"
DEFAULT_IMAGE = ASSETS / "000000390341.jpg"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "bisect_tinyvit_gpu"
GPU_CONFIG = {
    "PERFORMANCE_HINT": "LATENCY",
    "INFERENCE_PRECISION_HINT": "f32",
}
CHECKPOINT_NAMES = [
    "norm_pre_qkv",
    "qkv_linear",
    "attn_logits_bias",
    "attn_softmax",
    "attn_v_reshape",
    "proj_linear",
    "after_window_reverse",
    "after_spatial_slice",
    "after_token_view",
    "after_drop_path",
    "after_residual_add",
    "encoder_features",
]


class EncoderCheckpointWrapper(nn.Module):
    def __init__(
        self,
        predictor: nn.Module,
        layer_idx: int,
        block_idx: int,
        legacy_residual_add: bool = False,
    ) -> None:
        super().__init__()
        self.predictor = predictor
        self.image_encoder = predictor._predictor.model.image_encoder  # noqa: SLF001
        self.layer_idx = layer_idx
        self.block_idx = block_idx
        self.target_block = self.image_encoder.layers[layer_idx].blocks[block_idx]
        self.target_attention = self.target_block.attn
        self.legacy_residual_add = legacy_residual_add
        self._records: dict[str, torch.Tensor] = {}
        self._saved_block_forward = self.target_block.forward
        self._saved_attn_forward = self.target_attention.forward

    def _install_debug_patch(self) -> None:
        def debug_attn_forward(attn_self: nn.Module, x: torch.Tensor) -> torch.Tensor:
            bsz, tokens, _ = x.shape
            ln = attn_self.norm
            mean = x.mean(dim=-1, keepdim=True)
            var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
            normed = (x - mean) * torch.rsqrt(var + ln.eps) * ln.weight + ln.bias
            self._records["norm_pre_qkv"] = normed

            qkv = attn_self.qkv(normed)
            self._records["qkv_linear"] = qkv

            q, k, v = qkv.view(bsz, tokens, attn_self.num_heads, -1).split(
                [attn_self.key_dim, attn_self.key_dim, attn_self.d],
                dim=3,
            )
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)

            bias = getattr(attn_self, "ab")
            logits = (q @ k.transpose(-2, -1)) * attn_self.scale + bias
            self._records["attn_logits_bias"] = logits

            attn = logits.softmax(dim=-1)
            self._records["attn_softmax"] = attn

            attended = (attn @ v).transpose(1, 2).reshape(bsz, tokens, attn_self.dh)
            self._records["attn_v_reshape"] = attended

            projected = attn_self.proj(attended)
            self._records["proj_linear"] = projected
            return projected

        def debug_block_forward(block_self: nn.Module, x: torch.Tensor) -> torch.Tensor:
            h, w = block_self.input_resolution
            batch, length, channels = x.shape
            assert length == h * w, "input feature has wrong size"
            residual_tokens = x
            residual_spatial = residual_tokens.view(batch, h, w, channels)

            if h == block_self.window_size and w == block_self.window_size:
                x = block_self.attn(x).view(batch, h, w, channels)
                self._records["after_window_reverse"] = x
                self._records["after_spatial_slice"] = x
            else:
                x = x.view(batch, h, w, channels)
                pad_b = (block_self.window_size - h % block_self.window_size) % block_self.window_size
                pad_r = (block_self.window_size - w % block_self.window_size) % block_self.window_size
                padding = pad_b > 0 or pad_r > 0
                if padding:
                    x = torch.nn.functional.pad(x, (0, 0, 0, pad_r, 0, pad_b))

                ph, pw = h + pad_b, w + pad_r
                nh = ph // block_self.window_size
                nw = pw // block_self.window_size
                ws = block_self.window_size
                x = x.view(batch, nh, ws, nw, ws, channels).transpose(2, 3).reshape(batch * nh * nw, ws * ws, channels)
                x = block_self.attn(x)
                x = x.view(batch, nh, nw, ws, ws, channels).transpose(2, 3).reshape(batch, ph, pw, channels)
                self._records["after_window_reverse"] = x

                if padding:
                    x = x[:, :h, :w].contiguous()
                self._records["after_spatial_slice"] = x

            token_view = x.view(batch, length, channels)
            self._records["after_token_view"] = token_view

            if self.legacy_residual_add:
                dropped = block_self.drop_path(token_view)
                residual_add = residual_tokens + dropped
            else:
                dropped = block_self.drop_path(x)
                residual_add = residual_spatial.contiguous() + dropped.contiguous()

            self._records["after_drop_path"] = dropped.view(batch, length, channels)
            x = residual_add.view(batch, length, channels)
            self._records["after_residual_add"] = x

            x = x.transpose(1, 2).reshape(batch, channels, h, w)
            x = block_self.local_conv(x)
            x = x.view(batch, channels, length).transpose(1, 2)
            x = x + block_self.drop_path(block_self.mlp(x))
            return x

        self.target_attention.forward = types.MethodType(debug_attn_forward, self.target_attention)
        self.target_block.forward = types.MethodType(debug_block_forward, self.target_block)

    def _restore_debug_patch(self) -> None:
        self.target_block.forward = self._saved_block_forward
        self.target_attention.forward = self._saved_attn_forward

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        self._records = {}
        self._install_debug_patch()
        try:
            features, _ = self.image_encoder(x)
        finally:
            self._restore_debug_patch()
        self._records["encoder_features"] = features
        return tuple(self._records[name] for name in CHECKPOINT_NAMES)


def build_input_tensor(predictor: nn.Module, image_path: Path) -> tuple[torch.Tensor, np.ndarray]:
    image = read_image(str(image_path))
    transformed = predictor.transform.apply_image_torch(image).float().cpu()
    preprocessed = predictor._predictor.model.preprocess(transformed)  # noqa: SLF001
    return preprocessed, image.permute(1, 2, 0).cpu().numpy()


def summarize_diff(name: str, cpu_value: np.ndarray, gpu_value: np.ndarray) -> tuple[float, float]:
    diff = np.abs(cpu_value.astype(np.float32) - gpu_value.astype(np.float32))
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    print(
        f"[BISECT] {name:20s} shape={tuple(cpu_value.shape)!s:20s} "
        f"max_diff={max_diff:.6f} mean_diff={mean_diff:.6f}",
    )
    return max_diff, mean_diff


def export_model(wrapper: nn.Module, sample_input: torch.Tensor, onnx_path: Path, xml_path: Path) -> None:
    wrapper.eval()
    with torch.no_grad():
        patch_tinyvit_for_export(wrapper)
        try:
            torch.onnx.export(
                wrapper,
                args=(sample_input,),
                f=onnx_path,
                input_names=["image"],
                output_names=CHECKPOINT_NAMES,
                dynamo=False,
            )
        finally:
            unpatch_tinyvit(wrapper)

    ov_model = ov.Core().read_model(str(onnx_path))
    ov.save_model(ov_model, str(xml_path))


def print_block_graph_summary(onnx_path: Path, layer_idx: int, block_idx: int) -> None:
    model = onnx.load(str(onnx_path))
    prefix = f"/image_encoder/layers.{layer_idx}/blocks.{block_idx}"
    nodes = [node for node in model.graph.node if prefix in node.name]
    if not nodes:
        prefix = f"/layers.{layer_idx}/blocks.{block_idx}"
        nodes = [node for node in model.graph.node if prefix in node.name]
    print(f"[BISECT] ONNX nodes for {prefix} ({len(nodes)} total):")
    for node in nodes:
        print(f"[BISECT]   {node.name} :: {node.op_type}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--block", type=int, default=1)
    parser.add_argument(
        "--legacy-residual-add",
        action="store_true",
        help="Reproduce the old token-layout residual add instead of the patched 4D spatial add.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = args.output_dir / "tinyvit_encoder_debug.onnx"
    xml_path = args.output_dir / "tinyvit_encoder_debug.xml"

    predictor = load_sam_model(
        sam=SAMModelName.SAM_HQ_TINY,
        device="cpu",
        precision="fp32",
        compile_models=False,
    )
    wrapper = EncoderCheckpointWrapper(
        predictor,
        args.layer,
        args.block,
        legacy_residual_add=args.legacy_residual_add,
    ).cpu().eval()
    sample_input, image = build_input_tensor(predictor, args.image)

    print(f"[BISECT] image={args.image}")
    print(f"[BISECT] target block=layers.{args.layer}.blocks.{args.block}")
    print(f"[BISECT] transformed input shape={tuple(sample_input.shape)}")
    print(f"[BISECT] image shape={tuple(image.shape)}")
    print(f"[BISECT] legacy_residual_add={args.legacy_residual_add}")

    export_model(wrapper, sample_input, onnx_path, xml_path)
    print(f"[BISECT] exported ONNX: {onnx_path}")
    print(f"[BISECT] exported  IR : {xml_path}")
    print_block_graph_summary(onnx_path, args.layer, args.block)

    core = ov.Core()
    compiled_cpu = core.compile_model(core.read_model(str(xml_path)), "CPU")
    if "GPU" not in core.available_devices:
        print("[BISECT] GPU device is unavailable")
        return 1
    compiled_gpu = core.compile_model(core.read_model(str(xml_path)), "GPU", config=GPU_CONFIG)

    np_input = sample_input.cpu().numpy()
    cpu_outputs = compiled_cpu([np_input])
    gpu_outputs = compiled_gpu([np_input])

    first_bad: tuple[str, float, float] | None = None
    for name in CHECKPOINT_NAMES:
        cpu_value = np.array(cpu_outputs[compiled_cpu.output(name)])
        gpu_value = np.array(gpu_outputs[compiled_gpu.output(name)])
        max_diff, mean_diff = summarize_diff(name, cpu_value, gpu_value)
        if first_bad is None and max_diff > 1e-2:
            first_bad = (name, max_diff, mean_diff)

    if first_bad is None:
        print("[BISECT] No divergent checkpoint found (CPU/GPU stayed aligned across exported checkpoints).")
        return 0

    name, max_diff, mean_diff = first_bad
    print(f"[BISECT] First divergent checkpoint: {name} max_diff={max_diff:.6f} mean_diff={mean_diff:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
