# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Monkey-patches for TinyViT modules to produce GPU-safe ONNX graphs."""

from __future__ import annotations

import logging
import types
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)

_PATCHED_ATTR = "_tinyvit_original_forward"
_BIAS_BUFFER = "_ov_preexpanded_attention_bias"
_FUSED_BN_ATTR = "_tinyvit_fused_bn_list"

__all__ = ["patch_tinyvit_for_export", "patch_tinyvit_window_partition", "unpatch_tinyvit"]


# ---------------------------------------------------------------------------
# Patched Attention forward (eliminates LayerNorm op + Gather + Split)
# ---------------------------------------------------------------------------

def _make_attention_forward(module: nn.Module) -> Any:
    """Return a bound forward that uses manual LayerNorm and slice instead of split."""

    def patched_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape

        ln = self.norm
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        x = (x - mean) * torch.rsqrt(var + ln.eps) * ln.weight + ln.bias

        qkv = self.qkv(x)
        qkv_4d = qkv.view(B, N, self.num_heads, -1)
        q = qkv_4d[:, :, :, : self.key_dim]
        k = qkv_4d[:, :, :, self.key_dim : 2 * self.key_dim]
        v = qkv_4d[:, :, :, 2 * self.key_dim :]

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        bias = getattr(self, _BIAS_BUFFER).to(device=q.device, dtype=q.dtype)
        attn = (q @ k.transpose(-2, -1)) * self.scale + bias
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dh)
        x = self.proj(x)
        return x

    return types.MethodType(patched_forward, module)


# ---------------------------------------------------------------------------
# Patched TinyViTBlock forward (eliminates 6-D Transpose + MLP LayerNorm)
# ---------------------------------------------------------------------------

def _make_tinyvit_block_forward(module: nn.Module) -> Any:
    """Return a bound forward using ONNX-compatible window partition and manual LayerNorm for MLP."""
    import torch.nn.functional as F  # noqa: PLC0415

    def patched_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        res_x = x.view(B, H, W, C)

        if H == self.window_size and W == self.window_size:
            x = self.attn(x).view(B, H, W, C)
        else:
            x = x.view(B, H, W, C)
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            padding = pad_b > 0 or pad_r > 0
            if padding:
                x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
            pH, pW = H + pad_b, W + pad_r
            nH = pH // self.window_size
            nW = pW // self.window_size
            ws = self.window_size

            x = x.reshape(B * nH, ws, pW, C)
            # Decompose 5D permute into 4D transpose — avoids Intel B60 GPU bug.
            x = x.reshape(B * nH, ws, nW, ws * C)       # merge ws+C → [B*nH, ws, nW, ws*C]
            x = x.permute(0, 2, 1, 3).contiguous()      # 4D swap dims 1,2 → [B*nH, nW, ws, ws*C]
            x = x.reshape(B * nH * nW, ws * ws, C)

            x = self.attn(x)

            # Decompose 5D permute into 4D transpose — avoids Intel B60 GPU bug.
            x = x.reshape(B * nH, nW, ws, ws * C)       # merge ws+C → [B*nH, nW, ws, ws*C]
            x = x.permute(0, 2, 1, 3).contiguous()      # 4D swap dims 1,2 → [B*nH, ws, nW, ws*C]
            x = x.reshape(B, pH, pW, C)

            if padding:
                x = x[:, :H, :W].contiguous()

        # Keep the residual add in spatial layout. The Intel B60 GPU diverges on
        # the flattened [B, L, C] Add for TinyViT stage-1 blocks, while the 4D add
        # on [B, H, W, C] stays aligned with CPU.
        x = res_x.contiguous() + self.drop_path(x).contiguous()
        x = x.view(B, L, C)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.local_conv(x)
        x = x.view(B, C, L).transpose(1, 2)

        mlp = self.mlp
        mlp_x = x
        ln = mlp.norm
        mean = mlp_x.mean(dim=-1, keepdim=True)
        var = ((mlp_x - mean) ** 2).mean(dim=-1, keepdim=True)
        mlp_x = (mlp_x - mean) * torch.rsqrt(var + ln.eps) * ln.weight + ln.bias
        mlp_x = mlp.fc1(mlp_x)
        mlp_x = mlp.act(mlp_x)
        mlp_x = mlp.drop(mlp_x)
        mlp_x = mlp.fc2(mlp_x)
        mlp_x = mlp.drop(mlp_x)
        x = x + self.drop_path(mlp_x)
        return x

    return types.MethodType(patched_forward, module)


# ---------------------------------------------------------------------------
# Conv2d_BN fusion helpers
# ---------------------------------------------------------------------------

def _fuse_conv2d_bn_recursive(module: nn.Module, tiny_vit_sam: Any, saved: list) -> None:
    for child_name, child_module in list(module.named_children()):
        if isinstance(child_module, tiny_vit_sam.Conv2d_BN):
            with torch.no_grad():
                fused = child_module.fuse()
            fused.train(child_module.training)
            fused.requires_grad_(False)
            saved.append((module, child_name, child_module))
            setattr(module, child_name, fused)
        else:
            _fuse_conv2d_bn_recursive(child_module, tiny_vit_sam, saved)


def _apply_conv2d_bn_fusion(img_encoder: nn.Module, tiny_vit_sam: Any) -> None:
    if _FUSED_BN_ATTR in img_encoder.__dict__:
        return
    saved: list = []
    _fuse_conv2d_bn_recursive(img_encoder, tiny_vit_sam, saved)
    img_encoder.__dict__[_FUSED_BN_ATTR] = saved


def _restore_conv2d_bn_fusion(img_encoder: nn.Module) -> None:
    saved = img_encoder.__dict__.pop(_FUSED_BN_ATTR, None)
    if saved is None:
        return
    for parent_module, child_name, original in saved:
        setattr(parent_module, child_name, original)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _patch_module(module: nn.Module, tiny_vit_sam: Any, patch_window_partition: bool) -> tuple[bool, bool]:
    patched_attention = False
    patched_block = False

    if isinstance(module, tiny_vit_sam.Attention) and hasattr(module, "attention_bias_idxs"):
        if not hasattr(module, _BIAS_BUFFER):
            with torch.no_grad():
                preexpanded = module.attention_biases[:, module.attention_bias_idxs].detach().clone()
            module.register_buffer(_BIAS_BUFFER, preexpanded, persistent=False)
        if _PATCHED_ATTR not in module.__dict__:
            module.__dict__[_PATCHED_ATTR] = module.forward
            module.forward = _make_attention_forward(module)
        patched_attention = True

    if patch_window_partition and isinstance(module, tiny_vit_sam.TinyViTBlock):
        if _PATCHED_ATTR not in module.__dict__:
            module.__dict__[_PATCHED_ATTR] = module.forward
            module.forward = _make_tinyvit_block_forward(module)
        patched_block = True

    return patched_attention, patched_block


def _unpatch_module(module: nn.Module, tiny_vit_sam: Any) -> None:
    if isinstance(module, (tiny_vit_sam.Attention, tiny_vit_sam.TinyViTBlock)):
        if _PATCHED_ATTR in module.__dict__:
            module.forward = module.__dict__.pop(_PATCHED_ATTR)
        if _BIAS_BUFFER in module._buffers:  # noqa: SLF001
            module._buffers.pop(_BIAS_BUFFER)  # noqa: SLF001


def _get_sam_img_encoder(model: nn.Module) -> nn.Module | None:
    for module in model.modules():
        predictor = getattr(module, "_predictor", None)
        if predictor is None:
            continue
        sam_model = getattr(predictor, "model", None)
        if sam_model is None:
            continue
        img_encoder = getattr(sam_model, "image_encoder", None)
        if img_encoder is not None and isinstance(img_encoder, nn.Module):
            return img_encoder
    return None


def _iter_all_tinyvit_modules(model: nn.Module):
    img_encoder = _get_sam_img_encoder(model)
    yield from model.modules()
    if img_encoder is not None:
        yield from img_encoder.modules()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def patch_tinyvit_window_partition(model: nn.Module) -> None:
    """Patch only TinyViTBlock window partition/reverse ops."""
    try:
        import segment_anything_hq.modeling.tiny_vit_sam as tiny_vit_sam  # noqa: PLC0415
    except ImportError:
        return

    patched = 0
    seen: set[int] = set()
    for module in _iter_all_tinyvit_modules(model):
        mid = id(module)
        if mid in seen:
            continue
        seen.add(mid)
        if isinstance(module, tiny_vit_sam.TinyViTBlock):
            if _PATCHED_ATTR not in module.__dict__:
                module.__dict__[_PATCHED_ATTR] = module.forward
                module.forward = _make_tinyvit_block_forward(module)
            patched += 1

    if patched == 0:
        logger.warning("patch_tinyvit_window_partition: no TinyViT blocks found; skipping patch")


def patch_tinyvit_for_export(model: nn.Module, patch_window_partition: bool = True) -> None:
    """Monkey-patch TinyViT modules to eliminate GPU-unfriendly ONNX ops."""
    try:
        import segment_anything_hq.modeling.tiny_vit_sam as tiny_vit_sam  # noqa: PLC0415
    except ImportError:
        return

    seen: set[int] = set()
    patched_attention = 0
    for module in _iter_all_tinyvit_modules(model):
        mid = id(module)
        if mid in seen:
            continue
        seen.add(mid)
        did_patch_attention, _ = _patch_module(module, tiny_vit_sam, patch_window_partition)
        patched_attention += int(did_patch_attention)

    img_encoder = _get_sam_img_encoder(model)
    if img_encoder is not None:
        _apply_conv2d_bn_fusion(img_encoder, tiny_vit_sam)

    if patched_attention == 0:
        logger.warning("patch_tinyvit_for_export: no TinyViT attention modules found; skipping patch")


def unpatch_tinyvit(model: nn.Module) -> None:
    """Restore original forward methods on any patched TinyViT modules."""
    try:
        import segment_anything_hq.modeling.tiny_vit_sam as tiny_vit_sam  # noqa: PLC0415
    except ImportError:
        return

    seen: set[int] = set()
    for module in _iter_all_tinyvit_modules(model):
        mid = id(module)
        if mid in seen:
            continue
        seen.add(mid)
        _unpatch_module(module, tiny_vit_sam)

    img_encoder = _get_sam_img_encoder(model)
    if img_encoder is not None:
        _restore_conv2d_bn_fusion(img_encoder)
