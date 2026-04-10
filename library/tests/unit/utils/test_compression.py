# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the weight compression utility module."""

import numpy as np
import openvino as ov
import pytest

from instantlearn.utils.compression import compress_model
from instantlearn.utils.constants import CompressionMode


def _make_dummy_ov_model() -> ov.Model:
    """Create a minimal OpenVINO model (single linear layer) for testing."""
    from openvino.runtime import opset13 as opset  # noqa: PLC0415

    param = opset.parameter([1, 16], dtype=np.float32, name="input")
    weights = opset.constant(np.random.default_rng(42).standard_normal((16, 8)).astype(np.float32))
    matmul = opset.matmul(param, weights, False, False)
    result = opset.result(matmul, name="output")
    return ov.Model([result], [param], "test_model")


class TestCompressModel:
    """Tests for compress_model()."""

    def test_fp32_returns_model_unchanged(self) -> None:
        model = _make_dummy_ov_model()
        result = compress_model(model, CompressionMode.FP32)
        assert result is model

    def test_fp16_returns_model_unchanged(self) -> None:
        """FP16 is handled by openvino.save_model, not nncf. Model should pass through."""
        model = _make_dummy_ov_model()
        result = compress_model(model, CompressionMode.FP16)
        assert result is model

    def test_int8_sym_compresses(self) -> None:
        model = _make_dummy_ov_model()
        compressed = compress_model(model, CompressionMode.INT8_SYM)
        assert compressed is not None
        assert len(compressed.outputs) == len(model.outputs)

    def test_int8_asym_compresses(self) -> None:
        model = _make_dummy_ov_model()
        compressed = compress_model(model, CompressionMode.INT8_ASYM)
        assert compressed is not None
        assert len(compressed.outputs) == len(model.outputs)

    def test_int4_sym_compresses(self) -> None:
        model = _make_dummy_ov_model()
        compressed = compress_model(model, CompressionMode.INT4_SYM)
        assert compressed is not None

    def test_int4_asym_compresses(self) -> None:
        model = _make_dummy_ov_model()
        compressed = compress_model(model, CompressionMode.INT4_ASYM)
        assert compressed is not None

    def test_string_mode_accepted(self) -> None:
        model = _make_dummy_ov_model()
        compressed = compress_model(model, "int8_sym")
        assert compressed is not None

    def test_invalid_mode_raises(self) -> None:
        model = _make_dummy_ov_model()
        with pytest.raises(ValueError, match="not a valid CompressionMode"):
            compress_model(model, "invalid_mode")

    def test_compressed_model_produces_output(self) -> None:
        model = _make_dummy_ov_model()
        compressed = compress_model(model, CompressionMode.INT8_SYM)
        compiled = ov.compile_model(compressed)
        result = compiled(np.ones((1, 16), dtype=np.float32))
        assert result[0].shape == (1, 8)


class TestCompressionModeEnum:
    """Tests for the CompressionMode enum."""

    def test_all_values(self) -> None:
        expected = {"fp32", "fp16", "int8_sym", "int8_asym", "int4_sym", "int4_asym"}
        actual = {m.value for m in CompressionMode}
        assert actual == expected

    def test_string_construction(self) -> None:
        assert CompressionMode("fp32") == CompressionMode.FP32
        assert CompressionMode("int8_sym") == CompressionMode.INT8_SYM

    def test_invalid_string_raises(self) -> None:
        with pytest.raises(ValueError):
            CompressionMode("int16")
