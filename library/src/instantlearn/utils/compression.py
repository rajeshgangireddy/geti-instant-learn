# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OpenVINO model weight compression via NNCF."""

import logging
import time

import openvino as ov

from instantlearn.utils.constants import CompressionMode

logger = logging.getLogger(__name__)

# Mapping from CompressionMode to nncf.CompressWeightsMode.
# Resolved lazily so that nncf is only imported when actually used.
_NNCF_MODE_MAP: dict[CompressionMode, str] = {
    CompressionMode.INT8_SYM: "INT8_SYM",
    CompressionMode.INT8_ASYM: "INT8_ASYM",
    CompressionMode.INT4_SYM: "INT4_SYM",
    CompressionMode.INT4_ASYM: "INT4_ASYM",
}


def compress_model(
    ov_model: ov.Model,
    mode: str | CompressionMode = CompressionMode.FP32,
    group_size: int = 128,
) -> ov.Model:
    """Apply weight compression to an OpenVINO model.

    Args:
        ov_model: OpenVINO model to compress.
        mode: Compression mode. One of the :class:`CompressionMode` values.
        group_size: Group size for INT4 compression. Ignored for INT8/FP16/FP32.
            Smaller values preserve more accuracy at the cost of less compression.

    Returns:
        Compressed OpenVINO model. For FP32 the model is returned unchanged.
        For FP16 the caller should use ``openvino.save_model(..., compress_to_fp16=True)``
        instead — this function returns the model unchanged in that case too.

    Raises:
        ValueError: If *mode* is not a valid :class:`CompressionMode` value.
    """
    mode = CompressionMode(mode)

    if mode in (CompressionMode.FP32, CompressionMode.FP16):
        # FP32 = no-op. FP16 is handled natively by openvino.save_model(compress_to_fp16=True).
        return ov_model

    nncf_mode_name = _NNCF_MODE_MAP.get(mode)
    if nncf_mode_name is None:
        msg = f"Unsupported compression mode: {mode}"
        raise ValueError(msg)

    import nncf  # noqa: PLC0415

    nncf_mode = getattr(nncf.CompressWeightsMode, nncf_mode_name)

    if mode in (CompressionMode.INT4_SYM, CompressionMode.INT4_ASYM):
        logger.warning(
            "INT4 weight compression on small vision models may degrade accuracy. "
            "Benchmark the quantized model before deploying."
        )

    logger.info("Compressing model weights to %s ...", mode.value)
    start = time.time()

    kwargs: dict = {"mode": nncf_mode}
    if mode in (CompressionMode.INT4_SYM, CompressionMode.INT4_ASYM):
        kwargs["group_size"] = group_size

    ov_model = nncf.compress_weights(ov_model, **kwargs)

    elapsed = time.time() - start
    logger.info("Weight compression (%s) finished in %.1fs", mode.value, elapsed)

    return ov_model
