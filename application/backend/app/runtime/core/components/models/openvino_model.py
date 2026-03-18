# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import tempfile
import time
from collections import defaultdict

import numpy as np
import openvino
from instantlearn.data.base.batch import Batch
from instantlearn.models.base import Model
from instantlearn.utils.constants import Backend
from instantlearn.utils.utils import device_to_openvino_device, precision_to_openvino_type
from openvino import properties

from domain.services.schemas.processor import InputData
from runtime.core.components.base import ModelHandler

logger = logging.getLogger(__name__)

# Maps ONNX node name prefixes to logical pipeline components.
# Adjust prefixes to match your exported model's node naming convention.
_OV_NODE_PREFIX_TO_COMPONENT: dict[str, str] = {
    "encoder": "encoder",
    "dinov": "encoder",
    "prompt_generator": "prompt_generator",
    "sam_decoder": "decoder",
    "decoder": "decoder",
    "mask_decoder": "decoder",
}


def _aggregate_profiling_info(
    profiling_info: list,
) -> dict[str, float]:
    """Aggregate OpenVINO per-node profiling into logical component times.

    Args:
        profiling_info: List of ProfilingInfo objects from compiled_model.

    Returns:
        Dict mapping component name to total time in milliseconds.
    """
    component_times: dict[str, float] = defaultdict(float)
    for node_info in profiling_info:
        node_name = node_info.node_name.lower()
        real_time_us = (
            node_info.real_time.total_seconds() * 1e6
            if hasattr(node_info.real_time, "total_seconds")
            else float(node_info.real_time)
        )

        matched = False
        for prefix, component in _OV_NODE_PREFIX_TO_COMPONENT.items():
            if prefix in node_name:
                component_times[component] += real_time_us / 1000.0  # us → ms
                matched = True
                break
        if not matched:
            component_times["other"] += real_time_us / 1000.0

    return dict(component_times)


class OpenVINOModelHandler(ModelHandler):
    def __init__(self, model: Model, reference_batch: Batch, precision: str, enable_profiling: bool = False) -> None:
        self._model = model
        self._reference_batch = reference_batch
        self._precision = precision
        self._compiled_model = None
        self._enable_profiling = enable_profiling
        self._infer_request = None

    def initialise(self) -> None:
        self._model.fit(self._reference_batch)

        # Export on CPU to avoid XPU/CUDA compilation issues during tracing
        # The exported OpenVINO model can then be run on any device (CPU, GPU, etc.)
        original_device = next(self._model.parameters()).device
        self._model.cpu()
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                path = self._model.export(tmp_dir, Backend.OPENVINO)

                core = openvino.Core()
                ov_device = device_to_openvino_device("CPU")

                config: dict = {
                    properties.hint.inference_precision: precision_to_openvino_type(self._precision),
                }
                if self._enable_profiling:
                    config[properties.enable_profiling] = True

                core.set_property(ov_device, config)

                logger.debug(f"Compiling exported model from {path} for device {ov_device}...")
                logger.debug(f"Reading model {path}...")
                ov_model = core.read_model(str(path))
                logger.debug(f"Compiling model to {ov_device} (this may take a few minutes)...")

                start_time = time.time()
                self._compiled_model = core.compile_model(ov_model, ov_device)
                logger.debug(f"Model compilation finished in {time.time() - start_time:.2f}s.")

                if self._enable_profiling:
                    self._infer_request = self._compiled_model.create_infer_request()
        finally:
            self._model.to(original_device)

    def predict(self, inputs: list[InputData]) -> list[dict[str, np.ndarray]]:
        if self._compiled_model is None:
            raise RuntimeError("Model not initialised. Call initialise() first.")

        logger.debug("Inference started: model=%s batch size=%d", type(self._model).__name__, len(inputs))

        results: list[dict[str, np.ndarray]] = []

        for input_data in inputs:
            image = np.expand_dims(input_data.frame.transpose(2, 0, 1), axis=0)  # HWC -> 1CHW

            if self._enable_profiling and self._infer_request is not None:
                start = time.perf_counter()
                self._infer_request.infer(image)
                elapsed_ms = (time.perf_counter() - start) * 1000.0

                output = {name: self._infer_request.get_tensor(name).data for name in ["masks", "scores", "labels"]}

                # Extract per-node profiling and aggregate to components
                profiling_info = self._infer_request.get_profiling_info()
                component_times = _aggregate_profiling_info(profiling_info)
                logger.debug(
                    "OV profiling: total=%.1fms components=%s",
                    elapsed_ms,
                    {k: f"{v:.1f}ms" for k, v in component_times.items()},
                )

                # Inject into FrameTrace if available
                if input_data.trace is not None:
                    for comp, dur_ms in component_times.items():
                        input_data.trace.record_model_component(comp, dur_ms)
            else:
                output = self._compiled_model(image)

            results.append(
                {
                    "pred_masks": output["masks"],
                    "pred_scores": output["scores"],
                    "pred_labels": output["labels"],
                }
            )
        return results
