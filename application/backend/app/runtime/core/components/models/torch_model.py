# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

import numpy as np
import torch
from instantlearn.data.base.batch import Batch
from instantlearn.data.base.sample import Sample
from instantlearn.models.base import InferenceResult, Model
from torchvision import tv_tensors

from domain.services.schemas.processor import InputData
from runtime.core.components.base import ModelHandler

logger = logging.getLogger(__name__)


class TorchModelHandler(ModelHandler):
    def __init__(self, model: Model, reference_batch: Batch) -> None:
        self._model = model
        self._reference_batch = reference_batch

    def initialise(self) -> None:
        logger.info(
            "Initialising TorchModelHandler: model=%s, reference batch size=%d",
            type(self._model).__name__,
            len(self._reference_batch.samples),
        )
        self._model.fit(self._reference_batch)

    def predict(self, inputs: list[InputData]) -> list[dict[str, np.ndarray]]:
        logger.debug("Inference started: model=%s batch size=%d", type(self._model).__name__, len(inputs))
        batch = self._build_batch(inputs)
        inference_result = self._model.predict(batch)

        # Bridge model-level component timing into per-frame traces
        if isinstance(inference_result, InferenceResult) and inference_result.timing:
            timing = inference_result.timing
            logger.debug(
                "Model timing: %s (total=%.1f ms)",
                {k: f"{v:.1f}ms" for k, v in timing.component_times.items()},
                timing.total_ms,
            )
            # Inject sub-component timing into each input's FrameTrace
            for input_data in inputs:
                if input_data.trace is not None:
                    for component, duration_ms in timing.component_times.items():
                        input_data.trace.record_model_component(component, duration_ms)

        results = []
        for result in inference_result:
            batch_result = {}
            for k, v in result.items():
                tensor = v
                if isinstance(tensor, torch.Tensor) and tensor.dtype == torch.bfloat16:
                    tensor = tensor.to(torch.float32)
                batch_result[k] = tensor.detach().cpu().numpy()
            results.append(batch_result)
        return results

    @staticmethod
    def _build_batch(inputs: list[InputData]) -> Batch:
        samples = [Sample(image=tv_tensors.Image(torch.from_numpy(data.frame).permute(2, 0, 1))) for data in inputs]
        return Batch.collate(samples)
