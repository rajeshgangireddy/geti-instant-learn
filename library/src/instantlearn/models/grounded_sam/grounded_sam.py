# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""This model uses a zero-shot object detector (from Huggingface) to generate boxes for SAM."""

import time

from instantlearn.components import SamDecoder
from instantlearn.components.sam import load_sam_model
from instantlearn.data.base.batch import Batch, Collatable
from instantlearn.data.base.sample import Sample
from instantlearn.models.base import InferenceResult, Model
from instantlearn.utils.constants import SAMModelName

from .grounded import GroundingModel, TextToBoxPromptGenerator
from .prompt_filter import BoxPromptFilter


class GroundedSAM(Model):
    """This model uses a zero-shot object detector (from Huggingface) to generate boxes for SAM."""

    def __init__(
        self,
        sam: SAMModelName = SAMModelName.SAM_HQ_TINY,
        grounding_model: GroundingModel = GroundingModel.LLMDET_TINY,
        precision: str = "bf16",
        compile_models: bool = False,
        box_threshold: float = 0.4,
        text_threshold: float = 0.3,
        use_nms: bool = True,
        device: str = "cuda",
    ) -> None:
        """Initialize the model.

        Args:
            sam: The SAM model name.
            grounding_model: The grounding model to use.
            precision: The precision to use for the model.
            compile_models: Whether to compile the models.
            box_threshold: The box threshold.
            text_threshold: The text threshold.
            use_nms: Whether to use NMS in SamDecoder.
            device: The device to use.
        """
        super().__init__()
        self.sam_predictor = load_sam_model(
            sam,
            device=device,
            precision=precision,
            compile_models=compile_models,
        )
        self.prompt_generator: TextToBoxPromptGenerator = TextToBoxPromptGenerator(
            device=device,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            template=TextToBoxPromptGenerator.Template.specific_object,
            model_id=grounding_model,
            precision=precision,
            compile_models=compile_models,
        )
        self.segmenter: SamDecoder = SamDecoder(sam_predictor=self.sam_predictor, use_nms=use_nms)
        self.prompt_filter: BoxPromptFilter = BoxPromptFilter()

    def fit(self, reference: Sample | list[Sample] | Batch) -> None:
        """Perform learning step on the reference images and priors.

        Args:
            reference: Reference data to learn from. Accepts:
                - Sample: A single reference sample
                - list[Sample]: A list of reference samples
                - Batch: A batch of reference samples
        """
        reference_batch = Batch.collate(reference)
        self.category_mapping = {}
        for sample in reference_batch.samples:
            for category_id, category in zip(sample.category_ids, sample.categories, strict=False):
                if category not in self.category_mapping:
                    self.category_mapping[category] = int(category_id)

    def predict(self, target: Collatable) -> InferenceResult:
        """Perform inference step on the target images.

        Args:
            target: Target data to infer. Accepts:
                - Sample: A single target sample
                - list[Sample]: A list of target samples
                - Batch: A batch of target samples
                - str | Path: A single image path
                - list[str] | list[Path]: Multiple image paths

        Returns:
            InferenceResult with predictions and per-component timing.
        """
        wall_start = time.perf_counter()
        target_batch = Batch.collate(target)

        # Generate box prompts (tensor format)
        with self._time_component("grounding") as t_ground:
            box_prompts, category_ids = self.prompt_generator(
                target_batch.images,
                self.category_mapping,
            )

        # Filter box prompts
        with self._time_component("box_filter") as t_filter:
            box_prompts = self.prompt_filter(box_prompts)

        # Decode masks
        with self._time_component("decoder") as t_dec:
            predictions = self.segmenter(
                target_batch.images,
                category_ids,
                box_prompts=box_prompts,
            )

        total_ms = (time.perf_counter() - wall_start) * 1000.0
        timing = self._build_timing([t_ground, t_filter, t_dec], total_ms)
        return InferenceResult(predictions=predictions, timing=timing)
