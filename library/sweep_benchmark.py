# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Run full benchmark sweep across all encoder/decoder/model combinations."""

import gc
import itertools
from datetime import datetime, timezone
from logging import getLogger
from pathlib import Path

import polars as pl
import torch

from instantlearn.scripts.benchmark import load_dataset_by_name, predict_on_dataset
from instantlearn.utils.args import get_arguments
from instantlearn.utils.benchmark import (
    MODEL_ANNOTATION_MODES,
    _save_results,  # noqa: PLC2701
    load_model,
)
from instantlearn.utils.constants import MODEL_MAP, DatasetName, ModelName, SAMModelName
from instantlearn.utils.utils import setup_logger

logger = getLogger("Geti Instant Learn")

ENCODERS = [
    "dinov2_small",
    "dinov2_base",
    "dinov2_large",
    "dinov2_giant",
    "dinov3_small",
    "dinov3_small_plus",
    "dinov3_base",
    "dinov3_large",
    "dinov3_huge",
]

SAM_BACKBONES = list(SAMModelName)
MODELS = [ModelName.MATCHER, ModelName.SOFT_MATCHER]

DATASET_ROOT = "/home/rgangire/workspace/data/prompt/"
DEVICE = "cuda"
DATA_PATH = Path("~/data").expanduser()


def is_sam_weight_ready(sam: SAMModelName) -> bool:
    """Check if SAM model weights are fully downloaded."""
    info = MODEL_MAP[sam]
    path = DATA_PATH / info["local_filename"]
    if not path.exists():
        return False
    # Check file size is non-trivially large (at least 100MB for non-tiny)
    size_mb = path.stat().st_size / (1024 * 1024)
    if sam in {SAMModelName.SAM_HQ_TINY, SAMModelName.SAM2_TINY}:
        return size_mb > 30  # tiny models are ~40-150MB
    return size_mb > 100  # larger models should be >100MB


def run_sweep() -> None:
    """Run the full benchmark sweep."""
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    base_output = Path("~").expanduser() / "outputs" / f"sweep_{timestamp}"
    base_output.mkdir(parents=True, exist_ok=True)
    setup_logger(base_output, "INFO")

    device = torch.device(DEVICE)
    dataset_enum = DatasetName.LVIS

    # Pre-check which SAM backbones are available
    available_sams = [s for s in SAM_BACKBONES if is_sam_weight_ready(s)]
    skipped_sams = [s for s in SAM_BACKBONES if s not in available_sams]
    if skipped_sams:
        logger.warning("Skipping SAM backbones (weights missing): %s", [s.value for s in skipped_sams])
    logger.info("Available SAM backbones: %s", [s.value for s in available_sams])

    # Pre-load dataset once (reused for all runs)
    annotation_mode = MODEL_ANNOTATION_MODES.get(ModelName.MATCHER)
    dataset = load_dataset_by_name(
        dataset_enum.value,
        categories="default",
        n_shots=1,
        dataset_root=DATASET_ROOT,
        annotation_mode=annotation_mode,
    )

    all_results: list[pl.DataFrame] = []
    failed: list[str] = []
    combos = list(itertools.product(MODELS, available_sams, ENCODERS))
    total = len(combos)

    logger.info("Starting sweep: %d combinations", total)
    logger.info("Models: %s", [m.value for m in MODELS])
    logger.info("Encoders: %s", ENCODERS)

    for i, (model_enum, backbone_enum, encoder) in enumerate(combos):
        combo_name = f"{model_enum.value}_{backbone_enum.value}_{encoder}"
        logger.info("[%d/%d] Running: %s", i + 1, total, combo_name)

        model = None
        try:
            args = get_arguments([
                "--model",
                model_enum.value,
                "--sam",
                backbone_enum.value,
                "--dataset_name",
                dataset_enum.value,
                "--dataset_root",
                DATASET_ROOT,
                "--device",
                DEVICE,
                "--encoder_model",
                encoder,
                "--overwrite",
            ])

            model = load_model(sam=backbone_enum, model_name=model_enum, args=args)
            output_path = base_output / combo_name

            metrics_df = predict_on_dataset(
                args,
                model,
                dataset=dataset,
                output_path=output_path,
                dataset_name=dataset_enum.value,
                model_name=model_enum.value,
                backbone_name=backbone_enum.value,
                number_of_priors_tests=1,
                device=device,
            )

            metrics_df = metrics_df.with_columns(pl.lit(encoder).alias("encoder"))
            all_results.append(metrics_df)

            # Save incremental results
            _save_results(all_results, base_output, timestamp="incremental")
            logger.info("[%d/%d] Completed: %s", i + 1, total, combo_name)

        except Exception:
            logger.exception("[%d/%d] FAILED: %s", i + 1, total, combo_name)
            failed.append(combo_name)

        finally:
            if model is not None:
                del model
            gc.collect()
            torch.cuda.empty_cache()

    # Save final results
    if all_results:
        _save_results(all_results, base_output, timestamp="final")
    if failed:
        logger.warning("Failed combinations (%d): %s", len(failed), failed)
    logger.info("Sweep complete: %d succeeded, %d failed. Results at: %s", len(all_results), len(failed), base_output)


if __name__ == "__main__":
    run_sweep()
