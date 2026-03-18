# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Geti Instant Learn Benchmark Script."""

import itertools
from argparse import Namespace
from logging import getLogger
from pathlib import Path

import polars as pl
import torch
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from torch.utils.data import DataLoader
from torchmetrics import Metric

from instantlearn.data import Dataset, LVISDataset, PerSegDataset
from instantlearn.data.base import Batch
from instantlearn.data.lvis import LVISAnnotationMode
from instantlearn.models import SAM3, InferenceResult, Model
from instantlearn.utils import setup_logger
from instantlearn.utils.args import get_arguments, parse_experiment_args
from instantlearn.utils.benchmark import (
    MODEL_ANNOTATION_MODES,
    _get_output_path_for_experiment,
    _save_results,
    convert_masks_to_one_hot_tensor,
    create_metrics,
    load_model,
    parse_metrics_arg,
    prepare_output_directory,
)
from instantlearn.utils.constants import get_category_presets
from instantlearn.visualizer import Visualizer

logger = getLogger("Geti Instant Learn")


def predict_on_category(
    dataset: Dataset,
    model: Model,
    category_name: str,
    priors_batch_index: int,
    visualizer: Visualizer,
    metrics_calculators: dict[int, dict[str, Metric]],
    predict_timings: list[dict[str, float]],
    progress: Progress,
    batch_size: int,
    device: torch.device,
    visualize: bool = True,
) -> None:
    """Perform prediction on all samples of a category.

    Args:
        dataset: The dataset containing target samples
        model: The model to run
        category_name: The current category
        priors_batch_index: The current prior batch
        visualizer: The visualizer for exporting
        metrics_calculators: Per-prior dict of metric-name → Metric instances
        predict_timings: List to append per-batch timing dicts to
        progress: The progress bar
        batch_size: Batch size for DataLoader
        device: The device to use.
        visualize: Whether to visualize the results

    Raises:
        ValueError: If no target samples are found for the category.
    """
    # Get target samples for this category
    target_dataset = dataset.get_target_dataset(category=category_name)

    if len(target_dataset) == 0:
        msg = f"No target samples found for category: {category_name}"
        logger.warning(msg)
        raise ValueError(msg)

    # Create DataLoader
    dataloader = DataLoader(
        target_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=target_dataset.collate_fn,
    )

    # Task for batches for the current category and prior (transient)
    batches_task = progress.add_task(
        f"[magenta]Infer step: {category_name}",
        total=len(dataloader),
        transient=True,
    )

    # Create category ID to index mapping
    num_classes = len(dataset.categories)
    category_id_to_index = {dataset.get_category_id(cat_name): idx for idx, cat_name in enumerate(dataset.categories)}

    for batch in dataloader:
        # Run prediction
        if isinstance(model, SAM3):
            for sample in batch:
                sample.categories = ["visual"]  # Obscure category name to prevent cheating for SAM3

        result: InferenceResult = model.predict(batch)

        # Collect timing if available
        if result.timing is not None:
            timing_entry = result.timing.to_dict()
            timing_entry["category"] = category_name
            timing_entry["prior_index"] = priors_batch_index
            timing_entry["batch_size"] = len(batch)
            predict_timings.append(timing_entry)

        # Convert masks to one-hot boolean tensors for torchmetrics
        # Returns lists of tensors, each with shape (C, H, W)
        batch_pred_tensors, batch_gt_tensors = convert_masks_to_one_hot_tensor(
            predictions=result,
            ground_truths=batch,
            num_classes=num_classes,
            category_id_to_index=category_id_to_index,
            device=device,
        )

        # Update all metrics for each image in the batch
        # Metrics expect (N, C, H, W) but images have different sizes
        # So we update with (1, C, H, W) for each image
        for pred_tensor, gt_tensor in zip(batch_pred_tensors, batch_gt_tensors, strict=True):
            for metric in metrics_calculators[priors_batch_index].values():
                metric.update(pred_tensor, gt_tensor)

        if visualize:
            # Generate export paths
            file_names = [
                str(
                    Path("predictions") / f"priors_batch_{priors_batch_index}" / category_name / Path(img_path).name,
                )
                for img_path in batch.image_paths
            ]

            # Visualize predictions and ground truth
            visualizer.visualize(
                images=batch.images,
                predictions=result.predictions,
                file_names=file_names,
            )

        progress.update(batches_task, advance=1)

    progress.remove_task(batches_task)


def learn_from_category(dataset: Dataset, model: Model, category_name: str) -> None:
    """Learn from reference samples of a category.

    Args:
        dataset: The dataset containing reference samples
        model: The model to train
        category_name: The category to learn from
    """
    reference_dataset = dataset.get_reference_dataset(category=category_name)
    reference_batch = Batch.collate(list(reference_dataset))

    # Filter to only the requested category — images may contain annotations
    # for multiple categories in INSTANCE mode (e.g. LVIS).
    samples = [
        filtered
        for sample in reference_batch.samples
        if (filtered := sample.filter_by_category(category_name)) is not None
    ]
    reference_batch = Batch.collate(samples)

    # Learn
    model.fit(reference_batch)


def predict_on_dataset(
    args: Namespace,
    model: Model,
    dataset: Dataset,
    output_path: Path,
    dataset_name: str,
    model_name: str,
    backbone_name: str,
    number_of_priors_tests: int,
    device: torch.device,
    requested_metrics: list[str] | None = None,
) -> pl.DataFrame:
    """Run predictions on the dataset and evaluate them.

    Args:
        args: Args from the argparser.
        model: The model to use.
        dataset: The dataset (contains both reference and target samples)
        output_path: Output path
        dataset_name: The dataset name
        model_name: The algorithm name
        backbone_name: The model name
        number_of_priors_tests: The number of priors to try
        device: The device to use.
        requested_metrics: List of metric names to compute. Defaults to ["iou"].

    Returns:
        The results DataFrame
    """
    if requested_metrics is None:
        requested_metrics = ["iou"]

    output_path = prepare_output_directory(output_path, args.overwrite)
    msg = f"Output path: {output_path}"
    logger.info(msg)

    visualizer = Visualizer(
        output_folder=str(output_path),
        class_map={dataset.get_category_id(category): category for category in dataset.categories},
    )
    # Keep metrics per prior: dict[prior_index, dict[metric_name, Metric]]
    metrics_calculators: dict[int, dict[str, Metric]] = {}
    # Collect per-batch predict timings and per-category fit timings
    predict_timings: list[dict[str, float]] = []
    fit_timings: list[dict[str, float]] = []

    # Setup Rich Progress
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
    )

    # Get all unique categories
    categories = dataset.categories

    with progress:
        # Main task for categories (persistent)
        categories_task = progress.add_task(f"[cyan]Processing {dataset_name}", total=len(categories))

        # Iterate over all categories in the dataset
        for category_name in categories:
            # Task for priors for the current category (transient)
            priors_task = progress.add_task(f"[green]Learn step: {category_name}", total=1, transient=True)

            # For now, only use 1 prior batch (can be extended for multiple prior batches)
            for priors_batch_index in range(number_of_priors_tests):
                # Shuffle reference/target assignment for multi-prior runs
                if number_of_priors_tests > 1:
                    seed = getattr(args, "seed", 42)
                    dataset.shuffle_references(seed=seed, prior_index=priors_batch_index)

                # Initialize metrics for this prior if needed
                if priors_batch_index not in metrics_calculators:
                    metrics_calculators[priors_batch_index] = create_metrics(
                        num_classes=len(dataset.categories),
                        requested_metrics=requested_metrics,
                        device=device,
                    )

                # Learn from reference samples
                learn_from_category(
                    dataset=dataset,
                    model=model,
                    category_name=category_name,
                )

                # Collect fit timing
                if hasattr(model, "last_fit_timing") and model.last_fit_timing is not None:
                    fit_entry = model.last_fit_timing.to_dict()
                    fit_entry["category"] = category_name
                    fit_entry["prior_index"] = priors_batch_index
                    fit_timings.append(fit_entry)

                progress.update(priors_task, advance=1)

                # Predict on target samples
                predict_on_category(
                    dataset=dataset,
                    model=model,
                    category_name=category_name,
                    priors_batch_index=priors_batch_index,
                    visualizer=visualizer,
                    metrics_calculators=metrics_calculators,
                    predict_timings=predict_timings,
                    progress=progress,
                    batch_size=args.batch_size,
                    device=device,
                )

            progress.remove_task(priors_task)
            progress.update(categories_task, advance=1)

    # Construct the output metrics file from the calculated metrics
    return _aggregate_results(
        metrics_calculators=metrics_calculators,
        categories=categories,
        predict_timings=predict_timings,
        fit_timings=fit_timings,
        dataset_name=dataset_name,
        model_name=model_name,
        backbone_name=backbone_name,
    )


def _compute_per_class_metrics(metric_set: dict[str, Metric], num_categories: int) -> dict[str, list[float]]:
    """Compute per-class metric values from a set of torchmetric objects.

    Args:
        metric_set: Mapping from metric name to a computed Metric object.
        num_categories: Number of categories (used to index per-class tensors).

    Returns:
        Mapping from metric name to a list of per-class float values.
    """
    per_class_results: dict[str, list[float]] = {}
    for metric_name, metric_obj in metric_set.items():
        computed = metric_obj.compute()
        if metric_name == "precision_recall":
            # PixelPrecisionRecall returns dict with "precision" and "recall" tensors
            for sub_name, tensor in computed.items():
                per_class_results[sub_name] = [tensor[idx].item() for idx in range(num_categories)]
        elif isinstance(computed, torch.Tensor):
            # MeanIoU, DiceScore return a tensor of shape (C,)
            per_class_results[metric_name] = [computed[idx].item() for idx in range(num_categories)]
    return per_class_results


def _aggregate_results(
    metrics_calculators: dict[int, dict[str, Metric]],
    categories: list[str],
    predict_timings: list[dict[str, float]],
    fit_timings: list[dict[str, float]],
    dataset_name: str,
    model_name: str,
    backbone_name: str,
) -> pl.DataFrame:
    """Aggregate per-prior metrics and timing into a single results DataFrame."""
    all_metrics = None
    for prompt_index, metric_set in metrics_calculators.items():
        metrics: dict[str, list] = {"category": []}

        per_class_results = _compute_per_class_metrics(metric_set, len(categories))

        # Build per-category rows
        for idx, cat_name in enumerate(categories):
            metrics["category"].append(cat_name)
            for metric_name, values in per_class_results.items():
                if metric_name not in metrics:
                    metrics[metric_name] = []
                metrics[metric_name].append(values[idx])

        ln = len(metrics["category"])
        metrics["prior_index"] = [prompt_index] * ln

        # Aggregate timing: average predict time per sample for this prior
        prior_predict_timings = [t for t in predict_timings if t.get("prior_index") == prompt_index]
        if prior_predict_timings:
            total_predict_ms = sum(t.get("total_ms", 0) for t in prior_predict_timings)
            total_samples = sum(t.get("batch_size", 1) for t in prior_predict_timings)
            avg_predict_ms = total_predict_ms / max(total_samples, 1)
        else:
            avg_predict_ms = 0.0

        prior_fit_timings = [t for t in fit_timings if t.get("prior_index") == prompt_index]
        avg_fit_ms = sum(t.get("total_ms", 0) for t in prior_fit_timings) / max(len(prior_fit_timings), 1)

        metrics["fit_time_ms"] = [avg_fit_ms] * ln
        metrics["predict_time_ms_per_sample"] = [avg_predict_ms] * ln
        metrics["dataset_name"] = [dataset_name] * ln
        metrics["model_name"] = [model_name] * ln
        metrics["backbone_name"] = [backbone_name] * ln

        if all_metrics is None:
            all_metrics = metrics
        else:
            for key in all_metrics:
                all_metrics[key].extend(metrics[key])

    return pl.DataFrame(all_metrics)


def load_dataset_by_name(
    dataset_name: str,
    categories: list[str] | str | None = None,
    n_shots: int = 1,
    dataset_root: str | Path | None = None,
    annotation_mode: LVISAnnotationMode = LVISAnnotationMode.SEMANTIC,
) -> Dataset:
    """Load a dataset by name.

    Args:
        dataset_name: Name of the dataset (e.g., "PerSeg", "LVIS")
        categories: Categories to filter. Can be:
            - None: uses "default" preset if available, else all categories
            - "all": uses all available categories
            - "default": uses default preset (quick testing)
            - "benchmark": uses benchmark preset (comprehensive testing)
            - list[str]: explicit list of category names
        n_shots: Number of reference shots per category
        dataset_root: Root directory where datasets are stored. If None, uses defaults.
        annotation_mode: LVIS annotation mode. SEMANTIC merges instances per
            category (for Matcher/SoftMatcher), INSTANCE keeps per-instance
            masks and boxes (for SAM3). Ignored for non-LVIS datasets.

    Raises:
        ValueError: If the dataset name is unknown or preset is invalid.

    Returns:
        InstantLearnDataset instance

    Example:
        Load dataset with different category configurations:

        >>> # Use default preset (quick testing, 4 categories for LVIS)
        >>> dataset = load_dataset_by_name("lvis")
        >>> # Or explicitly:
        >>> dataset = load_dataset_by_name("lvis", categories="default")
        >>>
        >>> # Use benchmark preset (comprehensive testing, 92 categories on LVIS-92 fold 0)
        >>> dataset = load_dataset_by_name("lvis", categories="benchmark")
        >>>
        >>> # Use all available categories in the dataset
        >>> dataset = load_dataset_by_name("lvis", categories="all")
        >>>
        >>> # Use explicit category list
        >>> dataset = load_dataset_by_name("lvis", categories=["cat", "dog", "bird"])
        >>>
        >>> # Extend a preset by importing constants
        >>> from instantlearn.utils.constants import LVIS_DEFAULT_CATEGORIES
        >>> custom_categories = list(LVIS_DEFAULT_CATEGORIES) + ["tiger", "lion"]
        >>> dataset = load_dataset_by_name("lvis", categories=custom_categories)
        >>>
        >>> # Use benchmark categories from a specific fold
        >>> from instantlearn.utils.constants import LVIS_BENCHMARK_CATEGORIES
        >>> dataset = load_dataset_by_name("lvis", categories=LVIS_BENCHMARK_CATEGORIES["fold_1"])
        >>>
        >>> # Configure n_shots and custom dataset root
        >>> dataset = load_dataset_by_name(
        ...     "lvis",
        ...     categories="benchmark",
        ...     n_shots=3,
        ...     dataset_root="~/my_datasets"
        ... )
    """
    # Resolve category presets
    if categories is None:
        categories = "default"

    if isinstance(categories, str):
        preset_key = categories.lower()
        if preset_key == "all":
            resolved_categories = None  # Dataset will use all available categories
        else:
            category_presets = get_category_presets()
            if dataset_name.lower() in category_presets:
                if preset_key in category_presets[dataset_name.lower()]:
                    resolved_categories = category_presets[dataset_name.lower()][preset_key]
                else:
                    available_presets = list(category_presets[dataset_name.lower()].keys())
                    msg = f"Unknown preset '{categories}' for dataset '{dataset_name}'. Available: {available_presets}"
                    raise ValueError(msg)
            else:
                msg = f"No presets defined for dataset '{dataset_name}'"
                raise ValueError(msg)
    else:
        resolved_categories = categories

    if dataset_name.lower() == "perseg":
        root = (
            Path(dataset_root).expanduser() / "PerSeg"
            if dataset_root is not None
            else Path("~/datasets/PerSeg").expanduser()
        )
        return PerSegDataset(
            root=root,
            categories=resolved_categories,
            n_shots=n_shots,
        )
    if dataset_name.lower() == "lvis":
        root = (
            Path(dataset_root).expanduser() / "lvis"
            if dataset_root is not None
            else Path("~/datasets/lvis").expanduser()
        )
        return LVISDataset(
            root=root,
            categories=resolved_categories,
            n_shots=n_shots,
            annotation_mode=annotation_mode,
        )
    msg = f"Unknown dataset: {dataset_name}"
    raise ValueError(msg)


def perform_benchmark_experiment(args: Namespace | None = None) -> None:
    """Main function to run the experiments.

    This function initializes the arguments, determines which models, datasets, and models to process,
    and then iterates over all combinations to run the predictions and evaluate them.

    Args:
        args: The arguments to use.
    """
    # Initialization
    if args is None:
        args = get_arguments()

    base_output_path = Path("~").expanduser() / "outputs"
    # The final results path will include the experiment name if provided.
    final_results_path = base_output_path / args.experiment_name if args.experiment_name else base_output_path

    setup_logger(final_results_path, args.log_level)
    final_results_path.mkdir(parents=True, exist_ok=True)

    # Parse requested metrics from CLI
    metrics_arg = getattr(args, "metrics", "all")
    requested_metrics = parse_metrics_arg(metrics_arg)

    # Get experiment lists and generate a plan
    datasets_to_run, models_to_run, backbones_to_run = parse_experiment_args(args)
    experiments = list(itertools.product(datasets_to_run, models_to_run, backbones_to_run))

    # Execute experiments
    all_results = []
    for dataset_enum, model_enum, backbone_enum in experiments:
        msg = (
            f"Starting experiment with Dataset={dataset_enum.value}, "
            f"Model={model_enum.value}, Backbone={backbone_enum.value}",
        )
        logger.info(msg)

        # Parse categories from CLI argument
        # Support both presets (e.g., "default", "benchmark", "all") and explicit lists (e.g., "cat,dog,bird")
        if args.class_name:
            # Check if it's a preset or a comma-separated list
            if "," not in args.class_name and args.class_name.lower() in {"default", "benchmark", "all"}:
                categories_arg = args.class_name  # Pass preset string directly
            else:
                categories_arg = [c.strip() for c in args.class_name.split(",")]  # Split comma-separated list
        else:
            categories_arg = None  # Will default to "default" preset

        # Resolve annotation mode from the model
        annotation_mode = MODEL_ANNOTATION_MODES.get(model_enum, LVISAnnotationMode.SEMANTIC)

        # Load dataset using new API
        dataset = load_dataset_by_name(
            dataset_enum.value,
            categories=categories_arg,
            n_shots=args.n_shot,
            dataset_root=args.dataset_root,
            annotation_mode=annotation_mode,
        )

        model = load_model(sam=backbone_enum, model_name=model_enum, args=args)

        # Individual experiment artifacts are saved in a path derived from the base path.
        output_path = _get_output_path_for_experiment(
            base_output_path,
            args.experiment_name,
            dataset_enum,
            model_enum,
            backbone_enum,
        )

        all_metrics_df = predict_on_dataset(
            args,
            model,
            dataset=dataset,
            output_path=output_path,
            dataset_name=dataset_enum.value,
            model_name=model_enum.value,
            backbone_name=backbone_enum.value,
            number_of_priors_tests=args.num_priors,
            device=args.device,
            requested_metrics=requested_metrics,
        )
        all_results.append(all_metrics_df)

    # Save aggregated results to the final results path
    _save_results(all_results, final_results_path)
