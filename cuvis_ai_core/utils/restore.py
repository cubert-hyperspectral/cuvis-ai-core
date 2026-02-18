"""Utilities for restoring and running pipelines and trainruns."""

import time
from enum import Enum
from pathlib import Path
from typing import Literal

import torch
import yaml
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from cuvis_ai_core.data.datasets import SingleCu3sDataset
from cuvis_ai_core.data.datasets import SingleCu3sDataModule


from cuvis_ai_core.pipeline.factory import PipelineBuilder
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.training import GradientTrainer, StatisticalTrainer
from cuvis_ai_core.training.config import TrainRunConfig
from cuvis_ai_core.utils.config_helpers import resolve_config_with_hydra
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.execution import Context


class PipelineVisFormat(str, Enum):
    """Pipeline visualization export formats.

    Attributes
    ----------
    PNG : str
        Render pipeline as PNG image using Graphviz
    MD : str
        Export pipeline as Markdown with Mermaid diagram
    """

    PNG = "png"
    MD = "md"


def restore_pipeline(
    pipeline_path: str | Path,
    weights_path: str | Path | None = None,
    device: str = "auto",
    cu3s_file_path: str | Path | None = None,
    processing_mode: str = "Reflectance",
    annotation_json_path: str | Path | None = None,
    measurement_indices: list[int] | None = None,
    config_overrides: list[str] | None = None,
    plugins_path: str | Path | None = None,
    pipeline_vis_ext: PipelineVisFormat | None = None,
) -> CuvisPipeline:
    """Restore pipeline from configuration and weights for inference.

    Parameters
    ----------
    pipeline_path : str | Path
        Path to pipeline YAML configuration file
    weights_path : str | Path | None
        Optional path to weights file (.pt). If None, defaults to pipeline_path with .pt extension
    device : str
        Device to load weights to ('cpu', 'cuda', 'auto')
    cu3s_file_path : str | Path | None
        Optional path to .cu3s file for inference
    processing_mode : str
        Cuvis processing mode string ("Raw", "Reflectance")
    config_overrides : list[str] | None
        Optional list of config overrides in dot notation (e.g., ["nodes.10.hparams.output_dir=outputs/my_tb"])
    plugins_path : str | Path | None
        Optional path to plugins manifest YAML file for loading external plugin nodes
    pipeline_vis_ext : PipelineVisFormat | None
        Optional pipeline visualization export format.
        If provided, saves visualization next to the pipeline YAML file.
        PipelineVisFormat.PNG for rendered image, PipelineVisFormat.MD for Mermaid markdown.
        Default: None (no visualization)

    Returns
    -------
    CuvisPipeline
        Loaded pipeline ready for inference
    """
    pipeline_path = Path(pipeline_path)
    if weights_path is None:
        weights_path = pipeline_path.with_suffix(".pt")
    else:
        weights_path = Path(weights_path)

    logger.info(f"Loading pipeline from {pipeline_path}")

    # Load plugins if specified
    registry = None
    if plugins_path:
        from cuvis_ai_core.utils.node_registry import NodeRegistry

        registry = NodeRegistry()
        plugins_path = Path(plugins_path)
        if not plugins_path.exists():
            raise FileNotFoundError(f"Plugins manifest not found: {plugins_path}")
        registry.load_plugins(plugins_path)
        logger.info(f"Loaded plugins from: {plugins_path}")

    load_device = device if device != "auto" else None
    pipeline = CuvisPipeline.load_pipeline(
        str(pipeline_path),
        weights_path=str(weights_path) if weights_path.exists() else None,
        device=load_device,
        config_overrides=config_overrides,
        node_registry=registry,
    )

    # Generate pipeline visualization if requested
    if pipeline_vis_ext is not None:
        pipeline_path_obj = Path(pipeline_path)

        if pipeline_vis_ext == PipelineVisFormat.PNG:
            vis_output = pipeline_path_obj.with_suffix(".png")
            pipeline.visualize(format="render", output_path=vis_output)
            logger.info(f"Pipeline visualization (PNG) saved to: {vis_output}")
        elif pipeline_vis_ext == PipelineVisFormat.MD:
            vis_output = pipeline_path_obj.with_suffix(".md")
            pipeline.visualize(format="render_mermaid", output_path=vis_output)
            logger.info(f"Pipeline visualization (Markdown) saved to: {vis_output}")

    # If cu3s_file_path provided, setup data and run inference
    if cu3s_file_path:
        data = SingleCu3sDataset(
            cu3s_file_path=str(cu3s_file_path),
            processing_mode=processing_mode,
            annotation_json_path=str(annotation_json_path)
            if annotation_json_path
            else None,
            measurement_indices=measurement_indices,
        )
        dataloader = DataLoader(data, shuffle=False, batch_size=1, num_workers=0)

        for module in pipeline.torch_layers:
            module.eval()

        # Process all batches (outputs discarded — sink nodes write to disk)
        total_frames = len(data)
        global_step = 0
        forward_times = []
        with torch.no_grad():
            for batch in tqdm(
                dataloader, total=total_frames, desc="Inference", unit="frame"
            ):
                if load_device:
                    batch = {
                        k: v.to(load_device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }
                context = Context(
                    stage=ExecutionStage.INFERENCE,
                    batch_idx=global_step,
                    global_step=global_step,
                )
                t0 = time.perf_counter()
                pipeline.forward(batch=batch, context=context)
                t1 = time.perf_counter()
                forward_times.append(t1 - t0)
                global_step += 1

        avg_time = sum(forward_times) / len(forward_times)
        total_time = sum(forward_times)
        fps = global_step / total_time
        logger.info(
            f"Inference complete: {global_step} frames processed in {total_time:.2f}s "
            f"(avg forward call: {avg_time * 1000:.1f}ms, {fps:.1f} fps)"
        )

        # Finalize video outputs from any ToVideoNode in the pipeline
        for node in pipeline.nodes:
            if hasattr(node, "close") and hasattr(node, "output_video_path"):
                node.close()
                logger.success(f"Video saved: {node.output_video_path}")

    else:
        # Just display input/output specs
        input_specs = pipeline.get_input_specs()
        output_specs = pipeline.get_output_specs()

        print("\nInput Specs:")
        for name, spec in input_specs.items():
            print(f"  {name}: {spec}")

        print("\nOutput Specs:")
        for name, spec in output_specs.items():
            print(f"  {name}: {spec}")

    logger.info("Pipeline ready for inference")
    return pipeline


def _build_pipeline_from_config(
    trainrun_config: TrainRunConfig, device: str = "auto"
) -> CuvisPipeline:
    """Build pipeline from trainrun configuration.

    Parameters
    ----------
    trainrun_config : TrainRunConfig
        Trainrun configuration object
    device : str
        Device to load pipeline to ('cpu', 'cuda', 'auto')

    Returns
    -------
    CuvisPipeline
        Built pipeline ready for training
    """
    builder = PipelineBuilder()
    if trainrun_config.pipeline is None:
        raise ValueError("Pipeline configuration is missing in trainrun config.")
    pipeline_dict = trainrun_config.pipeline.to_dict()
    pipeline = builder.build_from_config(pipeline_dict)

    # Move pipeline to specified device if needed
    if device != "auto":
        pipeline = pipeline.to(device)

    return pipeline


def _create_datamodule_from_config(
    trainrun_config: TrainRunConfig,
) -> SingleCu3sDataModule:
    """Create datamodule from trainrun configuration.

    Parameters
    ----------
    trainrun_config : TrainRunConfig
        Trainrun configuration object

    Returns
    -------
    SingleCu3sDataModule
        Configured datamodule
    """
    datamodule = SingleCu3sDataModule(
        cu3s_file_path=trainrun_config.data.cu3s_file_path,
        annotation_json_path=trainrun_config.data.annotation_json_path,
        train_ids=trainrun_config.data.train_ids,
        val_ids=trainrun_config.data.val_ids,
        test_ids=trainrun_config.data.test_ids,
        batch_size=trainrun_config.data.batch_size,
        processing_mode=trainrun_config.data.processing_mode,
    )
    datamodule.setup(stage="fit")
    return datamodule


def restore_trainrun(
    trainrun_path: str | Path,
    mode: Literal["train", "validate", "test", "info"] = "info",
    checkpoint_path: str | Path | None = None,
    device: str = "auto",
    overrides: list[str] | None = None,
) -> None:
    """Restore and reproduce training run from configuration file.

    Intelligently detects whether to use GradientTrainer (for gradient-based training)
    or StatisticalTrainer (for statistical-only training) based on trainrun configuration.
    Always runs statistical initialization before gradient training.

    Execution order:
    1. Statistical initialization (if needed) - always runs first
    2. Gradient training (if enabled)
    3. Validation (to populate monitoring/TensorBoard)
    4. Test evaluation (to populate monitoring/TensorBoard)

    Parameters
    ----------
    trainrun_path : str | Path
        Path to trainrun YAML file
    mode : str
        Execution mode:
        - 'info': Display experiment information only
        - 'train': Re-run training from scratch or continue training
        - 'validate': Run validation on trained model
        - 'test': Run test evaluation on trained model
    checkpoint_path : str | Path | None
        Optional checkpoint path to resume training from (gradient training only)
    device : str
        Device to run on ('cpu', 'cuda', 'auto')
    overrides : list[str] | None
        Hydra-style config overrides (e.g., ["output_dir=outputs/custom", "data.batch_size=16"])
    """
    trainrun_path = Path(trainrun_path)
    if not trainrun_path.exists():
        raise FileNotFoundError(f"TrainRun file not found: {trainrun_path}")

    logger.info(f"Loading trainrun from: {trainrun_path}")

    # Check if config has Hydra defaults - if so, resolve them first
    with trainrun_path.open("r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    if isinstance(raw_config, dict) and "defaults" in raw_config:
        # Config has Hydra defaults - resolve them
        configs_dir = trainrun_path.parent.parent
        relative_path = trainrun_path.relative_to(configs_dir)
        config_path = str(relative_path.with_suffix("")).replace("\\", "/")

        config_dict = resolve_config_with_hydra(
            config_type="trainrun",
            config_path=config_path,
            search_paths=[str(configs_dir)],
            overrides=overrides,
        )
        trainrun_config = TrainRunConfig.model_validate(config_dict)
    else:
        # Config is already resolved - load directly
        trainrun_config: TrainRunConfig = TrainRunConfig.load_from_file(trainrun_path)

    # Build pipeline
    pipeline = _build_pipeline_from_config(trainrun_config, device=device)

    if mode == "info":
        logger.info("Info mode - displaying pipeline specifications")
        input_specs = pipeline.get_input_specs()
        output_specs = pipeline.get_output_specs()

        print("\nInput Specs:")
        for name, spec in input_specs.items():
            print(f"  {name}: {spec}")

        print("\nOutput Specs:")
        for name, spec in output_specs.items():
            print(f"  {name}: {spec}")

        logger.info("Info mode complete")
        return

    # Create datamodule
    datamodule = _create_datamodule_from_config(trainrun_config)
    output_dir = Path(trainrun_config.output_dir)

    # Detect training type: check if we have training config with trainer
    has_gradient_training = (
        trainrun_config.training is not None
        and trainrun_config.training.trainer is not None
    )

    # Check if statistical initialization is needed
    # Skip if weights are already loaded (indicates pre-trained pipeline)
    requires_static_fit = any(node.requires_initial_fit for node in pipeline.nodes())

    # Create trainers (both will be used based on mode)
    stat_trainer = StatisticalTrainer(pipeline=pipeline, datamodule=datamodule)
    grad_trainer = None

    if has_gradient_training:
        logger.info("Detected gradient training configuration")

        # Find loss and metric nodes by name
        loss_nodes = []
        metric_nodes = []
        for node in pipeline.nodes():
            if node.name in trainrun_config.loss_nodes:
                loss_nodes.append(node)
            if node.name in trainrun_config.metric_nodes:
                metric_nodes.append(node)

        # Update checkpoint directory to output_dir
        training_config = trainrun_config.training
        if training_config is None:
            raise ValueError("Training configuration is missing in trainrun config.")

        if (
            training_config.trainer.callbacks
            and training_config.trainer.callbacks.checkpoint
        ):
            training_config.trainer.callbacks.checkpoint.dirpath = str(
                output_dir / "checkpoints"
            )

        grad_trainer = GradientTrainer(
            pipeline=pipeline,
            datamodule=datamodule,
            loss_nodes=loss_nodes,
            metric_nodes=metric_nodes,
            trainer_config=training_config.trainer,
            optimizer_config=training_config.optimizer,
        )
    else:
        logger.info("Detected statistical-only training configuration")

    # === EXECUTION BASED ON MODE ===

    if mode == "train":
        logger.info("Training mode")

        # Step 1: Statistical initialization (always first, if needed)
        if requires_static_fit:
            logger.info("  Step 1: Statistical initialization...")
            stat_trainer.fit()

        # Step 2: Gradient training (if enabled)
        if grad_trainer is not None:
            logger.info("  Step 2: Gradient-based training...")

            # Unfreeze nodes for gradient training
            if trainrun_config.unfreeze_nodes:
                pipeline.unfreeze_nodes_by_name(trainrun_config.unfreeze_nodes)

            if checkpoint_path:
                logger.warning(
                    f"Checkpoint path provided: {checkpoint_path}. "
                    "Automatic checkpoint resumption is not yet implemented. "
                    "Please configure checkpoint resumption in trainer_config."
                )

            grad_trainer.fit()
        else:
            logger.info("  Step 2: Skipped (statistical-only)")

        # Step 3: Save trained pipeline
        logger.info("  Step 3: Saving trained pipeline...")
        restored_pipeline_path = (
            output_dir / "trained_models" / f"{pipeline.name}_restored.yaml"
        )
        restored_pipeline_path.parent.mkdir(parents=True, exist_ok=True)
        pipeline.save_to_file(str(restored_pipeline_path))
        logger.info(f"     Saved to {restored_pipeline_path}")

        # Step 4: Run validation (populate TensorBoard monitoring)
        if trainrun_config.data.val_ids:
            logger.info("  Step 4: Validation...")
            if grad_trainer is not None:
                grad_trainer.validate()
            else:
                stat_trainer.validate()
        else:
            logger.warning(
                "  Step 4: Validation skipped - no validation data provided (val_ids is empty)"
            )

        # Step 5: Run test evaluation (populate TensorBoard monitoring)
        if trainrun_config.data.test_ids:
            logger.info("  Step 5: Test evaluation...")
            if grad_trainer is not None:
                grad_trainer.test()
            else:
                stat_trainer.test()
        else:
            logger.warning(
                "  Step 5: Test evaluation skipped - no test data provided (test_ids is empty)"
            )

    elif mode == "validate":
        logger.info("Validation mode")

        # Run statistical initialization if needed
        if requires_static_fit:
            logger.info("  Running statistical initialization...")
            stat_trainer.fit()

        # Run validation
        if grad_trainer is not None:
            grad_trainer.validate()
        else:
            stat_trainer.validate()

    elif mode == "test":
        logger.info("Test mode")

        # Run statistical initialization if needed
        if requires_static_fit:
            logger.info("  Running statistical initialization...")
            stat_trainer.fit()

        # Run test
        if grad_trainer is not None:
            grad_trainer.test()
        else:
            stat_trainer.test()

    logger.info("Complete")


def restore_pipeline_cli() -> None:
    """CLI entry point for pipeline restoration."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Restore trained pipeline for inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Display pipeline info
  uv run restore-pipeline --pipeline-path configs/pipeline/adaclip_baseline.yaml

  # Run inference on CU3S file
  uv run restore-pipeline \\
    --pipeline-path configs/pipeline/adaclip_baseline.yaml \\
    --cu3s-file-path data/Lentils/Lentils_000.cu3s

  # Use custom device
  uv run restore-pipeline \\
    --pipeline-path configs/pipeline/adaclip_baseline.yaml \\
    --device cuda
        """,
    )

    parser.add_argument(
        "--pipeline-path",
        type=str,
        required=True,
        help="Path to pipeline YAML file",
    )
    parser.add_argument(
        "--weights-path",
        type=str,
        default=None,
        help="Path to weights (.pt) file (defaults to pipeline_path with .pt extension)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run on (default: auto)",
    )
    parser.add_argument(
        "--cu3s-file-path",
        type=str,
        default=None,
        help="Path to .cu3s file for inference",
    )
    parser.add_argument(
        "--processing-mode",
        type=str,
        default="Reflectance",
        choices=["Raw", "Reflectance", "SpectralRadiance"],
        help="Processing mode (default: Reflectance)",
    )
    parser.add_argument(
        "--annotation-json-path",
        type=str,
        default=None,
        help="Path to COCO annotation JSON file for mask loading",
    )
    parser.add_argument(
        "--measurement-indices",
        type=str,
        default=None,
        help="Comma-separated measurement indices to process (e.g., '0,304,530,568'). Default: all frames.",
    )
    parser.add_argument(
        "--override",
        action="append",
        help="Override config values in dot notation. Can be specified multiple times.",
    )
    parser.add_argument(
        "--plugins-path",
        type=str,
        default=None,
        help="Path to plugins manifest YAML file for loading external plugin nodes",
    )
    parser.add_argument(
        "--pipeline-vis-ext",
        type=str,
        choices=["png", "md"],
        default=None,
        help="Export pipeline visualization: 'png' (rendered image) or 'md' (Mermaid markdown)",
    )

    args = parser.parse_args()

    # Convert string argument to enum if provided
    vis_ext = None
    if args.pipeline_vis_ext is not None:
        vis_ext = PipelineVisFormat(args.pipeline_vis_ext)

    # Parse measurement indices from comma-separated string
    meas_indices = None
    if args.measurement_indices is not None:
        meas_indices = [int(x.strip()) for x in args.measurement_indices.split(",")]

    restore_pipeline(
        pipeline_path=args.pipeline_path,
        weights_path=args.weights_path,
        device=args.device,
        cu3s_file_path=args.cu3s_file_path,
        processing_mode=args.processing_mode,
        annotation_json_path=args.annotation_json_path,
        measurement_indices=meas_indices,
        config_overrides=args.override,
        plugins_path=args.plugins_path,
        pipeline_vis_ext=vis_ext,
    )


def restore_trainrun_cli() -> None:
    """CLI entry point for trainrun restoration."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Restore and reproduce training runs from saved configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Display trainrun info
  uv run restore-trainrun --trainrun-path outputs/gradient_based/trained_models/gradient_based_trainrun.yaml

  # Re-run training
  uv run restore-trainrun \\
    --trainrun-path outputs/gradient_based/trained_models/gradient_based_trainrun.yaml \\
    --mode train

  # Run validation
  uv run restore-trainrun \\
    --trainrun-path outputs/gradient_based/trained_models/gradient_based_trainrun.yaml \\
    --mode validate

  # Override data and training configs
  uv run restore-trainrun \\
    --trainrun-path outputs/.../trainrun.yaml \\
    --mode train \\
    --override data.batch_size=16 \\
    --override training.optimizer.lr=0.001
        """,
    )

    parser.add_argument(
        "--trainrun-path",
        type=str,
        required=True,
        help="Path to trainrun YAML file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["info", "train", "validate", "test"],
        default="info",
        help="Execution mode (default: info)",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Checkpoint path to resume training from",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run on (default: auto)",
    )
    parser.add_argument(
        "--override",
        action="append",
        help="Override config values in dot notation (e.g., data.batch_size=16). Can be specified multiple times.",
    )

    args = parser.parse_args()

    restore_trainrun(
        trainrun_path=args.trainrun_path,
        mode=args.mode,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        overrides=args.override,
    )
