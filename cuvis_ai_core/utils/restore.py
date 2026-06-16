"""Utilities for restoring and running pipelines and trainruns."""

from enum import Enum
from pathlib import Path
from typing import Literal

import torch
import yaml
from loguru import logger
from tqdm import tqdm

from cuvis_ai_core.data.datamodule import create_data_module
from cuvis_ai_core.pipeline.factory import PipelineBuilder
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.training import GradientTrainer, StatisticalTrainer
from cuvis_ai_core.training.config import TrainRunConfig
from cuvis_ai_core.utils.config_helpers import resolve_config_with_hydra
from cuvis_ai_core.utils.node_registry import NodeRegistry
from cuvis_ai_core.utils.plugin_resolver import resolve_pipeline_plugins
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.execution import Context
from cuvis_ai_schemas.pipeline import PipelineConfig
from cuvis_ai_schemas.training import DataConfig


def _discover_plugins_dirs(
    pipeline_path: Path,
    explicit_dirs: list[str | Path] | None,
) -> list[Path]:
    """Build the list of candidate plugins directories for the resolver.

    Resolution order (later entries win on plugin-name collisions):

    1. Any ``configs/plugins/`` discovered by walking upward from
       ``pipeline_path``'s parent.
    2. Any ``--plugins-dir`` values from the CLI.
    """
    candidates: list[Path] = []
    for ancestor in pipeline_path.resolve().parents:
        candidate = ancestor / "configs" / "plugins"
        if candidate.is_dir():
            candidates.append(candidate)
            break

    if explicit_dirs:
        candidates.extend(Path(p) for p in explicit_dirs)

    return candidates


def _resolve_pipeline_reference(reference: str, base_dir: Path | None) -> Path:
    """Resolve a trainrun's ``pipeline:`` reference to a pipeline YAML on disk.

    Resolution order (first hit wins), trying both the literal reference and a
    ``.yaml``-suffixed form:

    1. an absolute path,
    2. relative to ``base_dir`` (the trainrun file's directory),
    3. relative to ``base_dir``'s parent (the ``configs/`` root, since a
       trainrun typically lives in ``configs/trainrun/``),
    4. relative to the current working directory.
    """
    roots: list[Path] = []
    if base_dir is not None:
        roots.extend([base_dir, base_dir.parent])
    roots.append(Path.cwd())

    # Append (not replace via with_suffix) ``.yaml`` so a dotted stem like
    # ``foo.v1`` yields ``foo.v1.yaml``, not ``foo.yaml``.
    refs = [reference]
    if not reference.endswith((".yaml", ".yml")):
        refs.append(reference + ".yaml")

    candidates: list[Path] = []
    for ref_str in refs:
        ref = Path(ref_str)
        if ref.is_absolute():
            candidates.append(ref)
        else:
            candidates.extend(root / ref for root in roots)

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Pipeline reference {reference!r} could not be resolved (base_dir={base_dir}). "
        f"Tried: {tried}."
    )


def _load_data_module_plugin(
    registry: NodeRegistry, data_module_name: str, candidate_dirs: list[Path]
) -> None:
    """Find and load the plugin providing ``data_module_name`` into ``registry``.

    The dataloader plugin ships no node classes, so the pipeline-node resolver
    never pulls it in; the data module is selected explicitly (``--data-module``
    / ``DataConfig.data_module``), so we look it up by ``data_module_name`` in the
    plugins-dir catalog and materialise its plugin.
    """
    from cuvis_ai_core.utils.plugin_resolver import _build_catalog

    catalog = _build_catalog([Path(d) for d in candidate_dirs])
    for plugin_name, cfg in catalog.items():
        for entry in cfg.provides:
            if (
                getattr(entry, "kind", "node") == "data_module"
                and entry.data_module_name == data_module_name
            ):
                registry.register_plugin(plugin_name, cfg.model_dump())
                return
    raise ValueError(
        f"No plugin in {[str(d) for d in candidate_dirs]} provides data module "
        f"{data_module_name!r}. Pass the plugin's manifest dir via --plugins-dir."
    )


def _build_data_module(
    registry: NodeRegistry, data_config: DataConfig, candidate_dirs: list[Path]
):
    """Dispatch a DataModule from ``data_config``, loading its plugin if needed."""
    if data_config.data_module not in registry.data_modules:
        _load_data_module_plugin(registry, data_config.data_module, candidate_dirs)
    return create_data_module(registry, data_config)


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
    data_module: str | None = None,
    data_args: dict[str, str] | None = None,
    config_overrides: list[str] | None = None,
    plugins_dirs: list[str | Path] | None = None,
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
    data_module : str | None
        Optional DataModule name (its DATA_MODULE_NAME, e.g. "cu3s",
        "tiff_paired") to run inference over. The providing plugin is loaded
        from the plugins dirs by ``data_module_name``. When omitted, only the
        pipeline input/output specs are displayed.
    data_args : dict[str, str] | None
        Module-specific arguments for the selected DataModule (the ``--data-arg
        key=value`` pairs), passed through as ``DataConfig.params``.
    config_overrides : list[str] | None
        Optional list of config overrides in dot notation (e.g., ["nodes.10.hparams.output_dir=outputs/my_tb"])
    plugins_dirs : list[str | Path] | None
        Optional list of plugins directories to scan for per-plugin
        manifests. Used by the pipeline-driven plugin resolver
        When omitted, the resolver still discovers
        ``configs/plugins/`` siblings by walking up from the pipeline
        YAML.
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

    registry: NodeRegistry | None = None
    candidate_dirs = _discover_plugins_dirs(pipeline_path, plugins_dirs)

    if pipeline_path.is_file():
        # Pipeline-driven plugin resolution: materialise only what the pipeline declares.
        # Skip when the file doesn't exist on disk — mocked/programmatic
        # callers handle pipeline loading downstream without our pre-read.
        pipeline_cfg = PipelineConfig.load_from_file(pipeline_path)
        # Only engage the resolver when the user has declared plugins OR a
        # catalog dir is discoverable. A pipeline that uses only built-in
        # core nodes needs no resolver pass and loads with no plugin registry.
        if pipeline_cfg.plugins or candidate_dirs:
            resolved_plugins = resolve_pipeline_plugins(pipeline_cfg, candidate_dirs)
            if resolved_plugins:
                registry = NodeRegistry()
                for name, cfg in resolved_plugins.items():
                    registry.register_plugin(name, cfg.model_dump())
                logger.info(
                    f"Materialised plugins from pipeline declaration: {sorted(resolved_plugins)}"
                )

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

    # If a data module is selected, set up data and run inference.
    if data_module:
        if registry is None:
            registry = NodeRegistry()
        data_config = DataConfig(data_module=data_module, params=dict(data_args or {}))
        datamodule = _build_data_module(registry, data_config, candidate_dirs)
        datamodule.setup(stage="predict")
        dataloader = datamodule.predict_dataloader()

        for module in pipeline.torch_layers:
            module.eval()

        # Process all batches (outputs discarded; sink nodes write to disk).
        global_step = 0
        pipeline.set_profiling(enabled=True)
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Inference", unit="batch"):
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
                pipeline.forward(batch=batch, context=context)
                global_step += 1

        logger.info(
            pipeline.format_profiling_summary(
                stage=ExecutionStage.INFERENCE, total_frames=global_step
            )
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
    trainrun_config: TrainRunConfig,
    device: str = "auto",
    base_dir: Path | None = None,
) -> CuvisPipeline:
    """Build pipeline from a trainrun's ``pipeline:`` reference.

    Parameters
    ----------
    trainrun_config : TrainRunConfig
        Trainrun configuration object. Its ``pipeline`` is a path reference to
        a pipeline YAML, not an inline config.
    device : str
        Device to load pipeline to ('cpu', 'cuda', 'auto')
    base_dir : Path | None
        Directory the reference is resolved against (the trainrun file's
        directory). When omitted, only absolute / CWD-relative references resolve.

    Returns
    -------
    CuvisPipeline
        Built pipeline ready for training
    """
    builder = PipelineBuilder()
    if trainrun_config.pipeline is None:
        raise ValueError("Pipeline reference is missing in trainrun config.")
    pipeline_path = _resolve_pipeline_reference(trainrun_config.pipeline, base_dir)
    pipeline_cfg = PipelineConfig.load_from_file(pipeline_path)
    pipeline = builder.build_from_config(pipeline_cfg.to_dict())

    # Move pipeline to specified device if needed
    if device != "auto":
        pipeline = pipeline.to(device)

    return pipeline


def _create_datamodule_from_config(
    trainrun_config: TrainRunConfig,
    candidate_dirs: list[Path] | None = None,
):
    """Build the trainrun's DataModule via the registry dispatch and setup('fit').

    Parameters
    ----------
    trainrun_config : TrainRunConfig
        Trainrun configuration object (its ``data`` is a polymorphic DataConfig).
    candidate_dirs : list[Path] | None
        Plugins directories to resolve the ``data_module``'s providing plugin from.

    Returns
    -------
    pytorch_lightning.LightningDataModule
        Configured datamodule, already ``setup(stage="fit")``.
    """
    registry = NodeRegistry()
    datamodule = _build_data_module(
        registry, trainrun_config.data, candidate_dirs or []
    )
    datamodule.setup(stage="fit")
    return datamodule


def restore_trainrun(
    trainrun_path: str | Path,
    mode: Literal["train", "validate", "test", "info"] = "info",
    checkpoint_path: str | Path | None = None,
    device: str = "auto",
    overrides: list[str] | None = None,
    plugins_dirs: list[str | Path] | None = None,
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

    if (  # pragma: no cover - needs a Hydra config tree; covered by data-backed runs
        isinstance(raw_config, dict) and "defaults" in raw_config
    ):
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

    # Build pipeline (its `pipeline:` reference resolves relative to the trainrun dir)
    pipeline = _build_pipeline_from_config(
        trainrun_config, device=device, base_dir=trainrun_path.parent
    )

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

    # Create datamodule (resolve the data_module's plugin from discovered + explicit dirs)
    candidate_dirs = _discover_plugins_dirs(trainrun_path, plugins_dirs)
    datamodule = _create_datamodule_from_config(trainrun_config, candidate_dirs)
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

        # Step 4: Run validation (populate TensorBoard monitoring).
        # Gate on the resolved val dataset (built by setup("fit")); the old flat
        # ``data.val_ids`` field no longer exists on the polymorphic DataConfig.
        if datamodule.val_ds is not None:
            logger.info("  Step 4: Validation...")
            if grad_trainer is not None:
                grad_trainer.validate()
            else:
                stat_trainer.validate()
        else:
            logger.warning(
                "  Step 4: Validation skipped - no validation data resolved for this run"
            )

        # Step 5: Run test evaluation (populate TensorBoard monitoring).
        datamodule.setup(stage="test")
        if datamodule.test_ds is not None:
            logger.info("  Step 5: Test evaluation...")
            if grad_trainer is not None:
                grad_trainer.test()
            else:
                stat_trainer.test()
        else:
            logger.warning(
                "  Step 5: Test evaluation skipped - no test data resolved for this run"
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

  # Run inference with a data module
  uv run restore-pipeline \\
    --pipeline-path configs/pipeline/adaclip_baseline.yaml \\
    --plugins-dir   ../cuvis-ai-dataloader/configs/plugins \\
    --data-module cu3s \\
    --data-arg    cu3s_file_path=data/Lentils/Lentils_000.cu3s

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
        "--data-module",
        type=str,
        default=None,
        help="DataModule name (its DATA_MODULE_NAME, e.g. 'cu3s', 'tiff_paired') "
        "to run inference over. Its providing plugin is loaded from --plugins-dir.",
    )
    parser.add_argument(
        "--data-arg",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Module-specific argument for the selected --data-module. Repeatable, "
        "e.g. --data-arg cu3s_file_path=X.cu3s --data-arg annotation_json_path=Y.json.",
    )
    parser.add_argument(
        "--override",
        action="append",
        help="Override config values in dot notation. Can be specified multiple times.",
    )
    parser.add_argument(
        "--plugins-dir",
        action="append",
        default=None,
        help="Plugins directory containing per-plugin manifest YAML files. "
        "Can be specified multiple times. The pipeline's `plugins:` field "
        "(or auto-resolution) looks up entries against the merged catalog.",
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

    # Parse repeatable --data-arg KEY=VALUE pairs into the params dict.
    data_args: dict[str, str] = {}
    for pair in args.data_arg or []:
        if "=" not in pair:
            parser.error(f"--data-arg must be KEY=VALUE, got {pair!r}")
        key, value = pair.split("=", 1)
        data_args[key.strip()] = value

    restore_pipeline(
        pipeline_path=args.pipeline_path,
        weights_path=args.weights_path,
        device=args.device,
        data_module=args.data_module,
        data_args=data_args or None,
        config_overrides=args.override,
        plugins_dirs=args.plugins_dir,
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
    parser.add_argument(
        "--plugins-dir",
        action="append",
        default=None,
        help="Plugins directory containing per-plugin manifest YAML files. Can be specified "
        "multiple times. The trainrun's data_module is resolved against the merged catalog.",
    )

    args = parser.parse_args()

    restore_trainrun(
        trainrun_path=args.trainrun_path,
        mode=args.mode,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        overrides=args.override,
        plugins_dirs=args.plugins_dir,
    )
