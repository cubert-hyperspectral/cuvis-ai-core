"""External trainer orchestrators for cuvis.ai graphs."""

from collections.abc import Mapping, Sequence
from typing import Any

import pytorch_lightning as pl
import torch
from loguru import logger
from torch.optim import Optimizer

from cuvis_ai_core.node.node import Node
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.training.config import (
    TrainingConfig,
    create_callbacks_from_config,
)
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.execution import Context, InputStream


class GradientTrainer(pl.LightningModule):
    """Gradient-based trainer with explicit loss and metric handling.

    Parameters
    ----------
    pipeline : CuvisPipeline
        Computation pipeline
    datamodule : LightningDataModule
        Data module
    loss_nodes : list[Node]
        List of loss nodes whose outputs should be summed for backpropagation.
        Each loss node should output {"loss": scalar} or multiple scalar outputs.
    metric_nodes : list[Node], optional
        List of metric nodes whose outputs should be logged during validation/testing.
        Each metric node should output {"metrics": list[Metric]}.
    training_config : TrainingConfig, optional
        Full training configuration: the ``pl.Trainer`` keyword arguments plus the
        optimizer, scheduler, and callbacks. Defaults to ``TrainingConfig()``.
    monitors : list, optional
        List of monitoring objects (TensorBoard, etc.)

    Examples
    --------
    >>> # Define loss and metric nodes
    >>> bce_loss = AnomalyBCEWithLogits(name="bce", weight=1.0)
    >>> entropy_loss = SelectorEntropyRegularizer(name="entropy", weight=0.01)
    >>> iou_metric = AnomalyDetectionMetrics(name="detection")
    >>>
    >>> # Connect in pipeline
    >>> pipeline.connect(
    ...     (predictions, bce_loss.predictions),
    ...     (targets, bce_loss.targets),
    ...     (selector_weights, entropy_loss.weights),
    ...     (predictions, iou_metric.predictions),
    ...     (targets, iou_metric.targets),
    ... )
    >>>
    >>> # Create trainer with explicit node lists
    >>> trainer = GradientTrainer(
    ...     pipeline=pipeline,
    ...     datamodule=datamodule,
    ...     loss_nodes=[bce_loss, entropy_loss],
    ...     metric_nodes=[iou_metric],
    ...     training_config=cfg,
    ... )
    >>> trainer.fit()

    Logged Metrics
    --------------
    Training:
        - train_loss: Sum of all losses (for backpropagation)
        - train/{node.name}: Individual loss components (e.g., train/bce, train/entropy)
        - train/{node.name}_{port}: If loss node has multiple outputs (e.g., train/bce_loss1)

    Validation:
        - val_loss: Sum of all losses
        - val/{node.name}: Individual loss components
        - {node.name}_{metric.name}: All metrics (e.g., detection_anomaly/iou)
        - {node.name}_{metric.name}_1: If duplicate names occur

    Testing:
        - test_loss: Sum of all losses (if loss nodes execute in test stage)
        - test/{node.name}: Individual loss components
        - {node.name}_{metric.name}: All metrics
    """

    def __init__(
        self,
        pipeline: CuvisPipeline,
        datamodule: pl.LightningDataModule,
        loss_nodes: list[Node],
        metric_nodes: list[Node] | None = None,
        training_config: TrainingConfig | None = None,
        monitors: list | None = None,
        callbacks: list | None = None,
    ) -> None:
        super().__init__()

        # Register graph's modules so Lightning can move them to correct device
        # Without this, graph nodes stay on CPU while trainer moves to CUDA
        self.pipeline_modules = pipeline.torch_layers
        self.pipeline = pipeline
        self.datamodule = datamodule
        self.loss_nodes = loss_nodes
        self.metric_nodes = metric_nodes or []
        # A node declaring POOLED_METRIC_NAMES must also expose pooled_metrics(), or those
        # names would be skipped in the per-batch path and never logged at epoch end. Fail
        # loud on the mismatch instead of silently dropping the metric.
        for metric_node in self.metric_nodes:
            pooled_names = getattr(metric_node, "POOLED_METRIC_NAMES", frozenset())
            if pooled_names and not callable(
                getattr(metric_node, "pooled_metrics", None)
            ):
                raise TypeError(
                    f"Metric node '{metric_node.name}' declares POOLED_METRIC_NAMES "
                    f"{set(pooled_names)} but has no pooled_metrics() method, so those "
                    "metrics would never be logged."
                )
        self.training_config = training_config or TrainingConfig()
        self.optimizer_config = self.training_config.optimizer
        self.scheduler_config = self.training_config.scheduler
        self.monitors = monitors or []
        self.callbacks = callbacks or []
        self.trainer = None  # Store trainer instance for reuse in test()

    def fit(self) -> None:
        """Start training using PyTorch Lightning.

        Unified API - both GradientTrainer and StatisticalTrainer use .fit()
        """

        # Re-seed here as a fallback for direct GradientTrainer users who bypass
        # the restore/gRPC orchestration (which seeds before building the pipeline
        # and datamodule, the point where weight init and data splitting happen).
        pl.seed_everything(self.training_config.seed, workers=True)

        config_callbacks = self.training_config.callbacks
        logger.info("=" * 60)
        logger.info(
            f"Training: max_epochs={self.training_config.max_epochs}, "
            f"optimizer={self.optimizer_config.name}(lr={self.optimizer_config.lr})"
        )
        if config_callbacks and config_callbacks.early_stopping:
            es = config_callbacks.early_stopping[0]
            logger.info(
                f"EarlyStopping: {es.monitor} (patience={es.patience}, mode={es.mode})"
            )
        logger.info("=" * 60)

        # The pl.Trainer keyword arguments (explicit allowlist; callbacks handled below).
        trainer_kwargs = self.training_config.to_lightning_kwargs()

        # Callbacks: explicit ones passed to __init__ win (e.g. the gRPC progress
        # stream); otherwise build them from the config.
        if self.callbacks:
            trainer_kwargs["callbacks"] = self.callbacks
        elif config_callbacks is not None:
            trainer_kwargs["callbacks"] = create_callbacks_from_config(config_callbacks)

        # Automatic checkpointing: enable whenever a checkpoint callback is
        # configured, independent of how callbacks were supplied.
        if config_callbacks is not None and config_callbacks.checkpoint is not None:
            trainer_kwargs["enable_checkpointing"] = True

        self.trainer = pl.Trainer(**trainer_kwargs)
        self.trainer.fit(model=self, datamodule=self.datamodule)

    def validate(self, ckpt_path: str | None = None) -> Sequence[Mapping[str, float]]:
        """Run validation evaluation.

        Parameters
        ----------
        ckpt_path : str | None
            Checkpoint path to load. Use "best" to load the best checkpoint,
            or None to validate with current model state.

        Returns
        -------
        Sequence[Mapping[str, float]]
            Sequence of mappings containing validation metrics.
        """
        if self.trainer is None:
            raise RuntimeError(
                "validate() called before fit(). Please call fit() first."
            )

        self.datamodule.setup(stage="val")
        return self.trainer.validate(
            model=self, datamodule=self.datamodule, ckpt_path=ckpt_path
        )

    def test(self, ckpt_path: str | None = "best") -> Sequence[Mapping[str, float]]:
        """Run test evaluation, loading best checkpoint if available.

        Reuses the trainer instance from fit() to maintain callback state and
        enable checkpoint loading. If a ModelCheckpoint callback was used during
        training, this will automatically load the best checkpoint.

        Parameters
        ----------
        ckpt_path : str | None
            Checkpoint path to load. Use "best" to load the best checkpoint saved
            during training, or None to test with current model state.

        Returns
        -------
        Sequence[Mapping[str, float]]
            Sequence of mappings containing test metrics. Each mapping maps
            metric names (e.g., 'test_loss', 'metrics_anomaly/iou') to their float values.
            Typically contains one mapping with all test metrics.

        Raises
        ------
        RuntimeError
            If test() is called before fit()

        Examples
        --------
        >>> grad_trainer = GradientTrainer(...)
        >>> grad_trainer.fit()
        >>> test_results = grad_trainer.test()  # Loads best checkpoint if available
        >>> # test_results = [{'test_loss': 0.123, 'metrics_anomaly/iou': 0.95, ...}]
        """
        if self.trainer is None:
            raise RuntimeError(
                "test() called before fit(). Please call fit() first to initialize the trainer."
            )

        self.datamodule.setup(stage="test")
        return self.trainer.test(
            model=self, datamodule=self.datamodule, ckpt_path=ckpt_path
        )

    def _collect_losses(
        self,
        node_outputs: dict[str, dict[str, Any]],
        stage: str,
        epoch: int,
        batch_idx: int,
    ) -> torch.Tensor:
        """Collect and log losses from loss_nodes.

        Parameters
        ----------
        node_outputs : dict[str, dict[str, Any]]
            Restructured graph outputs with node_id as primary key.
            Created by restructure_output_to_node_dict() for O(1) access.
        stage : str
            Execution stage: "train", "val", "test"
        epoch : int
            Current epoch number
        batch_idx : int
            Current batch index

        Returns
        -------
        torch.Tensor
            Sum of all losses for backpropagation
        """
        all_losses = []
        loss_name_counter: dict[str, int] = {}  # Track duplicate names

        for loss_node in self.loss_nodes:
            # O(1) lookup instead of O(n_outputs) iteration
            loss_outputs = node_outputs.get(loss_node.name, {})

            # Warn on first batch of first epoch if multiple outputs
            if epoch == 0 and batch_idx == 0 and len(loss_outputs) > 1:
                logger.warning(
                    f"Loss node '{loss_node.name}' produces multiple outputs: "
                    f"{list(loss_outputs.keys())}. All outputs will be summed for backpropagation."
                )

            # Collect and log each loss output
            for port_name, loss_val in loss_outputs.items():
                all_losses.append(loss_val)

                # Generate base log name
                if len(loss_outputs) == 1:
                    base_name = f"{stage}/{loss_node.name}"
                else:
                    base_name = f"{stage}/{loss_node.name}_{port_name}"

                # Handle duplicate names with numeric increment
                if base_name in loss_name_counter:
                    loss_name_counter[base_name] += 1
                    log_name = f"{base_name}_{loss_name_counter[base_name]}"
                else:
                    loss_name_counter[base_name] = 0
                    log_name = base_name

                self.log(log_name, loss_val, prog_bar=True)

        # Sum all losses for backpropagation
        if all_losses:
            total_loss = torch.stack(all_losses).sum()
        elif stage == "train":
            raise ValueError("No losses collected for training stage!")
        else:
            total_loss = torch.tensor(0.0, device=self.device)
        self.log(f"{stage}_loss", total_loss, prog_bar=True)

        return total_loss

    def _collect_metrics(self, node_outputs: dict[str, dict[str, Any]]) -> None:
        """Collect and log metrics from metric_nodes.

        Parameters
        ----------
        node_outputs : dict[str, dict[str, Any]]
            Restructured graph outputs with node_id as primary key.
            Created by restructure_output_to_node_dict() for O(1) access.
        """
        if not self.metric_nodes:
            return

        metric_name_counter: dict[str, int] = {}  # Track duplicate names

        for metric_node in self.metric_nodes:
            # Metrics that pool over the whole epoch (e.g. AUROC, average
            # precision) must be reduced by a single compute() at epoch end, not
            # by averaging per-batch running values (batch-size-sensitive, and
            # badly biased at small batch sizes). Their names are skipped in the
            # per-batch path here and logged as live torchmetrics objects below.
            pooled_names = getattr(metric_node, "POOLED_METRIC_NAMES", frozenset())

            # O(1) lookup instead of O(n_outputs) iteration
            metric_outputs = node_outputs.get(metric_node.name, {}).get("metrics", [])

            # Each metric node outputs a list of Metric dataclass objects
            for metric in metric_outputs:  # list[Metric] objects
                if metric.name in pooled_names:
                    continue

                # Generate base log name: NodeName_metric_name
                base_name = f"{metric_node.name}/{metric.name}"

                # Handle duplicate names with numeric increment
                if base_name in metric_name_counter:
                    metric_name_counter[base_name] += 1
                    log_name = f"{base_name}_{metric_name_counter[base_name]}"
                else:
                    metric_name_counter[base_name] = 0
                    log_name = base_name

                self.log(log_name, metric.value, prog_bar=True)

            # Pooled metrics: log the live torchmetrics object with on_epoch=True
            # so Lightning accumulates over the epoch and logs a single pooled
            # compute() (then resets) at epoch end, exact and batch-size-invariant,
            # with no manual epoch-end hook. A metric node opts in by exposing
            # pooled_metrics() -> dict[str, torchmetrics.Metric] alongside
            # POOLED_METRIC_NAMES; nodes without it are unaffected.
            pooled_metrics = getattr(metric_node, "pooled_metrics", None)
            if pooled_metrics is not None:
                for pooled_name, metric_obj in pooled_metrics().items():
                    self.log(
                        f"{metric_node.name}/{pooled_name}",
                        metric_obj,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                    )

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """Execute graph and collect losses for training."""
        from cuvis_ai_core.utils.graph_helper import restructure_output_to_node_dict

        context = Context(
            stage=ExecutionStage.TRAIN,
            epoch=self.current_epoch,
            batch_idx=batch_idx,
            global_step=self.global_step,
        )

        # Execute graph
        outputs = self.pipeline.forward(batch=batch, context=context)

        # Transform outputs once for efficient access (O(n) operation)
        node_outputs = restructure_output_to_node_dict(outputs)

        # Collect losses with O(1) lookups per node
        total_loss = self._collect_losses(
            node_outputs, "train", self.current_epoch, batch_idx
        )

        # Log to external monitors
        for monitor in self.monitors:
            monitor.log("train/loss", total_loss, step=self.global_step)

        return total_loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        """Execute graph and collect losses + metrics for validation."""
        from cuvis_ai_core.utils.graph_helper import restructure_output_to_node_dict

        context = Context(
            stage=ExecutionStage.VAL,
            epoch=self.current_epoch,
            batch_idx=batch_idx,
            global_step=self.global_step,
        )

        # Execute graph
        outputs = self.pipeline.forward(batch=batch, context=context)

        # Transform outputs once for efficient access (O(n) operation)
        node_outputs = restructure_output_to_node_dict(outputs)

        # Collect losses and metrics with O(1) lookups per node
        total_loss = self._collect_losses(
            node_outputs, "val", self.current_epoch, batch_idx
        )
        self._collect_metrics(node_outputs)

        # Log to external monitors
        for monitor in self.monitors:
            monitor.log("val/loss", total_loss, step=self.global_step)

        return total_loss

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        """Execute graph and collect losses + metrics for testing."""
        from cuvis_ai_core.utils.graph_helper import restructure_output_to_node_dict

        context = Context(
            stage=ExecutionStage.TEST,
            epoch=self.current_epoch,
            batch_idx=batch_idx,
            global_step=self.global_step,
        )

        # Execute graph
        outputs = self.pipeline.forward(batch=batch, context=context)

        # Transform outputs once for efficient access (O(n) operation)
        node_outputs = restructure_output_to_node_dict(outputs)

        # Collect losses and metrics with O(1) lookups per node
        total_loss = self._collect_losses(
            node_outputs, "test", self.current_epoch, batch_idx
        )
        self._collect_metrics(node_outputs)

        return total_loss

    def configure_optimizers(self) -> Optimizer | dict:
        """Configure optimizer and optional scheduler via the OptimizerConfig.

        Returns
        -------
        Optimizer | dict
            If no scheduler is configured, returns the optimizer.
            If a scheduler is configured, returns a dictionary with optimizer and scheduler config.
        """
        from cuvis_ai_core.training.optimizer_registry import (
            create_optimizer,
            create_scheduler,
            wrap_scheduler_for_lightning,
        )

        params = [p for p in self.pipeline.parameters() if p.requires_grad]
        optimizer = create_optimizer(self.optimizer_config, params)

        scheduler = create_scheduler(
            self.scheduler_config,
            optimizer,
            self.training_config.max_epochs,
        )
        if scheduler is None:
            return optimizer

        monitor = self.scheduler_config.monitor if self.scheduler_config else None
        wrapped_scheduler = wrap_scheduler_for_lightning(scheduler, monitor)

        return {"optimizer": optimizer, "lr_scheduler": wrapped_scheduler}


class StatisticalTrainer:
    """External orchestrator for statistical initialization.

    NOT a Node - it's an external orchestrator that initializes
    statistical nodes using clean port-based streams.

    This trainer:
    - Finds all nodes with requires_initial_fit=True
    - Initializes them in topological order using node.statistical_initialization()
    - Uses executor with upto_node for clean data transformation

    Examples
    --------
    >>> # Statistical initialization with unified API
    >>> stat_trainer = StatisticalTrainer(graph=graph, datamodule=datamodule)
    >>> stat_trainer.fit()  # Unified API - same as GradientTrainer
    >>>
    >>> # Run validation and test
    >>> stat_trainer.validate()
    >>> stat_trainer.test()
    """

    def __init__(
        self, pipeline: CuvisPipeline, datamodule: pl.LightningDataModule
    ) -> None:
        """Initialize statistical trainer.

        Parameters
        ----------
        pipeline : CuvisPipeline
            The cuvis-ai pipeline containing statistical nodes
        datamodule : pl.LightningDataModule
            Data provider for initialization
        """
        self.pipeline = pipeline
        self.datamodule = datamodule

    def fit(self) -> None:
        """Train all statistical nodes in topological order.

        Unified API - both GradientTrainer and StatisticalTrainer use .fit()
        """
        # Setup datamodule
        self.datamodule.setup("fit")
        train_loader = self.datamodule.train_dataloader()

        # Find nodes requiring statistical initialization
        stat_nodes = [
            node
            for node in self.pipeline.nodes()
            if getattr(node, "requires_initial_fit", False)
        ]

        if not stat_nodes:
            logger.info("No statistical nodes to train")
            return

        logger.info(f"Training {len(stat_nodes)} statistical nodes...")

        # Train in topological order
        for node in self.pipeline._sorted_nodes:
            if node not in stat_nodes:
                continue

            logger.info(f"  Training {type(node).__name__}...")

            # Create port-based input stream using upto_node
            input_stream = self._create_input_stream(node, train_loader)

            # Initialize the node from data
            node.statistical_initialization(input_stream)

    def validate(self) -> None:
        """Run validation on the validation dataset.

        Returns
        -------
        list[dict[tuple[str, str], Any]]
            List of output dictionaries (one per batch), where each dict maps
            (node_id, port_name) tuples to output tensors
        """
        self.datamodule.setup(stage="val")
        val_loader = self.datamodule.val_dataloader()

        for batch_idx, batch in enumerate(val_loader):
            batch = self._move_batch_to_device(batch)
            context = Context(
                stage=ExecutionStage.VAL,
                epoch=0,
                batch_idx=batch_idx,
                global_step=batch_idx,
            )
            self.pipeline.forward(batch=batch, context=context)

    def test(self) -> None:
        """Run test on the test dataset.

        Returns
        -------
        list[dict[tuple[str, str], Any]]
            List of output dictionaries (one per batch), where each dict maps
            (node_id, port_name) tuples to output tensors
        """
        self.datamodule.setup(stage="test")
        test_loader = self.datamodule.test_dataloader()

        for batch_idx, batch in enumerate(test_loader):
            batch = self._move_batch_to_device(batch)
            context = Context(
                stage=ExecutionStage.TEST,
                epoch=0,
                batch_idx=batch_idx,
                global_step=batch_idx,
            )
            self.pipeline.forward(batch=batch, context=context)

    def _get_pipeline_device(self) -> torch.device:
        """Get the device of the pipeline from its parameters/buffers."""
        for node in self.pipeline._sorted_nodes:
            for param in node.parameters():
                return param.device
            for buf in node.buffers():
                return buf.device
        return torch.device("cpu")

    def _move_batch_to_device(self, batch: dict) -> dict:
        """Move batch tensors to the pipeline's device.

        Parameters
        ----------
        batch : dict
            Input batch dictionary with tensor values

        Returns
        -------
        dict
            Batch with all tensors moved to pipeline's device
        """
        device = self._get_pipeline_device()
        moved_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved_batch[key] = value.to(device)
            else:
                moved_batch[key] = value
        return moved_batch

    def _create_input_stream(self, target_node, dataloader) -> InputStream:
        """Create port-based input stream for statistical node.

        Uses graph.forward() with upto_node to get clean transformed inputs.
        Yields dicts matching target_node.INPUT_SPECS.

        Parameters
        ----------
        target_node : Node
            Node to create input stream for
        dataloader : DataLoader
            Source dataloader providing batches

        Yields
        ------
        dict[str, Any]
            Input dict with keys from target_node.INPUT_SPECS
        """
        for batch in dataloader:
            # Move batch to pipeline device BEFORE calling forward
            batch = self._move_batch_to_device(batch)

            # Execute ancestors, stop before target (using upto_node)
            outputs = self.pipeline.forward(
                batch=batch,
                stage=ExecutionStage.INFERENCE,
                upto_node=target_node,  # Partial execution
            )

            # Gather inputs for target node from predecessor outputs
            node_inputs = {}
            predecessors = list(self.pipeline._graph.predecessors(target_node))

            if not predecessors:
                # Entry node - get directly from batch
                for port_name in target_node.INPUT_SPECS:
                    if port_name in batch:
                        node_inputs[port_name] = batch[port_name]
            else:
                # Get from parent outputs via graph edges
                for parent_node in predecessors:
                    for edge_data in self.pipeline._graph[parent_node][
                        target_node
                    ].values():
                        from_port = edge_data["from_port"]
                        to_port = edge_data["to_port"]
                        node_inputs[to_port] = outputs[(parent_node.name, from_port)]

            yield node_inputs  # Clean dict matching INPUT_SPECS!
