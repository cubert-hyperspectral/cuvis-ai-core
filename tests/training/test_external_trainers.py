"""Test suite for external trainer orchestrators."""

import pytorch_lightning as pl
import pytest
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from cuvis_ai_core.node import Node
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.execution import Context, Metric
from cuvis_ai_schemas.pipeline import PortSpec

from cuvis_ai_core.data.datamodule import BaseCuvisAIDataModule
from cuvis_ai_core.training.config import OptimizerConfig, TrainerConfig
from cuvis_ai_core.training.trainers import GradientTrainer, StatisticalTrainer
from tests.fixtures import (
    MockStatisticalTrainableNode,
    SimpleLossNode,
    SoftChannelSelector,
)


@pytest.mark.slow
class TestGradientTrainer:
    """Test GradientTrainer implementation."""

    def test_gradient_trainer_requires_loss_nodes(self):
        """Test that GradientTrainer requires loss_nodes parameter."""

        class SimpleNode(Node):
            INPUT_SPECS = {}
            OUTPUT_SPECS = {"out": PortSpec(dtype=torch.float32, shape=())}

            def forward(self, **inputs):
                return {"out": torch.tensor(1.0)}

            def load(self, params: dict, serial_dir: str) -> None:
                pass

        pipeline = CuvisPipeline("test")

        class MockDataModule(pl.LightningDataModule):
            pass

        datamodule = MockDataModule()
        trainer_config = TrainerConfig(max_epochs=1)

        # Should raise TypeError - missing required parameter
        with pytest.raises(TypeError, match="loss_nodes"):
            GradientTrainer(
                pipeline=pipeline, datamodule=datamodule, trainer_config=trainer_config
            )

    def test_gradient_trainer_creates_executor_once(self):
        """Test that GradientTrainer is properly initialized."""

        class SourceNode(Node):
            INPUT_SPECS = {}
            OUTPUT_SPECS = {"out": PortSpec(dtype=torch.float32, shape=())}

            def forward(self, **inputs):
                return {"out": torch.tensor(1.0)}

            def load(self, params: dict, serial_dir: str) -> None:
                pass

        class SimpleLossNode(Node):
            INPUT_SPECS = {"value": PortSpec(dtype=torch.float32, shape=())}
            OUTPUT_SPECS = {"loss": PortSpec(dtype=torch.float32, shape=())}

            def forward(self, value, **kwargs):
                return {"loss": value}

            def load(self, params: dict, serial_dir: str) -> None:
                pass

        pipeline = CuvisPipeline("test")
        source = SourceNode()
        loss_node = SimpleLossNode(name="test_loss")
        loss_node.execution_stages = {ExecutionStage.TRAIN, ExecutionStage.VAL}

        # Nodes auto-added when connected
        pipeline.connect(source.outputs.out, loss_node.value)

        class MockDataModule(pl.LightningDataModule):
            pass

        trainer_config = TrainerConfig(max_epochs=1)
        trainer = GradientTrainer(
            pipeline=pipeline,
            datamodule=MockDataModule(),
            trainer_config=trainer_config,
            loss_nodes=[loss_node],
        )

        # Verify trainer was created successfully
        assert trainer is not None
        assert len(list(trainer.modules())) > 0  # Contains modules from graph

    def test_gradient_trainer_passes_context_to_executor(self):
        """Test that trainer creates Context and passes to executor.forward()."""

        class SourceNode(Node):
            INPUT_SPECS = {}
            OUTPUT_SPECS = {"out": PortSpec(dtype=torch.float32, shape=())}

            def __init__(self):
                super().__init__()
                self.received_context = None

            def forward(self, context=None, **inputs):
                self.received_context = context
                return {"out": torch.tensor(1.0)}

            def load(self, params: dict, serial_dir: str) -> None:
                pass

        class SimpleLossNode(Node):
            INPUT_SPECS = {"value": PortSpec(dtype=torch.float32, shape=())}
            OUTPUT_SPECS = {"loss": PortSpec(dtype=torch.float32, shape=())}

            def forward(self, value, **kwargs):
                return {"loss": value}

            def load(self, params: dict, serial_dir: str) -> None:
                pass

        pipeline = CuvisPipeline("test")
        source = SourceNode()
        loss_node = SimpleLossNode(name="test_loss")
        loss_node.execution_stages = {ExecutionStage.TRAIN, ExecutionStage.VAL}
        pipeline.connect(source.outputs.out, loss_node.value)

        class MockDataModule(pl.LightningDataModule):
            def train_dataloader(self):
                return [{"dummy": torch.tensor(1.0)}]

        trainer_config = TrainerConfig(max_epochs=1)
        trainer = GradientTrainer(
            pipeline=pipeline,
            datamodule=MockDataModule(),
            trainer_config=trainer_config,
            loss_nodes=[loss_node],
        )
        trainer.setup("fit")

        # Manually call training_step to test context passing
        # Suppress self.log() warning since we're calling training_step without PL Trainer
        import warnings

        batch = {"dummy": torch.tensor(1.0)}
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=r".*self\.log\(\).*", category=UserWarning
            )
            trainer.training_step(batch, 0)

        # Check that context was passed
        assert source.received_context is not None
        assert isinstance(source.received_context, Context)
        assert source.received_context.stage == ExecutionStage.TRAIN

    def test_gradient_training_reduces_loss_and_updates_weights(self):
        """Test complete gradient training workflow with statistical initialization.

        This integration test verifies:
        1. Statistical initialization works
        2. Gradient training reduces loss
        3. Trainable parameters are updated during training
        """

        # Define test nodes similar to example
        class SimpleStatisticalNode(Node):
            """Statistical node for normalization."""

            INPUT_SPECS = {"features": PortSpec(dtype=torch.Tensor, shape=(-1, -1))}
            OUTPUT_SPECS = {
                "normalized": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, -1))
            }

            def __init__(self):
                super().__init__()
                self.register_buffer("mean", torch.zeros(1))
                self.register_buffer("std", torch.ones(1))
                self._requires_initial_fit_override = True

            def statistical_initialization(self, input_stream):
                """Port-based training."""
                all_features = []
                for inputs in input_stream:
                    features = inputs["features"]
                    all_features.append(features)

                all_features = torch.cat(all_features, dim=0)
                self.mean = all_features.mean(dim=0, keepdim=True)
                self.std = all_features.std(dim=0, keepdim=True) + 1e-8
                self._requires_initial_fit_override = False
                self._statistically_initialized = True

            def forward(self, features, **kwargs):
                if not self._statistically_initialized:
                    raise RuntimeError("SimpleStatisticalNode not initialized")
                normalized = (features - self.mean) / self.std
                return {"normalized": normalized.unsqueeze(-1).unsqueeze(-1)}

            def load(self, params: dict, serial_dir: str) -> None:
                pass

        class TrainableProjection(Node):
            """Trainable projection layer."""

            INPUT_SPECS = {
                "features": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, -1))
            }
            OUTPUT_SPECS = {
                "projected": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, -1))
            }

            def __init__(self, input_dim: int = 10, output_dim: int = 3):
                super().__init__()
                self.projection = torch.nn.Linear(input_dim, output_dim)

            @property
            def is_trainable(self) -> bool:
                return True

            def forward(self, features, **kwargs):
                if features.dim() == 4:
                    features = features.squeeze(-1).squeeze(-1)
                projected = self.projection(features)
                return {"projected": projected.unsqueeze(-1).unsqueeze(-1)}

            def load(self, params: dict, serial_dir: str) -> None:
                pass

        class TestDataModule(BaseCuvisAIDataModule):
            """Simple test datamodule."""

            def __init__(self, batch_size: int = 16):
                super().__init__()
                self.batch_size = batch_size

            def setup(self, stage: str):
                X_train = torch.randn(100, 10)
                y_train = X_train.sum(dim=1, keepdim=True)
                self.train_dataset = TensorDataset(X_train, y_train)

                X_val = torch.randn(20, 10)
                y_val = X_val.sum(dim=1, keepdim=True)
                self.val_dataset = TensorDataset(X_val, y_val)

            def train_dataloader(self):
                return DataLoader(
                    self.train_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    collate_fn=self.collate_batch,
                )

            def val_dataloader(self):
                return DataLoader(
                    self.val_dataset,
                    batch_size=self.batch_size,
                    collate_fn=self.collate_batch,
                )

            def collate_batch(self, batch):
                features, targets = zip(*batch, strict=True)
                return {
                    "features": torch.stack(features),
                    "targets": torch.stack(targets),
                }

        # Build graph
        pipeline = CuvisPipeline("test_training")
        normalizer = SimpleStatisticalNode()
        projection = TrainableProjection(input_dim=10, output_dim=10)
        loss_node = SimpleLossNode(name="mse_loss", weight=1.0)
        loss_node.execution_stages = {ExecutionStage.TRAIN, ExecutionStage.VAL}

        # Connect nodes
        pipeline.connect(
            (normalizer.normalized, projection.features),
            (projection.outputs.projected, loss_node.predictions),
            (normalizer.normalized, loss_node.targets),
        )

        # Create datamodule
        datamodule = TestDataModule(batch_size=16)

        # Step 1: Statistical initialization
        stat_trainer = StatisticalTrainer(pipeline=pipeline, datamodule=datamodule)
        stat_trainer.fit()

        # Verify statistical initialization worked
        assert not normalizer.requires_initial_fit
        assert normalizer.mean.abs().sum() > 0  # Mean was computed

        # Step 2: Record initial state
        datamodule.setup("fit")
        initial_batch = next(iter(datamodule.train_dataloader()))

        initial_outputs = pipeline.forward(
            stage=ExecutionStage.TRAIN, batch=initial_batch
        )

        # Extract initial loss directly from the loss node
        loss_key = (loss_node.name, "loss")
        initial_loss = initial_outputs[loss_key].item()

        # Record initial parameters
        initial_params = {}
        projection_node_id = None
        for node in pipeline.nodes():
            if isinstance(node, TrainableProjection):
                initial_params[node.name] = {
                    name: param.clone().detach()
                    for name, param in node.named_parameters()
                }
                projection_node_id = node.name
                break

        assert projection_node_id is not None, "Could not find projection node"

        # Step 3: Gradient training
        trainer_config = TrainerConfig(
            max_epochs=3,
            enable_progress_bar=False,
            enable_checkpointing=False,
        )
        optimizer_config = OptimizerConfig(name="adam", lr=0.01)

        grad_trainer = GradientTrainer(
            pipeline=pipeline,
            datamodule=datamodule,
            trainer_config=trainer_config,
            optimizer_config=optimizer_config,
            loss_nodes=[loss_node],
        )

        # Override to disable validation
        grad_trainer.trainer_config.__dict__.update(
            {
                "num_sanity_val_steps": 0,
                "limit_val_batches": 0,
            }
        )

        grad_trainer.fit()

        # Step 4: Verify training effects
        final_outputs = pipeline.forward(
            stage=ExecutionStage.TRAIN, batch=initial_batch
        )

        # Extract final loss directly from the loss node
        loss_key = (loss_node.name, "loss")
        final_loss = final_outputs[loss_key].item()

        # Verify loss decreased
        assert final_loss < initial_loss, (
            f"Loss did not decrease: initial={initial_loss:.4f}, final={final_loss:.4f}"
        )

        # Verify parameters changed
        params_changed = False
        for node_id in initial_params:
            # Find node by id
            target_node = None
            for node in pipeline.nodes():
                if node.name == node_id:
                    target_node = node
                    break
            assert target_node is not None, f"Could not find node with id {node_id}"

            for param_name, initial_param in initial_params[node_id].items():
                final_param = dict(target_node.named_parameters())[param_name]
                param_diff = (final_param - initial_param).abs().sum().item()
                if param_diff > 1e-6:
                    params_changed = True
                    break

        assert params_changed, "Parameters did not change during training"


class TestStatisticalTrainer:
    """Test StatisticalTrainer implementation."""

    def test_statistical_trainer_initializes_in_topological_order(self):
        """Test that statistical nodes are initialized in correct order."""

        class StatNode(Node):
            INPUT_SPECS = {"data": PortSpec(dtype=torch.float32, shape=(-1,))}
            OUTPUT_SPECS = {"out": PortSpec(dtype=torch.float32, shape=(-1,))}

            def __init__(self):
                super().__init__()
                self._statistically_initialized = False

            def statistical_initialization(self, input_stream):
                """Port-based training method."""
                for inputs in input_stream:
                    _ = inputs["data"]  # From INPUT_SPECS
                self._statistically_initialized = True

            def forward(self, data, **kwargs):
                return {"out": data}

        pipeline = CuvisPipeline("test")
        node1: Node = StatNode()
        node2: Node = StatNode()

        # Nodes auto-added when connected
        pipeline.connect(node1.outputs.out, node2.data)

        class MockDataModule(pl.LightningDataModule):
            def setup(self, stage):
                pass

            def train_dataloader(self):
                return [{"data": torch.randn(10)}]

        trainer = StatisticalTrainer(pipeline=pipeline, datamodule=MockDataModule())

        trainer.fit()

        # Both nodes should be initialized
        assert node1._statistically_initialized, "Source node was not initialized"
        assert node2._statistically_initialized, "Dependent node was not initialized"

    def test_statistical_trainer_skips_nodes_without_initial_fit(self):
        """Test that nodes without requires_initial_fit are skipped."""

        class RegularNode(Node):
            INPUT_SPECS = {}
            OUTPUT_SPECS = {"out": PortSpec(dtype=torch.float32, shape=())}

            def __init__(self):
                super().__init__()

            @property
            def requires_initial_fit(self) -> bool:
                return False

            def forward(self, **inputs):
                return {"out": torch.tensor(1.0)}

        pipeline = CuvisPipeline("test")
        # Create a minimal node without connections
        RegularNode()

        class MockDataModule(pl.LightningDataModule):
            def setup(self, stage):
                pass

            def train_dataloader(self):
                return []

        trainer = StatisticalTrainer(pipeline, MockDataModule())

        # Should not raise error even with no statistical nodes
        trainer.fit()


class TestTrainerValidation:
    """Test trainer validation logic."""

    def test_trainer_requires_loss_nodes_parameter(self):
        """Test that GradientTrainer requires loss_nodes parameter."""
        pipeline = CuvisPipeline("test")

        class MockDataModule(pl.LightningDataModule):
            pass

        # Create a simple loss node for testing
        class SimpleLossNode(Node):
            INPUT_SPECS = {}
            OUTPUT_SPECS = {"loss": PortSpec(dtype=torch.float32, shape=())}

            def forward(self, **kwargs):
                return {"loss": torch.tensor(1.0)}

            def load(self, params: dict, serial_dir: str) -> None:
                pass

        # Test that providing no loss_nodes raises TypeError
        with pytest.raises(TypeError, match="loss_nodes"):
            GradientTrainer(pipeline, MockDataModule())


class TestEpochPooledMetrics:
    """Streaming val/test metrics are reduced by a single pooled compute() over
    the epoch — the trainer logs the torchmetrics *object* and Lightning pools +
    resets it — not by Lightning's per-batch epoch-mean.

    Covers the streaming-AUROC epoch-reduction bug: on separable data (true
    AUROC 1.0) the per-batch mean gave ~0.79 / 0.83 / 1.00 at batch_size
    1 / 4 / 12 (early single-class batches make the running metric undefined ->
    torchmetrics returns 0.0, dragging the mean). Also covers reset between runs:
    the object must be reset at each validation/test epoch end so repeated
    validate()/test() calls, and successive fit epochs, never pool onto each
    other.
    """

    SCORES = [0.10, 0.20, 0.30, 0.40, 0.45, 0.49, 0.51, 0.60, 0.70, 0.80, 0.90, 0.95]
    SEPARABLE = [0.0] * 6 + [1.0] * 6  # anomaly scores all exceed normal -> AUROC 1.0
    INVERTED = [1.0] * 6 + [0.0] * 6  # same scores, flipped labels        -> AUROC 0.0

    @staticmethod
    def _nodes():
        from torchmetrics.classification import BinaryAUROC

        class ScoreSource(Node):
            INPUT_SPECS = {
                "scores": PortSpec(dtype=torch.float32, shape=(-1,)),
                "targets": PortSpec(dtype=torch.float32, shape=(-1,)),
            }
            OUTPUT_SPECS = {
                "scores": PortSpec(dtype=torch.float32, shape=(-1,)),
                "targets": PortSpec(dtype=torch.float32, shape=(-1,)),
            }

            def forward(self, scores, targets, **kwargs):
                return {"scores": scores, "targets": targets}

            def load(self, params: dict, serial_dir: str) -> None:
                pass

        class StreamingAUROC(Node):
            """Minimal streaming metric node exposing its torchmetrics object.

            Deliberately does NOT self-reset, so the tests exercise Lightning's
            epoch-end reset of the logged object rather than any node-side logic.
            It also emits a per-batch running float under a *different* name
            ("auroc_running") so the plain per-batch path is exercised alongside
            the pooled object path without colliding on a log key.
            """

            INPUT_SPECS = {
                "scores": PortSpec(dtype=torch.float32, shape=(-1,)),
                "targets": PortSpec(dtype=torch.float32, shape=(-1,)),
            }
            OUTPUT_SPECS = {"metrics": PortSpec(dtype=list, shape=())}
            POOLED_METRIC_NAMES = frozenset({"auroc"})

            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.auroc = BinaryAUROC()  # thresholds=None -> exact

            def forward(self, scores, targets, context, **kwargs):
                self.auroc.update(scores.flatten(), targets.flatten().int())
                running = float(self.auroc.compute())
                return {
                    "metrics": [
                        Metric(
                            name="auroc",
                            value=running,
                            stage=context.stage,
                            epoch=context.epoch,
                        ),
                        Metric(
                            name="auroc_running",
                            value=running,
                            stage=context.stage,
                            epoch=context.epoch,
                        ),
                    ]
                }

            def pooled_metrics(self):
                return {"auroc": self.auroc}

            def load(self, params: dict, serial_dir: str) -> None:
                pass

        class Scaler(Node):
            """Trainable passthrough so fit() has a parameter to optimize."""

            INPUT_SPECS = {"scores": PortSpec(dtype=torch.float32, shape=(-1,))}
            OUTPUT_SPECS = {"scores": PortSpec(dtype=torch.float32, shape=(-1,))}

            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.w = torch.nn.Parameter(torch.tensor(1.0))

            @property
            def is_trainable(self) -> bool:
                return True

            def forward(self, scores, **kwargs):
                return {"scores": scores * self.w}

            def load(self, params: dict, serial_dir: str) -> None:
                pass

        class MeanLoss(Node):
            INPUT_SPECS = {"scores": PortSpec(dtype=torch.float32, shape=(-1,))}
            OUTPUT_SPECS = {"loss": PortSpec(dtype=torch.float32, shape=())}

            def forward(self, scores, **kwargs):
                return {"loss": scores.mean() ** 2}

            def load(self, params: dict, serial_dir: str) -> None:
                pass

        return ScoreSource, StreamingAUROC, Scaler, MeanLoss

    @classmethod
    def _build(cls, batch_size, *, train=None, val=None, test=None):
        ScoreSource, StreamingAUROC, Scaler, MeanLoss = cls._nodes()

        pipeline = CuvisPipeline("epoch_pooled_metrics")
        source = ScoreSource()
        auroc = StreamingAUROC(name="auroc_node")
        scaler = Scaler(name="scaler")
        loss = MeanLoss(name="mean_loss")
        auroc.execution_stages = {ExecutionStage.VAL, ExecutionStage.TEST}
        for node in (scaler, loss):
            node.execution_stages = {
                ExecutionStage.TRAIN,
                ExecutionStage.VAL,
                ExecutionStage.TEST,
            }
        # Metric reads the RAW source scores (so AUROC is decided purely by the
        # labels); the loss goes through the trainable scaler.
        pipeline.connect(
            (source.outputs.scores, auroc.scores),
            (source.outputs.targets, auroc.targets),
            (source.outputs.scores, scaler.inputs.scores),
            (scaler.outputs.scores, loss.scores),
        )

        data = {"train": train, "val": val, "test": test}

        class _DataModule(BaseCuvisAIDataModule):
            def setup(self, stage=None):
                self.train_dataset = self._ds("train")
                self.val_dataset = self._ds("val")
                self.test_dataset = self._ds("test")

            def _ds(self, split):
                spec = data.get(split)
                if spec is None:
                    return None
                scores, labels = spec
                return TensorDataset(torch.tensor(scores), torch.tensor(labels))

            def _dl(self, dataset):
                return DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=self.collate_batch,
                )

            def train_dataloader(self):
                return self._dl(self.train_dataset)

            def val_dataloader(self):
                return self._dl(self.val_dataset)

            def test_dataloader(self):
                return self._dl(self.test_dataset)

            def collate_batch(self, batch):
                s, t = zip(*batch, strict=True)
                return {"scores": torch.stack(s), "targets": torch.stack(t)}

        datamodule = _DataModule()
        trainer = GradientTrainer(
            pipeline=pipeline,
            datamodule=datamodule,
            loss_nodes=[loss],
            metric_nodes=[auroc],
            optimizer_config=OptimizerConfig(name="sgd", lr=0.01),
        )
        return trainer, datamodule

    @classmethod
    def _loader(cls, labels, batch_size):
        ds = TensorDataset(torch.tensor(cls.SCORES), torch.tensor(labels))

        def collate(batch):
            s, t = zip(*batch, strict=True)
            return {"scores": torch.stack(s), "targets": torch.stack(t)}

        return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    @staticmethod
    def _pl_trainer(**kw):
        return pl.Trainer(
            enable_checkpointing=False,
            enable_progress_bar=False,
            logger=False,
            **kw,
        )

    @pytest.mark.parametrize("batch_size", [1, 4, 12])
    def test_pooled_auroc_is_batch_size_invariant(self, batch_size):
        """Pooled epoch AUROC == 1.0 for every batch size on separable data."""
        trainer, datamodule = self._build(batch_size, val=(self.SCORES, self.SEPARABLE))
        pl_trainer = self._pl_trainer()
        pl_trainer.validate(model=trainer, datamodule=datamodule)

        # Pooled name: written only by the object path (the per-batch "auroc" is
        # skipped), so an exact 1.0 attributes the value to the pooled compute.
        value = pl_trainer.callback_metrics["auroc_node/auroc"].item()
        assert value == pytest.approx(1.0, abs=1e-6), (
            f"pooled AUROC at batch_size={batch_size} was {value}, expected 1.0"
        )
        # The non-pooled per-batch metric still reaches the trainer on the float
        # path (the pooled name was skipped there, not dropped entirely).
        assert "auroc_node/auroc_running" in pl_trainer.callback_metrics

    def test_repeated_validate_is_isolated(self):
        """Two validate() calls on one trainer must not pool onto each other.

        The object is reset at each validation epoch end: call 1 (separable) ->
        1.0; call 2 (same scores, inverted labels) -> 0.0. A leak would give ~0.5.
        The node deliberately does not self-reset, so this exercises Lightning's
        epoch-end reset of the logged object.
        """
        trainer, _ = self._build(4, val=(self.SCORES, self.SEPARABLE))
        pl_trainer = self._pl_trainer()

        pl_trainer.validate(trainer, dataloaders=self._loader(self.SEPARABLE, 4))
        first = pl_trainer.callback_metrics["auroc_node/auroc"].item()
        pl_trainer.validate(trainer, dataloaders=self._loader(self.INVERTED, 4))
        second = pl_trainer.callback_metrics["auroc_node/auroc"].item()

        assert first == pytest.approx(1.0, abs=1e-6)
        assert second == pytest.approx(0.0, abs=1e-6), (
            f"metric leaked across validate() calls: second run was {second}, "
            "expected 0.0 (~0.5 indicates no reset)"
        )

    def test_val_then_test_is_isolated(self):
        """test() after validate() must not inherit validation state."""
        trainer, _ = self._build(4, val=(self.SCORES, self.SEPARABLE))
        pl_trainer = self._pl_trainer()

        pl_trainer.validate(trainer, dataloaders=self._loader(self.SEPARABLE, 4))
        val_auroc = pl_trainer.callback_metrics["auroc_node/auroc"].item()
        pl_trainer.test(trainer, dataloaders=self._loader(self.INVERTED, 4))
        test_auroc = pl_trainer.callback_metrics["auroc_node/auroc"].item()

        assert val_auroc == pytest.approx(1.0, abs=1e-6)
        assert test_auroc == pytest.approx(0.0, abs=1e-6), (
            f"validation state leaked into test: {test_auroc}, expected 0.0"
        )

    def test_metric_available_during_multi_epoch_fit(self):
        """Object logging works through a 2-epoch fit and the pooled val metric
        lands in callback_metrics (i.e. is monitorable for checkpointing)."""
        trainer, datamodule = self._build(
            4,
            train=(self.SCORES, self.SEPARABLE),
            val=(self.SCORES, self.SEPARABLE),
        )
        pl_trainer = self._pl_trainer(max_epochs=2, num_sanity_val_steps=0)
        pl_trainer.fit(trainer, datamodule=datamodule)

        value = pl_trainer.callback_metrics["auroc_node/auroc"].item()
        assert value == pytest.approx(1.0, abs=1e-6)


class TestGraphInputValidation:
    """Tests covering executor validation of dataloader keys."""

    def test_executor_raises_when_batch_keys_do_not_match_graph_inputs(self):
        """Ensure missing required batch keys produce a clear failure."""

        class DictDataset(Dataset):
            def __init__(self, batch_key: str, num_samples: int = 6):
                self.batch_key = batch_key
                self.samples = torch.randn(num_samples, 4, 4, 8)

            def __len__(self) -> int:
                return len(self.samples)

            def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
                return {self.batch_key: self.samples[idx]}

        class MockDataModule(pl.LightningDataModule):
            def __init__(self, batch_key: str):
                super().__init__()
                self.dataset = DictDataset(batch_key)

            def train_dataloader(self) -> DataLoader:
                return DataLoader(self.dataset, batch_size=2, shuffle=False)

        pipeline = CuvisPipeline("selector_statistical_validation")
        soft_selector = SoftChannelSelector(
            n_select=2, input_channels=8
        )  # input is data, output is selected
        statistical_node = MockStatisticalTrainableNode(
            input_dim=2, hidden_dim=4
        )  # input is data, output is result
        pipeline.connect(soft_selector.selected, statistical_node.data)

        datamodule = MockDataModule(
            batch_key="cube"
        )  # we dont mention cube -> data mapping in pipeline, hence forward should fail
        bad_batch = next(iter(datamodule.train_dataloader()))

        error_match = "missing required inputs"

        with pytest.raises(RuntimeError, match=error_match):
            pipeline.forward(stage=ExecutionStage.TRAIN, batch=bad_batch)

        with pytest.raises(RuntimeError, match=error_match):
            pipeline.forward(
                stage=ExecutionStage.TRAIN,
                batch=bad_batch,
                upto_node=statistical_node,
            )
