"""Class-level execution stages: the class declaration is the only source of truth."""

from __future__ import annotations

import pytest
import torch
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.pipeline import PortSpec

from cuvis_ai_core.node import Node
from cuvis_ai_core.pipeline.factory import PipelineBuilder


class _Passthrough(Node):
    """Minimal concrete node with no stage declaration (inherits ALWAYS).

    Has its own ``__init__`` like every real node, so ``hparams`` reflects the
    subclass signature.
    """

    INPUT_SPECS = {"x": PortSpec(dtype=torch.float32, shape=(-1,))}
    OUTPUT_SPECS = {"y": PortSpec(dtype=torch.float32, shape=(-1,))}

    def __init__(self, scale: float = 1.0, **kwargs):
        self.scale = scale
        super().__init__(scale=scale, **kwargs)

    def forward(self, x, **_):
        return {"y": x * self.scale}


class _EvalOnly(_Passthrough):
    """Declares its stages as a set literal on the class body."""

    EXECUTION_STAGES = {ExecutionStage.VAL, ExecutionStage.TEST}


class _NamedStages(_Passthrough):
    """Declares its stages by name; normalized to enum members at class creation."""

    EXECUTION_STAGES = ["train", "val"]


class _Never(_Passthrough):
    """An empty declaration: the node runs in no stage at all."""

    EXECUTION_STAGES = set()


class _BareSink(Node):
    """A node without its own ``__init__`` (inherits the base signature)."""

    INPUT_SPECS = {"x": PortSpec(dtype=torch.float32, shape=(-1,))}
    OUTPUT_SPECS = {}

    def forward(self, x, **_):
        return {}


class TestClassDeclaration:
    def test_base_default_is_always(self):
        assert Node.EXECUTION_STAGES == frozenset({ExecutionStage.ALWAYS})
        assert _Passthrough().execution_stages == frozenset({ExecutionStage.ALWAYS})

    def test_set_literal_is_normalized_to_frozenset(self):
        assert isinstance(_EvalOnly.EXECUTION_STAGES, frozenset)
        assert _EvalOnly.EXECUTION_STAGES == {ExecutionStage.VAL, ExecutionStage.TEST}
        assert _EvalOnly.get_execution_stages() == _EvalOnly.EXECUTION_STAGES

    def test_stage_names_are_coerced_at_class_creation(self):
        assert _NamedStages.EXECUTION_STAGES == frozenset(
            {ExecutionStage.TRAIN, ExecutionStage.VAL}
        )

    def test_instance_reads_class_declaration(self):
        node = _EvalOnly()
        assert node.execution_stages == {ExecutionStage.VAL, ExecutionStage.TEST}
        assert isinstance(node.execution_stages, frozenset)
        assert node.execution_stages is _EvalOnly.EXECUTION_STAGES

    def test_subclass_inherits_parent_declaration(self):
        class _Child(_EvalOnly):
            pass

        assert _Child().execution_stages == {ExecutionStage.VAL, ExecutionStage.TEST}

    def test_empty_declaration_is_allowed(self):
        assert _Never.EXECUTION_STAGES == frozenset()

    def test_unknown_stage_name_on_class_raises(self):
        with pytest.raises(ValueError, match="_Broken.EXECUTION_STAGES.*'validation'"):

            class _Broken(_Passthrough):
                EXECUTION_STAGES = {"validation"}

    def test_non_iterable_declaration_raises(self):
        with pytest.raises(TypeError, match="_BrokenType.EXECUTION_STAGES"):

            class _BrokenType(_Passthrough):
                EXECUTION_STAGES = 5

    def test_bare_string_declaration_raises(self):
        with pytest.raises(TypeError, match="_BrokenStr.EXECUTION_STAGES"):

            class _BrokenStr(_Passthrough):
                EXECUTION_STAGES = "train"

    def test_no_mutable_opt_in_exists(self):
        assert not hasattr(Node, "EXECUTION_STAGES_MUTABLE")


class TestNoConstructorOverride:
    """``execution_stages`` is not a constructor argument any more."""

    def test_none_is_accepted_and_ignored(self):
        # What a pipeline yaml from an earlier release delivers
        # (`hparams: {execution_stages: null}`).
        node = _EvalOnly(execution_stages=None)
        assert node.execution_stages == {ExecutionStage.VAL, ExecutionStage.TEST}

    def test_none_inside_forwarded_kwargs_is_ignored(self):
        node = _EvalOnly(**{"execution_stages": None, "scale": 2.0})
        assert node.execution_stages == {ExecutionStage.VAL, ExecutionStage.TEST}
        assert node.scale == 2.0

    def test_none_never_reaches_hparams(self):
        assert _EvalOnly(name="n", execution_stages=None).hparams == {"scale": 1.0}

    def test_stage_names_raise_naming_the_node(self):
        with pytest.raises(TypeError) as excinfo:
            _EvalOnly(name="monitor", execution_stages=["inference"])
        message = str(excinfo.value)
        assert "_EvalOnly 'monitor'" in message
        assert "EXECUTION_STAGES" in message
        assert "no longer a constructor argument" in message

    def test_stage_members_raise_too(self):
        with pytest.raises(TypeError, match="no longer a constructor argument"):
            _EvalOnly(execution_stages={ExecutionStage.INFERENCE})

    def test_class_without_own_init_behaves_the_same(self):
        with pytest.raises(TypeError, match="_BareSink '_BareSink'"):
            _BareSink(execution_stages=["train"])
        assert _BareSink(execution_stages=None).execution_stages == {
            ExecutionStage.ALWAYS
        }

    def test_consume_base_kwargs_is_gone(self):
        assert not hasattr(Node, "consume_base_kwargs")


class TestReadOnly:
    def test_assignment_raises_and_leaves_the_declaration_alone(self):
        node = _EvalOnly()
        with pytest.raises(AttributeError):
            node.execution_stages = {ExecutionStage.TRAIN}
        assert node.execution_stages == {ExecutionStage.VAL, ExecutionStage.TEST}
        assert _EvalOnly.EXECUTION_STAGES == {ExecutionStage.VAL, ExecutionStage.TEST}

    def test_base_subclass_without_declaration_is_also_read_only(self):
        with pytest.raises(AttributeError):
            _Passthrough().execution_stages = set()


class TestShouldExecute:
    @pytest.mark.parametrize("stage", list(ExecutionStage))
    def test_always_runs_in_every_stage(self, stage):
        assert _Passthrough().should_execute(stage) is True
        assert _Passthrough().should_execute(stage.value) is True

    def test_declared_stages_gate_execution(self):
        node = _EvalOnly()
        assert node.should_execute(ExecutionStage.VAL)
        assert node.should_execute("test")
        assert not node.should_execute(ExecutionStage.TRAIN)
        assert not node.should_execute(ExecutionStage.INFERENCE)

    def test_unknown_stage_string_is_false(self):
        assert _Passthrough().should_execute("deploy") is False

    def test_empty_declaration_never_runs(self):
        node = _Never()
        assert not any(node.should_execute(stage) for stage in ExecutionStage)


class _LocalRegistry:
    """Registry stub the factory resolves this module's nodes from by class name."""

    _classes = {"_Passthrough": _Passthrough, "_EvalOnly": _EvalOnly}

    def get(self, class_identifier: str) -> type:
        return self._classes[class_identifier]


def _config(sink_hparams: dict | None) -> dict:
    sink: dict = {"name": "sink", "class_name": "_EvalOnly"}
    if sink_hparams is not None:
        sink["hparams"] = sink_hparams
    return {
        "metadata": {"name": "stages"},
        "nodes": [
            {"name": "source", "class_name": "_Passthrough", "hparams": {"scale": 2.0}},
            sink,
        ],
        "connections": [{"source": "source.outputs.y", "target": "sink.inputs.x"}],
    }


class TestPipelineLoad:
    """What a pipeline yaml's ``hparams`` can and cannot say about stages."""

    @staticmethod
    def _build(sink_hparams: dict | None):
        factory = PipelineBuilder(node_registry=_LocalRegistry())
        return factory.build_from_config(_config(sink_hparams))

    @staticmethod
    def _sink(pipeline):
        return next(node for node in pipeline.nodes if node.name == "sink")

    def test_absent_key_uses_the_class_declaration(self):
        sink = self._sink(self._build(None))
        assert sink.execution_stages == {ExecutionStage.VAL, ExecutionStage.TEST}

    def test_legacy_null_key_loads_and_is_dropped(self):
        sink = self._sink(self._build({"execution_stages": None}))
        assert sink.execution_stages == {ExecutionStage.VAL, ExecutionStage.TEST}
        assert "execution_stages" not in sink.hparams

    def test_stage_list_fails_at_load_naming_the_class(self):
        with pytest.raises(TypeError, match="_EvalOnly.*EXECUTION_STAGES"):
            self._build({"execution_stages": ["inference"]})

    def test_loaded_nodes_are_gated_by_their_classes(self):
        pipeline = self._build(None)
        runs_at = {
            stage: sorted(n.name for n in pipeline.nodes if n.should_execute(stage))
            for stage in (ExecutionStage.INFERENCE, ExecutionStage.VAL)
        }
        assert runs_at[ExecutionStage.INFERENCE] == ["source"]
        assert runs_at[ExecutionStage.VAL] == ["sink", "source"]
