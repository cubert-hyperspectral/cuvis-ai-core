"""Class-level execution stages: declaration, per-instance override, guarded setter."""

from __future__ import annotations

import pytest
import torch
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.pipeline import PortSpec

from cuvis_ai_core.node import Node


class _Passthrough(Node):
    """Minimal concrete node with no stage declaration (inherits ALWAYS).

    Has its own ``__init__`` like every real node, so ``hparams`` reflects the
    subclass signature (a class without one inherits the base signature and
    records ``name`` / ``execution_stages`` as ``None``; that is unchanged).
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


class _Movable(_EvalOnly):
    """Opts into instance-level reassignment."""

    EXECUTION_STAGES_MUTABLE = True


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

    def test_subclass_inherits_parent_declaration(self):
        assert _Movable().execution_stages == {ExecutionStage.VAL, ExecutionStage.TEST}

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

    def test_mutable_flag_must_be_bool(self):
        with pytest.raises(TypeError, match="EXECUTION_STAGES_MUTABLE must be a bool"):

            class _BadFlag(_Passthrough):
                EXECUTION_STAGES_MUTABLE = "yes"


class TestConstructorOverride:
    def test_kwarg_overrides_class_declaration(self):
        node = _EvalOnly(execution_stages={ExecutionStage.INFERENCE})
        assert node.execution_stages == frozenset({ExecutionStage.INFERENCE})
        # The class itself is untouched.
        assert _EvalOnly.EXECUTION_STAGES == {ExecutionStage.VAL, ExecutionStage.TEST}

    def test_stage_names_are_coerced(self):
        # What a pipeline yaml's `hparams: {execution_stages: [inference]}` delivers.
        node = _EvalOnly(execution_stages=["inference"])
        assert node.execution_stages == frozenset({ExecutionStage.INFERENCE})

    def test_misspelled_stage_raises_naming_the_node(self):
        with pytest.raises(ValueError) as excinfo:
            _EvalOnly(name="monitor", execution_stages=["Inference"])
        message = str(excinfo.value)
        assert "_EvalOnly 'monitor'" in message
        assert "'Inference'" in message
        assert "always, inference, test, train, val" in message

    def test_bare_string_raises(self):
        with pytest.raises(TypeError, match="iterable of ExecutionStage"):
            _EvalOnly(execution_stages="inference")

    def test_kwarg_forwarded_inside_kwargs_still_wins(self):
        # Subclasses that pass **kwargs through to Node.__init__ keep working.
        node = _EvalOnly(**{"execution_stages": [ExecutionStage.TRAIN]})
        assert node.execution_stages == frozenset({ExecutionStage.TRAIN})

    def test_override_is_not_an_hparam(self):
        node = _EvalOnly(name="n", execution_stages={ExecutionStage.INFERENCE})
        assert node.hparams == {"scale": 1.0}

    def test_consume_base_kwargs_still_pops(self):
        kwargs = {"name": "n", "execution_stages": {ExecutionStage.TRAIN}, "k": 1}
        name, stages = Node.consume_base_kwargs(kwargs, {ExecutionStage.VAL})
        assert (name, stages) == ("n", {ExecutionStage.TRAIN})
        assert kwargs == {"k": 1}
        assert Node.consume_base_kwargs({}, {ExecutionStage.VAL}) == (
            None,
            {ExecutionStage.VAL},
        )


class TestGuardedSetter:
    def test_reassignment_is_refused_by_default(self):
        node = _EvalOnly()
        with pytest.raises(AttributeError, match="EXECUTION_STAGES_MUTABLE"):
            node.execution_stages = {ExecutionStage.TRAIN}
        assert node.execution_stages == {ExecutionStage.VAL, ExecutionStage.TEST}

    def test_base_node_subclass_without_declaration_is_also_immutable(self):
        with pytest.raises(AttributeError):
            _Passthrough().execution_stages = set()

    def test_opt_in_allows_reassignment_and_coerces(self):
        node = _Movable()
        node.execution_stages = ["train", ExecutionStage.VAL]
        assert node.execution_stages == frozenset(
            {ExecutionStage.TRAIN, ExecutionStage.VAL}
        )
        node.execution_stages = set()
        assert node.execution_stages == frozenset()

    def test_opt_in_setter_rejects_unknown_names(self):
        node = _Movable(name="mover")
        with pytest.raises(ValueError, match="_Movable 'mover'.*'nope'"):
            node.execution_stages = {"nope"}


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

    def test_empty_stage_set_never_runs(self):
        node = _Movable()
        node.execution_stages = set()
        assert not any(node.should_execute(stage) for stage in ExecutionStage)
