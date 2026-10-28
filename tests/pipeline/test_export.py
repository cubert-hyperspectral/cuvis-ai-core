"""Span export: the cut between declared input and output ports of a pipeline."""

from __future__ import annotations

import pytest
import torch

from cuvis_ai_core.node import Node
from cuvis_ai_core.pipeline.export import PortRef, SpanSelectionError, select_span
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.pipeline import PortSpec

_CUBE = PortSpec(dtype=torch.float32, shape=(-1, -1, -1, -1))
_VEC = PortSpec(dtype=torch.float32, shape=(-1,))


class _Source(Node):
    """A data node: produces a cube and its wavelength grid."""

    OUTPUT_SPECS = {"cube": _CUBE, "wavelengths": _VEC}

    def __init__(self, channels: int = 4, **kwargs) -> None:
        super().__init__(channels=channels, **kwargs)
        self.channels = channels

    def forward(self, **_):
        return {
            "cube": torch.rand(1, 6, 6, self.channels),
            "wavelengths": torch.linspace(450.0, 950.0, self.channels),
        }


class _Scale(Node):
    """Multiplies by a fitted factor held in a buffer."""

    INPUT_SPECS = {"data": _CUBE}
    OUTPUT_SPECS = {"scaled": _CUBE}

    def __init__(self, factor: float = 2.0, **kwargs) -> None:
        super().__init__(factor=factor, **kwargs)
        self.register_buffer("factor_buf", torch.tensor(float(factor)))

    def forward(self, data, **_):
        return {"scaled": data * self.factor_buf}


class _Select(Node):
    """Picks the band closest to a target wavelength; ``wavelengths`` is a constant
    candidate at export time."""

    INPUT_SPECS = {"cube": _CUBE, "wavelengths": _VEC}
    OUTPUT_SPECS = {"band": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, 1))}

    def __init__(self, target_nm: float = 700.0, **kwargs) -> None:
        super().__init__(target_nm=target_nm, **kwargs)
        self.target_nm = float(target_nm)

    def forward(self, cube, wavelengths, **_):
        index = (wavelengths - self.target_nm).abs().argmin().reshape(1)
        return {"band": torch.index_select(cube, -1, index)}


class _Detect(Node):
    """Scores a band; an optional reference is subtracted when connected."""

    INPUT_SPECS = {
        "band": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, 1)),
        "reference": PortSpec(
            dtype=torch.float32, shape=(-1, -1, -1, 1), optional=True
        ),
    }
    OUTPUT_SPECS = {
        "scores": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, 1)),
        "anomaly_score": PortSpec(dtype=torch.float32, shape=(-1,)),
    }

    def forward(self, band, reference=None, **_):
        scores = band if reference is None else band - reference
        return {"scores": scores, "anomaly_score": scores.amax(dim=(1, 2, 3))}


class _Decide(Node):
    """A consumer outside the span."""

    INPUT_SPECS = {"scores": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, 1))}
    OUTPUT_SPECS = {"decisions": PortSpec(dtype=torch.bool, shape=(-1, -1, -1, 1))}

    def forward(self, scores, **_):
        return {"decisions": scores > 0.5}


class _FanIn(Node):
    INPUT_SPECS = {
        "items": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, -1), variadic=True)
    }
    OUTPUT_SPECS = {"merged": _CUBE}

    def forward(self, items, **_):
        return {"merged": torch.cat(items, dim=-1)}


class _TrainOnly(Node):
    EXECUTION_STAGES = frozenset({ExecutionStage.TRAIN})
    INPUT_SPECS = {"data": _CUBE}
    OUTPUT_SPECS = {"scaled": _CUBE}

    def forward(self, data, **_):
        return {"scaled": data}


def _anomaly_pipeline() -> tuple[CuvisPipeline, dict[str, Node]]:
    """source -> scale -> select -> detect -> decide, wavelengths from the source."""
    nodes = {
        "source": _Source(name="source"),
        "scale": _Scale(name="scale"),
        "select": _Select(name="select"),
        "detect": _Detect(name="detect"),
        "decide": _Decide(name="decide"),
    }
    pipeline = CuvisPipeline("anomaly")
    pipeline.connect(
        (nodes["source"].outputs.cube, nodes["scale"].inputs.data),
        (nodes["scale"].outputs.scaled, nodes["select"].inputs.cube),
        (nodes["source"].outputs.wavelengths, nodes["select"].inputs.wavelengths),
        (nodes["select"].outputs.band, nodes["detect"].inputs.band),
        (nodes["detect"].outputs.scores, nodes["decide"].inputs.scores),
    )
    return pipeline, nodes


def test_select_span_walks_back_from_the_outputs_to_the_declared_inputs() -> None:
    pipeline, _ = _anomaly_pipeline()

    selection = select_span(
        pipeline,
        inputs=["scale.data", "select.wavelengths"],
        outputs=["detect.scores", "detect.anomaly_score"],
    )

    assert [node.name for node in selection.nodes] == ["scale", "select", "detect"]
    assert selection.inputs == [
        PortRef("scale", "data"),
        PortRef("select", "wavelengths"),
    ]
    assert selection.outputs == [
        PortRef("detect", "scores"),
        PortRef("detect", "anomaly_score"),
    ]
    # Internal edges are wired from the producing output port; boundary inputs and
    # the unconnected optional port are not in the wiring.
    assert selection.wiring["select"] == {"cube": PortRef("scale", "scaled")}
    assert selection.wiring["detect"] == {"band": PortRef("select", "band")}
    assert selection.wiring["scale"] == {}


def test_select_span_refuses_an_unknown_port() -> None:
    pipeline, _ = _anomaly_pipeline()

    with pytest.raises(SpanSelectionError, match="detect.heatmap"):
        select_span(pipeline, inputs=["scale.data"], outputs=["detect.heatmap"])
    with pytest.raises(SpanSelectionError, match="nowhere.data"):
        select_span(pipeline, inputs=["nowhere.data"], outputs=["detect.scores"])


def test_select_span_refuses_a_required_input_that_would_fall_back_to_the_batch() -> (
    None
):
    """``select.wavelengths`` is fed by the source; without it as a declared input the
    walk would pull the data node in, and a port with no edge at all would read the batch."""
    scale, select = _Scale(name="scale"), _Select(name="select")
    pipeline = CuvisPipeline("loose")
    pipeline.connect(scale.outputs.scaled, select.inputs.cube)

    with pytest.raises(SpanSelectionError, match="select.wavelengths"):
        select_span(pipeline, inputs=["scale.data"], outputs=["select.band"])


def test_select_span_refuses_a_declared_input_the_outputs_do_not_reach() -> None:
    pipeline, _ = _anomaly_pipeline()

    with pytest.raises(SpanSelectionError, match="scale.data"):
        select_span(
            pipeline,
            inputs=["scale.data", "select.cube", "select.wavelengths"],
            outputs=["detect.scores"],
        )


def test_select_span_refuses_a_span_output_consumed_outside_but_not_declared() -> None:
    """``detect.scores`` feeds the decider; a span that keeps only ``anomaly_score`` would
    leave the decider without its input in the deploy graph."""
    pipeline, _ = _anomaly_pipeline()

    with pytest.raises(SpanSelectionError, match=r"detect.scores.*decide"):
        select_span(
            pipeline,
            inputs=["scale.data", "select.wavelengths"],
            outputs=["detect.anomaly_score"],
        )


def test_select_span_refuses_a_cut_whose_contraction_creates_a_cycle() -> None:
    """select -> detect inside the span, and select.band -> decide -> detect.reference
    outside: contracting the span into one node makes decide both its consumer and
    its producer."""
    scale, select, detect = (
        _Scale(name="scale"),
        _Select(name="select"),
        _Detect(name="detect"),
    )
    ref = _Scale(name="ref")
    source = _Source(name="source")
    pipeline = CuvisPipeline("loop")
    pipeline.connect(
        (source.outputs.cube, scale.inputs.data),
        (source.outputs.wavelengths, select.inputs.wavelengths),
        (scale.outputs.scaled, select.inputs.cube),
        (select.outputs.band, detect.inputs.band),
        (select.outputs.band, ref.inputs.data),
        (ref.outputs.scaled, detect.inputs.reference),
    )

    with pytest.raises(SpanSelectionError, match="ref"):
        select_span(
            pipeline,
            inputs=["scale.data", "select.wavelengths", "detect.reference"],
            outputs=["detect.scores", "select.band"],
        )


def test_select_span_keeps_variadic_fan_in_in_the_pipeline_order() -> None:
    a, b, c = _Scale(name="a"), _Scale(name="b"), _Scale(name="c")
    sink = _FanIn(name="sink")
    pipeline = CuvisPipeline("fan_in")
    pipeline.connect(b.outputs.scaled, sink.inputs.items)
    pipeline.connect(a.outputs.scaled, sink.inputs.items)
    pipeline.connect(c.outputs.scaled, sink.inputs.items)
    # The reference order is the pipeline's own gather order (``_gather_node_inputs``).
    expected = [
        PortRef(pred.name, edge["from_port"])
        for pred in pipeline._graph.predecessors(sink)
        for _, edge in pipeline._graph[pred][sink].items()
    ]
    assert len(expected) == 3

    selection = select_span(
        pipeline, inputs=["a.data", "b.data", "c.data"], outputs=["sink.merged"]
    )

    assert selection.wiring["sink"] == {"items": expected}
    assert selection.nodes[-1] is sink


def test_select_span_refuses_a_variadic_port_as_a_span_input() -> None:
    sink = _FanIn(name="sink")
    pipeline = CuvisPipeline("fan_in")
    pipeline.connect(_Scale(name="a").outputs.scaled, sink.inputs.items)

    with pytest.raises(SpanSelectionError, match="variadic"):
        select_span(pipeline, inputs=["sink.items"], outputs=["sink.merged"])


def test_select_span_refuses_a_node_that_does_not_run_at_inference() -> None:
    scale, train_only = _Scale(name="scale"), _TrainOnly(name="train_only")
    pipeline = CuvisPipeline("stages")
    pipeline.connect(scale.outputs.scaled, train_only.inputs.data)

    with pytest.raises(SpanSelectionError, match="train_only"):
        select_span(pipeline, inputs=["scale.data"], outputs=["train_only.scaled"])


# ---------------------------------------------------------------------------
# PipelineSpanModule: the static schedule the exporter traces
# ---------------------------------------------------------------------------


class _ItemGuard(Node):
    """Scales by a factor read with ``.item()`` (not traceable); the export adapter
    does the same with tensor ops."""

    INPUT_SPECS = {"data": _CUBE}
    OUTPUT_SPECS = {"scaled": _CUBE}

    def __init__(self, factor: float = 3.0, **kwargs) -> None:
        super().__init__(factor=factor, **kwargs)
        self.register_buffer("factor_buf", torch.tensor(float(factor)))

    def forward(self, data, **_):
        return {"scaled": data * self.factor_buf.item()}

    def export_module(self):
        return _ItemGuardAdapter(self)


class _ItemGuardAdapter(torch.nn.Module):
    def __init__(self, node: _ItemGuard) -> None:
        super().__init__()
        self.register_buffer("factor_buf", node.factor_buf.detach().clone())

    def forward(self, data):
        return {"scaled": data * self.factor_buf}


def _run_pipeline(pipeline: CuvisPipeline, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    return pipeline.forward(batch={}, stage=ExecutionStage.INFERENCE)


def test_node_export_module_defaults_to_none() -> None:
    assert _Scale(name="scale").export_module() is None


def test_span_module_reproduces_the_pipeline_forward() -> None:
    from cuvis_ai_core.pipeline.export import PipelineSpanModule

    pipeline, _ = _anomaly_pipeline()
    ports = _run_pipeline(pipeline)
    selection = select_span(
        pipeline,
        inputs=["scale.data", "select.wavelengths"],
        outputs=["detect.scores", "detect.anomaly_score"],
    )

    module = PipelineSpanModule(
        selection, constants={"select.wavelengths": ports[("source", "wavelengths")]}
    )
    scores, anomaly_score = module(ports[("source", "cube")])

    assert module.input_names == ["scale.data"]
    assert module.constant_names == ["select.wavelengths"]
    assert module.output_names == ["detect.scores", "detect.anomaly_score"]
    torch.testing.assert_close(scores, ports[("detect", "scores")], rtol=0, atol=0)
    torch.testing.assert_close(
        anomaly_score, ports[("detect", "anomaly_score")], rtol=0, atol=0
    )


def test_span_module_runs_the_export_adapter_instead_of_the_forward() -> None:
    from cuvis_ai_core.pipeline.export import PipelineSpanModule

    guard, select = _ItemGuard(name="guard"), _Select(name="select")
    source = _Source(name="source")
    pipeline = CuvisPipeline("adapted")
    pipeline.connect(
        (source.outputs.cube, guard.inputs.data),
        (guard.outputs.scaled, select.inputs.cube),
        (source.outputs.wavelengths, select.inputs.wavelengths),
    )
    ports = _run_pipeline(pipeline)
    selection = select_span(
        pipeline, inputs=["guard.data", "select.wavelengths"], outputs=["select.band"]
    )

    module = PipelineSpanModule(
        selection, constants={"select.wavelengths": ports[("source", "wavelengths")]}
    )
    (band,) = module(ports[("source", "cube")])

    assert isinstance(module.module_for("guard"), _ItemGuardAdapter)
    assert module.module_for("select") is not select  # a copy, never the live node
    torch.testing.assert_close(band, ports[("select", "band")], rtol=0, atol=0)


def test_span_module_copies_the_nodes_and_leaves_the_live_ones_in_their_mode() -> None:
    from cuvis_ai_core.pipeline.export import PipelineSpanModule

    pipeline, nodes = _anomaly_pipeline()
    nodes["scale"].train()
    selection = select_span(
        pipeline, inputs=["scale.data", "select.wavelengths"], outputs=["detect.scores"]
    )

    module = PipelineSpanModule(
        selection, constants={"select.wavelengths": torch.ones(4)}
    )
    module.module_for("scale").factor_buf.fill_(99.0)

    assert nodes["scale"].training is True
    assert module.module_for("scale").training is False
    assert float(nodes["scale"].factor_buf) == 2.0


def test_span_module_feeds_a_variadic_port_in_the_gather_order() -> None:
    from cuvis_ai_core.pipeline.export import PipelineSpanModule

    a, b, c = (
        _Scale(name="a", factor=1.0),
        _Scale(name="b", factor=10.0),
        _Scale(name="c"),
    )
    sink = _FanIn(name="sink")
    pipeline = CuvisPipeline("fan_in")
    pipeline.connect(b.outputs.scaled, sink.inputs.items)
    pipeline.connect(a.outputs.scaled, sink.inputs.items)
    pipeline.connect(c.outputs.scaled, sink.inputs.items)
    selection = select_span(
        pipeline, inputs=["a.data", "b.data", "c.data"], outputs=["sink.merged"]
    )
    x = torch.ones(1, 2, 2, 1)
    expected = torch.cat(
        [
            {"a": x, "b": x * 10.0, "c": x * 2.0}[ref.node]
            for ref in selection.wiring["sink"]["items"]
        ],
        dim=-1,
    )

    (merged,) = PipelineSpanModule(selection)(x, x, x)

    torch.testing.assert_close(merged, expected, rtol=0, atol=0)


def test_span_module_refuses_a_constant_that_is_not_a_span_input() -> None:
    from cuvis_ai_core.pipeline.export import PipelineSpanModule

    pipeline, _ = _anomaly_pipeline()
    selection = select_span(
        pipeline, inputs=["scale.data", "select.wavelengths"], outputs=["detect.scores"]
    )

    with pytest.raises(SpanSelectionError, match="detect.band"):
        PipelineSpanModule(selection, constants={"detect.band": torch.ones(1)})
