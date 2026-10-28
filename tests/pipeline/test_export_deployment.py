"""export_deployment: the ONNX file, its sidecar, the deploy yaml and the weights."""

from __future__ import annotations

import json

import pytest
import torch
import yaml

from cuvis_ai_core.node import Node
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.pipeline import PortSpec
from tests.pipeline.test_export import (
    _CUBE,
    _VEC,
    _ItemGuard,
    _ItemGuardAdapter,
    _Scale,
    _Select,
    _Source,
    _anomaly_pipeline,
    select_span,
)

pytest.importorskip("onnxruntime")

#: The deploy folder, set by the test that loads the deploy yaml. An environment
#: variable, because the registry imports this module by its dotted path, which
#: is a second module object under pytest's rootdir import.
_DEPLOY_DIR_ENV = "CUVIS_TEST_DEPLOY_DIR"


class _StandInSpan(Node):
    """What ``ExportedSpan`` (cuvis-ai) will be: runs the span's ONNX file."""

    INPUT_SPECS = {"data": _CUBE, "wavelengths": _VEC}
    OUTPUT_SPECS = {
        "scores": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, 1)),
        "anomaly_score": PortSpec(dtype=torch.float32, shape=(-1,)),
    }

    def __init__(self, onnx_path: str, backend: str = "onnxruntime", **kwargs) -> None:
        super().__init__(onnx_path=onnx_path, backend=backend, **kwargs)
        self.onnx_path = onnx_path
        self.backend = backend

    def forward(self, data, wavelengths, **_):
        import os

        import onnxruntime as ort

        session = ort.InferenceSession(
            os.path.join(os.environ[_DEPLOY_DIR_ENV], self.onnx_path),
            providers=["CPUExecutionProvider"],
        )
        scores, anomaly_score = session.run(None, {"scale.data": data.numpy()})
        return {
            "scores": torch.from_numpy(scores),
            "anomaly_score": torch.from_numpy(anomaly_score),
        }


class _WrongAdapter(_ItemGuard):
    """An export adapter that disagrees with the node's own forward."""

    def export_module(self):
        adapter = _ItemGuardAdapter(self)
        adapter.factor_buf.add_(1.0)
        return adapter


class _Untraceable(Node):
    """Reads a data-dependent Python float; neither exporter can carry that."""

    INPUT_SPECS = {"data": _CUBE}
    OUTPUT_SPECS = {"scaled": _CUBE}

    def forward(self, data, **_):
        total = float(data.sum())
        return {"scaled": data * (total / total)}


class _ListOut(Node):
    INPUT_SPECS = {"data": _CUBE}
    OUTPUT_SPECS = {"values": _VEC}

    def forward(self, data, **_):
        return {"values": [1.0, 2.0]}


_ANOMALY_CUT = {
    "inputs": ["scale.data", "select.wavelengths"],
    "outputs": ["detect.scores", "detect.anomaly_score"],
    "constants": ["select.wavelengths"],
}


def _examples(count: int = 2, seed: int = 1) -> list[dict[str, torch.Tensor]]:
    torch.manual_seed(seed)
    wavelengths = torch.linspace(450.0, 950.0, 4)
    return [
        {"scale.data": torch.rand(1, 6, 6, 4), "select.wavelengths": wavelengths}
        for _ in range(count)
    ]


@pytest.fixture(scope="module")
def deployment(tmp_path_factory):
    from cuvis_ai_core.pipeline.export import export_deployment

    pipeline, _ = _anomaly_pipeline()
    examples = _examples()
    result = export_deployment(
        pipeline,
        tmp_path_factory.mktemp("deploy"),
        examples=examples,
        span_class_name="tests.pipeline.test_export_deployment._StandInSpan",
        **_ANOMALY_CUT,
    )
    return result, pipeline, examples


def test_export_deployment_writes_the_onnx_file_sidecar_deploy_yaml_and_weights(
    deployment,
):
    result, _, _ = deployment
    out_dir = result.onnx_path.parent

    assert sorted(p.name for p in out_dir.iterdir()) == [
        "anomaly.onnx",
        "anomaly.onnx.json",
        "anomaly_deploy.pt",
        "anomaly_deploy.yaml",
    ]
    assert result.sidecar_path == out_dir / "anomaly.onnx.json"
    assert result.yaml_path == out_dir / "anomaly_deploy.yaml"
    assert result.weights_path == out_dir / "anomaly_deploy.pt"


def test_exported_graph_matches_the_span_in_onnx_runtime(deployment):
    import onnxruntime as ort

    from cuvis_ai_core.pipeline.export import PipelineSpanModule

    result, pipeline, examples = deployment
    torch.manual_seed(7)
    cube = torch.rand(1, 6, 6, 4)
    selection = select_span(pipeline, _ANOMALY_CUT["inputs"], _ANOMALY_CUT["outputs"])
    expected = PipelineSpanModule(
        selection, constants={"select.wavelengths": examples[0]["select.wavelengths"]}
    )(cube)

    session = ort.InferenceSession(
        str(result.onnx_path), providers=["CPUExecutionProvider"]
    )
    got = session.run(None, {"scale.data": cube.numpy()})

    assert [i.name for i in session.get_inputs()] == ["scale.data"]
    assert [o.name for o in session.get_outputs()] == [
        "detect.scores",
        "detect.anomaly_score",
    ]
    for want, have in zip(expected, got):
        torch.testing.assert_close(torch.from_numpy(have), want, atol=1e-6, rtol=1e-5)


def test_sidecar_records_ports_shapes_constants_and_fingerprints(deployment):
    from cuvis_ai_core import __version__
    from cuvis_ai_core.utils.trt_engine import file_md5

    result, _, examples = deployment
    sidecar = json.loads(result.sidecar_path.read_text(encoding="utf-8"))

    assert sidecar == result.sidecar
    assert sidecar["format"] == 1
    assert sidecar["pipeline"] == "anomaly"
    assert sidecar["span"]["nodes"] == [
        {"name": "scale", "class_name": "tests.pipeline.test_export._Scale"},
        {"name": "select", "class_name": "tests.pipeline.test_export._Select"},
        {"name": "detect", "class_name": "tests.pipeline.test_export._Detect"},
    ]
    assert sidecar["span"]["inputs"] == [
        {
            "port": "scale.data",
            "name": "data",
            "tensor": "scale.data",
            "dtype": "float32",
            "shape": [1, 6, 6, 4],
        }
    ]
    assert sidecar["span"]["outputs"] == [
        {
            "port": "detect.scores",
            "name": "scores",
            "tensor": "detect.scores",
            "dtype": "float32",
            "shape": [1, 6, 6, 1],
        },
        {
            "port": "detect.anomaly_score",
            "name": "anomaly_score",
            "tensor": "detect.anomaly_score",
            "dtype": "float32",
            "shape": [1],
        },
    ]
    (constant,) = sidecar["span"]["constants"]
    assert constant["port"] == "select.wavelengths"
    assert constant["name"] == "wavelengths"
    assert constant["dtype"] == "float32"
    assert constant["shape"] == [4]
    torch.testing.assert_close(
        torch.tensor(constant["value"]),
        examples[0]["select.wavelengths"],
        rtol=0,
        atol=0,
    )
    assert sidecar["exporter"] == {
        "kind": "dynamo",
        "opset": 18,
        "torch": torch.__version__,
        "cuvis_ai_core": __version__,
    }
    assert sidecar["onnx_md5"] == file_md5(result.onnx_path)
    assert len(sidecar["fingerprint"]) == 16
    assert sidecar["parity"]["examples"] == 2
    assert sidecar["parity"]["max_abs_diff"]["detect.scores"] < 1e-5


def test_deploy_yaml_replaces_the_span_with_one_node(deployment):
    result, _, _ = deployment
    config = yaml.safe_load(result.yaml_path.read_text(encoding="utf-8"))

    assert [n["name"] for n in config["nodes"]] == ["source", "exported_span", "decide"]
    span = config["nodes"][1]
    assert span["class_name"] == "tests.pipeline.test_export_deployment._StandInSpan"
    assert span["hparams"] == {"onnx_path": "anomaly.onnx", "backend": "onnxruntime"}
    assert config["nodes"][0]["hparams"] == {"channels": 4}
    assert {(c["source"], c["target"]) for c in config["connections"]} == {
        ("source.outputs.cube", "exported_span.inputs.data"),
        ("source.outputs.wavelengths", "exported_span.inputs.wavelengths"),
        ("exported_span.outputs.scores", "decide.inputs.scores"),
    }
    assert config["plugins"] == ["cuvis_ai_onnx"]
    assert config["metadata"]["name"] == "anomaly"


def test_deploy_weights_hold_the_remaining_nodes(deployment):
    result, _, _ = deployment
    checkpoint = torch.load(result.weights_path, weights_only=True)

    assert sorted(checkpoint["state_dict"]) == ["decide", "source"]
    assert checkpoint["metadata"]["name"] == "anomaly"


def test_deploy_folder_loads_and_runs_with_a_stand_in_span_node(
    deployment, monkeypatch
):
    result, pipeline, _ = deployment
    monkeypatch.setenv(_DEPLOY_DIR_ENV, str(result.yaml_path.parent))

    deployed = CuvisPipeline.load_pipeline(
        result.yaml_path, weights_path=result.weights_path
    )
    torch.manual_seed(3)
    outputs = deployed.forward(batch={}, stage=ExecutionStage.INFERENCE)
    torch.manual_seed(3)
    reference = pipeline.forward(batch={}, stage=ExecutionStage.INFERENCE)

    assert [n.name for n in deployed.nodes] == ["source", "exported_span", "decide"]
    torch.testing.assert_close(
        outputs[("exported_span", "scores")],
        reference[("detect", "scores")],
        atol=1e-6,
        rtol=1e-5,
    )
    assert torch.equal(
        outputs[("decide", "decisions")], reference[("decide", "decisions")]
    )


def _bad_pipeline(bad: Node) -> CuvisPipeline:
    source, scale, select = (
        _Source(name="source"),
        _Scale(name="scale"),
        _Select(name="select"),
    )
    pipeline = CuvisPipeline("bad")
    pipeline.connect(
        (source.outputs.cube, scale.inputs.data),
        (scale.outputs.scaled, bad.inputs.data),
        (bad.outputs.scaled, select.inputs.cube),
        (source.outputs.wavelengths, select.inputs.wavelengths),
    )
    return pipeline


_BAD_CUT = {
    "inputs": ["scale.data", "select.wavelengths"],
    "outputs": ["select.band"],
    "constants": ["select.wavelengths"],
}


def test_export_deployment_refuses_an_adapter_that_differs_from_the_node(tmp_path):
    from cuvis_ai_core.pipeline.export import SpanExportError, export_deployment

    pipeline = _bad_pipeline(_WrongAdapter(name="wrong"))

    with pytest.raises(SpanExportError, match=r"'wrong'.*adapter"):
        export_deployment(pipeline, tmp_path, examples=_examples(), **_BAD_CUT)
    assert not list(tmp_path.glob("*.onnx"))


def test_export_deployment_names_the_first_node_that_fails_on_its_own(tmp_path):
    from cuvis_ai_core.pipeline.export import SpanExportError, export_deployment

    pipeline = _bad_pipeline(_Untraceable(name="bad"))

    with pytest.raises(SpanExportError, match="'bad'") as info:
        export_deployment(pipeline, tmp_path, examples=_examples(), **_BAD_CUT)
    assert info.value.__cause__ is not None  # the span's own error is kept
    assert not list(tmp_path.glob("*.onnx"))


def test_export_deployment_has_no_fallback_when_the_dynamo_exporter_fails(
    tmp_path, monkeypatch
):
    """The dynamo exporter (``torch.export``) is the only exporter: a span it cannot
    trace is an error that names the node, never a graph from another tracer."""
    from cuvis_ai_core.pipeline import export as export_mod

    def no_dynamo(*_, **__):
        raise RuntimeError("dynamo exporter unavailable")

    monkeypatch.setattr(export_mod, "_export_dynamo", no_dynamo)
    pipeline, _ = _anomaly_pipeline()

    with pytest.raises(export_mod.SpanExportError, match="dynamo exporter") as info:
        export_mod.export_deployment(
            pipeline, tmp_path, examples=_examples(), **_ANOMALY_CUT
        )
    assert "legacy" not in str(info.value)
    assert isinstance(info.value.__cause__, RuntimeError)
    assert not list(tmp_path.iterdir())
    assert not hasattr(export_mod, "_export_legacy")


def test_export_deployment_refuses_a_constant_that_varies_across_examples(tmp_path):
    from cuvis_ai_core.pipeline.export import SpanExportError, export_deployment

    pipeline, _ = _anomaly_pipeline()
    examples = _examples()
    examples[1]["select.wavelengths"] = examples[1]["select.wavelengths"] + 1.0

    with pytest.raises(SpanExportError, match="select.wavelengths"):
        export_deployment(pipeline, tmp_path, examples=examples, **_ANOMALY_CUT)


def test_export_deployment_refuses_an_example_without_every_input(tmp_path):
    from cuvis_ai_core.pipeline.export import SpanExportError, export_deployment

    pipeline, _ = _anomaly_pipeline()
    examples = _examples()
    del examples[1]["scale.data"]

    with pytest.raises(SpanExportError, match="scale.data"):
        export_deployment(pipeline, tmp_path, examples=examples, **_ANOMALY_CUT)


def test_export_deployment_refuses_a_non_tensor_span_output(tmp_path):
    from cuvis_ai_core.pipeline.export import SpanExportError, export_deployment

    scale, listy = _Scale(name="scale"), _ListOut(name="listy")
    pipeline = CuvisPipeline("listy")
    pipeline.connect(scale.outputs.scaled, listy.inputs.data)

    with pytest.raises(SpanExportError, match="listy.values"):
        export_deployment(
            pipeline,
            tmp_path,
            inputs=["scale.data"],
            outputs=["listy.values"],
            examples=[{"scale.data": torch.rand(1, 2, 2, 4)}],
        )
