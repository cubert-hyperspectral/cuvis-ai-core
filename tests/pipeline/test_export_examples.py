"""collect_examples: real samples for the span inputs, taken from pipeline runs."""

from __future__ import annotations

import pytest
import torch

from cuvis_ai_core.pipeline.export import SpanExportError, collect_examples
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_schemas.enums import ExecutionStage
from tests.pipeline.test_export import _Scale, _Select, _anomaly_pipeline


def test_collect_examples_takes_the_values_the_pipeline_feeds_into_the_span_inputs():
    pipeline, _ = _anomaly_pipeline()
    torch.manual_seed(0)
    expected = [
        pipeline.forward(batch={}, stage=ExecutionStage.INFERENCE) for _ in range(2)
    ]

    torch.manual_seed(0)
    examples = collect_examples(
        pipeline, ["scale.data", "select.wavelengths"], [{}, {}]
    )

    assert len(examples) == 2
    for example, ports in zip(examples, expected):
        assert sorted(example) == ["scale.data", "select.wavelengths"]
        torch.testing.assert_close(
            example["scale.data"], ports[("source", "cube")], rtol=0, atol=0
        )
        torch.testing.assert_close(
            example["select.wavelengths"],
            ports[("source", "wavelengths")],
            rtol=0,
            atol=0,
        )


def _batch_fed_pipeline() -> CuvisPipeline:
    scale, select = _Scale(name="scale"), _Select(name="select")
    pipeline = CuvisPipeline("batch_fed")
    pipeline.connect(scale.outputs.scaled, select.inputs.cube)
    return pipeline


def test_collect_examples_reads_an_unconnected_input_from_the_batch():
    pipeline = _batch_fed_pipeline()
    batch = {
        "data": torch.rand(1, 3, 3, 4),
        "wavelengths": torch.linspace(450.0, 950.0, 4),
    }

    (example,) = collect_examples(
        pipeline, ["scale.data", "select.wavelengths"], [batch]
    )

    assert example["scale.data"] is batch["data"]
    assert example["select.wavelengths"] is batch["wavelengths"]


def test_collect_examples_refuses_an_input_the_batch_does_not_carry():
    pipeline = _batch_fed_pipeline()

    with pytest.raises(SpanExportError, match="select.wavelengths"):
        collect_examples(
            pipeline,
            ["scale.data", "select.wavelengths"],
            [{"data": torch.rand(1, 3, 3, 4)}],
        )


def test_collect_examples_stops_at_the_limit():
    pipeline, _ = _anomaly_pipeline()

    examples = collect_examples(
        pipeline, ["scale.data", "select.wavelengths"], iter([{}] * 5), limit=2
    )

    assert len(examples) == 2
