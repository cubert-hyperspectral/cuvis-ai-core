"""Characterization tests for ``_infer_tags_from_pipeline``: keyword buckets and their order."""

from __future__ import annotations

import pytest

from cuvis_ai_core.grpc.helpers import _infer_tags_from_pipeline


@pytest.mark.parametrize(
    ("node_types", "expected"),
    [
        ([], []),
        (["RXDetector"], ["anomaly"]),
        (["MaskCompositor"], ["segmentation"]),
        (
            ["ChannelSelector", "Dinomaly"],
            ["segmentation"],
        ),  # "dinomaly" lacks "anomaly"
        (
            ["gradient_based_thing", "lad_global", "segment_x"],
            ["anomaly", "segmentation"],
        ),
        (["Normalizer", "VideoWriter"], []),
    ],
)
def test_infer_tags_buckets_node_types(
    node_types: list[str], expected: list[str]
) -> None:
    data = {"nodes": [{"type": t} for t in node_types]}

    assert _infer_tags_from_pipeline(data) == expected


def test_infer_tags_ignores_nodes_that_are_not_dicts_and_missing_types() -> None:
    data = {
        "nodes": ["rx_detector", None, {"name": "no-type"}, {"type": "anomaly_map"}]
    }

    assert _infer_tags_from_pipeline(data) == ["anomaly"]


def test_infer_tags_without_nodes() -> None:
    assert _infer_tags_from_pipeline({}) == []
