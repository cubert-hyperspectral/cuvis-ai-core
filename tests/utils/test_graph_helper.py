"""Tests for :mod:`cuvis_ai_core.utils.graph_helper`."""

from cuvis_ai_core.utils.graph_helper import restructure_output_to_node_dict


def test_restructure_groups_ports_by_node_in_first_seen_order():
    outputs = {
        ("loss_node", "loss"): 0.5,
        ("metric_node", "metrics"): ["m1"],
        ("loss_node", "aux"): 0.25,
    }
    structured = restructure_output_to_node_dict(outputs)
    assert structured == {
        "loss_node": {"loss": 0.5, "aux": 0.25},
        "metric_node": {"metrics": ["m1"]},
    }
    assert list(structured) == ["loss_node", "metric_node"]


def test_restructure_empty_outputs():
    assert restructure_output_to_node_dict({}) == {}
