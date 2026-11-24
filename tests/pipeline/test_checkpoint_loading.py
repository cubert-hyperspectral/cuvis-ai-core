"""Checkpoint loading accepts tensors and plain containers only, and says why it refuses."""

import io
import pickle
from dataclasses import dataclass

import pytest
import torch

from cuvis_ai_core.pipeline.pipeline import CuvisPipeline


@dataclass
class _Canary:
    """A harmless class a checkpoint may not carry; the loader must refuse it."""

    label: str = "canary"


def _fitted_pipeline(node_cls):
    pipeline = CuvisPipeline("checkpoint_loading")
    node = node_cls(input_dim=3, hidden_dim=2)
    pipeline._graph.add_node(node)
    node.statistical_initialization(iter([{"data": torch.randn(1, 4, 4, 3)}]))
    return pipeline, node


def _save(pipeline, tmp_path):
    config_path = tmp_path / "pipeline.yaml"
    pipeline.save_to_file(config_path)
    return config_path, config_path.with_suffix(".pt")


@pytest.fixture
def load_kwargs(monkeypatch):
    """Record the keyword arguments of every torch.load call."""
    seen: list[dict] = []
    real_load = torch.load

    def spy(*args, **kwargs):
        seen.append(kwargs)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", spy)
    return seen


def test_restore_loads_with_weights_only(
    mock_statistical_trainable_node, tmp_path, load_kwargs
):
    pipeline, _ = _fitted_pipeline(mock_statistical_trainable_node)
    _, weights = _save(pipeline, tmp_path)

    pipeline._restore_weights_from_checkpoint(weights)

    assert [kwargs.get("weights_only") for kwargs in load_kwargs] == [True]


def test_restore_refuses_a_non_tensor_object(mock_statistical_trainable_node, tmp_path):
    pipeline, node = _fitted_pipeline(mock_statistical_trainable_node)
    before = node.fitted_mean.clone()
    weights = tmp_path / "canary.pt"
    torch.save({"state_dict": {node.name: {"fitted_mean": _Canary()}}}, weights)

    with pytest.raises(pickle.UnpicklingError, match="_Canary"):
        pipeline._restore_weights_from_checkpoint(weights)

    assert torch.equal(node.fitted_mean, before)


@pytest.mark.parametrize(
    ("payload", "found"),
    [
        (torch.zeros(2), "Tensor"),
        ([torch.zeros(2)], "list"),
        ({"state_dict": [torch.zeros(2)]}, "list"),
    ],
    ids=["tensor", "list", "state-dict-list"],
)
def test_restore_rejects_a_malformed_checkpoint(
    mock_statistical_trainable_node, tmp_path, payload, found
):
    pipeline, _ = _fitted_pipeline(mock_statistical_trainable_node)
    weights = tmp_path / "malformed.pt"
    torch.save(payload, weights)

    with pytest.raises(ValueError, match=f"found {found}"):
        pipeline._restore_weights_from_checkpoint(weights)


def test_restore_without_state_dict_loads_nothing(
    mock_statistical_trainable_node, tmp_path
):
    """A dict without ``state_dict`` keeps today's behaviour: every node is missing."""
    pipeline, node = _fitted_pipeline(mock_statistical_trainable_node)
    weights = tmp_path / "empty.pt"
    torch.save({}, weights)

    assert pipeline._restore_weights_from_checkpoint(weights) == [node.name]


def test_restore_from_a_file_object_matches_the_path_route(
    mock_statistical_trainable_node, tmp_path
):
    pipeline, node = _fitted_pipeline(mock_statistical_trainable_node)
    config_path, weights = _save(pipeline, tmp_path)
    rebuilt = CuvisPipeline.load_pipeline(config_path)
    rebuilt_node = list(rebuilt.nodes)[0]

    missing = rebuilt._restore_weights_from_checkpoint(io.BytesIO(weights.read_bytes()))

    assert missing == []
    assert torch.equal(rebuilt_node.fitted_mean, node.fitted_mean)
    assert torch.equal(rebuilt_node.fitted_transform, node.fitted_transform)


def test_restore_from_a_missing_path_raises(mock_statistical_trainable_node, tmp_path):
    pipeline, _ = _fitted_pipeline(mock_statistical_trainable_node)

    with pytest.raises(FileNotFoundError):
        pipeline._restore_weights_from_checkpoint(tmp_path / "absent.pt")


def test_load_trainer_state_uses_weights_only(tmp_path, load_kwargs):
    path = tmp_path / "trainer.pt"
    torch.save(
        {
            "metadata": {"epoch": 3},
            "optimizer_state": {"state": {}, "param_groups": [{"lr": 0.1}]},
        },
        path,
    )

    state = CuvisPipeline.load_trainer_state(path)

    assert [kwargs.get("weights_only") for kwargs in load_kwargs] == [True]
    assert state["metadata"] == {"epoch": 3}
    assert state["optimizer_state"]["param_groups"] == [{"lr": 0.1}]


def test_load_trainer_state_refuses_a_non_tensor_object(tmp_path):
    path = tmp_path / "trainer.pt"
    torch.save({"metadata": {"note": _Canary()}}, path)

    with pytest.raises(pickle.UnpicklingError, match="_Canary"):
        CuvisPipeline.load_trainer_state(path)


def test_load_trainer_state_rejects_a_non_dict_checkpoint(tmp_path):
    path = tmp_path / "trainer.pt"
    torch.save(torch.zeros(1), path)

    with pytest.raises(ValueError, match="found Tensor"):
        CuvisPipeline.load_trainer_state(path)
