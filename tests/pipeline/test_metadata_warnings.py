"""Tests for the pipeline-add-node metadata warning (ALL-5187 phase 2)."""

from __future__ import annotations

import warnings

import pytest
import torch

from cuvis_ai_core.node.node import Node
from cuvis_ai_core.pipeline import _metadata_warnings
from cuvis_ai_core.pipeline._metadata_warnings import MissingNodeMetadataWarning
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec


@pytest.fixture(autouse=True)
def _reset_warned_classes():
    """Each test gets a clean dedup set so order-dependent runs stay independent."""
    _metadata_warnings._warned_classes.clear()
    yield
    _metadata_warnings._warned_classes.clear()


def _bare_unannotated_class():
    class Bare(Node):
        INPUT_SPECS: dict[str, PortSpec] = {}
        OUTPUT_SPECS = {"data": PortSpec(torch.Tensor, (-1,))}

        def forward(self, **_):
            return {"data": torch.zeros(1)}

    return Bare


def _annotated_class():
    class Annotated(Node):
        INPUT_SPECS: dict[str, PortSpec] = {}
        OUTPUT_SPECS = {"data": PortSpec(torch.Tensor, (-1,))}
        _category = NodeCategory.SOURCE
        _tags = frozenset({NodeTag.HYPERSPECTRAL})

        def forward(self, **_):
            return {"data": torch.zeros(1)}

    return Annotated


def test_warning_fires_once_for_unannotated_class():
    """First instance added emits the warning; mentions both _category and _tags."""
    Bare = _bare_unannotated_class()
    pipeline = CuvisPipeline("p")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", MissingNodeMetadataWarning)
        pipeline._assign_counter_and_add_node(Bare(name="b1"))

    relevant = [w for w in caught if issubclass(w.category, MissingNodeMetadataWarning)]
    assert len(relevant) == 1, [str(w.message) for w in caught]
    msg = str(relevant[0].message)
    assert "_category" in msg
    assert "_tags" in msg


def test_warning_deduplicates_per_class():
    """Adding a second instance of the same unannotated class is silent."""
    Bare = _bare_unannotated_class()
    pipeline = CuvisPipeline("p")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", MissingNodeMetadataWarning)
        pipeline._assign_counter_and_add_node(Bare(name="b1"))
        pipeline._assign_counter_and_add_node(Bare(name="b2"))

    relevant = [w for w in caught if issubclass(w.category, MissingNodeMetadataWarning)]
    assert len(relevant) == 1, [str(w.message) for w in relevant]


def test_warning_silent_for_fully_annotated_class():
    """A class with both _category and _tags set never warns."""
    Annotated = _annotated_class()
    pipeline = CuvisPipeline("p")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", MissingNodeMetadataWarning)
        pipeline._assign_counter_and_add_node(Annotated(name="a"))

    relevant = [w for w in caught if issubclass(w.category, MissingNodeMetadataWarning)]
    assert relevant == []


def test_warning_treats_unspecified_tag_as_missing():
    """``_tags = frozenset({NodeTag.UNSPECIFIED})`` is semantically empty — still warn."""

    class FakeAnnotated(Node):
        INPUT_SPECS: dict[str, PortSpec] = {}
        OUTPUT_SPECS = {"data": PortSpec(torch.Tensor, (-1,))}
        _category = NodeCategory.SOURCE
        _tags = frozenset({NodeTag.UNSPECIFIED})

        def forward(self, **_):
            return {"data": torch.zeros(1)}

    pipeline = CuvisPipeline("p")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", MissingNodeMetadataWarning)
        pipeline._assign_counter_and_add_node(FakeAnnotated(name="x"))

    relevant = [w for w in caught if issubclass(w.category, MissingNodeMetadataWarning)]
    assert len(relevant) == 1
    assert "_tags" in str(relevant[0].message)
    assert "_category" not in str(relevant[0].message)


def test_warning_suppressible_via_catch_warnings():
    """Tests can opt out by installing a filter; pipeline-add becomes silent."""
    Bare = _bare_unannotated_class()
    pipeline = CuvisPipeline("p")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("ignore", MissingNodeMetadataWarning)
        pipeline._assign_counter_and_add_node(Bare(name="b"))

    relevant = [w for w in caught if issubclass(w.category, MissingNodeMetadataWarning)]
    assert relevant == []


def test_warning_message_points_at_phase_3_doc():
    """The remediation hint references the phase 3 decision flowchart."""
    Bare = _bare_unannotated_class()
    pipeline = CuvisPipeline("p")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", MissingNodeMetadataWarning)
        pipeline._assign_counter_and_add_node(Bare(name="b"))

    relevant = [w for w in caught if issubclass(w.category, MissingNodeMetadataWarning)]
    assert len(relevant) == 1
    assert "ALL-5187" in str(relevant[0].message)
