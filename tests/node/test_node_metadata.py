"""Direct coverage for the Node-level metadata accessors and ClassVar defaults."""

from __future__ import annotations

from cuvis_ai_schemas.enums import NodeCategory, NodeTag

from cuvis_ai_core.node import Node


class _AnnotatedNode(Node):
    INPUT_SPECS: dict = {}
    OUTPUT_SPECS: dict = {}
    _category = NodeCategory.MODEL
    _tags = frozenset({NodeTag.HYPERSPECTRAL, NodeTag.LEARNABLE})
    _icon_name = "custom_icon"

    def forward(self, **inputs):
        return {}


class _BareNode(Node):
    INPUT_SPECS: dict = {}
    OUTPUT_SPECS: dict = {}

    def forward(self, **inputs):
        return {}


def test_get_category_returns_overridden_value() -> None:
    assert _AnnotatedNode.get_category() is NodeCategory.MODEL


def test_get_tags_returns_overridden_value() -> None:
    assert _AnnotatedNode.get_tags() == frozenset(
        {NodeTag.HYPERSPECTRAL, NodeTag.LEARNABLE}
    )


def test_get_icon_name_returns_overridden_value() -> None:
    assert _AnnotatedNode.get_icon_name() == "custom_icon"


def test_get_category_default_is_unspecified() -> None:
    assert _BareNode.get_category() is NodeCategory.UNSPECIFIED


def test_get_tags_default_is_empty_frozenset() -> None:
    assert _BareNode.get_tags() == frozenset()


def test_get_icon_name_default_is_none() -> None:
    assert _BareNode.get_icon_name() is None


def test_classmethods_callable_from_instance() -> None:
    """`get_*` classmethods are callable from instances too — same return values."""
    instance = _AnnotatedNode(name="annotated_instance")
    assert instance.get_category() is NodeCategory.MODEL
    assert instance.get_tags() == frozenset({NodeTag.HYPERSPECTRAL, NodeTag.LEARNABLE})
    assert instance.get_icon_name() == "custom_icon"
