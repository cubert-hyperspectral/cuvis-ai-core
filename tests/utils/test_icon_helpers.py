"""Tests for the icon resolution chain."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import pytest

from cuvis_ai_core.utils.icon_helpers import get_node_icon
from cuvis_ai_schemas.enums import NodeCategory


def _bundled_icon_bytes(category: NodeCategory) -> bytes:
    asset = (
        resources.files("cuvis_ai_schemas.extensions.ui")
        / "icons"
        / f"{category.value}.svg"
    )
    return asset.read_bytes()


def test_get_node_icon_prefers_per_node_svg(tmp_path: Path) -> None:
    """A file at <package_root>/assets/node_icons/<class>.svg wins over the schemas default."""
    icons_dir = tmp_path / "assets" / "node_icons"
    icons_dir.mkdir(parents=True)
    custom = b"<svg>per-node</svg>"
    (icons_dir / "MyNode.svg").write_bytes(custom)

    out = get_node_icon(
        class_name="MyNode",
        icon_name=None,
        category=NodeCategory.MODEL,
        package_root=tmp_path,
    )
    assert out == custom


def test_get_node_icon_uses_explicit_icon_name(tmp_path: Path) -> None:
    """When ``icon_name`` is set it overrides the class name for filename lookup."""
    icons_dir = tmp_path / "assets" / "node_icons"
    icons_dir.mkdir(parents=True)
    custom = b"<svg>aliased</svg>"
    (icons_dir / "alias.svg").write_bytes(custom)

    out = get_node_icon(
        class_name="DifferentClassName",
        icon_name="alias",
        category=NodeCategory.MODEL,
        package_root=tmp_path,
    )
    assert out == custom


def test_get_node_icon_falls_back_to_schemas_default(tmp_path: Path) -> None:
    """No per-node file → return the category-default SVG bundled by schemas."""
    out = get_node_icon(
        class_name="UnknownClass",
        icon_name=None,
        category=NodeCategory.MODEL,
        package_root=tmp_path,  # exists but has no assets/node_icons folder
    )
    expected = _bundled_icon_bytes(NodeCategory.MODEL)
    assert out == expected
    assert len(out) > 0


@pytest.mark.parametrize("category", list(NodeCategory))
def test_get_node_icon_resolves_for_every_category(category: NodeCategory) -> None:
    """Schemas ships 13 SVGs (12 named + unspecified) — every NodeCategory hits a real file."""
    out = get_node_icon(
        class_name="X",
        icon_name=None,
        category=category,
        package_root=None,
    )
    assert out == _bundled_icon_bytes(category)
    assert len(out) > 0


def test_get_node_icon_returns_empty_bytes_when_schemas_assets_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the schemas package can't be located the helper returns b\"\" rather than raising."""

    def _raise(_pkg: str):
        raise ModuleNotFoundError("cuvis_ai_schemas.extensions.ui not on sys.path")

    monkeypatch.setattr(resources, "files", _raise)

    out = get_node_icon(
        class_name="X",
        icon_name=None,
        category=NodeCategory.MODEL,
        package_root=None,
    )
    assert out == b""


def test_node_classmethods_default_to_unannotated() -> None:
    """A bare ``Node`` subclass inherits the additive defaults (UNSPECIFIED / empty / None)."""
    from cuvis_ai_core.node.node import Node

    class Bare(Node):
        def forward(self, **_):
            return {}

    assert Bare.get_category() is NodeCategory.UNSPECIFIED
    assert Bare.get_tags() == frozenset()
    assert Bare.get_icon_name() is None


def test_node_classmethods_pick_up_subclass_overrides() -> None:
    """Setting the ClassVars on a subclass body is reflected by the classmethods."""
    from cuvis_ai_core.node.node import Node
    from cuvis_ai_schemas.enums import NodeTag

    class Annotated(Node):
        _category = NodeCategory.MODEL
        _tags = frozenset({NodeTag.HYPERSPECTRAL, NodeTag.LEARNABLE})
        _icon_name = "custom"

        def forward(self, **_):
            return {}

    assert Annotated.get_category() is NodeCategory.MODEL
    assert Annotated.get_tags() == frozenset({NodeTag.HYPERSPECTRAL, NodeTag.LEARNABLE})
    assert Annotated.get_icon_name() == "custom"
