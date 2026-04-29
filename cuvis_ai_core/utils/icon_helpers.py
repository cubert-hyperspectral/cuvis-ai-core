"""Icon resolution: per-node SVG → category-default SVG → empty bytes."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

from cuvis_ai_schemas.enums import NodeCategory


def get_node_icon(
    *,
    class_name: str,
    icon_name: str | None,
    category: NodeCategory,
    package_root: Path | None,
) -> bytes:
    """Resolve the SVG bytes for a node.

    Resolution order:
      1. Per-node SVG: ``{package_root}/assets/node_icons/{icon_name or class_name}.svg``
      2. Category-default SVG: ``cuvis_ai_schemas/extensions/ui/icons/{category.value}.svg``
         (Phase 1 ships all 13 defaults including ``unspecified.svg``, so this branch
         hits in production for every category.)
      3. Empty bytes — only when both branches miss (e.g. someone has stripped the
         icons folder from the schemas wheel).
    """
    if package_root is not None:
        per_node = (
            package_root / "assets" / "node_icons" / f"{icon_name or class_name}.svg"
        )
        if per_node.is_file():
            return per_node.read_bytes()

    asset_name = f"{category.value}.svg"
    try:
        asset = resources.files("cuvis_ai_schemas.extensions.ui") / "icons" / asset_name
        if asset.is_file():
            return asset.read_bytes()
    except (ModuleNotFoundError, FileNotFoundError):
        pass

    return b""
