"""Server-side wrapper around the static node catalog.

The wire shape — :class:`CatalogPortSpec`, :class:`CatalogNodeEntry`,
:class:`CatalogPluginEntry` — lives in :mod:`cuvis_ai_schemas.catalog`.
The catalog is carried **inline** in each plugin manifest entry: the
entry's ``provides`` list *is* the node catalog (every item is a
:class:`CatalogNodeEntry` — an FQCN ``class_name`` plus optional palette
metadata). This module only adds the server-side glue: build the catalog
from a registered manifest entry without importing the plugin package.

The parent gRPC server has no business importing plugin packages: the
orchestrator confines plugin code to child venvs. The GUI's node-palette
RPC needs to enumerate every plugin's node classes before any pipeline
has been built — the inline catalog closes that gap.
"""

from __future__ import annotations

from typing import Any

from cuvis_ai_schemas.catalog import (
    SUPPORTED_SCHEMA_VERSIONS,
    CatalogNodeEntry,
    CatalogPluginEntry,
    CatalogPortSpec,
)


def load_catalog_entry(
    plugin_name: str,
    config_dict: dict[str, Any],
) -> CatalogPluginEntry | None:
    """Build a plugin's static node catalog without importing plugin code.

    The catalog comes from the manifest entry's inline ``provides`` list.
    Returns ``None`` when the entry provides no nodes, so the caller
    surfaces nothing in the palette for it.

    Raises:
        ValueError: the ``provides`` payload fails Pydantic validation
            (the schema module raises ``pydantic.ValidationError``, a
            ``ValueError`` subclass).
    """
    return CatalogPluginEntry.from_manifest_entry(plugin_name, config_dict)


__all__ = [
    "CatalogNodeEntry",
    "CatalogPluginEntry",
    "CatalogPortSpec",
    "SUPPORTED_SCHEMA_VERSIONS",
    "load_catalog_entry",
]
