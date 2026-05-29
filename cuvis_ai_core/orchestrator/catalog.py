"""Server-side I/O wrapper around the static node catalog.

The wire shape — :class:`CatalogPortSpec`, :class:`CatalogNodeEntry`,
:class:`CatalogPluginEntry` — lives in :mod:`cuvis_ai_schemas.catalog`
so plugin repos can emit ``metadata.json`` against the same Pydantic
models the server reads back. This module only adds the server-side
glue: read ``metadata_path`` off a plugin manifest entry, resolve it,
and dispatch to the schemas-provided validator.

The parent gRPC server has no business importing plugin packages: the
orchestrator confines plugin code to child venvs. The GUI's node-palette
RPC needs to enumerate every plugin's node classes before any pipeline
has been built — the static catalog closes that gap.
"""

from __future__ import annotations

from pathlib import Path
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
    """Load a plugin's static node catalog without importing plugin code.

    Returns ``None`` when the plugin's manifest entry carries no
    ``metadata_path``; the caller should then fall back to import-based
    discovery (a transitional path the next phase removes entirely).

    Raises:
        ValueError: ``metadata_path`` is set but not absolute, or the
            JSON payload fails Pydantic validation (the schema module
            raises ``pydantic.ValidationError``, which is a ``ValueError``
            subclass).
        FileNotFoundError: ``metadata_path`` is set but the file is
            missing on disk.
    """
    metadata_path = config_dict.get("metadata_path")
    if not metadata_path:
        return None

    path = Path(metadata_path)
    if not path.is_absolute():
        raise ValueError(
            f"Plugin '{plugin_name}' metadata_path must be absolute "
            f"(got: {metadata_path!r}). Relative paths in YAML manifests are "
            "resolved by PluginManifest.from_yaml; manifests submitted via the "
            "LoadPlugins RPC must use an absolute path."
        )

    return CatalogPluginEntry.from_metadata_file(path)


__all__ = [
    "CatalogNodeEntry",
    "CatalogPluginEntry",
    "CatalogPortSpec",
    "SUPPORTED_SCHEMA_VERSIONS",
    "load_catalog_entry",
]
