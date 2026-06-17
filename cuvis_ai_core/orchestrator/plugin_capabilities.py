"""Server-side helper for a plugin's declared capabilities.

The wire shape — :class:`NodePortSpec`, :class:`PluginCapabilityEntry`,
:class:`PluginCapabilities` — lives in :mod:`cuvis_ai_schemas.plugin`. A plugin
declares its capabilities in its manifest: the ``capabilities`` list (each item a
:class:`PluginCapabilityEntry`, an FQCN ``class_name`` plus optional palette
metadata). This module only adds the server-side glue: build the capability set
from a registered manifest dump without importing the plugin package.

The parent gRPC server has no business importing plugin packages: the
orchestrator confines plugin code to child venvs. The GUI's node-palette RPC
needs to enumerate every plugin's node classes before any pipeline has been
built; the declared capabilities close that gap.
"""

from __future__ import annotations

from cuvis_ai_schemas.plugin import (
    SUPPORTED_SCHEMA_VERSIONS,
    NodePortSpec,
    PluginCapabilities,
    PluginCapabilityEntry,
    parse_plugin_manifest,
)


def load_capabilities(config_dict: dict[str, object]) -> PluginCapabilities | None:
    """Build a plugin's capability set from a stored manifest dump.

    ``config_dict`` is a manifest's ``model_dump`` (as held in
    ``SessionState.registered_plugins``). It is re-validated into a manifest and
    stripped of its install source. Returns ``None`` when the plugin declares no
    capabilities, so the caller surfaces nothing in the palette for it.

    Raises:
        ValueError: the dump fails Pydantic validation (``pydantic.ValidationError``
            is a ``ValueError`` subclass).
    """
    manifest = parse_plugin_manifest(config_dict)
    return PluginCapabilities.from_manifest(manifest)


__all__ = [
    "NodePortSpec",
    "PluginCapabilities",
    "PluginCapabilityEntry",
    "SUPPORTED_SCHEMA_VERSIONS",
    "load_capabilities",
]
