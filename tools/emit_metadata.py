"""Emit a static ``metadata.json`` for a plugin from its node class specs.

Plugin release CI calls this once a release tag is cut, against the
plugin manifest YAML that ships in the host repo (or in the plugin's
own configs). The script imports every class in the manifest's
``provides:`` list, walks ``INPUT_SPECS`` / ``OUTPUT_SPECS`` /
``get_category()`` / ``get_tags()`` / ``get_icon_name()`` / docstring,
and writes a :class:`cuvis_ai_schemas.catalog.CatalogPluginEntry` to
disk.

The emitted JSON is the single source of truth for the GUI's node
palette: at server boot the parent reads it via the static catalog
loader, so the plugin's Python modules never get imported on the
parent side.

Usage::

    uv run python -m tools.emit_metadata \\
        --manifest configs/plugins/cuvis_ai_builtin.yaml \\
        --plugin cuvis_ai_builtin \\
        --output configs/plugins/cuvis_ai_builtin.metadata.json

CI guard (in the plugin repo's release workflow)::

    uv run python -m tools.emit_metadata --manifest ... --plugin ... --output build/metadata.json
    diff build/metadata.json configs/plugins/<name>.metadata.json
    # non-zero exit → the committed JSON has drifted from the live specs
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from loguru import logger

from cuvis_ai_core.utils.icon_helpers import get_node_icon
from cuvis_ai_schemas.catalog import (
    SUPPORTED_SCHEMA_VERSIONS,
    CatalogNodeEntry,
    CatalogPluginEntry,
    CatalogPortSpec,
)
from cuvis_ai_schemas.enums import NodeCategory
from cuvis_ai_schemas.pipeline import PortSpec


_TORCH_DTYPE_NAMES = {
    torch.float32: "float32",
    torch.float64: "float64",
    torch.float16: "float16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.uint8: "uint8",
    torch.uint16: "uint16",
    torch.bool: "bool",
}


def _dtype_to_string(dtype: Any) -> str:
    """Convert a node-class dtype to a NumPy-style string ('float32', ...).

    ``INPUT_SPECS`` / ``OUTPUT_SPECS`` accept a wide range of dtype values
    (torch.dtype, np.dtype instance, NumPy scalar class, torch.Tensor as
    a generic marker). Catalog JSON normalises everything to the string
    form ``np.dtype(...)`` accepts. Unknown / generic dtypes serialise as
    an empty string so the server can fall back to UNSPECIFIED.
    """
    if isinstance(dtype, torch.dtype):
        return _TORCH_DTYPE_NAMES.get(dtype, str(dtype).split(".", 1)[-1])
    if isinstance(dtype, np.dtype):
        return dtype.name
    if isinstance(dtype, type) and issubclass(dtype, np.generic):
        return np.dtype(dtype).name
    if dtype is torch.Tensor or dtype is None:
        return ""
    return ""


def _shape_to_int_list(shape: tuple) -> list[int]:
    """Replace symbolic-dim strings with -1; keep ints as-is."""
    out: list[int] = []
    for dim in shape:
        if isinstance(dim, int):
            out.append(dim)
        else:
            out.append(-1)
    return out


def _port_spec_to_catalog(spec: PortSpec) -> CatalogPortSpec:
    return CatalogPortSpec(
        dtype=_dtype_to_string(spec.dtype),
        shape=_shape_to_int_list(tuple(spec.shape)),
        optional=spec.optional,
        description=spec.description,
    )


def _specs_map_to_catalog(
    specs_dict: dict | None,
) -> dict[str, list[CatalogPortSpec]]:
    if not specs_dict:
        return {}
    out: dict[str, list[CatalogPortSpec]] = {}
    for port_name, spec in specs_dict.items():
        if isinstance(spec, list):
            out[port_name] = [_port_spec_to_catalog(s) for s in spec]
        else:
            out[port_name] = [_port_spec_to_catalog(spec)]
    return out


def _resolve_package_root(node_class: type) -> Path | None:
    """Walk up from the class's source file to find an ``assets/node_icons/`` folder."""
    try:
        source_path = Path(inspect.getfile(node_class)).resolve()
    except (TypeError, OSError):
        return None
    for ancestor in (source_path, *source_path.parents)[:9]:
        if (ancestor / "assets" / "node_icons").is_dir():
            return ancestor
    return None


def _category_for(node_class: type, class_name: str) -> NodeCategory:
    try:
        return node_class.get_category()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"{class_name}.get_category() failed: {exc}")
        return NodeCategory.UNSPECIFIED


def _tags_for(node_class: type, class_name: str) -> list[str]:
    try:
        return [t.value for t in node_class.get_tags()]
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"{class_name}.get_tags() failed: {exc}")
        return []


def _icon_svg_for(node_class: type, class_name: str, category: NodeCategory) -> str:
    try:
        icon_name = node_class.get_icon_name()
    except Exception:
        icon_name = None
    try:
        svg_bytes = get_node_icon(
            class_name=class_name,
            icon_name=icon_name,
            category=category,
            package_root=_resolve_package_root(node_class),
        )
    except Exception as exc:
        logger.warning(f"{class_name} icon resolution failed: {exc}")
        return ""
    return svg_bytes.decode("utf-8") if svg_bytes else ""


def _doc_summary_for(node_class: type) -> str:
    doc = inspect.getdoc(node_class)
    if not doc:
        return ""
    # First paragraph of the docstring, single-line.
    first_para = doc.split("\n\n", 1)[0]
    return " ".join(first_para.split())


def _import_class(fqcn: str) -> type:
    module_path, _, class_name = fqcn.rpartition(".")
    if not module_path:
        raise ValueError(f"Invalid FQCN '{fqcn}' — must be 'pkg.module.Class'")
    module = importlib.import_module(module_path)
    try:
        cls = getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(
            f"Module '{module_path}' has no attribute '{class_name}'"
        ) from exc
    if not isinstance(cls, type):
        raise TypeError(f"'{fqcn}' resolves to {type(cls).__name__}, not a class")
    return cls


def _node_entry(fqcn: str) -> CatalogNodeEntry:
    node_class = _import_class(fqcn)
    class_name = node_class.__name__
    category = _category_for(node_class, class_name)
    return CatalogNodeEntry(
        class_name=class_name,
        full_path=fqcn,
        category=category.value,
        tags=_tags_for(node_class, class_name),
        icon_svg=_icon_svg_for(node_class, class_name, category),
        input_specs=_specs_map_to_catalog(getattr(node_class, "INPUT_SPECS", {})),
        output_specs=_specs_map_to_catalog(getattr(node_class, "OUTPUT_SPECS", {})),
        doc_summary=_doc_summary_for(node_class),
    )


def emit(
    manifest_path: Path,
    plugin_name: str,
    output_path: Path,
    *,
    plugin_version: str = "",
) -> CatalogPluginEntry:
    """Build and write the metadata.json for ``plugin_name`` from ``manifest_path``."""
    manifest_data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    plugins_block = (manifest_data or {}).get("plugins", {})
    if plugin_name not in plugins_block:
        raise KeyError(
            f"Plugin '{plugin_name}' not found in manifest {manifest_path}. "
            f"Known plugins: {sorted(plugins_block)}"
        )
    provides = plugins_block[plugin_name].get("provides", [])
    if not provides:
        raise ValueError(
            f"Plugin '{plugin_name}' has empty 'provides:' in {manifest_path}"
        )

    nodes: list[CatalogNodeEntry] = []
    failures: list[tuple[str, str]] = []
    for fqcn in provides:
        try:
            nodes.append(_node_entry(fqcn))
        except Exception as exc:
            failures.append((fqcn, f"{type(exc).__name__}: {exc}"))
            logger.error(f"Skipping '{fqcn}': {exc}")

    if failures:
        logger.warning(f"{len(failures)}/{len(provides)} classes failed; check log above")
        if len(failures) == len(provides):
            raise RuntimeError("All provided classes failed to import; refusing to write empty catalog")

    entry = CatalogPluginEntry(
        schema_version=SUPPORTED_SCHEMA_VERSIONS[-1],
        plugin_name=plugin_name,
        plugin_version=plugin_version,
        nodes=nodes,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(json.loads(entry.model_dump_json()), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    logger.info(
        f"Wrote {output_path} — {len(nodes)} node(s), "
        f"{len(failures)} failure(s)"
    )
    return entry


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Emit a plugin metadata.json")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--plugin", required=True, help="Plugin name (key in 'plugins:')")
    parser.add_argument("--output", type=Path, help="Output JSON path (default: <manifest_dir>/<plugin>.metadata.json)")
    parser.add_argument("--plugin-version", default="", help="Plugin version string")
    args = parser.parse_args(argv)

    output_path = args.output or args.manifest.with_name(f"{args.plugin}.metadata.json")
    try:
        emit(args.manifest, args.plugin, output_path, plugin_version=args.plugin_version)
    except (KeyError, ValueError, RuntimeError) as exc:
        logger.error(f"Emit failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(_main())
