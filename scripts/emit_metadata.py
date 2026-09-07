"""Regenerate a plugin's inline node metadata inside its manifest YAML.

A plugin manifest's ``capabilities`` list *is* its node catalog: every
item carries an FQCN ``class_name`` plus, for nodes, optional palette
metadata (port specs, category, tags, icon, doc summary). This tool
imports every node class named in ``capabilities`` and rewrites each
node entry with freshly introspected metadata, in place, preserving the
manifest's comments and structure via a ruamel round-trip. ``data_module``
entries carry no palette metadata and are left untouched.

It also projects the plugin's model-weight declarations into the manifest's
``weights:`` block: the ``WEIGHTS`` tuple of ``<package>.weights`` (the import
root of the first capability, or ``--weights-module pkg.mod``), a
side-effect-free module of ``PluginWeightEntry`` rows. Every ``selected_by`` and
``explicit_path_hparams`` value must be a constructor parameter of one of the
plugin's node classes, so a manifest can never name a hyper-parameter the
nodes do not have. A plugin without a weights module leaves ``weights`` alone.

One yaml file is one plugin (a bare manifest: ``name`` + source +
``capabilities``), so there is no plugin selector. Release CI runs this
once a tag is cut; the ``--check`` mode is the drift guard (non-zero exit
when the committed metadata no longer matches the live node specs or the
live weight declarations).

Usage::

    uv run python -m scripts.emit_metadata \\
        --manifest configs/plugins/cuvis_ai_builtin.yaml

CI guard (in the plugin repo's release workflow)::

    uv run python -m scripts.emit_metadata --manifest ... --check
    # non-zero exit → the committed metadata has drifted from the live specs
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from ruamel.yaml import YAML

from cuvis_ai_core.utils.icon_helpers import get_node_icon
from cuvis_ai_schemas.enums import NodeCategory
from cuvis_ai_schemas.pipeline import PortSpec
from cuvis_ai_schemas.plugin import (
    NodePortSpec,
    PluginCapabilityEntry,
    PluginWeightEntry,
)


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
    a generic marker). Capability entries normalise everything to the string
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


def _port_spec_to_node(spec: PortSpec) -> NodePortSpec:
    return NodePortSpec(
        dtype=_dtype_to_string(spec.dtype),
        shape=_shape_to_int_list(tuple(spec.shape)),
        optional=spec.optional,
        description=spec.description,
        variadic=getattr(spec, "variadic", False),
    )


def _specs_map_to_node(
    specs_dict: dict | None,
) -> dict[str, NodePortSpec]:
    if not specs_dict:
        return {}
    return {
        port_name: _port_spec_to_node(spec) for port_name, spec in specs_dict.items()
    }


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
        # Sort for a deterministic catalog: get_tags() returns an unordered
        # set, so without this the emitted order varies per process and the
        # --check drift guard would flag spurious staleness on every run.
        return sorted(t.value for t in node_class.get_tags())
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


def _node_entry(fqcn: str) -> PluginCapabilityEntry:
    """Introspect one node class into a PluginCapabilityEntry (class_name = FQCN)."""
    return _entry_from_class(fqcn, _import_class(fqcn))


def _entry_from_class(fqcn: str, node_class: type) -> PluginCapabilityEntry:
    """Introspect an already imported node class into its capability entry."""
    short_name = node_class.__name__
    category = _category_for(node_class, short_name)
    return PluginCapabilityEntry(
        class_name=fqcn,
        category=category.value,
        tags=_tags_for(node_class, short_name),
        icon_svg=_icon_svg_for(node_class, short_name, category),
        input_specs=_specs_map_to_node(getattr(node_class, "INPUT_SPECS", {})),
        output_specs=_specs_map_to_node(getattr(node_class, "OUTPUT_SPECS", {})),
        doc_summary=_doc_summary_for(node_class),
    )


def _plainify(value: Any) -> Any:
    """ruamel CommentedMap/Seq/ScalarString → plain dict/list/str."""
    return json.loads(json.dumps(value))


def _is_node_entry(item: Any) -> bool:
    """A capability item is a node unless it explicitly declares another kind."""
    kind = item.get("kind", "node") if hasattr(item, "get") else "node"
    return (kind or "node") == "node"


def _entry_to_manifest_dict(entry: PluginCapabilityEntry) -> dict:
    """Capability entry → manifest 'capabilities' dict, dropping empty/default fields.

    ``class_name`` is always present; the rest is emitted only when it
    carries information, so minimal nodes stay terse and rich nodes stay
    complete. The dropped defaults round-trip back to the same model
    (``PluginCapabilityEntry`` fills them), so ``--check`` stays exact.
    """

    def _drop_default_variadic(specs: dict) -> dict:
        # `variadic` is an opt-in input flag; only emit it when True.
        return {
            port: {
                k: v for k, v in spec.items() if not (k == "variadic" and v is False)
            }
            for port, spec in specs.items()
        }

    full = json.loads(entry.model_dump_json())
    out: dict[str, Any] = {"class_name": full["class_name"]}
    if full.get("category") and full["category"] != "unspecified":
        out["category"] = full["category"]
    if full.get("tags"):
        out["tags"] = full["tags"]
    if full.get("icon_svg"):
        out["icon_svg"] = full["icon_svg"]
    if full.get("input_specs"):
        out["input_specs"] = _drop_default_variadic(full["input_specs"])
    if full.get("output_specs"):
        out["output_specs"] = _drop_default_variadic(full["output_specs"])
    if full.get("doc_summary"):
        out["doc_summary"] = full["doc_summary"]
    return out


# Fields of a PluginWeightEntry that the manifest omits when they hold the default;
# PluginWeightEntry fills them back in, so --check compares equal models.
_WEIGHT_DEFAULTS: dict[str, Any] = {
    "summary": "",
    "kind": "weights",
    "aux_files": [],
    "license_file": None,
    "aliases": [],
    "selected_by": None,
    "default": False,
    "explicit_path_hparams": [],
    "description": "",
}


def _weight_to_manifest_dict(entry: PluginWeightEntry) -> dict[str, Any]:
    """Weight entry → manifest ``weights`` dict, dropping fields at their default."""
    full = json.loads(entry.model_dump_json())
    return {
        key: value
        for key, value in full.items()
        if key not in _WEIGHT_DEFAULTS or value != _WEIGHT_DEFAULTS[key]
    }


def _weights_module_name(
    capabilities: Sequence[Any], override: str | None
) -> str | None:
    """``--weights-module`` if given, else ``<import root of the first capability>.weights``."""
    if override:
        return override
    for item in capabilities:
        fqcn = item.get("class_name") if hasattr(item, "get") else None
        if fqcn:
            return f"{fqcn.split('.', 1)[0]}.weights"
    return None


def _load_weights(module_name: str) -> tuple[PluginWeightEntry, ...] | None:
    """The ``WEIGHTS`` tuple of ``module_name``; ``None`` when the plugin has no such module."""
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            return None  # the plugin declares no weights
        raise
    weights = getattr(module, "WEIGHTS", None)
    if weights is None:
        raise ValueError(f"{module_name} exists but defines no WEIGHTS tuple")
    entries = tuple(weights)
    for entry in entries:
        if not isinstance(entry, PluginWeightEntry):
            raise TypeError(
                f"{module_name}.WEIGHTS must hold PluginWeightEntry rows, got "
                f"{type(entry).__name__}"
            )
    return entries


def _constructor_params(node_classes: Iterable[type]) -> set[str]:
    """Names every node class of the plugin accepts in its constructor."""
    names: set[str] = set()
    for node_class in node_classes:
        try:
            signature = inspect.signature(node_class.__init__)
        except (
            TypeError,
            ValueError,
        ):  # pragma: no cover - builtins without signatures
            continue
        names.update(
            param.name
            for param in signature.parameters.values()
            if param.kind
            in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            and param.name != "self"
        )
    return names


def _check_weight_references(
    entries: Iterable[PluginWeightEntry], params: set[str], module_name: str
) -> None:
    """Every selector / explicit-path hyper-parameter must exist on some node class."""
    for entry in entries:
        referenced = list(entry.explicit_path_hparams)
        if entry.selected_by is not None:
            referenced.insert(0, entry.selected_by)
        for hparam in referenced:
            if hparam not in params:
                raise ValueError(
                    f"{module_name}: weight '{entry.name}' names hyper-parameter "
                    f"'{hparam}', which no node class of this plugin accepts in its "
                    "constructor"
                )


def _describe_weight_drift(
    committed: Sequence[PluginWeightEntry] | None, fresh: Sequence[PluginWeightEntry]
) -> str:
    """One line naming what differs between the committed and the live weights."""
    if committed is None:
        return "the manifest has no weights block"
    by_name_committed = {e.name: e for e in committed}
    by_name_fresh = {e.name: e for e in fresh}
    notes: list[str] = []
    for name in sorted(set(by_name_committed) - set(by_name_fresh)):
        notes.append(f"'{name}' only in the manifest")
    for name in sorted(set(by_name_fresh) - set(by_name_committed)):
        notes.append(f"'{name}' only in WEIGHTS")
    for name in sorted(set(by_name_committed) & set(by_name_fresh)):
        old = by_name_committed[name].model_dump(mode="json")
        new = by_name_fresh[name].model_dump(mode="json")
        fields = sorted(key for key in new if old.get(key) != new[key])
        if fields:
            notes.append(f"'{name}': {', '.join(fields)}")
    if not notes and [e.name for e in committed] != [e.name for e in fresh]:
        notes.append("row order")
    return "; ".join(notes) or "no difference"


def _yaml() -> YAML:
    y = YAML()
    y.preserve_quotes = True
    # None = leaf (scalar-only) collections like `shape` / `tags` render inline
    # (`shape: [-1, -1, -1, 3]`); maps and the capability list stay block-style.
    y.default_flow_style = None
    y.width = 1_000_000  # never line-wrap long scalars (e.g. icon_svg)
    y.indent(mapping=2, sequence=2, offset=0)
    return y


def emit(
    manifest_path: Path, *, check: bool = False, weights_module: str | None = None
) -> bool:
    """Regenerate (or, with ``check``, verify) a manifest's node metadata and weights.

    Reads each node entry's ``class_name`` (FQCN) from the bare manifest's
    ``capabilities`` list, imports the class, and rewrites the entry with
    freshly introspected palette metadata, preserving the manifest's
    comments + structure. ``data_module`` entries are left untouched. Then
    projects the plugin's ``WEIGHTS`` declarations (``weights_module``, or the
    ``<package>.weights`` convention) into the ``weights:`` block, after
    checking that every selector and explicit-path hyper-parameter exists on
    one of the imported node classes; a plugin without a weights module
    leaves the block untouched. Returns True on success (or, in check mode,
    when the committed metadata is already in sync); False when ``check``
    finds drift in either block.
    """
    yaml_rt = _yaml()
    doc = yaml_rt.load(manifest_path.read_text(encoding="utf-8"))

    capabilities = (doc or {}).get("capabilities")
    if not capabilities:
        raise ValueError(f"Manifest {manifest_path} has empty 'capabilities:'")

    node_items = [item for item in capabilities if _is_node_entry(item)]

    fresh_by_fqcn: dict[str, PluginCapabilityEntry] = {}
    node_classes: dict[str, type] = {}
    failures: list[tuple[str, str]] = []
    for item in node_items:
        fqcn = item.get("class_name") if hasattr(item, "get") else None
        if not fqcn:
            raise ValueError(
                f"A 'capabilities' entry is missing 'class_name' in {manifest_path}"
            )
        try:
            node_classes[fqcn] = _import_class(fqcn)
            fresh_by_fqcn[fqcn] = _entry_from_class(fqcn, node_classes[fqcn])
        except Exception as exc:
            failures.append((fqcn, f"{type(exc).__name__}: {exc}"))
            logger.error(f"Skipping '{fqcn}': {exc}")

    if node_items and failures and len(failures) == len(node_items):
        raise RuntimeError("All node classes failed to import; refusing to rewrite")
    if failures:
        logger.warning(
            f"{len(failures)}/{len(node_items)} node classes failed; check log above"
        )

    # Drift check: committed node entry (validated) vs freshly introspected one.
    in_sync = True
    for item in node_items:
        fqcn = item["class_name"]
        fresh = fresh_by_fqcn.get(fqcn)
        if fresh is None:
            continue  # import failed — leave the committed entry untouched
        committed = PluginCapabilityEntry.model_validate(_plainify(item))
        if committed != fresh:
            in_sync = False

    # Weights: the plugin's declaration is the source; the manifest is its projection.
    module_name = _weights_module_name(capabilities, weights_module)
    fresh_weights = _load_weights(module_name) if module_name else None
    committed_weights: list[PluginWeightEntry] | None = None
    weights_in_sync = True
    if fresh_weights is not None:
        assert module_name is not None
        _check_weight_references(
            fresh_weights, _constructor_params(node_classes.values()), module_name
        )
        raw_committed = doc.get("weights")
        if raw_committed is not None:
            committed_weights = [
                PluginWeightEntry.model_validate(_plainify(item))
                for item in raw_committed
            ]
        weights_in_sync = committed_weights == list(fresh_weights)
        if not weights_in_sync:
            logger.info(
                f"{manifest_path} weights differ from {module_name}.WEIGHTS: "
                + _describe_weight_drift(committed_weights, fresh_weights)
            )

    if check:
        if in_sync and weights_in_sync:
            logger.info(f"{manifest_path} capabilities and weights are in sync")
        else:
            stale = " and ".join(
                part
                for part, ok in (
                    ("capabilities", in_sync),
                    ("weights", weights_in_sync),
                )
                if not ok
            )
            logger.error(
                f"{manifest_path} {stale} are stale — "
                "re-run emit_metadata without --check to regenerate"
            )
        return in_sync and weights_in_sync

    new_capabilities = [
        _entry_to_manifest_dict(fresh_by_fqcn[item["class_name"]])
        if _is_node_entry(item) and item["class_name"] in fresh_by_fqcn
        else _plainify(item)
        for item in capabilities
    ]
    doc["capabilities"] = new_capabilities

    if fresh_weights is not None:
        new_weights = [_weight_to_manifest_dict(entry) for entry in fresh_weights]
        if "weights" in doc:
            doc["weights"] = new_weights
        else:
            # Right after capabilities, where a reader expects it.
            position = list(doc.keys()).index("capabilities") + 1
            doc.insert(position, "weights", new_weights)

    with manifest_path.open("w", encoding="utf-8") as f:
        yaml_rt.dump(doc, f)
    weights_note = (
        f", {len(fresh_weights)} weight(s)" if fresh_weights is not None else ""
    )
    logger.info(
        f"Updated {manifest_path}: {len(fresh_by_fqcn)} node(s){weights_note}, "
        f"{len(failures)} failure(s)"
    )
    return True


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate a plugin manifest's inline node metadata"
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the committed metadata matches live specs; do not write.",
    )
    parser.add_argument(
        "--weights-module",
        default=None,
        help=(
            "Module holding the plugin's WEIGHTS tuple (default: "
            "<import root of the first capability>.weights)."
        ),
    )
    args = parser.parse_args(argv)

    try:
        ok = emit(args.manifest, check=args.check, weights_module=args.weights_module)
    except (KeyError, ValueError, TypeError, RuntimeError) as exc:
        logger.error(f"Emit failed: {exc}")
        return 1
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_main())
