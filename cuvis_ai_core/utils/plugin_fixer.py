"""Fix-it tool for the mandatory ``plugins:`` field in pipeline yamls.

A pipeline yaml that omits the top-level ``plugins:`` field is rejected
by ``LoadPipeline`` / ``restore-pipeline`` with a fix-it hint pointing
the user at this module's CLI:

    uv run suggest-plugins-fix --pipeline-path foo.yaml

The CLI runs the exact-match resolver heuristic in a non-fatal mode and
emits a patched yaml on stdout (default), a unified diff, or a
machine-readable JSON envelope. The user pipes / applies the result to
silence the error.

The reorder helper is the canonical implementation that the one-off
backfill script (``cuvis-ai/scripts/backfill_pipeline_plugins.py``)
also imports.
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from cuvis_ai_core.utils.plugin_resolver import (
    _build_catalog,
    _compute_auto_resolution,
)
from cuvis_ai_schemas.pipeline import PipelineConfig
from cuvis_ai_schemas.plugin import PluginManifest

# Canonical leading order for known top-level keys. Unknown keys (e.g.
# a future ``version`` field) are appended in their original order so the
# rewrite is non-destructive.
_KNOWN_LEADING_KEYS: tuple[str, ...] = ("metadata", "plugins", "nodes", "connections")


def reorder_pipeline_with_plugins(
    original: Mapping[str, Any],
    plugin_names: list[str],
) -> dict[str, Any]:
    """Build a new top-level pipeline-yaml dict with ``plugins:`` in canonical position.

    Known keys (``metadata`` / ``plugins`` / ``nodes`` / ``connections``)
    come first in that order. Any remaining keys from ``original`` are
    appended in their original iteration order so unknown fields survive.

    This is the single source of truth for "where does ``plugins:`` go in
    the yaml"; both the fix-it CLI and the one-off backfill script in
    cuvis-ai import this helper.
    """
    rebuilt: dict[str, Any] = {}
    for key in _KNOWN_LEADING_KEYS:
        if key == "plugins":
            rebuilt["plugins"] = plugin_names
        elif key in original:
            rebuilt[key] = original[key]
    for key, value in original.items():
        if key in _KNOWN_LEADING_KEYS:
            continue
        rebuilt[key] = value
    return rebuilt


def suggest_plugins_field(
    pipeline_config: PipelineConfig,
    raw_pipeline_dict: Mapping[str, Any],
    plugins_dirs: list[Path],
) -> tuple[list[str], dict[str, Any]]:
    """Return ``(plugin_names, patched_dict)`` for a pipeline missing ``plugins:``.

    Runs the auto-resolution heuristic and produces a patched pipeline-config
    dict ready to be serialised back to yaml. Used by the
    ``suggest-plugins-fix`` CLI and by the backfill script.

    Raises ``ValueError`` if the heuristic fails (ambiguous match, missing
    class, empty catalog) or if the pipeline already declares ``plugins:``.
    """
    if pipeline_config.plugins:
        msg = (
            "Pipeline already declares 'plugins:' — nothing to suggest. "
            "Remove the field to ask for a fresh suggestion."
        )
        raise ValueError(msg)

    catalog: dict[str, PluginManifest] = _build_catalog(plugins_dirs)
    class_names = [node.class_name for node in pipeline_config.nodes]
    resolved = _compute_auto_resolution(class_names, catalog, plugins_dirs)
    plugin_names = sorted(resolved)
    patched = reorder_pipeline_with_plugins(raw_pipeline_dict, plugin_names)
    return plugin_names, patched


def _discover_plugins_dirs(pipeline_path: Path, explicit: list[Path]) -> list[Path]:
    """Same discovery rules as ``restore-pipeline`` ('last entry wins')."""
    candidates: list[Path] = []
    for ancestor in pipeline_path.resolve().parents:
        candidate = ancestor / "configs" / "plugins"
        if candidate.is_dir():
            candidates.append(candidate)
            break
    candidates.extend(explicit)
    return candidates


def suggest_plugins_fix_cli(argv: list[str] | None = None) -> int:
    """CLI entry point for ``suggest-plugins-fix``.

    Wired into ``cuvis-ai-core/pyproject.toml`` ``[project.scripts]``.
    """
    parser = argparse.ArgumentParser(
        prog="suggest-plugins-fix",
        description=(
            "Suggest a 'plugins:' field for a pipeline yaml that is missing "
            "it. Emits a patched yaml (default), a unified diff, or JSON."
        ),
    )
    parser.add_argument(
        "--pipeline-path",
        required=True,
        type=Path,
        help="Path to the pipeline yaml to inspect.",
    )
    parser.add_argument(
        "--plugins-dir",
        action="append",
        default=[],
        type=Path,
        help=(
            "Directory containing per-plugin manifests (repeatable). If "
            "omitted, walks upward from the pipeline yaml looking for a "
            "sibling 'configs/plugins/'."
        ),
    )
    parser.add_argument(
        "--output",
        choices=("yaml", "diff", "json"),
        default="yaml",
        help=(
            "Output form. yaml: patched pipeline yaml on stdout (pipe to "
            "file). diff: unified diff fragment. json: {pipeline, plugins, "
            "diff} envelope."
        ),
    )
    args = parser.parse_args(argv)

    pipeline_path: Path = args.pipeline_path
    if not pipeline_path.is_file():
        logger.error(f"Pipeline yaml not found: {pipeline_path}")
        return 1

    raw_text = pipeline_path.read_text(encoding="utf-8")
    raw_dict = yaml.safe_load(raw_text)
    if not isinstance(raw_dict, dict):
        logger.error(f"Top-level of {pipeline_path} is not a mapping.")
        return 1

    pipeline_config = PipelineConfig.load_from_file(pipeline_path)

    if pipeline_config.plugins:
        logger.info(
            f"{pipeline_path}: 'plugins:' already declared "
            f"({pipeline_config.plugins}); nothing to do."
        )
        return 0

    plugins_dirs = _discover_plugins_dirs(pipeline_path, args.plugins_dir)
    try:
        plugin_names, patched = suggest_plugins_field(
            pipeline_config, raw_dict, plugins_dirs
        )
    except ValueError as exc:
        logger.error(str(exc))
        return 1

    patched_text = yaml.safe_dump(patched, sort_keys=False, default_flow_style=False)

    if args.output == "yaml":
        sys.stdout.write(patched_text)
    elif args.output == "diff":
        diff_lines = difflib.unified_diff(
            raw_text.splitlines(keepends=True),
            patched_text.splitlines(keepends=True),
            fromfile=str(pipeline_path),
            tofile=str(pipeline_path),
        )
        sys.stdout.writelines(diff_lines)
    else:  # json
        diff_text = "".join(
            difflib.unified_diff(
                raw_text.splitlines(keepends=True),
                patched_text.splitlines(keepends=True),
                fromfile=str(pipeline_path),
                tofile=str(pipeline_path),
            )
        )
        sys.stdout.write(
            json.dumps(
                {
                    "pipeline": str(pipeline_path),
                    "plugins": plugin_names,
                    "diff": diff_text,
                },
                indent=2,
            )
            + "\n"
        )
    return 0


if __name__ == "__main__":
    sys.exit(suggest_plugins_fix_cli())


__all__ = [
    "reorder_pipeline_with_plugins",
    "suggest_plugins_field",
    "suggest_plugins_fix_cli",
]
