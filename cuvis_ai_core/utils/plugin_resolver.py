"""Pure resolver for pipeline plugin sets.

* If ``pipeline_config.plugins`` is set, look up each bare plugin name in
  the merged catalog and return its :class:`GitPluginConfig` or
  :class:`LocalPluginConfig` (core-side types). A name with no catalog
  manifest is an error.
* If ``pipeline_config.plugins`` is None/empty, the production wrapper
  ``_auto_resolve`` hard-fails with a fix-it message pointing at
  ``suggest-plugins-fix``; the pure heuristic ``_compute_auto_resolution``
  stays callable for the fix-it tool itself.

This module has **no side effects** — no install, no import, no
``NodeRegistry`` mutation. The caller decides how to materialise the
returned plugin set.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from loguru import logger

from cuvis_ai_schemas.plugin import GitPluginConfig, LocalPluginConfig, PluginManifest
from cuvis_ai_schemas.pipeline.config import PipelineConfig

PluginConfig = GitPluginConfig | LocalPluginConfig


def _build_catalog(plugins_dirs: list[Path]) -> dict[str, PluginConfig]:
    """Merge per-plugin manifests from one or more dirs into a single catalog.

    Later dirs win on plugin-name collisions; the override is logged.
    Non-existent dirs are silently skipped (covers the common case where
    a caller passes a list of candidate dirs and some don't exist).

    For :class:`LocalPluginConfig` entries, the relative ``path`` is
    resolved against the manifest file's parent directory and stored as
    an absolute path in the returned catalog. This means downstream
    consumers don't need ``manifest_dir`` context to load the plugin.
    """
    catalog: dict[str, PluginConfig] = {}
    for plugins_dir in plugins_dirs:
        if not plugins_dir.exists() or not plugins_dir.is_dir():
            logger.debug(f"Plugins dir not found or not a directory: {plugins_dir}")
            continue
        for manifest_path in sorted(plugins_dir.glob("*.yaml")):
            manifest = PluginManifest.from_yaml(manifest_path)
            manifest_dir = manifest_path.parent
            for name, cfg in manifest.plugins.items():
                if isinstance(cfg, LocalPluginConfig):
                    cfg = cfg.model_copy(
                        update={"path": str(cfg.resolve_path(manifest_dir))}
                    )
                if name in catalog:
                    logger.info(
                        f"Plugin '{name}' overridden by manifest at {manifest_path} "
                        "(was provided by an earlier plugins dir)"
                    )
                catalog[name] = cfg
    return catalog


def _ref_to_core(
    ref: str,
    catalog: dict[str, PluginConfig],
) -> tuple[str, PluginConfig]:
    """Materialise a single bare plugin name into ``(name, core-side config)``.

    Raises ``ValueError`` if the name has no manifest in the catalog.
    """
    if ref not in catalog:
        msg = (
            f"Plugin '{ref}' is referenced in 'plugins:' but is not in the "
            f"catalog. Known plugins: {sorted(catalog)}"
        )
        raise ValueError(msg)
    return ref, catalog[ref]


def _compute_auto_resolution(
    class_names: list[str],
    catalog: dict[str, PluginConfig],
    plugins_dirs: list[Path],
) -> dict[str, PluginConfig]:
    """Pure heuristic: map class_names to plugins by exact-match against provides.

    Used by both the production hard-fail wrapper (``_auto_resolve``) and the
    fix-it CLI (``plugin_fixer.suggest_plugins_field``). No logging side
    effects; raises ``ValueError`` on ambiguous matches, missing classes,
    and an empty catalog.
    """
    if not catalog:
        msg = (
            f"Pipeline omits 'plugins:' and no plugin catalog was found in "
            f"{[str(p) for p in plugins_dirs] or '[]'}. Add 'plugins:' to the "
            "pipeline YAML or set a plugins directory (--plugins-dir on the CLI, "
            "or SetSessionSearchPaths over gRPC)."
        )
        raise ValueError(msg)

    provides_to_plugins: dict[str, list[str]] = defaultdict(list)
    for plugin_name, cfg in catalog.items():
        for node in cfg.provides:
            if getattr(node, "kind", "node") != "node":
                continue  # data_module entries are selected by name, not node coverage
            provides_to_plugins[node.class_name].append(plugin_name)

    resolved: dict[str, PluginConfig] = {}
    for class_name in class_names:
        owners = provides_to_plugins.get(class_name, [])
        if not owners:
            msg = (
                f"class_name '{class_name}' is not provided by any plugin in "
                f"{[str(p) for p in plugins_dirs]}. Add an explicit 'plugins:' "
                "entry (a plugin name) or extend the catalog."
            )
            raise ValueError(msg)
        if len(owners) > 1:
            msg = (
                f"class_name '{class_name}' is ambiguous — provided by multiple "
                f"catalog plugins: {owners}. Add an explicit 'plugins:' field to "
                "the pipeline YAML to disambiguate."
            )
            raise ValueError(msg)
        plugin_name = owners[0]
        resolved[plugin_name] = catalog[plugin_name]

    return resolved


def _auto_resolve(
    class_names: list[str],
    catalog: dict[str, PluginConfig],
    plugins_dirs: list[Path],
) -> dict[str, PluginConfig]:
    """Production auto-resolve: heuristic + hard-fail with fix-it hint.

    The ``plugins:`` field is mandatory in pipeline yamls. If a pipeline
    reaches this path it has omitted the field; we run the heuristic to
    suggest names, then raise ``ValueError`` pointing the caller at
    ``suggest-plugins-fix``.
    """
    suggested = _compute_auto_resolution(class_names, catalog, plugins_dirs)
    msg = (
        "Pipeline is missing the mandatory 'plugins:' field. Run\n"
        "    uv run suggest-plugins-fix --pipeline-path <yaml>\n"
        "to generate the field and patch the yaml. Auto-resolution suggests: "
        f"{sorted(suggested)}."
    )
    raise ValueError(msg)


def _validate_coverage(
    class_names: list[str],
    resolved: dict[str, PluginConfig],
) -> None:
    """Ensure every class_name is provided by some plugin in the resolved set."""
    provided = {
        node.class_name
        for cfg in resolved.values()
        for node in cfg.provides
        if getattr(node, "kind", "node") == "node"
    }
    missing = [c for c in class_names if c not in provided]
    if missing:
        msg = (
            f"The following node class_names are not provided by any plugin in "
            f"the resolved set {sorted(resolved)}: {missing}. Add the providing "
            "plugin to the pipeline's 'plugins:' field."
        )
        raise ValueError(msg)


def _union_data_module_plugin(
    resolved: dict[str, PluginConfig],
    catalog: dict[str, PluginConfig],
    data_module: str,
) -> None:
    """Add the plugin providing ``data_module`` to ``resolved`` (in place).

    A dataloader plugin ships no node classes, so the node-coverage resolver
    never pulls it in; a run selects its data module explicitly (DataConfig /
    --data-module), so we look it up by ``data_module_name`` in the catalog and
    union its plugin into the compose set. No-op if already present or unfound.
    """
    for plugin_name, cfg in catalog.items():
        for entry in cfg.provides:
            if (
                getattr(entry, "kind", "node") == "data_module"
                and getattr(entry, "data_module_name", "") == data_module
            ):
                resolved.setdefault(plugin_name, cfg)
                return


def resolve_pipeline_plugins(
    pipeline_config: PipelineConfig,
    plugins_dirs: list[Path],
    data_module: str | None = None,
) -> dict[str, PluginConfig]:
    """Resolve the plugin set a pipeline depends on.

    Pure function: input is the parsed ``PipelineConfig`` plus a list of
    candidate plugins directories; output is a ``dict`` mapping plugin
    name to its core-side ``GitPluginConfig`` / ``LocalPluginConfig``.
    No side effects — does not install, import, or mutate any global
    state.

    Resolution:

    * If ``pipeline_config.plugins`` is set, every bare name is looked up
      in the merged catalog and materialised into its core-side config.
      Duplicate names collapse to a single entry.
    * Otherwise run the exact-match heuristic to produce a fix-it hint,
      then raise ``ValueError`` because ``plugins:`` is mandatory.

    Final coverage check: every ``class_name`` in the pipeline must be
    provided by at least one entry in the resolved set.
    """
    catalog = _build_catalog(plugins_dirs)
    class_names = [node.class_name for node in pipeline_config.nodes]

    if not pipeline_config.plugins:
        resolved = _auto_resolve(class_names, catalog, plugins_dirs)
    else:
        resolved = {}
        for ref in pipeline_config.plugins:
            name, cfg = _ref_to_core(ref, catalog)
            resolved[name] = cfg  # duplicate names collapse to one entry

    _validate_coverage(class_names, resolved)
    # Union the data-module plugin (selected by DataConfig.data_module): it ships
    # no node classes, so coverage never pulls it in on its own.
    if data_module:
        _union_data_module_plugin(resolved, catalog, data_module)
    return resolved


__all__ = ["resolve_pipeline_plugins"]
