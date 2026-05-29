"""Pure resolver for pipeline plugin sets.

* If ``pipeline_config.plugins`` is set, materialise each ``PluginRef``
  into a :class:`GitPluginConfig` or :class:`LocalPluginConfig` (core-side
  types) by either looking up the bare name / catalog ref in the catalog
  or accepting the inline form directly.
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

from cuvis_ai_core.utils.plugin_config import (
    GitPluginConfig,
    LocalPluginConfig,
    PluginManifest,
)
from cuvis_ai_schemas.pipeline.config import (
    CatalogPluginRef,
    InlineGitPluginRef,
    InlineLocalPluginRef,
    PipelineConfig,
    PluginRef,
)

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
    ref: PluginRef,
    catalog: dict[str, PluginConfig],
) -> tuple[str, PluginConfig]:
    """Materialise a single ``PluginRef`` into ``(name, core-side config)``.

    Raises ``ValueError`` on unresolvable bare names or unknown forms.
    """
    if isinstance(ref, str):
        if ref not in catalog:
            msg = (
                f"Plugin '{ref}' is referenced in 'plugins:' but is not in the "
                f"catalog. Known plugins: {sorted(catalog)}"
            )
            raise ValueError(msg)
        return ref, catalog[ref]

    if isinstance(ref, CatalogPluginRef):
        if ref.name not in catalog:
            msg = (
                f"Plugin '{ref.name}' is referenced in 'plugins:' but is not in "
                f"the catalog. Known plugins: {sorted(catalog)}"
            )
            raise ValueError(msg)
        base = catalog[ref.name]
        if ref.tag is not None:
            if not isinstance(base, GitPluginConfig):
                msg = (
                    f"Plugin '{ref.name}' has a tag override in the pipeline YAML "
                    "but the catalog entry is a local-path plugin (tag does not "
                    "apply)."
                )
                raise ValueError(msg)
            base = base.model_copy(update={"tag": ref.tag})
        return ref.name, base

    if isinstance(ref, InlineGitPluginRef):
        data = ref.model_dump(exclude={"name"}, mode="json")
        return ref.name, GitPluginConfig.model_validate(data)

    if isinstance(ref, InlineLocalPluginRef):
        data = ref.model_dump(exclude={"name"}, mode="json")
        return ref.name, LocalPluginConfig.model_validate(data)

    msg = f"Unknown PluginRef shape: {type(ref).__name__}"
    raise TypeError(msg)


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
        for class_path in cfg.provides:
            provides_to_plugins[class_path].append(plugin_name)

    resolved: dict[str, PluginConfig] = {}
    for class_name in class_names:
        owners = provides_to_plugins.get(class_name, [])
        if not owners:
            msg = (
                f"class_name '{class_name}' is not provided by any plugin in "
                f"{[str(p) for p in plugins_dirs]}. Add an explicit 'plugins:' "
                "entry (inline or catalog reference) or extend the catalog."
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
    provided = {p for cfg in resolved.values() for p in cfg.provides}
    missing = [c for c in class_names if c not in provided]
    if missing:
        msg = (
            f"The following node class_names are not provided by any plugin in "
            f"the resolved set {sorted(resolved)}: {missing}. Add the providing "
            "plugin to the pipeline's 'plugins:' field."
        )
        raise ValueError(msg)


def resolve_pipeline_plugins(
    pipeline_config: PipelineConfig,
    plugins_dirs: list[Path],
) -> dict[str, PluginConfig]:
    """Resolve the plugin set a pipeline depends on.

    Pure function: input is the parsed ``PipelineConfig`` plus a list of
    candidate plugins directories; output is a ``dict`` mapping plugin
    name to its core-side ``GitPluginConfig`` / ``LocalPluginConfig``.
    No side effects — does not install, import, or mutate any global
    state.

    Resolution:

    * If ``pipeline_config.plugins`` is set, every entry is materialised
      (catalog lookup for bare names and :class:`CatalogPluginRef`;
      inline-as-is for :class:`InlineGitPluginRef` /
      :class:`InlineLocalPluginRef`). Same-name conflicts with diverging
      config raise ``ValueError``.
    * Otherwise auto-resolve from ``nodes[*].class_name`` against the
      catalog (exact match on ``provides`` entries) and emit a deprecation
      warning.

    Final coverage check: every ``class_name`` in the pipeline must be
    provided by at least one entry in the resolved set.

    Single-version-per-plugin-name: two ``PluginRef``s resolving to the
    same name with diverging ``tag`` / ``repo`` / ``path`` / ``provides``
    raise a hard error. Two-version coexistence will arrive when the
    per-env plugin cache lands separately.
    """
    catalog = _build_catalog(plugins_dirs)
    class_names = [node.class_name for node in pipeline_config.nodes]

    if not pipeline_config.plugins:
        resolved = _auto_resolve(class_names, catalog, plugins_dirs)
    else:
        resolved = {}
        for ref in pipeline_config.plugins:
            name, cfg = _ref_to_core(ref, catalog)
            if name in resolved:
                if resolved[name].model_dump() != cfg.model_dump():
                    msg = (
                        f"Plugin '{name}' is declared more than once in 'plugins:' "
                        f"with diverging configurations: {resolved[name].model_dump()} "
                        f"vs {cfg.model_dump()}. Same-name multi-version coexistence "
                        "is not supported here."
                    )
                    raise ValueError(msg)
                continue  # identical duplicate — ignore
            resolved[name] = cfg

    _validate_coverage(class_names, resolved)
    return resolved


__all__ = ["resolve_pipeline_plugins"]
