"""Tests for cuvis_ai_core.utils.plugin_resolver."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from cuvis_ai_core.utils.plugin_config import GitPluginConfig, LocalPluginConfig
from cuvis_ai_core.utils.plugin_resolver import resolve_pipeline_plugins
from cuvis_ai_schemas.pipeline import (
    CatalogPluginRef,
    InlineGitPluginRef,
    InlineLocalPluginRef,
    NodeConfig,
    PipelineConfig,
)


def _write_manifest(path: Path, body: str) -> None:
    path.write_text(dedent(body), encoding="utf-8")


@pytest.fixture
def plugins_dir(tmp_path: Path) -> Path:
    """A plugins/ dir holding two per-plugin manifests."""
    pdir = tmp_path / "plugins"
    pdir.mkdir()
    _write_manifest(
        pdir / "cuvis_ai_builtin.yaml",
        """
        plugins:
          cuvis_ai_builtin:
            path: "../.."
            provides:
              - cuvis_ai.node.anomaly.rx_detector.RXGlobal
              - cuvis_ai.node.normalization.MinMaxNormalizer
        """,
    )
    _write_manifest(
        pdir / "adaclip.yaml",
        """
        plugins:
          adaclip:
            repo: "https://github.com/example/adaclip.git"
            tag: "v0.1.2"
            provides:
              - cuvis_ai_adaclip.node.AdaCLIPDetector
        """,
    )
    return pdir


def _pipeline_with(class_names: list[str], plugins=None) -> PipelineConfig:
    return PipelineConfig(
        nodes=[NodeConfig(name=f"n{i}", class_name=c) for i, c in enumerate(class_names)],
        connections=[],
        plugins=plugins,
    )


# ---------------------------------------------------------------------------
# Declared plugins (Phase 1)
# ---------------------------------------------------------------------------


def test_declared_bare_name(plugins_dir: Path):
    """Bare string in plugins: resolves against the catalog."""
    pipeline = _pipeline_with(
        ["cuvis_ai.node.anomaly.rx_detector.RXGlobal"],
        plugins=["cuvis_ai_builtin"],
    )
    resolved = resolve_pipeline_plugins(pipeline, [plugins_dir])
    assert set(resolved) == {"cuvis_ai_builtin"}
    assert isinstance(resolved["cuvis_ai_builtin"], LocalPluginConfig)


def test_declared_catalog_ref_with_tag_override(plugins_dir: Path):
    """CatalogPluginRef with tag overrides the catalog entry's tag."""
    pipeline = _pipeline_with(
        ["cuvis_ai_adaclip.node.AdaCLIPDetector"],
        plugins=[CatalogPluginRef(name="adaclip", tag="v0.9.9")],
    )
    resolved = resolve_pipeline_plugins(pipeline, [plugins_dir])
    assert isinstance(resolved["adaclip"], GitPluginConfig)
    assert resolved["adaclip"].tag == "v0.9.9"


def test_declared_inline_git_no_catalog_needed():
    """Inline git refs do not require a catalog dir."""
    inline = InlineGitPluginRef(
        name="private",
        repo="https://github.com/example/private.git",
        tag="v0.0.1",
        provides=["pkg.module.Foo"],
    )
    pipeline = _pipeline_with(["pkg.module.Foo"], plugins=[inline])
    resolved = resolve_pipeline_plugins(pipeline, [])
    assert "private" in resolved
    assert resolved["private"].repo == "https://github.com/example/private.git"


def test_declared_inline_local_no_catalog_needed(tmp_path: Path):
    """Inline local refs do not require a catalog dir."""
    inline = InlineLocalPluginRef(
        name="dev_plug",
        path=str(tmp_path),
        provides=["pkg.module.Bar"],
    )
    pipeline = _pipeline_with(["pkg.module.Bar"], plugins=[inline])
    resolved = resolve_pipeline_plugins(pipeline, [])
    assert "dev_plug" in resolved
    assert resolved["dev_plug"].path == str(tmp_path)


def test_declared_unknown_bare_name(plugins_dir: Path):
    pipeline = _pipeline_with(
        ["cuvis_ai_adaclip.node.AdaCLIPDetector"],
        plugins=["does_not_exist"],
    )
    with pytest.raises(ValueError, match="does_not_exist"):
        resolve_pipeline_plugins(pipeline, [plugins_dir])


def test_declared_same_name_conflict(plugins_dir: Path):
    """Same-name PluginRefs with diverging config raise."""
    pipeline = _pipeline_with(
        ["cuvis_ai_adaclip.node.AdaCLIPDetector"],
        plugins=[
            "adaclip",
            CatalogPluginRef(name="adaclip", tag="v0.9.9"),
        ],
    )
    with pytest.raises(ValueError, match="adaclip.*diverging"):
        resolve_pipeline_plugins(pipeline, [plugins_dir])


def test_declared_same_name_identical_duplicate_ok(plugins_dir: Path):
    """Identical duplicates are silently deduped."""
    pipeline = _pipeline_with(
        ["cuvis_ai_adaclip.node.AdaCLIPDetector"],
        plugins=["adaclip", "adaclip"],
    )
    resolved = resolve_pipeline_plugins(pipeline, [plugins_dir])
    assert set(resolved) == {"adaclip"}


def test_declared_missing_class_in_resolved_set(plugins_dir: Path):
    """If plugins: forgets to include a class, the coverage check fails."""
    pipeline = _pipeline_with(
        [
            "cuvis_ai.node.anomaly.rx_detector.RXGlobal",  # in cuvis_ai_builtin
            "cuvis_ai_adaclip.node.AdaCLIPDetector",  # in adaclip — missing from plugins:
        ],
        plugins=["cuvis_ai_builtin"],
    )
    with pytest.raises(ValueError, match="AdaCLIPDetector"):
        resolve_pipeline_plugins(pipeline, [plugins_dir])


# ---------------------------------------------------------------------------
# Auto-resolution (Phase 2)
# ---------------------------------------------------------------------------


def test_phase4_missing_plugins_field_raises(plugins_dir: Path):
    """ALL-5349 Phase 4: pipelines missing 'plugins:' now hard-fail with a
    fix-it hint pointing at the suggest-plugins-fix CLI. Phase 1+2's
    'warn-and-continue' path is gone."""
    pipeline = _pipeline_with(
        [
            "cuvis_ai.node.anomaly.rx_detector.RXGlobal",
            "cuvis_ai.node.normalization.MinMaxNormalizer",
        ],
    )
    with pytest.raises(ValueError) as excinfo:
        resolve_pipeline_plugins(pipeline, [plugins_dir])
    msg = str(excinfo.value)
    assert "mandatory 'plugins:' field" in msg
    assert "suggest-plugins-fix" in msg
    # The hint surfaces the auto-resolution suggestion so users can paste it back.
    assert "cuvis_ai_builtin" in msg


def test_auto_resolve_unknown_class(plugins_dir: Path):
    pipeline = _pipeline_with(["pkg.unknown.Class"])
    with pytest.raises(ValueError, match="pkg.unknown.Class"):
        resolve_pipeline_plugins(pipeline, [plugins_dir])


def test_auto_resolve_ambiguous(tmp_path: Path):
    """A class provided by two plugins raises."""
    pdir = tmp_path / "plugins"
    pdir.mkdir()
    _write_manifest(
        pdir / "a.yaml",
        """
        plugins:
          a:
            path: "."
            provides: [pkg.module.Shared]
        """,
    )
    _write_manifest(
        pdir / "b.yaml",
        """
        plugins:
          b:
            path: "."
            provides: [pkg.module.Shared]
        """,
    )
    pipeline = _pipeline_with(["pkg.module.Shared"])
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_pipeline_plugins(pipeline, [pdir])


def test_auto_resolve_empty_catalog():
    """No catalog AND no declared plugins → explicit error."""
    pipeline = _pipeline_with(["pkg.module.Foo"])
    with pytest.raises(ValueError, match="no plugin catalog"):
        resolve_pipeline_plugins(pipeline, [])


# ---------------------------------------------------------------------------
# Catalog merging across multiple dirs
# ---------------------------------------------------------------------------


def test_multi_dir_override_logged(tmp_path: Path):
    """Later plugins dirs override earlier ones on plugin-name collision."""
    old_target = tmp_path / "old_plugin"
    old_target.mkdir()
    new_target = tmp_path / "new_plugin"
    new_target.mkdir()
    dir1 = tmp_path / "dir1"
    dir1.mkdir()
    _write_manifest(
        dir1 / "p.yaml",
        f"""
        plugins:
          p:
            path: {old_target.as_posix()!r}
            provides: [pkg.X]
        """,
    )
    dir2 = tmp_path / "dir2"
    dir2.mkdir()
    _write_manifest(
        dir2 / "p.yaml",
        f"""
        plugins:
          p:
            path: {new_target.as_posix()!r}
            provides: [pkg.X]
        """,
    )
    pipeline = _pipeline_with(["pkg.X"], plugins=["p"])
    resolved = resolve_pipeline_plugins(pipeline, [dir1, dir2])
    assert Path(resolved["p"].path) == new_target.resolve()


def test_nonexistent_dirs_silently_skipped(tmp_path: Path):
    """Non-existent dirs in the candidate list are ignored."""
    pipeline = _pipeline_with(
        ["pkg.module.Foo"],
        plugins=[
            InlineGitPluginRef(
                name="p",
                repo="https://x/y.git",
                tag="v1",
                provides=["pkg.module.Foo"],
            )
        ],
    )
    resolved = resolve_pipeline_plugins(pipeline, [tmp_path / "does_not_exist"])
    assert "p" in resolved
