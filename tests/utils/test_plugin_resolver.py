"""Tests for cuvis_ai_core.utils.plugin_resolver."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from cuvis_ai_core.utils.plugin_config import LocalPluginConfig
from cuvis_ai_core.utils.plugin_resolver import resolve_pipeline_plugins
from cuvis_ai_schemas.pipeline import (
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
              - class_name: cuvis_ai.node.anomaly.rx_detector.RXGlobal
              - class_name: cuvis_ai.node.normalization.MinMaxNormalizer
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
              - class_name: cuvis_ai_adaclip.node.AdaCLIPDetector
        """,
    )
    return pdir


def _pipeline_with(class_names: list[str], plugins=None) -> PipelineConfig:
    return PipelineConfig(
        nodes=[
            NodeConfig(name=f"n{i}", class_name=c) for i, c in enumerate(class_names)
        ],
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


def test_declared_unknown_bare_name(plugins_dir: Path):
    pipeline = _pipeline_with(
        ["cuvis_ai_adaclip.node.AdaCLIPDetector"],
        plugins=["does_not_exist"],
    )
    with pytest.raises(ValueError, match="does_not_exist"):
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


def test_missing_plugins_field_raises_with_fix_it_hint(plugins_dir: Path):
    """Pipelines missing 'plugins:' hard-fail with a fix-it hint pointing at
    the suggest-plugins-fix CLI."""
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
            provides:
              - class_name: pkg.module.Shared
        """,
    )
    _write_manifest(
        pdir / "b.yaml",
        """
        plugins:
          b:
            path: "."
            provides:
              - class_name: pkg.module.Shared
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
            provides:
              - class_name: pkg.X
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
            provides:
              - class_name: pkg.X
        """,
    )
    pipeline = _pipeline_with(["pkg.X"], plugins=["p"])
    resolved = resolve_pipeline_plugins(pipeline, [dir1, dir2])
    assert Path(resolved["p"].path) == new_target.resolve()


def test_nonexistent_dirs_silently_skipped(plugins_dir: Path):
    """Non-existent dirs in the candidate list are ignored; real dirs still resolve."""
    pipeline = _pipeline_with(
        ["cuvis_ai_adaclip.node.AdaCLIPDetector"],
        plugins=["adaclip"],
    )
    resolved = resolve_pipeline_plugins(
        pipeline, [plugins_dir.parent / "does_not_exist", plugins_dir]
    )
    assert "adaclip" in resolved
