"""Tests for the data-module pip-extras path through the composer + resolver."""

from __future__ import annotations

from pathlib import Path

from cuvis_ai_schemas.plugin import LocalPluginManifest, PluginCapabilityEntry

from cuvis_ai_core.orchestrator.cache_key import (
    CoreSource,
    ResolvedGitPlugin,
    ResolvedLocalPlugin,
)
from cuvis_ai_core.orchestrator.runtime_project import (
    _active_extras,
    _plugin_source_entry,
    build_runtime_pyproject,
)
from cuvis_ai_core.utils.plugin_resolver import _union_data_module_plugin


def _dataloader_cfg() -> LocalPluginManifest:
    return LocalPluginManifest(
        name="cuvis_ai_dataloader",
        path=".",
        package_name="cuvis-ai-dataloader",
        capabilities=[
            PluginCapabilityEntry(
                class_name="cuvis_ai_dataloader.data.datamodule_cu3s.Cu3sDataModule",
                kind="data_module",
                data_module_name="cu3s",
                extras=["cu3s", "coco"],
            ),
            PluginCapabilityEntry(
                class_name="cuvis_ai_dataloader.data.datamodule_tiff_paired.TiffPairedDataModule",
                kind="data_module",
                data_module_name="tiff_paired",
                extras=["tiff"],
            ),
        ],
    )


def test_active_extras_scopes_to_selected_module():
    cfg = _dataloader_cfg()
    assert _active_extras(cfg, "cu3s") == ("coco", "cu3s")  # sorted
    assert _active_extras(cfg, "tiff_paired") == ("tiff",)
    assert _active_extras(cfg, None) == ()
    assert _active_extras(cfg, "envi") == ()  # unknown -> no extras


def test_plugin_source_entry_emits_extras():
    p = ResolvedLocalPlugin(
        name="dl",
        path=Path("/x"),
        package_name="cuvis-ai-dataloader",
        pyproject_sha256="abc",
        git_head=None,
        dirty=False,
        extras=("coco", "cu3s"),
    )
    dependency_string, source_key, entry = _plugin_source_entry(p)
    assert dependency_string == "cuvis-ai-dataloader[coco,cu3s]"
    assert source_key == "cuvis-ai-dataloader"  # uv.sources keyed by the bare name
    assert entry["editable"] is True


def test_build_runtime_pyproject_includes_extras():
    p = ResolvedLocalPlugin(
        name="dl",
        path=Path("/x"),
        package_name="cuvis-ai-dataloader",
        pyproject_sha256="abc",
        git_head=None,
        dirty=False,
        extras=("coco", "cu3s"),
    )
    toml = build_runtime_pyproject(
        core_source=CoreSource(kind="pypi", identity="cuvis-ai-core==0.7.3"),
        plugins=(p,),
        python_requires=">=3.11,<3.12",
    )
    assert "cuvis-ai-dataloader[coco,cu3s]" in toml  # extras in [project].dependencies
    # The uv.sources entry is keyed by the BARE name (extras compose with the
    # path/git override); parse the toml back and check both shapes precisely.
    import tomllib

    doc = tomllib.loads(toml)
    assert "cuvis-ai-dataloader[coco,cu3s]" in doc["project"]["dependencies"]
    assert "cuvis-ai-dataloader" in doc["tool"]["uv"]["sources"]
    assert "cuvis-ai-dataloader[coco,cu3s]" not in doc["tool"]["uv"]["sources"]


def test_plugin_source_entry_ref_default_sha_vs_tag():
    """Regression: the composer default pins git plugins to the resolved sha;
    the provision helper's ref='tag' emits the manifest tag instead. The default
    must stay 'sha' so composed child envs remain cache-stable and reproducible."""
    repo = "https://github.com/cubert-hyperspectral/cuvis-ai-sam3.git"
    sha = "9f3c1a2b" * 5  # 40 hex chars
    p = ResolvedGitPlugin(
        name="sam3",
        repo=repo,
        sha=sha,
        tag="v0.1.6",
        package_name="cuvis-ai-sam3",
        extras=(),
    )
    _dep, _key, entry_default = _plugin_source_entry(p)
    assert entry_default == {"git": repo, "rev": sha}  # composer default unchanged
    assert _plugin_source_entry(p, ref="sha")[2] == entry_default
    _d, _k, entry_tag = _plugin_source_entry(p, ref="tag")
    assert entry_tag == {"git": repo, "tag": "v0.1.6"}  # provision env file


def test_union_data_module_plugin():
    catalog = {"cuvis_ai_dataloader": _dataloader_cfg()}
    resolved: dict = {}
    _union_data_module_plugin(resolved, catalog, "cu3s")
    assert "cuvis_ai_dataloader" in resolved
    # unknown module -> no-op
    other: dict = {}
    _union_data_module_plugin(other, catalog, "envi")
    assert other == {}
