"""Tests for the data-module pip-extras path through the composer + resolver."""

from __future__ import annotations

from pathlib import Path

import pytest
from cuvis_ai_schemas.plugin import LocalPluginSource, PluginCapabilityEntry

from unittest.mock import patch

from cuvis_ai_core.orchestrator.cache_key import (
    CoreSource,
    ResolvedGitPlugin,
    ResolvedLocalPlugin,
    compute_cache_key,
    spec_hash_of,
)
from cuvis_ai_core.orchestrator.runtime_project import (
    RuntimeProjectError,
    _plugin_extras,
    _plugin_source_entry,
    build_runtime_pyproject,
    check_locked_extras,
    merge_by_package,
    resolve_plugin_sources,
)
from cuvis_ai_core.utils.plugin_resolver import _union_data_module_plugin


def _dataloader_cfg() -> LocalPluginSource:
    return LocalPluginSource(
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


def _rfdetr_cfg(name="rfdetr", extras=(), package_name="cuvis-ai-rfdetr"):
    return LocalPluginSource(
        name=name,
        path=".",
        package_name=package_name,
        extras=list(extras),
        capabilities=[
            PluginCapabilityEntry(
                class_name="cuvis_ai_rfdetr.node.rfdetr_segmenter.RFDETRSegmenter"
            )
        ],
    )


def test_plugin_extras_scopes_data_module_extras_to_selected_module():
    cfg = _dataloader_cfg()
    assert _plugin_extras(cfg, "cu3s") == ("coco", "cu3s")  # sorted
    assert _plugin_extras(cfg, "tiff_paired") == ("tiff",)
    assert _plugin_extras(cfg, None) == ()
    assert _plugin_extras(cfg, "envi") == ()  # unknown -> no extras


def test_plugin_extras_unions_manifest_extras_with_the_data_module():
    cfg = _dataloader_cfg().model_copy(update={"extras": ["profiling"]})
    assert _plugin_extras(cfg, None) == ("profiling",)
    assert _plugin_extras(cfg, "cu3s") == ("coco", "cu3s", "profiling")
    assert _plugin_extras(_rfdetr_cfg(extras=["tensorrt"]), None) == ("tensorrt",)


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
    _union_data_module_plugin(resolved, catalog, "cu3s", [])
    assert "cuvis_ai_dataloader" in resolved
    # unknown module -> raises (mirrors restore._load_data_module_plugin)
    with pytest.raises(ValueError, match="envi"):
        _union_data_module_plugin({}, catalog, "envi", [])


# ---------------------------------------------------------------------------
# Manifest-level extras: stamping, merging by package, the cache key, the lock check
# ---------------------------------------------------------------------------
_RFDETR_REPO = "https://github.com/cubert-hyperspectral/cuvis-ai-rfdetr.git"
_RFDETR_SHA = "b" * 40


def _git(
    name, package_name="cuvis-ai-rfdetr", extras=(), repo=_RFDETR_REPO, sha=_RFDETR_SHA
):
    return ResolvedGitPlugin(
        name=name,
        repo=repo,
        sha=sha,
        tag="v0.5.1",
        package_name=package_name,
        extras=tuple(extras),
    )


def _loc(name, package_name="cuvis-ai-rfdetr", extras=(), path="/x"):
    return ResolvedLocalPlugin(
        name=name,
        path=Path(path),
        package_name=package_name,
        pyproject_sha256="h",
        git_head=None,
        dirty=False,
        extras=tuple(extras),
    )


def test_resolve_plugin_sources_stamps_manifest_extras():
    cfgs = {"rfdetr_seg_trt": _rfdetr_cfg(name="rfdetr_seg_trt", extras=["tensorrt"])}
    with patch(
        "cuvis_ai_core.orchestrator.runtime_project.local_plugin_provenance",
        return_value=("h", None, False),
    ):
        (p,) = resolve_plugin_sources(cfgs)
    assert p.extras == ("tensorrt",)


def test_merge_by_package_unions_extras_of_one_package_and_keeps_others():
    merged = merge_by_package(
        (
            _git("rfdetr"),
            _git("rfdetr_seg_trt", extras=("tensorrt",)),
            _git("sam3", package_name="cuvis-ai-sam3"),
        )
    )
    assert [(p.name, p.package_name, p.extras) for p in merged] == [
        ("rfdetr", "cuvis-ai-rfdetr", ("tensorrt",)),
        ("sam3", "cuvis-ai-sam3", ()),
    ]


def test_merge_by_package_sorts_deduplicates_and_groups_canonical_names():
    merged = merge_by_package(
        (
            _git("a", package_name="cuvis_ai_rfdetr", extras=("train", "tensorrt")),
            _git("b", package_name="Cuvis-AI-RFDETR", extras=("tensorrt",)),
        )
    )
    assert len(merged) == 1
    assert merged[0].name == "a"
    assert merged[0].extras == ("tensorrt", "train")


@pytest.mark.parametrize(
    "other",
    [
        _git("v", sha="c" * 40),  # another commit of the same repo
        _loc("v"),  # a checkout instead of the repo
        _git(
            "v", repo="git@github.com:cubert-hyperspectral/cuvis-ai-rfdetr.git"
        ),  # another transport
    ],
)
def test_merge_by_package_rejects_one_package_from_different_sources(other):
    with pytest.raises(RuntimeProjectError, match="different sources") as excinfo:
        merge_by_package((_git("rfdetr"), other))
    assert "'rfdetr'" in str(excinfo.value)
    assert "'v'" in str(excinfo.value)


def test_build_runtime_pyproject_merges_manifests_of_one_package():
    import tomllib

    core = CoreSource(kind="pypi", identity="cuvis-ai-core==0.18.0")
    merged = merge_by_package(
        (_git("rfdetr"), _git("rfdetr_seg_trt", extras=("tensorrt",)))
    )
    doc = tomllib.loads(
        build_runtime_pyproject(
            core_source=core, plugins=merged, python_requires=">=3.11,<3.12"
        )
    )
    assert [d for d in doc["project"]["dependencies"] if "rfdetr" in d] == [
        "cuvis-ai-rfdetr[tensorrt]"
    ]
    assert list(doc["tool"]["uv"]["sources"]).count("cuvis-ai-rfdetr") == 1


def test_build_runtime_pyproject_refuses_unmerged_duplicates():
    core = CoreSource(kind="pypi", identity="cuvis-ai-core==0.18.0")
    with pytest.raises(RuntimeProjectError, match="merge_by_package"):
        build_runtime_pyproject(
            core_source=core,
            plugins=(_git("rfdetr"), _git("rfdetr_seg_trt", extras=("tensorrt",))),
            python_requires=">=3.11,<3.12",
        )


def test_cache_key_splits_on_manifest_extras_and_on_the_manifest_pair():
    core = CoreSource(kind="pypi", identity="cuvis-ai-core==0.18.0")

    def digest_for(*plugins):
        content = build_runtime_pyproject(
            core_source=core,
            plugins=merge_by_package(plugins),
            python_requires=">=3.11,<3.12",
        )
        return compute_cache_key(
            core_source=core, plugins=plugins, spec_hash=spec_hash_of(content)
        ).digest

    plain = digest_for(_git("rfdetr"))
    assert digest_for(_git("rfdetr")) == plain
    assert digest_for(_git("rfdetr", extras=("tensorrt",))) != plain
    assert (
        digest_for(_git("rfdetr"), _git("rfdetr_seg_trt", extras=("tensorrt",)))
        != plain
    )


def _lock(tmp_path: Path, *entries: tuple[str, list[str]]) -> Path:
    body = "version = 1\n\n"
    for name, extras in entries:
        body += (
            f'[[package]]\nname = "{name}"\nversion = "0.5.1"\n'
            f'source = {{ git = "{_RFDETR_REPO}?rev={_RFDETR_SHA}#{_RFDETR_SHA}" }}\n'
        )
        if extras:
            body += "\n[package.optional-dependencies]\n"
            body += "".join(f"{e} = []\n" for e in extras)
        body += "\n"
    lock = tmp_path / "uv.lock"
    lock.write_text(body, encoding="utf-8")
    return lock


def test_check_locked_extras_passes_declared_and_skips_unlocked_packages(
    tmp_path: Path,
):
    lock = _lock(tmp_path, ("cuvis-ai-rfdetr", ["tensorrt", "train"]))
    check_locked_extras(
        lock,
        (
            _git("rfdetr_seg_trt", extras=("tensorrt",)),
            _git("other", package_name="not-locked", extras=("x",)),
        ),
    )


def test_check_locked_extras_names_the_unknown_extra_and_the_declared_ones(
    tmp_path: Path,
):
    lock = _lock(tmp_path, ("cuvis-ai-rfdetr", ["tensorrt", "train"]))
    with pytest.raises(RuntimeProjectError) as excinfo:
        check_locked_extras(lock, (_git("rfdetr_seg_trt", extras=("tensort",)),))
    msg = str(excinfo.value)
    assert "'rfdetr_seg_trt'" in msg
    assert "'tensort'" in msg
    assert "cuvis-ai-rfdetr" in msg
    assert "tensorrt, train" in msg
    assert "[project.optional-dependencies]" in msg


def test_check_locked_extras_reports_a_package_without_extras(tmp_path: Path):
    lock = _lock(tmp_path, ("cuvis-ai-rfdetr", []))
    with pytest.raises(RuntimeProjectError, match="resolved none of its extras"):
        check_locked_extras(lock, (_git("rfdetr", extras=("tensorrt",)),))


def test_check_locked_extras_requires_the_lock_only_when_extras_are_requested(
    tmp_path: Path,
):
    check_locked_extras(
        tmp_path / "uv.lock", (_git("rfdetr"),)
    )  # nothing requested, nothing read
    with pytest.raises(RuntimeProjectError, match="uv.lock"):
        check_locked_extras(
            tmp_path / "uv.lock", (_git("rfdetr", extras=("tensorrt",)),)
        )
