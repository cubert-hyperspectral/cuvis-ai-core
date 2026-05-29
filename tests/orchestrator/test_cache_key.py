"""Cache-key correctness tests.

Each of the six key components must, when changed in isolation,
produce a different cache directory (no false reuse).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cuvis_ai_core.orchestrator.cache_key import (
    COMPOSER_SCHEMA_VERSION,
    CacheKey,
    CoreSource,
    ResolvedGitPlugin,
    ResolvedLocalPlugin,
    compute_cache_key,
    spec_hash_of,
)


CORE = CoreSource(kind="pypi", identity="cuvis-ai-core==0.7.3")
PLUGIN_A = ResolvedGitPlugin(
    name="cuvis_ai_builtin",
    repo="https://github.com/cubert/cuvis-ai-builtin.git",
    sha="a" * 40,
    tag="v0.7.3",
)
PLUGIN_B = ResolvedGitPlugin(
    name="cuvis_ai_detr",
    repo="https://github.com/cubert/cuvis-ai-detr.git",
    sha="b" * 40,
    tag="v0.1.0",
)
SPEC_HASH = "a" * 64


def _base_key() -> CacheKey:
    return compute_cache_key(
        core_source=CORE,
        plugins=(PLUGIN_A, PLUGIN_B),
        spec_hash=SPEC_HASH,
        python_version="3.11.12",
        platform_tag="win-amd64",
    )


def test_directory_name_is_deterministic():
    a = _base_key().directory_name()
    b = _base_key().directory_name()
    assert a == b


def test_python_version_change_changes_key():
    base = _base_key().directory_name()
    bumped = compute_cache_key(
        core_source=CORE,
        plugins=(PLUGIN_A, PLUGIN_B),
        spec_hash=SPEC_HASH,
        python_version="3.12.4",
        platform_tag="win-amd64",
    ).directory_name()
    assert base != bumped


def test_platform_tag_change_changes_key():
    base = _base_key().directory_name()
    other_plat = compute_cache_key(
        core_source=CORE,
        plugins=(PLUGIN_A, PLUGIN_B),
        spec_hash=SPEC_HASH,
        python_version="3.11.12",
        platform_tag="linux-x86_64",
    ).directory_name()
    assert base != other_plat


def test_core_source_change_changes_key():
    base = _base_key().directory_name()
    git_core = CoreSource(
        kind="git", identity="https://github.com/cubert/cuvis-ai-core.git@deadbeef"
    )
    other = compute_cache_key(
        core_source=git_core,
        plugins=(PLUGIN_A, PLUGIN_B),
        spec_hash=SPEC_HASH,
        python_version="3.11.12",
        platform_tag="win-amd64",
    ).directory_name()
    assert base != other


def test_plugin_source_change_changes_key():
    base = _base_key().directory_name()
    bumped_plugin = ResolvedGitPlugin(
        name="cuvis_ai_detr",
        repo="https://github.com/cubert/cuvis-ai-detr.git",
        sha="c" * 40,  # different sha
        tag="v0.1.1",
    )
    other = compute_cache_key(
        core_source=CORE,
        plugins=(PLUGIN_A, bumped_plugin),
        spec_hash=SPEC_HASH,
        python_version="3.11.12",
        platform_tag="win-amd64",
    ).directory_name()
    assert base != other


def test_spec_hash_change_changes_key():
    base = _base_key().directory_name()
    other = compute_cache_key(
        core_source=CORE,
        plugins=(PLUGIN_A, PLUGIN_B),
        spec_hash="b" * 64,
        python_version="3.11.12",
        platform_tag="win-amd64",
    ).directory_name()
    assert base != other


def test_schema_version_change_changes_key():
    base = _base_key().directory_name()
    other = compute_cache_key(
        core_source=CORE,
        plugins=(PLUGIN_A, PLUGIN_B),
        spec_hash=SPEC_HASH,
        python_version="3.11.12",
        platform_tag="win-amd64",
        schema_version=COMPOSER_SCHEMA_VERSION + 1,
    ).directory_name()
    assert base != other


def test_dirty_local_plugin_forces_unique_directory(tmp_path: Path):
    dirty = ResolvedLocalPlugin(
        name="local_plugin",
        path=tmp_path,
        pyproject_sha256="x" * 64,
        git_head=None,
        dirty=True,
    )
    a = compute_cache_key(
        core_source=CORE,
        plugins=(dirty,),
        spec_hash=SPEC_HASH,
        python_version="3.11.12",
        platform_tag="win-amd64",
    ).directory_name()
    b = compute_cache_key(
        core_source=CORE,
        plugins=(dirty,),
        spec_hash=SPEC_HASH,
        python_version="3.11.12",
        platform_tag="win-amd64",
    ).directory_name()
    assert a != b  # dirty_suffix is random per call


def test_clean_local_plugin_caches_normally(tmp_path: Path):
    clean = ResolvedLocalPlugin(
        name="local_plugin",
        path=tmp_path,
        pyproject_sha256="x" * 64,
        git_head="d" * 40,
        dirty=False,
    )
    a = compute_cache_key(
        core_source=CORE,
        plugins=(clean,),
        spec_hash=SPEC_HASH,
        python_version="3.11.12",
        platform_tag="win-amd64",
    ).directory_name()
    b = compute_cache_key(
        core_source=CORE,
        plugins=(clean,),
        spec_hash=SPEC_HASH,
        python_version="3.11.12",
        platform_tag="win-amd64",
    ).directory_name()
    assert a == b


def test_serialise_roundtrip_is_canonical():
    key = _base_key()
    payload = key.serialise()
    assert payload["python_version"] == "3.11.12"
    assert payload["platform_tag"] == "win-amd64"
    assert payload["core_source"]["kind"] == "pypi"
    assert len(payload["plugins"]) == 2
    assert payload["plugins"][0]["kind"] == "git"
    assert payload["spec_hash"] == SPEC_HASH
    assert payload["schema_version"] == COMPOSER_SCHEMA_VERSION


def test_spec_hash_of_is_stable_and_content_sensitive():
    a = spec_hash_of("[project]\nname = 'x'\n")
    b = spec_hash_of("[project]\nname = 'x'\n")
    c = spec_hash_of("[project]\nname = 'y'\n")
    assert a == b
    assert a != c
    assert len(a) == 64  # sha256 hex


@pytest.mark.parametrize(
    "platform,expected_prefix",
    [
        ("win-amd64", "win-amd64"),
        ("linux-x86_64", "linux-x86_64"),
        ("macosx-14.0-arm64", "macosx-14.0-arm64"),
    ],
)
def test_directory_name_keeps_platform_segment_readable(
    platform: str, expected_prefix: str
):
    key = compute_cache_key(
        core_source=CORE,
        plugins=(PLUGIN_A,),
        spec_hash=SPEC_HASH,
        python_version="3.11.12",
        platform_tag=platform,
    )
    assert expected_prefix in key.directory_name()
