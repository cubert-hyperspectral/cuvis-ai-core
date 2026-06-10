"""Cache-key correctness tests.

Each of the six key components must, when changed in isolation,
produce a different cache directory (no false reuse).
"""

from __future__ import annotations

import re
from pathlib import Path

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
        package_name="local-plugin",
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
        package_name="local-plugin",
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


def test_directory_name_is_hash_only():
    name = _base_key().directory_name()
    assert re.fullmatch(r"[0-9a-f]{12}", name), name
    # No human-readable prefix or separator leaks into the name, however
    # many plugins the pipeline declares.
    assert "__" not in name
    for fragment in ("win", "amd64", "py3", "core", "cuvis", "builtin", "detr"):
        assert fragment not in name


def test_human_manifest_lists_intended_libraries():
    plugin_c = ResolvedGitPlugin(
        name="sam3",
        repo="https://github.com/cubert/cuvis-ai-sam3.git",
        sha="c" * 40,
        tag="v1.0.0",
        package_name="cuvis-ai-sam3",  # distinct from the manifest key
    )
    key = compute_cache_key(
        core_source=CORE,
        plugins=(PLUGIN_A, PLUGIN_B, plugin_c),
        spec_hash=SPEC_HASH,
        python_version="3.11.12",
        platform_tag="win-amd64",
    )
    manifest = key.human_manifest()

    # Identity + environment facts.
    assert key.digest in manifest
    assert "3.11.12" in manifest
    assert "win-amd64" in manifest
    assert "cuvis-ai-core==0.7.3" in manifest  # core identity

    # Each git plugin: manifest name, source URL, tag, short sha.
    for plugin in (PLUGIN_A, PLUGIN_B, plugin_c):
        assert plugin.name in manifest
        assert plugin.repo in manifest
        assert plugin.tag in manifest
        assert plugin.sha[:8] in manifest

    # package_name is surfaced distinctly from the manifest key — the
    # value uv installs, deliberately absent from key.json.
    assert "cuvis-ai-sam3" in manifest


def test_human_manifest_marks_local_plugin_state(tmp_path: Path):
    clean = ResolvedLocalPlugin(
        name="local_plugin",
        path=tmp_path,
        package_name="local-plugin",
        pyproject_sha256="x" * 64,
        git_head="d" * 40,
        dirty=False,
    )
    manifest = compute_cache_key(
        core_source=CORE,
        plugins=(clean,),
        spec_hash=SPEC_HASH,
        python_version="3.11.12",
        platform_tag="win-amd64",
    ).human_manifest()
    assert "local_plugin" in manifest  # manifest name
    assert "local-plugin" in manifest  # package_name
    assert str(tmp_path) in manifest  # source path
    assert "local (clean)" in manifest


# ---------------------------------------------------------------------------
# human_manifest edge rows + helper functions
# ---------------------------------------------------------------------------


def test_human_manifest_with_no_plugins_lists_none():
    manifest = compute_cache_key(
        core_source=CORE,
        plugins=(),
        spec_hash=SPEC_HASH,
        python_version="3.11.12",
        platform_tag="win-amd64",
    ).human_manifest()
    assert "_None._" in manifest


def test_human_manifest_marks_dirty_local_plugin(tmp_path: Path):
    dirty = ResolvedLocalPlugin(
        name="local_plugin",
        path=tmp_path,
        package_name="local-plugin",
        pyproject_sha256="x" * 64,
        git_head=None,
        dirty=True,
    )
    key = compute_cache_key(
        core_source=CORE,
        plugins=(dirty,),
        spec_hash=SPEC_HASH,
        python_version="3.11.12",
        platform_tag="win-amd64",
    )
    # A dirty local plugin mints a per-run suffix and stamps the dev-mode note.
    assert key.dirty_suffix is not None
    manifest = key.human_manifest()
    assert "Dev mode" in manifest
    assert "local (dirty)" in manifest


def test_short_core_identity_handles_git_and_local():
    from cuvis_ai_core.orchestrator.cache_key import _short_core_identity

    git_with_sha = CoreSource(kind="git", identity="https://x/core.git@" + "c" * 40)
    assert _short_core_identity(git_with_sha) == "c" * 8

    git_no_sha = CoreSource(kind="git", identity="weird-identity@")
    assert _short_core_identity(git_no_sha) == "weird-identity@"

    local = CoreSource(kind="local", identity="/somewhere/cuvis-ai-core")
    assert _short_core_identity(local) == "local"


def test_current_python_version_and_platform_tag():
    from cuvis_ai_core.orchestrator.cache_key import (
        current_platform_tag,
        current_python_version,
    )

    version = current_python_version()
    assert re.match(r"^\d+\.\d+\.\d+$", version)
    assert current_platform_tag()  # non-empty


def test_current_platform_tag_appends_machine_when_absent(monkeypatch):
    from cuvis_ai_core.orchestrator import cache_key as cache_key_mod

    monkeypatch.setattr(cache_key_mod.sysconfig, "get_platform", lambda: "linux-x86_64")
    monkeypatch.setattr(cache_key_mod.platform, "machine", lambda: "ppc64le")
    assert cache_key_mod.current_platform_tag() == "linux-x86_64-ppc64le"


# ---------------------------------------------------------------------------
# local_plugin_provenance
# ---------------------------------------------------------------------------


def test_local_plugin_provenance_no_pyproject_no_repo(tmp_path: Path, monkeypatch):
    from cuvis_ai_core.orchestrator import cache_key as cache_key_mod

    # No git available → head None, dirty True; no pyproject → empty-bytes hash.
    monkeypatch.setattr(cache_key_mod, "_git", lambda args, cwd: None)
    sha, head, dirty = cache_key_mod.local_plugin_provenance(tmp_path)
    import hashlib

    assert sha == hashlib.sha256(b"").hexdigest()
    assert head is None
    assert dirty is True


def test_local_plugin_provenance_clean_and_dirty(tmp_path: Path, monkeypatch):
    from cuvis_ai_core.orchestrator import cache_key as cache_key_mod

    (tmp_path / "pyproject.toml").write_bytes(b"[project]\nname = 'x'\n")

    def _git_clean(args, cwd):
        return "abc123" if args[0] == "rev-parse" else ""

    monkeypatch.setattr(cache_key_mod, "_git", _git_clean)
    sha, head, dirty = cache_key_mod.local_plugin_provenance(tmp_path)
    assert head == "abc123"
    assert dirty is False
    assert sha != ""

    def _git_dirty(args, cwd):
        return "abc123" if args[0] == "rev-parse" else " M changed.py"

    monkeypatch.setattr(cache_key_mod, "_git", _git_dirty)
    _sha, _head, dirty2 = cache_key_mod.local_plugin_provenance(tmp_path)
    assert dirty2 is True
