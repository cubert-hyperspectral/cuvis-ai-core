"""Runtime project generation tests.

Covers tag→SHA resolution, repo-scheme preservation, and the
canonical pyproject.toml output.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from unittest.mock import patch

import pytest

from cuvis_ai_core.orchestrator.cache_key import (
    CoreSource,
    ResolvedGitPlugin,
    ResolvedLocalPlugin,
)
from cuvis_ai_core.orchestrator.runtime_project import (
    RuntimeProjectError,
    _read_local_package_name,
    build_runtime_pyproject,
    git_source_url,
    resolve_git_tag,
    resolve_plugin_sources,
)
from cuvis_ai_schemas.plugin import GitPluginSource, LocalPluginSource

PYPI_CORE = CoreSource(kind="pypi", identity="cuvis-ai-core==0.7.3")


# ---------------------------------------------------------------------------
# git_source_url — URL scheme preservation
# ---------------------------------------------------------------------------


def test_git_source_url_https_stays_https():
    out = git_source_url("https://github.com/cubert/cuvis-ai-detr.git", "abc123")
    assert out == "git+https://github.com/cubert/cuvis-ai-detr.git@abc123"


def test_git_source_url_http_stays_http():
    out = git_source_url("http://example.com/repo.git", "abc123")
    assert out == "git+http://example.com/repo.git@abc123"


def test_git_source_url_ssh_short_form_rewritten_to_full_ssh():
    out = git_source_url("git@gitlab.com:org/repo.git", "abc123")
    assert out == "git+ssh://git@gitlab.com/org/repo.git@abc123"


def test_git_source_url_ssh_full_form_preserved():
    out = git_source_url("ssh://git@gitlab.com/org/repo.git", "abc123")
    assert out == "git+ssh://git@gitlab.com/org/repo.git@abc123"


def test_git_source_url_unknown_scheme_raises():
    with pytest.raises(RuntimeProjectError):
        git_source_url("ftp://example.com/repo.git", "abc")


def test_git_source_url_malformed_ssh_raises():
    with pytest.raises(RuntimeProjectError):
        git_source_url("git@nohostpath", "abc")


# ---------------------------------------------------------------------------
# resolve_git_tag — git ls-remote behaviour
# ---------------------------------------------------------------------------


def test_resolve_git_tag_picks_peeled_sha_for_annotated_tags():
    raw = (
        "1111111111111111111111111111111111111111\trefs/tags/v0.1.0\n"
        "2222222222222222222222222222222222222222\trefs/tags/v0.1.0^{}\n"
    )
    with patch(
        "cuvis_ai_core.orchestrator.runtime_project.subprocess.check_output",
        return_value=raw,
    ):
        assert resolve_git_tag("https://example.com/repo.git", "v0.1.0") == "2" * 40


def test_resolve_git_tag_uses_first_line_for_lightweight_tags():
    raw = "3333333333333333333333333333333333333333\trefs/tags/v0.1.0\n"
    with patch(
        "cuvis_ai_core.orchestrator.runtime_project.subprocess.check_output",
        return_value=raw,
    ):
        assert resolve_git_tag("https://example.com/repo.git", "v0.1.0") == "3" * 40


def test_resolve_git_tag_rejects_missing_tag():
    with patch(
        "cuvis_ai_core.orchestrator.runtime_project.subprocess.check_output",
        return_value="",
    ):
        with pytest.raises(RuntimeProjectError, match="not found"):
            resolve_git_tag("https://example.com/repo.git", "main")


def test_resolve_git_tag_surfaces_subprocess_failure():
    import subprocess as _sp

    with patch(
        "cuvis_ai_core.orchestrator.runtime_project.subprocess.check_output",
        side_effect=_sp.CalledProcessError(
            returncode=128, cmd=["git", "ls-remote"], stderr="auth failed"
        ),
    ):
        with pytest.raises(RuntimeProjectError, match="auth failed"):
            resolve_git_tag("https://example.com/repo.git", "v0.1.0")


def test_resolve_git_tag_names_git_when_missing():
    # A machine without git raises FileNotFoundError from subprocess; the
    # translated error must name the tool instead of leaking [WinError 2].
    with patch(
        "cuvis_ai_core.orchestrator.runtime_project.subprocess.check_output",
        side_effect=FileNotFoundError(2, "No such file or directory"),
    ):
        with pytest.raises(RuntimeProjectError, match="'git' was not found"):
            resolve_git_tag("https://example.com/repo.git", "v0.1.0")


# ---------------------------------------------------------------------------
# resolve_plugin_sources — git + local end-to-end
# ---------------------------------------------------------------------------


def test_resolve_plugin_sources_resolves_git_tag_and_sorts_by_name(tmp_path: Path):
    configs = {
        "z_plugin": GitPluginSource(
            name="z_plugin",
            repo="https://example.com/z.git",
            tag="v0.1.0",
            capabilities=[{"class_name": "z.Node"}],
        ),
        "a_plugin": GitPluginSource(
            name="a_plugin",
            repo="https://example.com/a.git",
            tag="v0.2.0",
            capabilities=[{"class_name": "a.Node"}],
        ),
    }
    with patch(
        "cuvis_ai_core.orchestrator.runtime_project.resolve_git_tag",
        side_effect=lambda repo, tag: ("a" * 40) if "a.git" in repo else ("z" * 40),
    ):
        resolved = resolve_plugin_sources(configs)
    assert [p.name for p in resolved] == ["a_plugin", "z_plugin"]
    assert isinstance(resolved[0], ResolvedGitPlugin)
    assert resolved[0].sha == "a" * 40
    assert resolved[1].sha == "z" * 40


def test_resolve_plugin_sources_stamps_local_provenance(tmp_path: Path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("[project]\nname = 'x'\n", encoding="utf-8")
    configs = {
        "my_local": LocalPluginSource(
            name="my_local",
            path=str(tmp_path),
            capabilities=[{"class_name": "x.Node"}],
        )
    }
    with patch(
        "cuvis_ai_core.orchestrator.runtime_project.local_plugin_provenance",
        return_value=("hash_x", "deadbeef", False),
    ):
        resolved = resolve_plugin_sources(configs)
    assert len(resolved) == 1
    p = resolved[0]
    assert isinstance(p, ResolvedLocalPlugin)
    assert p.pyproject_sha256 == "hash_x"
    assert p.git_head == "deadbeef"
    assert p.dirty is False


# ---------------------------------------------------------------------------
# build_runtime_pyproject — canonical TOML output
# ---------------------------------------------------------------------------


def test_build_runtime_pyproject_pypi_core_no_source_block():
    plugins = (
        ResolvedGitPlugin(
            name="cuvis_ai_detr",
            repo="https://github.com/cubert/cuvis-ai-detr.git",
            sha="a" * 40,
            tag="v0.1.0",
        ),
    )
    content = build_runtime_pyproject(
        core_source=PYPI_CORE,
        plugins=plugins,
        python_requires=">=3.11,<3.14",
    )
    doc = tomllib.loads(content)
    assert doc["project"]["name"] == "cuvis-ai-runtime-project"
    assert doc["project"]["requires-python"] == ">=3.11,<3.14"
    assert doc["project"]["dependencies"] == [
        "cuvis-ai-core==0.7.3",
        "cuvis_ai_detr",
    ]
    sources = doc["tool"]["uv"]["sources"]
    assert "cuvis-ai-core" not in sources  # pypi pin needs no source override
    assert sources["cuvis_ai_detr"] == {
        "git": "https://github.com/cubert/cuvis-ai-detr.git",
        "rev": "a" * 40,
    }


def test_build_runtime_pyproject_local_core_uses_path_source():
    local_core = CoreSource(kind="local", identity="/abs/path/to/cuvis-ai-core")
    content = build_runtime_pyproject(
        core_source=local_core,
        plugins=(),
        python_requires=">=3.11,<3.14",
    )
    doc = tomllib.loads(content)
    assert doc["project"]["dependencies"] == ["cuvis-ai-core"]
    assert doc["tool"]["uv"]["sources"]["cuvis-ai-core"] == {
        "path": "/abs/path/to/cuvis-ai-core",
        "editable": True,
    }


def test_build_runtime_pyproject_ssh_plugin_normalised_to_ssh_url():
    plugins = (
        ResolvedGitPlugin(
            name="private_plugin",
            repo="git@gitlab.com:org/repo.git",
            sha="b" * 40,
            tag="v1.0",
        ),
    )
    content = build_runtime_pyproject(
        core_source=PYPI_CORE,
        plugins=plugins,
        python_requires=">=3.11,<3.14",
    )
    doc = tomllib.loads(content)
    assert doc["tool"]["uv"]["sources"]["private_plugin"] == {
        "git": "ssh://git@gitlab.com/org/repo.git",
        "rev": "b" * 40,
    }


def test_build_runtime_pyproject_local_plugin_uses_editable_path(tmp_path: Path):
    """Local plugins are pinned by their real [project] name, NOT the
    manifest key, so uv's metadata check passes when the manifest key
    differs from the actual package name."""
    plugin = ResolvedLocalPlugin(
        name="local_p",
        path=tmp_path,
        package_name="real-package-name",
        pyproject_sha256="x" * 64,
        git_head=None,
        dirty=True,
    )
    content = build_runtime_pyproject(
        core_source=PYPI_CORE,
        plugins=(plugin,),
        python_requires=">=3.11,<3.14",
    )
    doc = tomllib.loads(content)
    assert "real-package-name" in doc["project"]["dependencies"]
    assert "local_p" not in doc["project"]["dependencies"]
    assert doc["tool"]["uv"]["sources"]["real-package-name"] == {
        "path": str(tmp_path),
        "editable": True,
    }
    assert "local_p" not in doc["tool"]["uv"]["sources"]


def test_build_runtime_pyproject_pins_resolution_to_host_platform():
    """The generated project declares the composing host as a required
    environment so uv resolves a wheel that exists for it (e.g. on Windows
    it backtracks cuvis-il to a version with a win_amd64 wheel)."""
    import sys

    content = build_runtime_pyproject(
        core_source=PYPI_CORE,
        plugins=(),
        python_requires=">=3.11,<3.14",
    )
    doc = tomllib.loads(content)
    assert doc["tool"]["uv"]["required-environments"] == [
        f"sys_platform == '{sys.platform}'"
    ]


def test_build_runtime_pyproject_is_byte_stable_for_same_input(tmp_path: Path):
    plugins = (
        ResolvedGitPlugin(
            name="a",
            repo="https://example.com/a.git",
            sha="a" * 40,
            tag="v1",
        ),
        ResolvedGitPlugin(
            name="b",
            repo="https://example.com/b.git",
            sha="b" * 40,
            tag="v2",
        ),
    )
    a = build_runtime_pyproject(
        core_source=PYPI_CORE, plugins=plugins, python_requires=">=3.11,<3.14"
    )
    b = build_runtime_pyproject(
        core_source=PYPI_CORE, plugins=plugins, python_requires=">=3.11,<3.14"
    )
    assert a == b


# ---------------------------------------------------------------------------
# _read_local_package_name — local plugin pyproject inspection
# ---------------------------------------------------------------------------


def test_read_local_package_name_returns_project_name(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "real-package"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )
    assert _read_local_package_name(tmp_path, manifest_key="logical") == "real-package"


def test_read_local_package_name_missing_pyproject_raises(tmp_path: Path):
    with pytest.raises(RuntimeProjectError, match="has no pyproject.toml"):
        _read_local_package_name(tmp_path, manifest_key="logical")


def test_read_local_package_name_missing_project_name_raises(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text(
        '[build-system]\nrequires = ["setuptools"]\n', encoding="utf-8"
    )
    with pytest.raises(RuntimeProjectError, match="declares no '\\[project\\] name'"):
        _read_local_package_name(tmp_path, manifest_key="logical")


def test_resolve_plugin_sources_reads_local_package_name(tmp_path: Path):
    """The package_name field must come from the plugin's pyproject,
    not the manifest key — this is what fixes the
    'Package metadata name X does not match given name Y' failure
    when the manifest groups several classes under a logical name.
    """
    plugin_dir = tmp_path / "plugin-checkout"
    plugin_dir.mkdir()
    (plugin_dir / "pyproject.toml").write_text(
        '[project]\nname = "actual-package"\nversion = "0.0.1"\n',
        encoding="utf-8",
    )

    config = LocalPluginSource(
        name="manifest_key_only",
        path=str(plugin_dir),
        capabilities=[{"class_name": "actual.module.Foo"}],
    )
    resolved = resolve_plugin_sources({"manifest_key_only": config})

    assert len(resolved) == 1
    plugin = resolved[0]
    assert isinstance(plugin, ResolvedLocalPlugin)
    assert plugin.name == "manifest_key_only"
    assert plugin.package_name == "actual-package"
