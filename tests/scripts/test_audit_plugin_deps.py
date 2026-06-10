"""Tests for the dependency-floor audit script."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner
from packaging.requirements import Requirement
from packaging.version import Version

# The audit lives in the top-level (namespace) ``scripts`` package; make it
# importable regardless of whether the editable install has been refreshed.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import audit_plugin_deps as apd  # noqa: E402


def _make_repo(
    tmp_path: Path,
    deps: list[str],
    locked: dict[str, str],
    sources_toml: str = "",
) -> Path:
    """Write a minimal pyproject.toml + uv.lock and return the repo dir."""
    lines = ["[project]", 'name = "sample"', "dependencies = ["]
    lines += [f'    "{d}",' for d in deps]
    lines.append("]")
    pyproject = "\n".join(lines) + "\n" + sources_toml
    (tmp_path / "pyproject.toml").write_text(pyproject, encoding="utf-8")

    lock = ""
    for name, ver in locked.items():
        lock += f'[[package]]\nname = "{name}"\nversion = "{ver}"\n\n'
    (tmp_path / "uv.lock").write_text(lock, encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Pillow", "pillow"),
        ("scikit_image", "scikit-image"),
        ("PyYAML", "pyyaml"),
        ("huggingface-hub", "huggingface-hub"),
    ],
)
def test_normalize(raw: str, expected: str) -> None:
    assert apd.normalize(raw) == expected


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (">=2.0", Version("2.0")),
        (">2.0", Version("2.0")),
        ("==2.0", Version("2.0")),
        ("~=2.0", Version("2.0")),
        ("==2.5.*", Version("2.5")),
        ("!=2.0", None),
        ("<3.0", None),
        ("", None),
    ],
)
def test_floor_of(spec: str, expected: Version | None) -> None:
    assert apd.floor_of(Requirement(f"pkg{spec}")) == expected


def test_floor_lags_ignores_local_version() -> None:
    # A CUDA local build satisfies a matching public floor.
    assert not apd.floor_lags(Version("2.11.0"), Version("2.11.0+cu128"))
    # A genuinely lower floor still lags the same locked build.
    assert apd.floor_lags(Version("2.0"), Version("2.11.0+cu128"))


def test_host_flags_stale_floor(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, ["transformers>=4.36.0"], {"transformers": "5.7.0"})
    findings, warnings = apd.check_host(repo)
    assert [(f.name, f.kind) for f in findings] == [("transformers", "stale")]
    assert warnings == []


def test_host_clean_when_floor_matches_lock(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, ["numpy>=2.4.1"], {"numpy": "2.4.1"})
    findings, _ = apd.check_host(repo)
    assert findings == []


def test_host_flags_missing_floor(tmp_path: Path) -> None:
    repo = _make_repo(
        tmp_path,
        ["matplotlib", "markupsafe!=3.0.2"],
        {
            "matplotlib": "3.10.9",
            "markupsafe": "3.0.3",
        },
    )
    findings, _ = apd.check_host(repo)
    assert {f.name for f in findings} == {"matplotlib", "markupsafe"}
    assert all(f.kind == "missing-floor" for f in findings)


def test_host_local_version_floor_not_stale(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, ["torch>=2.11.0"], {"torch": "2.11.0+cu128"})
    findings, _ = apd.check_host(repo)
    assert findings == []


def test_host_skips_editable_sibling(tmp_path: Path) -> None:
    sources = (
        '\n[tool.uv.sources]\ncuvis-ai-core = { path = "../core", editable = true }\n'
    )
    repo = _make_repo(
        tmp_path,
        ["cuvis-ai-core>=0.5.3"],
        {"cuvis-ai-core": "0.6.0"},
        sources_toml=sources,
    )
    findings, warnings = apd.check_host(repo)
    assert findings == []
    assert warnings == []  # skipped entirely, not even a not-in-lock note


def test_host_filters_inapplicable_marker(tmp_path: Path) -> None:
    # python_version < 3.0 never holds, so the dep must be ignored.
    repo = _make_repo(
        tmp_path,
        ["legacy>=1.0; python_version < '3.0'"],
        {"legacy": "9.9.9"},
    )
    findings, warnings = apd.check_host(repo)
    assert findings == []
    assert warnings == []


def test_host_not_in_lock_is_warning_not_finding(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, ["ghost>=1.0"], {"numpy": "2.4.1"})
    findings, warnings = apd.check_host(repo)
    assert findings == []
    assert len(warnings) == 1 and "ghost" in warnings[0]


def test_host_matches_normalized_name(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, ["Pillow>=9.0"], {"pillow": "12.2.0"})
    findings, _ = apd.check_host(repo)
    assert [f.name for f in findings] == ["Pillow"]


def test_cli_strict_exits_nonzero_on_findings(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, ["transformers>=4.36.0"], {"transformers": "5.7.0"})
    result = CliRunner().invoke(
        apd.main,
        ["--check", "host", "--project-dir", str(repo), "--strict"],
    )
    assert result.exit_code == 1
    assert "STALE" in result.output


def test_cli_exits_zero_when_clean(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, ["numpy>=2.4.1"], {"numpy": "2.4.1"})
    result = CliRunner().invoke(
        apd.main,
        ["--check", "host", "--project-dir", str(repo), "--strict"],
    )
    assert result.exit_code == 0


def test_cli_json_format(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, ["transformers>=4.36.0"], {"transformers": "5.7.0"})
    result = CliRunner().invoke(
        apd.main,
        ["--check", "host", "--project-dir", str(repo), "--format", "json"],
    )
    assert result.exit_code == 0  # no --strict
    payload = json.loads(result.output)
    assert payload["host"]["findings"][0]["name"] == "transformers"
    assert payload["host"]["findings"][0]["kind"] == "stale"


def test_cli_plugins_requires_dir(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, ["numpy>=2.4.1"], {"numpy": "2.4.1"})
    result = CliRunner().invoke(
        apd.main,
        ["--check", "plugins", "--project-dir", str(repo)],
    )
    assert result.exit_code != 0
    assert "--plugins-dir" in result.output


def _write_plugin_pyproject(tmp_path: Path, name: str, deps: list[str]) -> Path:
    lines = ["[project]", f'name = "{name}"', "dependencies = ["]
    lines += [f'    "{d}",' for d in deps]
    lines.append("]")
    pp = tmp_path / "pyproject.toml"
    pp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return pp


def test_load_installed_versions_includes_packaging() -> None:
    # Smoke: the audit's own dependency must be discoverable in this env.
    installed = apd.load_installed_versions()
    assert "packaging" in installed
    assert isinstance(installed["packaging"], Version)


def test_resolve_core_lock_installed_matches_metadata() -> None:
    lock = apd.resolve_core_lock("installed", Path("."))
    assert "packaging" in lock


def test_resolve_core_lock_path_reads_uv_lock(tmp_path: Path) -> None:
    core = _make_repo(tmp_path, [], {"numpy": "2.4.1"})
    lock = apd.resolve_core_lock(str(core), tmp_path / "elsewhere")
    assert lock["numpy"] == Version("2.4.1")


def test_check_plugin_pyproject_flags_incompatible_dep(tmp_path: Path) -> None:
    pp = _write_plugin_pyproject(tmp_path, "cuvis-ai-sam3", ["pillow>=12.2.0"])
    core_lock = {"pillow": Version("12.1.0")}  # core locks an older pillow
    findings, warnings = apd.check_plugin_pyproject(pp, core_lock)
    assert warnings == []
    assert [(f.plugin, f.name) for f in findings] == [("cuvis-ai-sam3", "pillow")]


def test_check_plugin_pyproject_clean_when_satisfied(tmp_path: Path) -> None:
    pp = _write_plugin_pyproject(tmp_path, "cuvis-ai-sam3", ["pillow>=12.0.0"])
    core_lock = {"pillow": Version("12.2.0")}
    findings, warnings = apd.check_plugin_pyproject(pp, core_lock)
    assert findings == []
    assert warnings == []


def test_check_plugin_pyproject_missing_file_warns(tmp_path: Path) -> None:
    findings, warnings = apd.check_plugin_pyproject(
        tmp_path / "nope.toml", {"pillow": Version("12.2.0")}
    )
    assert findings == []
    assert len(warnings) == 1 and "not found" in warnings[0]


def test_cli_plugin_pyproject_against_path_core(tmp_path: Path) -> None:
    # core checkout locks pillow 12.1.0; plugin demands >=12.2.0 -> mismatch.
    core_dir = tmp_path / "core"
    core_dir.mkdir()
    core = _make_repo(core_dir, [], {"pillow": "12.1.0"})
    plugin_dir = tmp_path / "plugin"
    plugin_dir.mkdir()
    _write_plugin_pyproject(plugin_dir, "cuvis-ai-sam3", ["pillow>=12.2.0"])
    result = CliRunner().invoke(
        apd.main,
        [
            "--check",
            "plugins",
            "--plugin-pyproject",
            str(plugin_dir / "pyproject.toml"),
            "--against-core",
            str(core),
            "--strict",
        ],
    )
    assert result.exit_code == 1
    assert "MISMATCH" in result.output and "pillow" in result.output


def test_check_plugins_skips_local_path_entries(tmp_path: Path) -> None:
    # A local-path manifest entry (dev / built-in) is skipped with a note, never
    # audited against the core lock — even when its pyproject exists on disk.
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    builtin = tmp_path / "host_repo"
    builtin.mkdir()
    _write_plugin_pyproject(builtin, "cuvis-ai", ["pillow>=99.0.0"])  # would mismatch
    (plugins_dir / "cuvis_ai_builtin.yaml").write_text(
        'plugins:\n  cuvis_ai_builtin:\n    path: "../host_repo"\n', encoding="utf-8"
    )
    findings, warnings = apd.check_plugins(plugins_dir, {"pillow": Version("12.2.0")})
    assert findings == []
    assert any("local-path entry" in w for w in warnings)


def test_cli_plugin_check_accepts_pyproject_without_dir(tmp_path: Path) -> None:
    # Providing --plugin-pyproject satisfies the plugin check (no --plugins-dir).
    core_dir = tmp_path / "core"
    core_dir.mkdir()
    core = _make_repo(core_dir, [], {"pillow": "12.2.0"})
    plugin_dir = tmp_path / "plugin"
    plugin_dir.mkdir()
    _write_plugin_pyproject(plugin_dir, "cuvis-ai-sam3", ["pillow>=12.0.0"])
    result = CliRunner().invoke(
        apd.main,
        [
            "--check",
            "plugins",
            "--plugin-pyproject",
            str(plugin_dir / "pyproject.toml"),
            "--against-core",
            str(core),
            "--strict",
        ],
    )
    assert result.exit_code == 0
