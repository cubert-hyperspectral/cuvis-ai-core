"""Tests for the provision helper (cuvis-ai-core)."""

from __future__ import annotations

from pathlib import Path

import pytest

from cuvis_ai_core.orchestrator.cache_key import ResolvedGitPlugin, ResolvedLocalPlugin
from cuvis_ai_core.utils import provision as prov

_REPO = "https://github.com/cubert-hyperspectral/cuvis-ai-sam3.git"


def _git(extras=()):
    return ResolvedGitPlugin(
        name="sam3",
        repo=_REPO,
        sha="a" * 40,
        tag="v0.1.6",
        package_name="cuvis-ai-sam3",
        extras=tuple(extras),
    )


def _local(path, extras=()):
    return ResolvedLocalPlugin(
        name="dl",
        path=Path(path),
        package_name="cuvis-ai-dataloader",
        pyproject_sha256="x",
        git_head=None,
        dirty=False,
        extras=tuple(extras),
    )


def test_spec_for_git_tag_and_sha():
    p = _git(extras=("coco", "cu3s"))
    assert prov._spec_for(p, "tag") == f"cuvis-ai-sam3[coco,cu3s] @ git+{_REPO}@v0.1.6"
    assert (
        prov._spec_for(p, "sha") == f"cuvis-ai-sam3[coco,cu3s] @ git+{_REPO}@{'a' * 40}"
    )


def test_spec_for_local_is_file_uri(tmp_path):
    spec = prov._spec_for(_local(tmp_path), "tag")
    assert spec.startswith("cuvis-ai-dataloader @ file://")
    assert tmp_path.resolve().as_uri() in spec


def test_format_install_command():
    specs = ["pkg @ git+https://x@v1"]
    assert (
        prov.format_install_command(specs) == "uv pip install 'pkg @ git+https://x@v1'"
    )
    assert (
        prov.format_install_command(specs, magic=True)
        == "%pip install 'pkg @ git+https://x@v1'"
    )
    assert "nothing to install" in prov.format_install_command([])


def test_env_file_text_is_pyproject_shaped():
    import tomllib

    text = prov._env_file_text(["pkg @ git+https://x@v1"], "mypipe")
    doc = tomllib.loads(text)
    assert doc["project"]["name"] == "mypipe-env"
    assert "pkg @ git+https://x@v1" in doc["project"]["dependencies"]
    assert doc["tool"]["uv"]["required-environments"]


def test_provision_environment_writes_toml_and_txt(tmp_path):
    specs = ["pkg @ git+https://x@v1"]
    toml_path = tmp_path / "env.toml"
    prov.provision_environment(specs, env_file=toml_path, pipeline_name="p")
    assert toml_path.exists()
    assert "[project]" in toml_path.read_text()

    txt_path = tmp_path / "env.txt"
    prov.provision_environment(specs, env_file=txt_path)
    assert txt_path.read_text().strip() == "pkg @ git+https://x@v1"


def test_provision_environment_rejects_conflicting_modes(tmp_path):
    with pytest.raises(ValueError):
        prov.provision_environment(["x"], env_file=tmp_path / "e.toml", apply=True)
    with pytest.raises(ValueError):
        prov.provision_environment(["x"], env_file=tmp_path / "e.toml", notebook=True)


def test_provision_environment_print_returns_command(capsys):
    cmd = prov.provision_environment(["pkg @ git+https://x@v1"])
    assert cmd.startswith("uv pip install")
    assert "uv pip install" in capsys.readouterr().out


def test_provision_notebook_print_only(capsys):
    line = prov.provision_environment(["pkg @ git+https://x@v1"], notebook=True)
    assert line.startswith("%pip install")
    assert "%pip install" in capsys.readouterr().out


def test_resolve_install_specs_skips_satisfied(monkeypatch):
    """Drops plugins already importable; pins to the tag by default."""
    git_p = _git()
    local_p = _local("/tmp/dl")

    class _Entry:
        def __init__(self, top):
            self.class_name = f"{top}.X"

    class _Cfg:
        def __init__(self, top):
            self.provides = [_Entry(top)]

    monkeypatch.setattr(
        prov.PipelineConfig, "load_from_file", staticmethod(lambda p: object())
    )
    monkeypatch.setattr(
        prov,
        "resolve_pipeline_plugins",
        lambda cfg, dirs, dm: {
            "sam3": _Cfg("cuvis_ai_sam3"),
            "dl": _Cfg("cuvis_ai_dataloader"),
        },
    )
    monkeypatch.setattr(
        prov,
        "resolve_plugin_sources",
        lambda cfgs, active_data_module=None: (git_p, local_p),
    )
    # sam3 already importable; dataloader not.
    monkeypatch.setattr(
        prov,
        "_is_satisfied",
        lambda cfg: cfg.provides[0].class_name.startswith("cuvis_ai_sam3"),
    )

    specs = prov.resolve_install_specs("pipe.yaml", ["dir"], data_module="cu3s")
    assert len(specs) == 1
    assert specs[0].startswith("cuvis-ai-dataloader @ file://")
