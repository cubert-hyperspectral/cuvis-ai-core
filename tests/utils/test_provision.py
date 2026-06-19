"""Tests for the provision helper (cuvis-ai-core)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

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

    monkeypatch.setattr(
        prov.PipelineConfig, "load_from_file", staticmethod(lambda p: object())
    )
    monkeypatch.setattr(
        prov,
        "resolve_pipeline_plugins",
        lambda cfg, dirs, dm: {
            "sam3": _cfg("cuvis_ai_sam3.X"),
            "dl": _cfg("cuvis_ai_dataloader.X"),
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
        lambda cfg: cfg.capabilities[0].class_name.startswith("cuvis_ai_sam3"),
    )

    specs = prov.resolve_install_specs("pipe.yaml", ["dir"], data_module="cu3s")
    assert len(specs) == 1
    assert specs[0].startswith("cuvis-ai-dataloader @ file://")


# ---------------------------------------------------------------------------
# _top_import_module / _is_satisfied (the real implementation; the resolver
# test above monkeypatches _is_satisfied away, so these pin its actual
# behaviour: importable -> True, missing / no-module / lookup error -> False).
# ---------------------------------------------------------------------------


def _cfg(*class_names):
    """Fake resolved config exposing ``.capabilities[*].class_name``."""
    return SimpleNamespace(
        capabilities=[SimpleNamespace(class_name=cn) for cn in class_names]
    )


def test_top_import_module_first_capability_and_empty():
    assert prov._top_import_module(_cfg("pkg.mod.Node", "x.Y")) == "pkg"
    assert prov._top_import_module(_cfg()) is None


def test_is_satisfied_true_when_importable(monkeypatch):
    monkeypatch.setattr(prov.importlib.util, "find_spec", lambda mod: object())
    assert prov._is_satisfied(_cfg("pkg.mod.Node")) is True


def test_is_satisfied_false_when_missing(monkeypatch):
    monkeypatch.setattr(prov.importlib.util, "find_spec", lambda mod: None)
    assert prov._is_satisfied(_cfg("pkg.mod.Node")) is False


def test_is_satisfied_false_without_module():
    assert prov._is_satisfied(_cfg()) is False


def test_is_satisfied_false_when_find_spec_raises(monkeypatch):
    def _boom(mod):
        raise ValueError("bad module name")

    monkeypatch.setattr(prov.importlib.util, "find_spec", _boom)
    assert prov._is_satisfied(_cfg("pkg.mod.Node")) is False


# ---------------------------------------------------------------------------
# _provision_notebook apply=True (the in-kernel %pip path)
# ---------------------------------------------------------------------------


def _inject_ipython(monkeypatch, ip):
    """Make ``from IPython import get_ipython`` return a fake yielding ``ip``."""
    monkeypatch.setitem(sys.modules, "IPython", SimpleNamespace(get_ipython=lambda: ip))


def test_provision_notebook_apply_runs_magic(monkeypatch, capsys):
    calls: list[tuple[str, str]] = []
    _inject_ipython(
        monkeypatch, SimpleNamespace(run_line_magic=lambda m, a: calls.append((m, a)))
    )
    assert prov._provision_notebook(["a", "b"], apply=True) is None
    assert calls == [("pip", "install 'a' 'b'")]
    assert "Installed 2 plugin" in capsys.readouterr().out


def test_provision_notebook_apply_no_specs(monkeypatch, capsys):
    _inject_ipython(monkeypatch, SimpleNamespace(run_line_magic=lambda m, a: None))
    assert prov._provision_notebook([], apply=True) is None
    assert "already provisioned" in capsys.readouterr().out


def test_provision_notebook_apply_without_kernel_raises(monkeypatch):
    _inject_ipython(monkeypatch, None)
    with pytest.raises(RuntimeError, match="running"):
        prov._provision_notebook(["a"], apply=True)


# ---------------------------------------------------------------------------
# provision_environment: apply + env-file sync (subprocess-backed paths)
# ---------------------------------------------------------------------------


def test_provision_environment_apply_installs(monkeypatch):
    calls: list[tuple] = []
    monkeypatch.setattr(prov.subprocess, "run", lambda *a, **k: calls.append((a, k)))
    assert prov.provision_environment(["a", "b"], apply=True) is None
    assert calls[0][0][0] == ["uv", "pip", "install", "a", "b"]


def test_provision_environment_apply_no_specs_skips_install(monkeypatch):
    calls: list[tuple] = []
    monkeypatch.setattr(prov.subprocess, "run", lambda *a, **k: calls.append(a))
    assert prov.provision_environment([], apply=True) is None
    assert calls == []


def test_provision_environment_env_file_sync_installs(monkeypatch, tmp_path):
    calls: list[tuple] = []
    monkeypatch.setattr(prov.subprocess, "run", lambda *a, **k: calls.append((a, k)))
    out = tmp_path / "env.toml"
    prov.provision_environment(["a"], env_file=out, sync=True, pipeline_name="P")
    assert out.exists()
    assert calls[0][0][0][:4] == ["uv", "pip", "install", "-r"]
