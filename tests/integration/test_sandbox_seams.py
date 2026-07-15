"""Sandbox seams: invariants the future runtime sandbox depends on.

The orchestrator must remain the single seam through which child
runtimes are launched, and the spawner-env policy must continue to
strip credentials / agent sockets / PYTHONPATH from what the child
inherits. These tests fail loudly if either property regresses, so
the work in item 06 (process / FS sandbox) stays bounded.

The child runs untrusted plugin code, so it never receives an HF token:
``HF_TOKEN`` / ``HUGGINGFACE_HUB_TOKEN`` are stripped and the child runs
``HF_HUB_OFFLINE=1`` against the shared model cache. Gated weights are
provisioned into that cache out-of-band by a trusted tool (the
``download-model`` CLI or the CuvisNEXT action).
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

from cuvis_ai_core.orchestrator import spawner as spawner_module
from cuvis_ai_core.orchestrator.spawner import (
    DeclaredPaths,
    LocalChildRuntimeSpawner,
    _DENY_EXACT,
    _DENY_PREFIXES,
    _is_denied,
)


PACKAGE_ROOT = Path(spawner_module.__file__).resolve().parents[1]
_POPEN_RE = re.compile(r"subprocess\s*\.\s*Popen\s*\(")


def _iter_package_python_files() -> list[Path]:
    return [p for p in PACKAGE_ROOT.rglob("*.py") if "__pycache__" not in p.parts]


def test_no_rogue_subprocess_popen_outside_spawner():
    """Only the orchestrator spawner is allowed to subprocess.Popen.

    Future sandbox work wraps :class:`LocalChildRuntimeSpawner` — every
    child process launch must funnel through that one site. If a new
    call to ``subprocess.Popen`` appears anywhere else in
    ``cuvis_ai_core``, the sandbox seam is broken.
    """
    allowed = (PACKAGE_ROOT / "orchestrator" / "spawner.py").resolve()
    offenders: list[str] = []
    for source in _iter_package_python_files():
        if source.resolve() == allowed:
            continue
        text = source.read_text(encoding="utf-8")
        if _POPEN_RE.search(text):
            offenders.append(str(source.relative_to(PACKAGE_ROOT)))
    assert not offenders, (
        "Only orchestrator/spawner.py may call subprocess.Popen. "
        f"Other call sites found: {offenders}"
    )


def test_deny_list_strips_known_secret_env_vars(monkeypatch, tmp_path):
    """``_build_child_env`` must remove every var the deny-list covers.

    The list is the contract item 06 inherits — adding a new bleed
    point (a leaked AWS profile, an editor's API token, a forgotten
    PYTHONPATH override) must show up here, not in production.
    """
    monkeypatch.setenv("PYTHONPATH", "/some/path")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/agent.sock")
    monkeypatch.setenv("SSH_AGENT_PID", "12345")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "should-not-leak")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "should-not-leak")
    monkeypatch.setenv("GITHUB_TOKEN", "should-not-leak")
    monkeypatch.setenv("GH_TOKEN", "should-not-leak")
    monkeypatch.setenv("GITLAB_TOKEN", "should-not-leak")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "should-not-leak")
    monkeypatch.setenv("OPENAI_API_KEY", "should-not-leak")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/etc/gcp.json")
    monkeypatch.setenv("AZURE_CLIENT_SECRET", "should-not-leak")

    output_dir = tmp_path / "out"
    scratch_dir = tmp_path / "scratch"
    output_dir.mkdir()
    scratch_dir.mkdir()
    venv_path = tmp_path / "venv"
    venv_path.mkdir()

    spawner = LocalChildRuntimeSpawner()
    env = spawner._build_child_env(
        venv_path=venv_path,
        declared_paths=DeclaredPaths(output_dir=output_dir, scratch_dir=scratch_dir),
        request_gpu=False,
    )

    for var in (
        "PYTHONPATH",
        "SSH_AUTH_SOCK",
        "SSH_AGENT_PID",
        "AWS_ACCESS_KEY_ID",
        "AWS_SESSION_TOKEN",
        "GITHUB_TOKEN",
        "GH_TOKEN",
        "GITLAB_TOKEN",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "AZURE_CLIENT_SECRET",
    ):
        assert var not in env, f"{var} leaked into child env"


def test_declared_paths_redirect_home_and_temp(tmp_path):
    """HOME / USERPROFILE / TEMP point at the declared scratch tree.

    A future FS sandbox uses these as the bind-mount set; nothing the
    child writes should land outside them.
    """
    output_dir = tmp_path / "out"
    scratch_dir = tmp_path / "scratch"
    output_dir.mkdir()
    scratch_dir.mkdir()
    venv_path = tmp_path / "venv"
    venv_path.mkdir()

    env = LocalChildRuntimeSpawner()._build_child_env(
        venv_path=venv_path,
        declared_paths=DeclaredPaths(output_dir=output_dir, scratch_dir=scratch_dir),
        request_gpu=False,
    )

    assert env["HOME"] == str(output_dir / ".home")
    assert env["USERPROFILE"] == str(output_dir / ".home")
    assert env["TEMP"] == str(scratch_dir)
    assert env["TMP"] == str(scratch_dir)
    assert env["TMPDIR"] == str(scratch_dir)
    # Never inherit the real user HOME or USERPROFILE.
    assert env["HOME"] != os.environ.get("HOME", "")
    assert env["USERPROFILE"] != os.environ.get("USERPROFILE", "")


def test_cuda_vars_stripped_when_gpu_not_requested(monkeypatch, tmp_path):
    """CUDA_* must be removed unless the caller opts in via request_gpu."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("CUDA_HOME", "/usr/local/cuda")
    monkeypatch.setenv("CUDA_PATH", "/usr/local/cuda")
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "all")
    if sys.platform != "win32":
        monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/local/cuda/lib64")

    venv_path = tmp_path / "venv"
    venv_path.mkdir()
    declared = DeclaredPaths(
        output_dir=tmp_path / "out", scratch_dir=tmp_path / "scratch"
    )
    declared.output_dir.mkdir()
    declared.scratch_dir.mkdir()

    no_gpu = LocalChildRuntimeSpawner()._build_child_env(
        venv_path=venv_path, declared_paths=declared, request_gpu=False
    )
    for var in (
        "CUDA_VISIBLE_DEVICES",
        "CUDA_HOME",
        "CUDA_PATH",
        "NVIDIA_VISIBLE_DEVICES",
    ):
        assert var not in no_gpu, f"{var} should be stripped when request_gpu=False"

    with_gpu = LocalChildRuntimeSpawner()._build_child_env(
        venv_path=venv_path, declared_paths=declared, request_gpu=True
    )
    assert "CUDA_VISIBLE_DEVICES" in with_gpu
    assert "CUDA_HOME" in with_gpu


@pytest.mark.parametrize(
    "var",
    [
        "PYTHONPATH",
        "SSH_AUTH_SOCK",
        "AWS_PROFILE",
        "GITHUB_REF",
        "ANTHROPIC_API_KEY",
        "AZURE_TENANT_ID",
    ],
)
def test_is_denied_matches_expected_patterns(var):
    """Sanity-check the prefix/exact match logic on representative names."""
    assert _is_denied(var)


def test_deny_patterns_constant_contains_required_classes():
    """Lock in the high-level categories the deny-list covers."""
    required_exact = {
        "PYTHONPATH",
        "SSH_AUTH_SOCK",
        "SSH_AGENT_PID",
        "GH_TOKEN",
        "GITLAB_TOKEN",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS",
    }
    required_prefixes = {"AWS_", "GITHUB_", "AZURE_"}
    missing_exact = required_exact - set(_DENY_EXACT)
    missing_prefixes = required_prefixes - set(_DENY_PREFIXES)
    assert not missing_exact, f"Deny-list lost exact entries: {missing_exact}"
    assert not missing_prefixes, f"Deny-list lost prefixes: {missing_prefixes}"


def test_hf_token_denied():
    """HF token env vars are stripped: the untrusted child never gets a credential.

    huggingface_hub reads ``HF_TOKEN`` or the legacy ``HUGGING_FACE_HUB_TOKEN``
    (with the underscore); both must be denied. The misspelled
    ``HUGGINGFACE_HUB_TOKEN`` is kept as harmless belt-and-suspenders.
    """
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        assert _is_denied(var)
        assert var in _DENY_EXACT


def test_hf_token_stripped_and_model_cache_injected(monkeypatch, tmp_path):
    """The child gets the shared cache + offline mode, and no HF credential.

    Gated weights are provisioned into the shared cache out-of-band by a trusted
    tool; the child (untrusted plugin code) runs ``HF_HUB_OFFLINE=1`` against that
    cache and never receives an HF token env var. The token FILE vector is closed
    by pinning ``HF_TOKEN_PATH`` at a guaranteed-absent scratch file, so an
    operator ``HF_TOKEN_PATH`` never reaches the child.
    """
    monkeypatch.setenv("HF_TOKEN", "must-not-leak")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "must-not-leak")
    monkeypatch.setenv("HUGGINGFACE_HUB_TOKEN", "must-not-leak")
    # An operator-set token-file locator must not survive into the child.
    monkeypatch.setenv("HF_TOKEN_PATH", str(tmp_path / "operator_token"))
    monkeypatch.setenv("CUVIS_MODEL_CACHE_DIR", str(tmp_path / "model_cache"))
    for var in (
        "HF_HOME",
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "TORCH_HOME",
        "HF_HUB_OFFLINE",
    ):
        monkeypatch.delenv(var, raising=False)

    output_dir = tmp_path / "out"
    scratch_dir = tmp_path / "scratch"
    output_dir.mkdir()
    scratch_dir.mkdir()
    venv_path = tmp_path / "venv"
    venv_path.mkdir()

    env = LocalChildRuntimeSpawner()._build_child_env(
        venv_path=venv_path,
        declared_paths=DeclaredPaths(output_dir=output_dir, scratch_dir=scratch_dir),
        request_gpu=False,
    )

    model_cache = str(tmp_path / "model_cache")
    assert "HF_TOKEN" not in env
    assert "HUGGING_FACE_HUB_TOKEN" not in env
    assert "HUGGINGFACE_HUB_TOKEN" not in env
    # The token file is pinned at an absent scratch path, overriding the operator's.
    assert env["HF_TOKEN_PATH"] == str(scratch_dir / ".no-hf-token")
    assert not Path(env["HF_TOKEN_PATH"]).exists()
    assert env["CUVIS_MODEL_CACHE_DIR"] == model_cache
    assert env["HF_HUB_CACHE"].startswith(model_cache)
    assert env["HUGGINGFACE_HUB_CACHE"] == env["HF_HUB_CACHE"]
    assert env["TORCH_HOME"].startswith(model_cache)
    assert env["HF_HUB_OFFLINE"] == "1"


def test_model_cache_respects_operator_hf_cache(monkeypatch, tmp_path):
    """An operator-set HF cache / offline flag is not overridden by injection."""
    operator_hf = str(tmp_path / "operator_hf")
    monkeypatch.setenv("HF_HUB_CACHE", operator_hf)
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("CUVIS_MODEL_CACHE_DIR", str(tmp_path / "model_cache"))
    monkeypatch.delenv("TORCH_HOME", raising=False)

    output_dir = tmp_path / "out"
    scratch_dir = tmp_path / "scratch"
    output_dir.mkdir()
    scratch_dir.mkdir()
    venv_path = tmp_path / "venv"
    venv_path.mkdir()

    env = LocalChildRuntimeSpawner()._build_child_env(
        venv_path=venv_path,
        declared_paths=DeclaredPaths(output_dir=output_dir, scratch_dir=scratch_dir),
        request_gpu=False,
    )

    assert env["HF_HUB_CACHE"] == operator_hf
    # The operator's explicit offline choice wins over the injected default.
    assert env["HF_HUB_OFFLINE"] == "0"
    # TORCH_HOME was unset, so it is still injected under the model cache.
    assert env["TORCH_HOME"].startswith(str(tmp_path / "model_cache"))


def test_model_cache_points_hub_cache_at_hf_home(monkeypatch, tmp_path):
    """With only HF_HOME set, the child's HF_HUB_CACHE targets ``$HF_HOME/hub``.

    ``download-model`` resolves the identical directory, so a pre-provisioned
    weight and the offline child agree even when the operator manages their
    cache via ``HF_HOME`` rather than ``HF_HUB_CACHE``. ``HF_HUB_CACHE`` outranks
    ``HF_HOME/hub`` in ``huggingface_hub``, so the child reads it regardless of
    its wiped HOME.
    """
    hf_home = tmp_path / "hf_home"
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.setenv("CUVIS_MODEL_CACHE_DIR", str(tmp_path / "model_cache"))

    output_dir = tmp_path / "out"
    scratch_dir = tmp_path / "scratch"
    output_dir.mkdir()
    scratch_dir.mkdir()
    venv_path = tmp_path / "venv"
    venv_path.mkdir()

    env = LocalChildRuntimeSpawner()._build_child_env(
        venv_path=venv_path,
        declared_paths=DeclaredPaths(output_dir=output_dir, scratch_dir=scratch_dir),
        request_gpu=False,
    )

    expected = str(hf_home / "hub")
    assert env["HF_HUB_CACHE"] == expected
    assert env["HUGGINGFACE_HUB_CACHE"] == expected
    # HF_HOME is read to resolve the cache, then dropped: the child keeps the
    # resolved HF_HUB_CACHE but never a pointer into the operator's HF home
    # (which is also the default token-file location).
    assert "HF_HOME" not in env
    # Still offline, and the shared root is still exported for non-HF plugins.
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["CUVIS_MODEL_CACHE_DIR"] == str(tmp_path / "model_cache")
