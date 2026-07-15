"""Unit tests for the shared HF cache resolution (``hf_cache_dir``).

The provisioner (``download-model``) and the sandboxed child must resolve the
SAME Hugging Face hub cache, or a pre-provisioned gated weight and the offline
child disagree. ``hf_cache_dir`` is that single resolver; it mirrors
``huggingface_hub``'s precedence and falls back to the shared model cache.
"""

from __future__ import annotations

from cuvis_ai_core.orchestrator.model_cache import hf_cache_dir


def test_hf_cache_dir_prefers_explicit_hub_cache(tmp_path):
    hub = tmp_path / "explicit_hub"
    env = {"HF_HUB_CACHE": str(hub), "HF_HOME": str(tmp_path / "home")}
    assert hf_cache_dir(env) == hub


def test_hf_cache_dir_uses_legacy_hub_cache(tmp_path):
    hub = tmp_path / "legacy_hub"
    assert hf_cache_dir({"HUGGINGFACE_HUB_CACHE": str(hub)}) == hub


def test_hf_cache_dir_falls_back_to_hf_home_hub(tmp_path):
    home = tmp_path / "hfhome"
    assert hf_cache_dir({"HF_HOME": str(home)}) == home / "hub"


def test_hf_cache_dir_defaults_to_model_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("CUVIS_MODEL_CACHE_DIR", str(tmp_path / "mc"))
    # No HF vars in the passed env -> the shared model cache's hf subdir.
    assert hf_cache_dir({}) == tmp_path / "mc" / "hf"


def test_hf_cache_dir_explicit_beats_home_and_default(monkeypatch, tmp_path):
    monkeypatch.setenv("CUVIS_MODEL_CACHE_DIR", str(tmp_path / "mc"))
    hub = tmp_path / "win"
    env = {"HF_HUB_CACHE": str(hub), "HF_HOME": str(tmp_path / "home")}
    assert hf_cache_dir(env) == hub
