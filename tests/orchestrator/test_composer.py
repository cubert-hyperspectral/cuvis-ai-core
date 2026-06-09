"""Composer cache + build behaviour tests.

Network and uv invocations are mocked — these tests exercise the
cache-key plumbing, atomic publish, half-built recovery, and the
per-key build lock without depending on the real uv binary or any
external git remote.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cuvis_ai_core.orchestrator import composer as composer_mod
from cuvis_ai_core.orchestrator.cache_key import CoreSource
from cuvis_ai_core.orchestrator.composer import ComposerError, compose_env
from cuvis_ai_core.orchestrator.uv_runner import UvRunnerError
from cuvis_ai_schemas.plugin import GitPluginConfig


PYPI_CORE = CoreSource(kind="pypi", identity="cuvis-ai-core==0.7.3")
FAKE_SHA = "a" * 40


@pytest.fixture(autouse=True)
def _isolate_in_process_locks():
    """Reset the module-level in-process lock map between tests."""
    composer_mod._in_process_locks.clear()
    yield
    composer_mod._in_process_locks.clear()


def _simple_plugin() -> dict:
    return {
        "p": GitPluginConfig(
            repo="https://example.com/p.git",
            tag="v0.1.0",
            provides=[{"class_name": "p.Node"}],
        )
    }


def _patch_resolve_and_uv(*, sync_side_effect=None):
    """Common patch: stub git ls-remote and uv lock/sync."""
    return (
        patch(
            "cuvis_ai_core.orchestrator.runtime_project.resolve_git_tag",
            return_value=FAKE_SHA,
        ),
        patch(
            "cuvis_ai_core.orchestrator.composer.uv_lock",
            new=MagicMock(),
        ),
        patch(
            "cuvis_ai_core.orchestrator.composer.uv_sync",
            new=MagicMock(side_effect=sync_side_effect)
            if sync_side_effect
            else MagicMock(),
        ),
    )


def test_in_process_lock_map_is_weak_and_evicts_unreferenced_locks():
    """Same digest shares one lock while referenced; the map evicts it after.

    Guards against the lock map growing without bound on a long-lived
    server — dirty local plugins mint a fresh digest per run.
    """
    import gc

    composer_mod._in_process_locks.clear()
    lock_a = composer_mod._in_process_lock_for("digestX")
    lock_b = composer_mod._in_process_lock_for("digestX")
    assert lock_a is lock_b
    assert "digestX" in composer_mod._in_process_locks

    del lock_a, lock_b
    gc.collect()
    assert "digestX" not in composer_mod._in_process_locks


def test_compose_env_publishes_venv_path_and_writes_key_json(tmp_path: Path):
    resolve_patch, lock_patch, sync_patch = _patch_resolve_and_uv()
    with resolve_patch, lock_patch as lock_mock, sync_patch as sync_mock:
        # Make uv_sync also create .venv (real uv would do this).
        def fake_sync(project_dir: Path):
            (project_dir / ".venv").mkdir(exist_ok=True)

        sync_mock.side_effect = fake_sync

        venv = compose_env(
            _simple_plugin(),
            core_source=PYPI_CORE,
            cache_root=tmp_path,
        )

    assert lock_mock.call_count == 1
    assert sync_mock.call_count == 1
    assert venv.parent.exists()
    assert (venv.parent / ".ready").exists()
    assert (venv.parent / "pyproject.toml").exists()

    payload = json.loads((venv.parent / "key.json").read_text())
    assert payload["core_source"]["identity"] == "cuvis-ai-core==0.7.3"
    assert payload["plugins"][0]["sha"] == FAKE_SHA

    # The human-readable companion names the resolved core + plugin.
    manifest = (venv.parent / "env_desc.md").read_text()
    assert "cuvis-ai-core==0.7.3" in manifest
    assert "https://example.com/p.git" in manifest
    assert "v0.1.0" in manifest
    assert FAKE_SHA[:8] in manifest


def test_compose_env_cache_hit_skips_uv(tmp_path: Path):
    # First call materialises the entry.
    resolve_patch, lock_patch, sync_patch = _patch_resolve_and_uv()
    with resolve_patch, lock_patch as lock_mock, sync_patch as sync_mock:
        sync_mock.side_effect = lambda project_dir: (project_dir / ".venv").mkdir()
        first = compose_env(
            _simple_plugin(),
            core_source=PYPI_CORE,
            cache_root=tmp_path,
        )
        assert lock_mock.call_count == 1
        assert sync_mock.call_count == 1

    # Second call must reuse the published entry without re-running uv.
    resolve_patch2, lock_patch2, sync_patch2 = _patch_resolve_and_uv()
    with resolve_patch2, lock_patch2 as lock_mock2, sync_patch2 as sync_mock2:
        second = compose_env(
            _simple_plugin(),
            core_source=PYPI_CORE,
            cache_root=tmp_path,
        )
        assert lock_mock2.call_count == 0
        assert sync_mock2.call_count == 0
    assert first == second


def test_compose_env_half_built_recovery(tmp_path: Path):
    """If uv_sync crashes after .venv exists but before .ready, the next
    attempt must rename the broken dir aside and rebuild cleanly."""
    call_count = {"n": 0}

    def crashing_then_ok(project_dir: Path):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # First call: leave a half-built .venv then crash.
            (project_dir / ".venv").mkdir(exist_ok=True)
            raise UvRunnerError("simulated crash mid-sync")
        # Second call: succeed.
        (project_dir / ".venv").mkdir(exist_ok=True)

    resolve_patch, lock_patch, sync_patch = _patch_resolve_and_uv()
    with resolve_patch, lock_patch, sync_patch as sync_mock:
        sync_mock.side_effect = crashing_then_ok
        with pytest.raises(UvRunnerError):
            compose_env(
                _simple_plugin(),
                core_source=PYPI_CORE,
                cache_root=tmp_path,
            )

    # After the crash a half-built .building.* dir remains; the next
    # attempt must NOT reuse it as if ready.
    resolve_patch2, lock_patch2, sync_patch2 = _patch_resolve_and_uv()
    with resolve_patch2, lock_patch2 as lock_mock2, sync_patch2 as sync_mock2:
        sync_mock2.side_effect = lambda project_dir: (project_dir / ".venv").mkdir(
            exist_ok=True
        )
        venv = compose_env(
            _simple_plugin(),
            core_source=PYPI_CORE,
            cache_root=tmp_path,
        )
        assert lock_mock2.call_count == 1
        assert sync_mock2.call_count == 1

    assert (venv.parent / ".ready").exists()


def test_compose_env_renames_published_dir_without_ready_aside(tmp_path: Path):
    """Defense in depth: if the published cache dir exists without
    a .ready sentinel, the composer must move it aside and rebuild."""

    # Pre-create a fake published dir that looks complete but has no .ready.
    resolve_patch, lock_patch, sync_patch = _patch_resolve_and_uv()
    with resolve_patch, lock_patch, sync_patch as sync_mock:
        sync_mock.side_effect = lambda project_dir: (project_dir / ".venv").mkdir(
            exist_ok=True
        )

        # Compute the expected cache dir name by running compose once,
        # then deleting only .ready to simulate the failure state.
        venv = compose_env(
            _simple_plugin(),
            core_source=PYPI_CORE,
            cache_root=tmp_path,
        )
        cache_dir = venv.parent
        (cache_dir / ".ready").unlink()

    # Next attempt should rename the broken dir aside and rebuild.
    resolve_patch2, lock_patch2, sync_patch2 = _patch_resolve_and_uv()
    with resolve_patch2, lock_patch2 as lock_mock2, sync_patch2 as sync_mock2:
        sync_mock2.side_effect = lambda project_dir: (project_dir / ".venv").mkdir(
            exist_ok=True
        )
        venv2 = compose_env(
            _simple_plugin(),
            core_source=PYPI_CORE,
            cache_root=tmp_path,
        )
        assert lock_mock2.call_count == 1
    assert venv2.parent.exists()
    assert (venv2.parent / ".ready").exists()
    # The broken dir was moved aside with a .broken.<ts> suffix.
    broken_dirs = [p for p in tmp_path.iterdir() if ".broken." in p.name]
    assert len(broken_dirs) == 1


def test_compose_env_moving_tag_rejected_via_resolver(tmp_path: Path):
    """If the git tag does not resolve, the composer surfaces a
    RuntimeProjectError (not a generic uv failure)."""
    from cuvis_ai_core.orchestrator.runtime_project import RuntimeProjectError

    with patch(
        "cuvis_ai_core.orchestrator.runtime_project.resolve_git_tag",
        side_effect=RuntimeProjectError(
            "Tag 'main' not found in https://example.com/p.git. "
            "Branches and moving refs are not accepted."
        ),
    ):
        with pytest.raises(RuntimeProjectError, match="moving refs"):
            compose_env(
                {
                    "p": GitPluginConfig(
                        repo="https://example.com/p.git",
                        tag="main",
                        provides=[{"class_name": "p.Node"}],
                    )
                },
                core_source=PYPI_CORE,
                cache_root=tmp_path,
            )


def test_compose_env_two_threads_serialise_on_same_key(tmp_path: Path):
    """Two concurrent calls with the same key: only one runs uv;
    the other observes the cache hit."""
    sync_calls = []
    sync_calls_lock = threading.Lock()
    barrier = threading.Barrier(2)

    def slow_sync(project_dir: Path):
        # Wait for both threads to be inside compose_env before any sync
        # completes, so we genuinely race on the lock and don't just
        # happen to land sequentially.
        with sync_calls_lock:
            sync_calls.append(project_dir)
        (project_dir / ".venv").mkdir(exist_ok=True)

    results = {}
    errors = {}

    def worker(idx: int):
        try:
            resolve_patch, lock_patch, sync_patch = _patch_resolve_and_uv()
            with resolve_patch, lock_patch, sync_patch as sync_mock:
                sync_mock.side_effect = slow_sync
                barrier.wait(timeout=5)
                results[idx] = compose_env(
                    _simple_plugin(),
                    core_source=PYPI_CORE,
                    cache_root=tmp_path,
                )
        except Exception as exc:  # pragma: no cover - signals test failure
            errors[idx] = exc

    t1 = threading.Thread(target=worker, args=(1,))
    t2 = threading.Thread(target=worker, args=(2,))
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert not errors, errors
    assert results[1] == results[2]
    # Only one of the two threads should have run uv_sync.
    assert len(sync_calls) == 1


def test_compose_env_respects_env_var_for_cache_root(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CUVIS_RUN_CACHE_DIR", str(tmp_path))
    resolve_patch, lock_patch, sync_patch = _patch_resolve_and_uv()
    with resolve_patch, lock_patch, sync_patch as sync_mock:
        sync_mock.side_effect = lambda project_dir: (project_dir / ".venv").mkdir(
            exist_ok=True
        )
        venv = compose_env(_simple_plugin(), core_source=PYPI_CORE)
    assert tmp_path in venv.parents


def test_compose_env_keeps_failed_build_dir_for_forensics(tmp_path: Path):
    """When uv_sync fails, the .building.* dir is left in place so the
    user can inspect logs; only the sweep removes it later."""
    resolve_patch, lock_patch, sync_patch = _patch_resolve_and_uv()
    with resolve_patch, lock_patch, sync_patch as sync_mock:
        sync_mock.side_effect = UvRunnerError("network down")
        with pytest.raises(UvRunnerError):
            compose_env(
                _simple_plugin(),
                core_source=PYPI_CORE,
                cache_root=tmp_path,
            )
    building = [p for p in tmp_path.iterdir() if ".building." in p.name]
    assert len(building) == 1


def test_compose_error_class_is_runtime_error_subclass():
    assert issubclass(ComposerError, RuntimeError)


# ---------------------------------------------------------------------------
# _build_lock timeout, cache-root default, stale-partial sweep, _rmtree
# ---------------------------------------------------------------------------


def test_build_lock_times_out_raises_composer_error(tmp_path: Path, monkeypatch):
    class _StuckLock:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def acquire(self, timeout=None):
            raise composer_mod.Timeout("held elsewhere")

        def release(self) -> None:  # pragma: no cover - not reached on timeout
            pass

    monkeypatch.setattr(composer_mod, "FileLock", _StuckLock)
    with pytest.raises(ComposerError, match="Timed out"):
        with composer_mod._build_lock("deadbeef", tmp_path):
            pass


def test_resolve_cache_root_defaults_without_override_or_env(monkeypatch):
    monkeypatch.delenv("CUVIS_RUN_CACHE_DIR", raising=False)
    assert composer_mod._resolve_cache_root(None) == composer_mod._DEFAULT_CACHE_ROOT


def test_sweep_stale_partials_noop_when_root_missing(tmp_path: Path):
    # Must not raise when the cache root has never been created.
    composer_mod._sweep_stale_partials(tmp_path / "never_created")


def test_sweep_stale_partials_removes_old_and_keeps_fresh(tmp_path: Path):
    import os

    stale = tmp_path / f"abc{composer_mod._BUILDING_TAG}123.deadbe"
    fresh = tmp_path / f"def{composer_mod._BUILDING_TAG}456.beadfe"
    stale.mkdir()
    fresh.mkdir()
    old = time.time() - composer_mod._STALE_PARTIAL_AGE_SECONDS - 100
    os.utime(stale, (old, old))

    composer_mod._sweep_stale_partials(tmp_path)

    assert not stale.exists()
    assert fresh.exists()


def test_rmtree_swallows_oserror(tmp_path: Path, monkeypatch):
    def _boom(path, ignore_errors=False):
        raise OSError("device busy")

    monkeypatch.setattr(composer_mod.shutil, "rmtree", _boom)
    # Logged, not raised.
    composer_mod._rmtree(tmp_path)
