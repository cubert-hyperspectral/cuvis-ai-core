"""Composer cache + build behaviour tests.

Network and uv invocations are mocked — these tests exercise the
cache-key plumbing, atomic publish, half-built recovery, and the
per-key build lock without depending on the real uv binary or any
external git remote.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import Iterator
from unittest.mock import MagicMock, patch

import pytest
from loguru import logger

from cuvis_ai_core.orchestrator import composer as composer_mod
from cuvis_ai_core.orchestrator import leases as leases_mod
from cuvis_ai_core.orchestrator.cache_key import COMPOSER_SCHEMA_VERSION, CoreSource
from cuvis_ai_core.orchestrator.composer import ComposerError, compose_env
from cuvis_ai_core.orchestrator.uv_runner import UvCacheBusyError, UvRunnerError
from cuvis_ai_schemas.plugin import GitPluginSource


PYPI_CORE = CoreSource(kind="pypi", identity="cuvis-ai-core==0.7.3")
FAKE_SHA = "a" * 40


@pytest.fixture(autouse=True)
def _isolate_in_process_locks():
    """Reset the module-level in-process lock map between tests."""
    composer_mod._in_process_locks.clear()
    yield
    composer_mod._in_process_locks.clear()


@pytest.fixture(autouse=True)
def _never_shell_to_real_uv_prune(monkeypatch):
    """The deleter's post-batch prune must never hit the real uv binary.

    A real ``uv cache prune`` walks the user's multi-GB uv cache and can
    run for minutes on the single deleter thread, starving every later
    ``wait_for_deleter`` call in the suite. Tests that assert prune
    behaviour override this stub with their own recorder.
    """
    monkeypatch.setattr(composer_mod, "uv_cache_prune", lambda: None)


def _simple_plugin() -> dict:
    return {
        "p": GitPluginSource(
            name="p",
            repo="https://example.com/p.git",
            tag="v0.1.0",
            capabilities=[{"class_name": "p.Node"}],
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
    server — every new dependency set mints a fresh digest.
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


def test_compose_env_cache_hit_touches_ready_mtime(tmp_path: Path):
    """Every hit refreshes the .ready mtime — the LRU timestamp for eviction."""
    import os

    resolve_patch, lock_patch, sync_patch = _patch_resolve_and_uv()
    with resolve_patch, lock_patch, sync_patch as sync_mock:
        sync_mock.side_effect = lambda project_dir: (project_dir / ".venv").mkdir()
        venv = compose_env(
            _simple_plugin(),
            core_source=PYPI_CORE,
            cache_root=tmp_path,
        )
    ready = venv.parent / ".ready"
    old = time.time() - 10_000
    os.utime(ready, (old, old))

    resolve_patch2, lock_patch2, sync_patch2 = _patch_resolve_and_uv()
    with resolve_patch2, lock_patch2, sync_patch2:
        compose_env(_simple_plugin(), core_source=PYPI_CORE, cache_root=tmp_path)
    assert ready.stat().st_mtime > old + 5_000


def test_compose_env_cache_hit_survives_utime_failure(tmp_path: Path, monkeypatch):
    """A read-only cache still serves hits: the touch is best-effort."""
    resolve_patch, lock_patch, sync_patch = _patch_resolve_and_uv()
    with resolve_patch, lock_patch, sync_patch as sync_mock:
        sync_mock.side_effect = lambda project_dir: (project_dir / ".venv").mkdir()
        first = compose_env(
            _simple_plugin(),
            core_source=PYPI_CORE,
            cache_root=tmp_path,
        )

    def _refuse(*args, **kwargs):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(composer_mod.os, "utime", _refuse)
    resolve_patch2, lock_patch2, sync_patch2 = _patch_resolve_and_uv()
    with resolve_patch2, lock_patch2, sync_patch2:
        second = compose_env(
            _simple_plugin(), core_source=PYPI_CORE, cache_root=tmp_path
        )
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
                    "p": GitPluginSource(
                        name="p",
                        repo="https://example.com/p.git",
                        tag="main",
                        capabilities=[{"class_name": "p.Node"}],
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
    assert composer_mod.resolve_cache_root(None) == composer_mod._DEFAULT_CACHE_ROOT


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


# ---------------------------------------------------------------------------
# evict_run_cache — candidate policies, protection stack, deletion protocol
# ---------------------------------------------------------------------------

# Far above any real pid on Windows or Linux; psutil raises NoSuchProcess.
DEAD_PID = 0x7FFF_FFF0


def _marked_root(tmp_path: Path) -> Path:
    """A cache root carrying the composer marker (eviction allowed)."""
    root = tmp_path / "cache"
    root.mkdir()
    leases_mod.ensure_root_marker(root)
    return root


def _make_entry(
    root: Path,
    name: str,
    *,
    age_seconds: float = 0.0,
    schema_version: int = COMPOSER_SCHEMA_VERSION,
    key_json: str | None = None,
) -> Path:
    """A published cache entry; ``key_json`` overrides the payload verbatim."""
    entry = root / name
    (entry / ".venv").mkdir(parents=True)
    if key_json is not None:
        (entry / "key.json").write_text(key_json, encoding="utf-8")
    else:
        (entry / "key.json").write_text(
            json.dumps({"schema_version": schema_version}), encoding="utf-8"
        )
    ready = entry / ".ready"
    ready.write_text("ok", encoding="utf-8")
    if age_seconds:
        stamp = time.time() - age_seconds
        os.utime(ready, (stamp, stamp))
    return entry


def _raw_final_lease(
    root: Path, session_id: str, digest: str, *, child_pid: int
) -> Path:
    """A final lease written directly (finalize_lease refuses dead children)."""
    leases_mod.write_intent_lease(root, session_id, digest)
    path = leases_mod.leases_dir(root) / f"{session_id}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(phase="final", child_pid=child_pid, child_create_time=time.time())
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture
def fast_snapshot(monkeypatch):
    """Replace the psutil process scan with an empty snapshot.

    Keeps each eviction test milliseconds instead of a full same-user
    process walk; the scan itself is exercised by the dedicated
    live-process-protection test below.
    """
    monkeypatch.setattr(
        composer_mod.leases,
        "take_process_snapshot",
        lambda: leases_mod.ProcessSnapshot(infos=()),
    )


def _evict(root: Path, **policy) -> list[str]:
    """One eviction pass, deleter drained so filesystem asserts are stable.

    Unspecified policies default to 0 (disabled) so each test opts into
    exactly the policy under test.
    """
    policy.setdefault("max_entries", 0)
    policy.setdefault("max_age_days", 0)
    policy.setdefault("min_idle_seconds", 0)
    evicted = composer_mod.evict_run_cache(root, **policy)
    assert composer_mod.wait_for_deleter(15)
    return evicted


def test_evict_count_cap_lru_oldest_first(tmp_path: Path, fast_snapshot):
    """The count cap drops the least-recently-used entries by .ready mtime."""
    root = _marked_root(tmp_path)
    _make_entry(root, "aaa", age_seconds=300)
    _make_entry(root, "bbb", age_seconds=200)
    _make_entry(root, "ccc", age_seconds=100)
    # Pre-existing lock files of entries the pass does not evict must
    # survive it: the evictor itself never unlinks lock files (on POSIX
    # that would split the lock domain and allow duplicate builds; the
    # filelock library's own Windows release-time delete is handle-safe).
    locks_dir = root / ".locks"
    locks_dir.mkdir()
    (locks_dir / "bbb.lock").touch()
    (locks_dir / "ccc.lock").touch()
    assert _evict(root, max_entries=2) == ["aaa"]
    assert not (root / "aaa").exists()
    assert (root / "bbb").exists()
    assert (root / "ccc").exists()
    assert not [p for p in root.iterdir() if ".evicting." in p.name]
    assert (locks_dir / "bbb.lock").exists()
    assert (locks_dir / "ccc.lock").exists()


def test_evict_age_cap(tmp_path: Path, fast_snapshot):
    """Entries older than the age cap are evicted; younger ones stay."""
    root = _marked_root(tmp_path)
    _make_entry(root, "old", age_seconds=40 * 86_400)
    _make_entry(root, "young", age_seconds=5 * 86_400)
    assert _evict(root, max_age_days=30) == ["old"]
    assert (root / "young").exists()


def test_evict_schema_stale_immediately(tmp_path: Path, fast_snapshot):
    """Pre-v-current entries are candidates regardless of age or count."""
    root = _marked_root(tmp_path)
    _make_entry(root, "stale", schema_version=COMPOSER_SCHEMA_VERSION - 1)
    _make_entry(root, "current")
    assert _evict(root) == ["stale"]
    assert (root / "current").exists()


def test_evict_zero_policies_disable_eviction(tmp_path: Path, fast_snapshot):
    """max_entries=0 and max_age_days=0 switch those policies off."""
    root = _marked_root(tmp_path)
    for offset, name in enumerate(["a1", "b2", "c3"]):
        _make_entry(root, name, age_seconds=(400 - offset) * 86_400)
    assert _evict(root) == []


def test_evict_hot_floor_protects_recent_entries(tmp_path: Path, fast_snapshot):
    """An entry used within the hot floor survives even the schema policy."""
    root = _marked_root(tmp_path)
    _make_entry(
        root, "stale", schema_version=COMPOSER_SCHEMA_VERSION - 1, age_seconds=100
    )
    assert _evict(root, min_idle_seconds=3600) == []
    assert (root / "stale").exists()


def test_evict_policy_resolved_from_env_knobs(
    tmp_path: Path, fast_snapshot, monkeypatch
):
    """Without explicit args the pass reads the CUVIS_RUN_CACHE_* knobs."""
    monkeypatch.setenv("CUVIS_RUN_CACHE_MAX_ENTRIES", "1")
    monkeypatch.setenv("CUVIS_RUN_CACHE_MAX_AGE_DAYS", "0")
    monkeypatch.setenv("CUVIS_RUN_CACHE_MIN_IDLE_SECONDS", "0")
    root = _marked_root(tmp_path)
    _make_entry(root, "older", age_seconds=200)
    _make_entry(root, "newer", age_seconds=100)
    evicted = composer_mod.evict_run_cache(root)
    assert composer_mod.wait_for_deleter(15)
    assert evicted == ["older"]


def test_evict_final_lease_with_live_child_protects(tmp_path: Path, fast_snapshot):
    """A final lease keyed to a live child pins its entry."""
    root = _marked_root(tmp_path)
    entry = _make_entry(root, "leased", age_seconds=90 * 86_400)
    leases_mod.write_intent_lease(root, "sid-live", "leased")
    leases_mod.finalize_lease(
        root,
        "sid-live",
        "leased",
        child_pid=os.getpid(),
        session_root=tmp_path / "sess",
    )
    assert _evict(root, max_age_days=30) == []
    assert entry.exists()


def test_evict_intent_lease_with_live_parent_protects(tmp_path: Path, fast_snapshot):
    """An intent lease protects through the compose-to-spawn window."""
    root = _marked_root(tmp_path)
    entry = _make_entry(root, "spawning", age_seconds=90 * 86_400)
    leases_mod.write_intent_lease(root, "sid-intent", "spawning")
    assert _evict(root, max_age_days=30) == []
    assert entry.exists()


def test_evict_gcs_dead_child_lease_and_reclaims_entry(tmp_path: Path, fast_snapshot):
    """A final lease whose child died (parent alive) is garbage-collected."""
    root = _marked_root(tmp_path)
    _make_entry(root, "was-leased", age_seconds=90 * 86_400)
    lease_path = _raw_final_lease(root, "sid-dead", "was-leased", child_pid=DEAD_PID)
    assert _evict(root, max_age_days=30) == ["was-leased"]
    assert not lease_path.exists()


def test_evict_live_process_scan_protects_unleased_children(
    tmp_path: Path, monkeypatch
):
    """A live process executing from the entry protects it without any lease.

    This is the guard for children that predate leases (schema v4 era),
    mid-spawn children, and torn leases.
    """
    root = _marked_root(tmp_path)
    entry = _make_entry(root, "v4child", age_seconds=90 * 86_400)
    snapshot = leases_mod.ProcessSnapshot(
        infos=(
            leases_mod.ProcessInfo(
                pid=1,
                create_time=0.0,
                ppid=0,
                exe=str(entry / ".venv" / "python.exe"),
                cwd=None,
                is_run_runtime=True,
            ),
        )
    )
    monkeypatch.setattr(composer_mod.leases, "take_process_snapshot", lambda: snapshot)
    assert _evict(root, max_age_days=30) == []
    assert entry.exists()


def test_evict_never_considers_model_cache(tmp_path: Path, fast_snapshot):
    """model_cache holds shared weights and is never an eviction candidate."""
    root = _marked_root(tmp_path)
    model = root / "model_cache"
    model.mkdir()
    ready = model / ".ready"
    ready.write_text("ok", encoding="utf-8")
    stamp = time.time() - 400 * 86_400
    os.utime(ready, (stamp, stamp))
    _make_entry(root, "real", age_seconds=400 * 86_400)
    assert _evict(root, max_age_days=30, max_entries=1) == ["real"]
    assert model.exists()


def test_evict_os_replace_failure_skips_entry(
    tmp_path: Path, fast_snapshot, monkeypatch
):
    """Windows refuses to rename an in-use dir — the pass skips, never fails."""
    root = _marked_root(tmp_path)
    entry = _make_entry(root, "busy", age_seconds=90 * 86_400)

    def _refuse(src, dst):
        raise OSError("directory in use")

    monkeypatch.setattr(composer_mod.os, "replace", _refuse)
    assert _evict(root, max_age_days=30) == []
    assert entry.exists()


def test_evict_skips_entry_whose_lock_is_held(tmp_path: Path, fast_snapshot):
    """A held digest lock (builder or sibling evictor) means non-blocking skip."""
    root = _marked_root(tmp_path)
    entry = _make_entry(root, "locked", age_seconds=90 * 86_400)
    lock = composer_mod._in_process_lock_for("locked")
    lock.acquire()
    try:
        assert _evict(root, max_age_days=30) == []
    finally:
        lock.release()
    assert entry.exists()


def test_evict_corrupt_key_json_not_immortal(tmp_path: Path, fast_snapshot):
    """Corrupt key.json exempts only the schema policy — age still applies."""
    root = _marked_root(tmp_path)
    _make_entry(root, "corrupt-old", key_json="{not json", age_seconds=40 * 86_400)
    _make_entry(root, "corrupt-new", key_json="{not json")
    assert _evict(root, max_age_days=30) == ["corrupt-old"]
    assert (root / "corrupt-new").exists()


def test_evict_reverify_under_lock_aborts_on_new_lease(
    tmp_path: Path, fast_snapshot, monkeypatch
):
    """A lease that lands between the scan and the lock wins the race."""
    root = _marked_root(tmp_path)
    entry = _make_entry(root, "racy", age_seconds=90 * 86_400)
    calls = {"n": 0}

    def _racing(root_arg):
        calls["n"] += 1
        return set() if calls["n"] == 1 else {"racy"}

    monkeypatch.setattr(composer_mod, "_lease_protected_digests", _racing)
    assert _evict(root, max_age_days=30) == []
    assert entry.exists()
    assert calls["n"] >= 2


def test_evict_prunes_uv_cache_once_after_batch_drains(
    tmp_path: Path, fast_snapshot, monkeypatch
):
    """One prune per batch, strictly after the rmtrees (hardlinks pin blobs)."""
    root = _marked_root(tmp_path)
    _make_entry(root, "one", age_seconds=90 * 86_400)
    _make_entry(root, "two", age_seconds=90 * 86_400)
    events: list[str] = []
    monkeypatch.setattr(composer_mod, "_rmtree", lambda path: events.append("rmtree"))
    monkeypatch.setattr(composer_mod, "uv_cache_prune", lambda: events.append("prune"))
    evicted = _evict(root, max_age_days=30)
    assert sorted(evicted) == ["one", "two"]
    assert events == ["rmtree", "rmtree", "prune"]


def test_evict_prune_failure_is_non_fatal(tmp_path: Path, fast_snapshot, monkeypatch):
    """A busy global uv-cache lock must not kill the deleter worker."""
    root = _marked_root(tmp_path)
    _make_entry(root, "gone", age_seconds=90 * 86_400)

    def _busy():
        raise UvRunnerError("uv cache lock busy")

    monkeypatch.setattr(composer_mod, "uv_cache_prune", _busy)
    assert _evict(root, max_age_days=30) == ["gone"]


def test_evict_refuses_unmarked_root(tmp_path: Path, fast_snapshot):
    """No composer marker means no eviction — mispointed roots stay intact."""
    root = tmp_path / "not-a-cache"
    root.mkdir()
    entry = _make_entry(root, "data", age_seconds=90 * 86_400)
    assert _evict(root, max_age_days=30) == []
    assert entry.exists()
    # The guard fires before any side effect on the foreign directory.
    assert not (root / ".locks").exists()


def test_evict_adopts_marker_on_composer_shaped_root(tmp_path: Path, fast_snapshot):
    """A legacy pre-marker cache root (has .locks) is adopted, then bounded."""
    root = tmp_path / "legacy"
    (root / ".locks").mkdir(parents=True)
    _make_entry(root, "v4", schema_version=4, age_seconds=90 * 86_400)
    assert _evict(root) == ["v4"]
    assert (root / leases_mod.ROOT_MARKER_NAME).exists()


def test_evict_missing_root_is_noop(tmp_path: Path):
    """A cache root that does not exist yields an empty pass, no error."""
    assert composer_mod.evict_run_cache(tmp_path / "never") == []


def test_evict_pass_sweeps_stale_broken_and_evicting_dirs(
    tmp_path: Path, fast_snapshot
):
    """Old .broken/.evicting remnants drain through the deleter queue."""
    root = _marked_root(tmp_path)
    old_broken = root / "x.broken.123"
    old_evicting = root / "y.evicting.123.abc"
    fresh_broken = root / "z.broken.456"
    for path in (old_broken, old_evicting, fresh_broken):
        path.mkdir()
    stamp = time.time() - composer_mod._STALE_PARTIAL_AGE_SECONDS - 100
    os.utime(old_broken, (stamp, stamp))
    os.utime(old_evicting, (stamp, stamp))
    assert _evict(root) == []
    assert not old_broken.exists()
    assert not old_evicting.exists()
    assert fresh_broken.exists()


def test_compose_build_runs_eviction_pass_hit_does_not(tmp_path: Path, monkeypatch):
    """A publish triggers one eviction pass; a cache hit triggers none."""
    calls: list[Path] = []
    monkeypatch.setattr(
        composer_mod,
        "evict_run_cache",
        lambda root, **kwargs: calls.append(root),
    )
    resolve_patch, lock_patch, sync_patch = _patch_resolve_and_uv()
    with resolve_patch, lock_patch, sync_patch as sync_mock:
        sync_mock.side_effect = lambda project_dir: (project_dir / ".venv").mkdir(
            exist_ok=True
        )
        compose_env(_simple_plugin(), core_source=PYPI_CORE, cache_root=tmp_path)
    assert calls == [tmp_path]

    resolve_patch2, lock_patch2, sync_patch2 = _patch_resolve_and_uv()
    with resolve_patch2, lock_patch2, sync_patch2:
        compose_env(_simple_plugin(), core_source=PYPI_CORE, cache_root=tmp_path)
    assert calls == [tmp_path]


# ---------------------------------------------------------------------------
# Eviction edge branches — deleter idle paths, vanishing entries, torn and
# stuck leases, cross-process lock contention, remnant-sweep skips
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _captured_log(level: str) -> Iterator[list[str]]:
    """Collect loguru messages at ``level`` and above emitted inside the block."""
    messages: list[str] = []
    handler = logger.add(
        lambda msg: messages.append(msg.record["message"]), level=level
    )
    try:
        yield messages
    finally:
        logger.remove(handler)


def _recorded_sweep_queue(monkeypatch) -> queue.Queue:
    """Redirect the sweep's deleter hand-off into a fresh, inspectable queue."""
    recorded: queue.Queue = queue.Queue()
    monkeypatch.setattr(composer_mod, "_deleter_queue", recorded)
    monkeypatch.setattr(composer_mod, "_ensure_deleter", lambda: None)
    return recorded


def _root_with_unstatable(root: Path, name: str) -> Path:
    """``root`` retyped so ``stat`` on its child ``name`` raises OSError.

    ``is_dir`` keeps answering truthfully via ``os.path.isdir`` — pathlib's
    own implementation routes through ``stat`` and would report "not a
    directory" instead, diverting to the wrong branch.
    """

    class _Flaky(type(Path())):
        def is_dir(self, **kwargs):
            return os.path.isdir(self)

        def stat(self, *args, **kwargs):
            if self.name == name:
                raise OSError("vanished under us")
            return super().stat(*args, **kwargs)

    return _Flaky(root)


def test_run_uv_cache_prune_busy_lock_is_logged_and_swallowed(monkeypatch):
    # Pins the UvCacheBusyError branch: info-level skip, no warning, no raise.
    def _busy():
        raise UvCacheBusyError("uv cache lock busy")

    monkeypatch.setattr(composer_mod, "uv_cache_prune", _busy)
    with _captured_log("INFO") as messages:
        composer_mod._run_uv_cache_prune()
    assert "uv cache prune skipped: uv cache lock busy" in messages
    assert not any("failed" in m for m in messages)


@pytest.mark.parametrize("worker", ["never_started", "exited"])
def test_wait_for_deleter_without_live_worker_reports_queue_state(monkeypatch, worker):
    # Pins the early return: with no live worker nothing could ever signal,
    # so the answer is simply whether the queue is already drained. A fresh
    # queue keeps the real worker (started by earlier tests) from consuming
    # anything and faking a signal round-trip.
    thread = None
    if worker == "exited":
        thread = threading.Thread(target=lambda: None)
        thread.start()
        thread.join()
        assert not thread.is_alive()
    monkeypatch.setattr(composer_mod, "_deleter_thread", thread)
    idle_queue: queue.Queue = queue.Queue()
    monkeypatch.setattr(composer_mod, "_deleter_queue", idle_queue)
    assert composer_mod.wait_for_deleter(0.01) is True
    assert idle_queue.empty()  # no signal was queued
    idle_queue.put(("rmtree", Path("never-processed")))
    assert composer_mod.wait_for_deleter(0.01) is False
    assert idle_queue.qsize() == 1  # pending item reported, not consumed


def test_ready_entries_ignores_dirs_without_ready_marker(tmp_path: Path, fast_snapshot):
    # Pins the stat-OSError `continue` in the candidate scan: a dir with no
    # published .ready (or one vanishing mid-scan) is never a candidate.
    root = _marked_root(tmp_path)
    _make_entry(root, "published", age_seconds=90 * 86_400)
    unpublished = root / "unpublished"
    (unpublished / ".venv").mkdir(parents=True)
    scanned = [
        entry.name for entry, _mtime, _schema in composer_mod._ready_entries(root)
    ]
    assert scanned == ["published"]
    assert _evict(root, evict_all=True) == ["published"]
    assert unpublished.exists()


def test_lease_protection_skips_corrupt_lease_files(tmp_path: Path):
    # Pins the corrupt-lease `continue`: a torn lease neither protects anything
    # nor aborts the scan of the leases sorted behind it.
    root = _marked_root(tmp_path)
    leases_mod.write_intent_lease(root, "sid-ok", "kept")
    torn = leases_mod.leases_dir(root) / "aaa-torn.json"  # sorts first
    torn.write_text("{not json", encoding="utf-8")
    assert composer_mod._lease_protected_digests(root) == {"kept"}
    assert torn.exists()  # quarantine is the reaper's job, not the evictor's


def test_lease_gc_survives_unlink_failure(tmp_path: Path, monkeypatch):
    # Pins the OSError branch of the dead-child lease unlink: warn and carry
    # on, and the dead lease still protects nothing.
    root = _marked_root(tmp_path)
    lease_path = _raw_final_lease(root, "sid-dead", "orphaned", child_pid=DEAD_PID)
    real_unlink = Path.unlink

    def _refuse(self, missing_ok=False):
        if self == lease_path:
            raise PermissionError("lease held open by another process")
        return real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", _refuse)
    with _captured_log("WARNING") as messages:
        assert composer_mod._lease_protected_digests(root) == set()
    assert lease_path.exists()
    assert any("Could not remove dead lease sid-dead.json" in m for m in messages)


def test_evict_leaves_entry_whose_ready_marker_vanished_mid_pass(
    tmp_path: Path, fast_snapshot, monkeypatch
):
    # Pins the stat-OSError branch of the protection check: an entry whose
    # .ready disappears between the scan and the check (a sibling evictor
    # won) counts as protected rather than being renamed away.
    root = _marked_root(tmp_path)
    entry = _make_entry(root, "vanishing", age_seconds=90 * 86_400)
    ready = entry / ".ready"

    def _sibling_won(root_arg):
        ready.unlink(missing_ok=True)
        return set()

    monkeypatch.setattr(composer_mod, "_lease_protected_digests", _sibling_won)
    assert _evict(root, max_age_days=30) == []
    assert entry.exists()
    assert not [p for p in root.iterdir() if ".evicting." in p.name]
    assert composer_mod._entry_protected(
        entry,
        now=time.time(),
        min_idle_seconds=0,
        protected_digests=set(),
        snapshot=leases_mod.ProcessSnapshot(infos=()),
    )


def test_evict_skips_entry_whose_file_lock_is_held(tmp_path: Path, fast_snapshot):
    # Pins the filelock Timeout branch of _try_build_lock: a digest lock held
    # by another process (builder or sibling evictor) means a non-blocking
    # skip. A second FileLock on the same path conflicts even in-process
    # (separate fds under both flock and msvcrt.locking).
    root = _marked_root(tmp_path)
    entry = _make_entry(root, "locked", age_seconds=90 * 86_400)
    locks_dir = root / ".locks"
    locks_dir.mkdir()
    held = composer_mod.FileLock(str(locks_dir / "locked.lock"))
    held.acquire(timeout=1)
    try:
        assert _evict(root, max_age_days=30) == []
    finally:
        held.release()
    assert entry.exists()
    # The in-process half of the lock was released on the way out.
    in_proc = composer_mod._in_process_lock_for("locked")
    assert in_proc.acquire(blocking=False)
    in_proc.release()


def test_sweep_failed_dirs_skips_tagged_plain_files(tmp_path: Path, monkeypatch):
    # Pins the `not entry.is_dir()` continue: only directories are remnants.
    root = _marked_root(tmp_path)
    stray_file = root / "x.broken.123"
    stray_file.write_text("crash note", encoding="utf-8")
    remnant = root / "y.evicting.123.abc"
    remnant.mkdir()
    stamp = time.time() - composer_mod._STALE_PARTIAL_AGE_SECONDS - 100
    os.utime(stray_file, (stamp, stamp))
    os.utime(remnant, (stamp, stamp))
    recorded = _recorded_sweep_queue(monkeypatch)
    composer_mod._sweep_failed_dirs(root, time.time())
    assert recorded.get_nowait() == ("rmtree", remnant)
    assert recorded.empty()
    assert stray_file.exists()


def test_sweep_failed_dirs_skips_remnant_that_vanishes_under_stat(
    tmp_path: Path, monkeypatch
):
    # Pins the stat-OSError continue: a remnant a sibling deleter removes
    # between iterdir and stat is skipped while the rest of the sweep proceeds.
    root = _marked_root(tmp_path)
    ghost = root / "g.evicting.1.abc"
    remnant = root / "r.evicting.1.abc"
    ghost.mkdir()
    remnant.mkdir()
    stamp = time.time() - composer_mod._STALE_PARTIAL_AGE_SECONDS - 100
    os.utime(remnant, (stamp, stamp))
    recorded = _recorded_sweep_queue(monkeypatch)
    composer_mod._sweep_failed_dirs(
        _root_with_unstatable(root, ghost.name), time.time()
    )
    assert recorded.get_nowait() == ("rmtree", remnant)
    assert recorded.empty()
