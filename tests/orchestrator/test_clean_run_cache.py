"""``clean-run-cache`` CLI behaviour.

Everything runs against pytest tmp roots with the psutil scan and the
uv binary stubbed — the CLI must never touch real machine state here.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from click.testing import CliRunner

from cuvis_ai_core.orchestrator import clean_run_cache as cli_mod
from cuvis_ai_core.orchestrator import composer as composer_mod
from cuvis_ai_core.orchestrator import leases as leases_mod
from cuvis_ai_core.orchestrator.cache_key import COMPOSER_SCHEMA_VERSION

# Far above any real pid on Windows or Linux; psutil raises NoSuchProcess.
DEAD_PID = 0x7FFF_FFF0


@pytest.fixture
def sandbox(monkeypatch, tmp_path):
    """Stub the process scan, temp dir, and uv binary for CLI runs.

    Returns the recorded ``uv cache prune`` invocations. The temp-dir
    patch keeps the reaper's scratch sweep inside pytest tmp instead of
    the real %TEMP%.
    """
    fake_temp = tmp_path / "faketemp"
    (fake_temp / leases_mod.SCRATCH_ROOT_NAME).mkdir(parents=True)
    monkeypatch.setattr(leases_mod.tempfile, "gettempdir", lambda: str(fake_temp))
    monkeypatch.setattr(
        leases_mod,
        "take_process_snapshot",
        lambda: leases_mod.ProcessSnapshot(infos=()),
    )
    prune_calls: list[str] = []
    monkeypatch.setattr(cli_mod, "uv_cache_prune", lambda: prune_calls.append("cli"))
    monkeypatch.setattr(
        composer_mod, "uv_cache_prune", lambda: prune_calls.append("deleter")
    )
    return prune_calls


def _marked_root(tmp_path: Path) -> Path:
    """A cache root carrying the composer marker (cleanup allowed)."""
    root = tmp_path / "cache"
    root.mkdir()
    leases_mod.ensure_root_marker(root)
    return root


def _entry(root: Path, name: str, *, age_days: float = 0.0) -> Path:
    """A published, schema-current cache entry aged via its .ready mtime."""
    entry = root / name
    (entry / ".venv").mkdir(parents=True)
    (entry / "key.json").write_text(
        f'{{"schema_version": {COMPOSER_SCHEMA_VERSION}}}', encoding="utf-8"
    )
    ready = entry / ".ready"
    ready.write_text("ok", encoding="utf-8")
    if age_days:
        stamp = time.time() - age_days * 86_400
        os.utime(ready, (stamp, stamp))
    return entry


def _scratch_dir(fake_temp_root: Path, name: str, *, age_days: float) -> Path:
    """A per-session scratch dir under the (fake) temp sweep root."""
    path = fake_temp_root / leases_mod.SCRATCH_ROOT_NAME / name
    path.mkdir(parents=True, exist_ok=True)
    stamp = time.time() - age_days * 86_400
    os.utime(path, (stamp, stamp))
    return path


def _snapshot_with(monkeypatch, **info_fields) -> None:
    """Point the process scan at a single fabricated live process."""
    defaults = dict(
        pid=DEAD_PID,
        create_time=0.0,
        ppid=1,
        exe=None,
        cwd=None,
        is_run_runtime=False,
    )
    defaults.update(info_fields)
    snapshot = leases_mod.ProcessSnapshot(infos=(leases_mod.ProcessInfo(**defaults),))
    monkeypatch.setattr(leases_mod, "take_process_snapshot", lambda: snapshot)


def test_dry_run_lists_and_preserves(tmp_path, sandbox):
    """--dry-run reports the victims, removes nothing, and skips prune."""
    root = _marked_root(tmp_path)
    entry = _entry(root, "victim", age_days=40)
    result = CliRunner().invoke(cli_mod.main, ["--root", str(root), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "Would evict 1" in result.output
    assert "victim" in result.output
    assert "reaping skipped" in result.output
    assert "prune skipped" in result.output
    assert entry.exists()
    assert not [p for p in root.iterdir() if ".evicting." in p.name]
    assert sandbox == []


def test_default_pass_evicts_by_policy_and_prunes(tmp_path, sandbox):
    """The flagless pass applies the configured knobs and ends in a prune."""
    root = _marked_root(tmp_path)
    _entry(root, "old", age_days=40)
    _entry(root, "new")
    result = CliRunner().invoke(cli_mod.main, ["--root", str(root)])
    assert result.exit_code == 0, result.output
    assert "Evicted 1" in result.output
    assert not (root / "old").exists()
    assert (root / "new").exists()
    assert len(sandbox) >= 1


def test_busy_uv_cache_is_reported_as_skipped_prune(tmp_path, sandbox, monkeypatch):
    """Another uv process holding the cache lock is a skip, not a failure."""
    from cuvis_ai_core.orchestrator.uv_runner import UvCacheBusyError

    def _busy() -> None:
        raise UvCacheBusyError("uv cache is in use by another uv process")

    monkeypatch.setattr(cli_mod, "uv_cache_prune", _busy)
    root = _marked_root(tmp_path)
    _entry(root, "kept")
    result = CliRunner().invoke(cli_mod.main, ["--root", str(root)])
    assert result.exit_code == 0, result.output
    assert "uv cache prune skipped" in result.output
    assert "failed" not in result.output


def test_failed_uv_cache_prune_is_reported_non_fatal(tmp_path, sandbox, monkeypatch):
    """A generic uv failure during prune is echoed and does not fail the pass."""
    # Pins the UvRunnerError branch (distinct from the busy-lock skip).
    from cuvis_ai_core.orchestrator.uv_runner import UvRunnerError

    def _broken() -> None:
        raise UvRunnerError("uv exited with status 2")

    monkeypatch.setattr(cli_mod, "uv_cache_prune", _broken)
    root = _marked_root(tmp_path)
    _entry(root, "kept")
    result = CliRunner().invoke(cli_mod.main, ["--root", str(root)])
    assert result.exit_code == 0, result.output
    assert "uv cache prune failed (non-fatal): uv exited with status 2" in result.output
    assert "prune skipped" not in result.output
    assert "prune completed" not in result.output


def test_slow_deleter_drain_warns_but_exits_cleanly(tmp_path, sandbox, monkeypatch):
    """A deleter still busy after the drain timeout is a warning, not an error."""
    # Pins the wait_for_deleter(...) -> False branch.
    waited: list[float] = []

    def _still_draining(timeout: float) -> bool:
        waited.append(timeout)
        return False

    monkeypatch.setattr(cli_mod, "wait_for_deleter", _still_draining)
    root = _marked_root(tmp_path)
    _entry(root, "kept")
    result = CliRunner().invoke(cli_mod.main, ["--root", str(root)])
    assert result.exit_code == 0, result.output
    assert waited == [600.0]
    assert "deletions are still draining" in result.output
    # Nothing was evicted, so the CLI still falls through to its own prune
    # (exactly once; the real deleter is not drained here, so a stray
    # deleter-side prune from elsewhere in the process must not matter).
    assert "uv cache prune completed" in result.output
    assert sandbox.count("cli") == 1


def test_all_respects_live_leases_and_processes(tmp_path, sandbox, monkeypatch):
    """--all ignores the caps but never touches an in-use entry."""
    root = _marked_root(tmp_path)
    _entry(root, "leased")
    busy = _entry(root, "busy")
    _entry(root, "free")
    leases_mod.write_intent_lease(root, "sid-live", "leased")
    leases_mod.finalize_lease(
        root,
        "sid-live",
        "leased",
        child_pid=os.getpid(),
        session_root=tmp_path / "sess",
    )
    _snapshot_with(monkeypatch, exe=str(busy / ".venv" / "python.exe"))
    result = CliRunner().invoke(cli_mod.main, ["--root", str(root), "--all"])
    assert result.exit_code == 0, result.output
    assert "Evicted 1" in result.output
    assert not (root / "free").exists()
    assert (root / "leased").exists()
    assert (root / "busy").exists()


def test_sessions_sweep_honors_cwd_guard(tmp_path, sandbox, monkeypatch):
    """--sessions never sweeps a scratch dir a live process works in."""
    root = _marked_root(tmp_path)
    fake_temp = tmp_path / "faketemp"
    live = _scratch_dir(fake_temp, "live-session", age_days=10)
    dead = _scratch_dir(fake_temp, "dead-session", age_days=10)
    _snapshot_with(monkeypatch, cwd=str(live / "work"))
    result = CliRunner().invoke(cli_mod.main, ["--root", str(root), "--sessions"])
    assert result.exit_code == 0, result.output
    assert live.exists()
    assert not dead.exists()


def test_all_sessions_relaxes_age_floor(tmp_path, sandbox):
    """A fresh lease-less scratch dir goes only under --all --sessions."""
    root = _marked_root(tmp_path)
    fake_temp = tmp_path / "faketemp"
    fresh = _scratch_dir(fake_temp, "fresh-session", age_days=0.1)

    result = CliRunner().invoke(cli_mod.main, ["--root", str(root), "--sessions"])
    assert result.exit_code == 0, result.output
    assert fresh.exists()

    result = CliRunner().invoke(
        cli_mod.main, ["--root", str(root), "--sessions", "--all"]
    )
    assert result.exit_code == 0, result.output
    assert not fresh.exists()


def test_refuses_unmarked_root(tmp_path, sandbox):
    """A directory without the composer marker is refused outright."""
    root = tmp_path / "not-a-cache"
    root.mkdir()
    entry = _entry(root, "data", age_days=400)
    result = CliRunner().invoke(cli_mod.main, ["--root", str(root), "--all"])
    assert result.exit_code != 0
    assert "not a cuvis run-cache root" in result.output
    assert entry.exists()


def test_missing_root_is_noop(tmp_path, sandbox):
    """A nonexistent root reports and exits cleanly."""
    result = CliRunner().invoke(cli_mod.main, ["--root", str(tmp_path / "never")])
    assert result.exit_code == 0, result.output
    assert "does not exist" in result.output


def test_empty_root_is_noop_not_an_error(tmp_path, sandbox):
    """An empty root is a fresh install (nothing composed yet), not a bad path."""
    root = tmp_path / "fresh"
    root.mkdir()
    result = CliRunner().invoke(cli_mod.main, ["--root", str(root)])
    assert result.exit_code == 0, result.output
    assert "nothing composed yet" in result.output
