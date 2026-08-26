"""Per-pipeline child venv composer.

Public entry point: :func:`compose_env` takes a plugin set and returns
the path to a ready ``.venv`` containing exactly that set. Cache hits
are O(filesystem stat); misses run ``uv lock`` + ``uv sync`` once and
publish atomically.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import secrets
import shutil
import sys
import threading
import time
import weakref
from pathlib import Path
from typing import Iterator, Mapping

from filelock import FileLock, Timeout
from loguru import logger

from cuvis_ai_core.orchestrator import leases
from cuvis_ai_core.orchestrator.cache_key import (
    COMPOSER_SCHEMA_VERSION,
    CacheKey,
    CoreSource,
    compute_cache_key,
    spec_hash_of,
)
from cuvis_ai_core.orchestrator.env_config import number_from_env
from cuvis_ai_core.orchestrator.runtime_project import (
    PluginManifest,
    build_runtime_pyproject,
    resolve_plugin_sources,
)
from cuvis_ai_core.orchestrator.uv_runner import (
    UvCacheBusyError,
    UvRunnerError,
    uv_cache_prune,
    uv_lock,
    uv_sync,
)

_DEFAULT_CACHE_ROOT_ENV = "CUVIS_RUN_CACHE_DIR"
_DEFAULT_CACHE_ROOT = Path.home() / ".cuvis_runs"
_LOCK_TIMEOUT_SECONDS = 1800  # cold-start install can take a long time

# Pin composed child envs to the composing interpreter's minor version. Leaving
# the range open (e.g. ">=3.11,<3.14") let uv pick a newer Python (3.13) whose
# matplotlib wheel ships a broken ft2font on Windows, crashing the child runtime.
# The child must track the parent stack's Python, so derive it from sys.
_PARENT_PYTHON_REQUIRES = (
    f">={sys.version_info.major}.{sys.version_info.minor},"
    f"<{sys.version_info.major}.{sys.version_info.minor + 1}"
)
_STALE_PARTIAL_AGE_SECONDS = 6 * 60 * 60  # sweep half-built dirs older than 6h

# Cache-protocol filenames/markers. The writer constructs them and the
# sweeper/cache-hit check recognise them; sharing the constants keeps the
# two sides from drifting.
_LOCKS_DIRNAME = ".locks"
_READY_MARKER = ".ready"
_PYPROJECT_NAME = "pyproject.toml"
_KEY_JSON_NAME = "key.json"
_MANIFEST_NAME = "env_desc.md"
_BUILDING_TAG = ".building."
_BROKEN_TAG = ".broken."
_EVICTING_TAG = ".evicting."

# Eviction policy knobs. 0 disables the corresponding dimension;
# schema-stale eviction (entries whose key.json predates the current
# COMPOSER_SCHEMA_VERSION — dead by construction) is always active.
_MAX_ENTRIES_ENV = "CUVIS_RUN_CACHE_MAX_ENTRIES"
_MAX_ENTRIES_DEFAULT = 10
_MAX_AGE_DAYS_ENV = "CUVIS_RUN_CACHE_MAX_AGE_DAYS"
_MAX_AGE_DAYS_DEFAULT = 30.0
# Hot floor: entries whose .ready was touched more recently than this are
# never evicted — it closes the window between another server's cache hit
# and its lease write. Tests and the count-cap E2E set it to 0.
_MIN_IDLE_ENV = "CUVIS_RUN_CACHE_MIN_IDLE_SECONDS"
_MIN_IDLE_DEFAULT = 3600.0


class ComposerError(RuntimeError):
    """Raised when a composed env cannot be produced."""


# Per-key in-process mutex layered on top of the cross-process file
# lock so two threads in the same process serialise cheaply rather
# than thrashing the OS lock primitive. A WeakValueDictionary keeps the
# map from growing without bound on a long-lived server: while a build
# holds (or waits on) a key's lock the caller's local reference keeps it
# alive, so concurrent callers share the same object; once no one holds
# it the entry is garbage-collected — the digest space is unbounded over
# a server's lifetime (every new dependency set mints one), so an
# unevicted dict would slowly leak locks.
_in_process_locks: "weakref.WeakValueDictionary[str, threading.Lock]" = (
    weakref.WeakValueDictionary()
)
_in_process_locks_guard = threading.Lock()


def _in_process_lock_for(digest: str) -> threading.Lock:
    with _in_process_locks_guard:
        lock = _in_process_locks.get(digest)
        if lock is None:
            lock = threading.Lock()
            _in_process_locks[digest] = lock
        return lock


def _build_dir_name(final_name: str) -> str:
    """Unique temp-dir name for an in-progress build of ``final_name``."""
    return f"{final_name}{_BUILDING_TAG}{os.getpid()}.{secrets.token_hex(3)}"


def _is_partial_build_dir(name: str) -> bool:
    """True if ``name`` is an in-progress or abandoned build dir."""
    return _BUILDING_TAG in name


@contextlib.contextmanager
def _build_lock(digest: str, locks_dir: Path) -> Iterator[None]:
    """Serialise builds of one cache key.

    Layers the per-key in-process mutex over the cross-process file lock,
    then yields with both held. Releasing happens in reverse on exit.
    """
    in_proc_lock = _in_process_lock_for(digest)
    file_lock = FileLock(str(locks_dir / f"{digest}.lock"))
    with in_proc_lock:
        try:
            file_lock.acquire(timeout=_LOCK_TIMEOUT_SECONDS)
        except Timeout as exc:
            raise ComposerError(
                f"Timed out after {_LOCK_TIMEOUT_SECONDS}s waiting for build "
                f"lock on cache key {digest}."
            ) from exc
        try:
            yield
        finally:
            file_lock.release()


def compose_env(
    plugin_configs: Mapping[str, PluginManifest],
    *,
    core_source: CoreSource,
    cache_root: Path | None = None,
    python_requires: str = _PARENT_PYTHON_REQUIRES,
    active_data_module: str | None = None,
) -> Path:
    """Materialise (or reuse) a cached venv for ``plugin_configs``.

    Returns the path to the ``.venv`` directory inside the published
    cache entry. The caller spawns ``venv_python(...)`` against this
    path. ``active_data_module`` scopes which plugin's data-module pip
    extras are installed (a tiff_paired run never pulls a cu3s module's
    ``cuvis`` extra).
    """
    resolved = resolve_plugin_sources(
        plugin_configs, active_data_module=active_data_module
    )
    pyproject_content = build_runtime_pyproject(
        core_source=core_source,
        plugins=resolved,
        python_requires=python_requires,
    )
    spec_hash = spec_hash_of(pyproject_content)
    key = compute_cache_key(
        core_source=core_source,
        plugins=resolved,
        spec_hash=spec_hash,
    )

    root = resolve_cache_root(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    leases.ensure_root_marker(root)
    locks_dir = root / _LOCKS_DIRNAME
    locks_dir.mkdir(exist_ok=True)

    final_dir = root / key.directory_name()
    venv_dir = final_dir / ".venv"

    _sweep_stale_partials(root)

    with _build_lock(key.digest, locks_dir):
        venv_path, built = _build_or_reuse(
            final_dir=final_dir,
            venv_dir=venv_dir,
            root=root,
            key=key,
            pyproject_content=pyproject_content,
        )
    if built:
        # A publish is when the entry count can have grown. Run the pass
        # AFTER the build lock releases — eviction must never extend the
        # lock a concurrent hit is waiting on — and the pass itself only
        # scans and renames; the multi-GB deletes happen on the deleter
        # thread.
        evict_run_cache(root)
    return venv_path


def _build_or_reuse(
    *,
    final_dir: Path,
    venv_dir: Path,
    root: Path,
    key: CacheKey,
    pyproject_content: str,
) -> tuple[Path, bool]:
    """Return ``(venv_dir, built)`` — ``built`` is False on a cache hit."""
    ready = final_dir / _READY_MARKER
    if ready.exists():
        # The .ready mtime is the entry's last-used timestamp — eviction
        # orders by it (LRU) and the hot floor reads it, so every hit must
        # touch it. Best-effort: a read-only cache still serves hits.
        try:
            os.utime(ready)
        except OSError as exc:
            logger.warning(f"Could not touch {ready}: {exc}")
        logger.info(f"Cache hit: {final_dir.name}")
        return venv_dir, False

    # Defense in depth: a published directory without ``.ready`` is
    # broken — rename it aside and rebuild.
    if final_dir.exists():
        broken = root / f"{final_dir.name}{_BROKEN_TAG}{int(time.time())}"
        logger.warning(
            f"Cache dir {final_dir.name} exists without {_READY_MARKER}; moving to "
            f"{broken.name} and rebuilding."
        )
        final_dir.rename(broken)

    build_dir = root / _build_dir_name(final_dir.name)
    build_dir.mkdir(parents=True, exist_ok=False)
    (build_dir / _PYPROJECT_NAME).write_text(pyproject_content, encoding="utf-8")
    (build_dir / _KEY_JSON_NAME).write_text(
        json.dumps(key.serialise(), indent=2), encoding="utf-8"
    )
    # Human-readable companion to key.json: the dir name is an opaque
    # hash, so this records which libraries the env was composed for.
    (build_dir / _MANIFEST_NAME).write_text(key.human_manifest(), encoding="utf-8")

    try:
        logger.info(f"Building cache entry {key.digest} in {build_dir.name}")
        uv_lock(build_dir)
        uv_sync(build_dir)
        (build_dir / _READY_MARKER).write_text("ok", encoding="utf-8")
    except Exception:
        logger.exception(
            f"uv lock/sync failed for {build_dir.name}; leaving for sweep."
        )
        raise

    os.replace(build_dir, final_dir)
    logger.info(f"Published cache entry {final_dir.name}")
    return venv_dir, True


def resolve_cache_root(override: Path | None = None) -> Path:
    """Resolve the composed-env / model cache root.

    Resolves ``override`` -> ``$CUVIS_RUN_CACHE_DIR`` -> the default
    ``~/.cuvis_runs``. Shared with the model-weight cache (``model_cache``) so
    the venv and weight caches sit under one root.
    """
    if override is not None:
        return Path(override)
    env_val = os.environ.get(_DEFAULT_CACHE_ROOT_ENV)
    if env_val:
        return Path(env_val)
    return _DEFAULT_CACHE_ROOT


def _sweep_stale_partials(root: Path) -> None:
    """Remove ``.building.*`` directories older than the staleness threshold."""
    if not root.exists():
        return
    now = time.time()
    cutoff = now - _STALE_PARTIAL_AGE_SECONDS
    for entry in root.iterdir():
        if not _is_partial_build_dir(entry.name) or not entry.is_dir():
            continue
        try:
            mtime = entry.stat().st_mtime
        except OSError:
            continue
        if mtime < cutoff:
            logger.warning(f"Sweeping stale partial cache dir {entry.name}")
            _rmtree(entry)


def _rmtree(path: Path) -> None:
    """Best-effort recursive delete; failures are logged but non-fatal."""
    try:
        shutil.rmtree(path, ignore_errors=False)
    except OSError as exc:
        logger.warning(f"Failed to remove {path}: {exc}")


# ---------------------------------------------------------------------------
# Eviction — bound the cache by schema version, age, and entry count.
#
#   evict_run_cache: scan candidates -> per victim: non-blocking digest
#   lock -> RE-VERIFY protection under the lock -> os.replace to
#   <digest>.evicting.<ts> (Windows refuses while a child holds the venv)
#   -> hand to ONE deleter worker.  The worker rmtrees, then runs a single
#   `uv cache prune` per drained batch (prune before the rmtrees complete
#   would find the blobs still hardlinked and keep them).
#
# Protection, checked before AND re-checked under the lock:
#   - lease with a live child / intent lease with a live parent
#   - .ready younger than the hot floor
#   - any live same-user process executing from inside the entry (covers
#     children that predate leases, mid-spawn children, torn leases)
# ---------------------------------------------------------------------------

# One deleter worker per process: multi-GB rmtrees must never run on the
# compose/request path, and a thread per victim would turn a large backlog
# into an I/O storm. Concurrent servers each run one worker — safe, the
# os.replace loser skips — bounded by server count.
_deleter_queue: "queue.Queue[tuple[str, object]]" = queue.Queue()
_deleter_started = threading.Lock()
_deleter_thread: threading.Thread | None = None


def _ensure_deleter() -> None:
    """Start the singleton deleter worker thread on first use."""
    global _deleter_thread
    with _deleter_started:
        if _deleter_thread is not None and _deleter_thread.is_alive():
            return
        _deleter_thread = threading.Thread(
            target=_deleter_loop, daemon=True, name="cuvis-cache-deleter"
        )
        _deleter_thread.start()


def _deleter_loop() -> None:
    """Drain the deletion queue forever (daemon thread)."""
    while True:
        kind, payload = _deleter_queue.get()
        try:
            if kind == "rmtree":
                _rmtree(Path(str(payload)))
            elif kind == "prune":
                _run_uv_cache_prune()
            elif kind == "signal" and isinstance(payload, threading.Event):
                payload.set()
        except Exception as exc:  # pragma: no cover - worker must survive
            logger.warning(f"Cache deleter task {kind} failed: {exc}")
        finally:
            _deleter_queue.task_done()


def _run_uv_cache_prune() -> None:
    """Run ``uv cache prune`` non-fatally (the global uv lock may be busy)."""
    try:
        uv_cache_prune()
        logger.info("uv cache prune completed after eviction batch.")
    except UvCacheBusyError as exc:
        # Expected on dev machines where a long-running uvx tool holds the
        # lock; the next eviction batch (or clean-run-cache) retries.
        logger.info(f"uv cache prune skipped: {exc}")
    except UvRunnerError as exc:
        logger.warning(f"uv cache prune failed (non-fatal): {exc}")


def wait_for_deleter(timeout: float = 30.0) -> bool:
    """Block until every queued deletion has been processed (tests / CLI)."""
    if _deleter_thread is None or not _deleter_thread.is_alive():
        return _deleter_queue.empty()
    done = threading.Event()
    _deleter_queue.put(("signal", done))
    return done.wait(timeout)


def _resolved_eviction_policy(
    max_entries: int | None,
    max_age_days: float | None,
    min_idle_seconds: float | None,
) -> tuple[int, float, float]:
    """Explicit args win; otherwise the env knobs (warn-and-default)."""
    if max_entries is None:
        max_entries = number_from_env(_MAX_ENTRIES_ENV, _MAX_ENTRIES_DEFAULT, cast=int)
    if max_age_days is None:
        max_age_days = number_from_env(
            _MAX_AGE_DAYS_ENV, _MAX_AGE_DAYS_DEFAULT, cast=float
        )
    if min_idle_seconds is None:
        min_idle_seconds = number_from_env(_MIN_IDLE_ENV, _MIN_IDLE_DEFAULT, cast=float)
    return max_entries, max_age_days, min_idle_seconds


def _ready_entries(root: Path) -> list[tuple[Path, float, int | None]]:
    """Published entries as ``(path, ready_mtime, schema_version|None)``.

    Only directories holding a ``.ready`` marker qualify; dot-prefixed
    dirs (``.locks``, ``.leases``, ``.crash_logs``), ``model_cache``, and
    tagged dirs (building/broken/evicting) never do. A corrupt or missing
    ``key.json`` yields ``None`` — such an entry is excluded from the
    schema-stale fast path but still ages out normally (a corrupt 4-byte
    file must not make a multi-GB entry immortal).
    """
    entries: list[tuple[Path, float, int | None]] = []
    for entry in root.iterdir():
        name = entry.name
        if (
            name.startswith(".")
            or name == "model_cache"
            or _BUILDING_TAG in name
            or _BROKEN_TAG in name
            or _EVICTING_TAG in name
            or not entry.is_dir()
        ):
            continue
        ready = entry / _READY_MARKER
        try:
            mtime = ready.stat().st_mtime
        except OSError:
            continue  # not published (or vanishing) — not a candidate
        schema_version: int | None = None
        try:
            payload = json.loads((entry / _KEY_JSON_NAME).read_text(encoding="utf-8"))
            schema_version = int(payload["schema_version"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning(f"Unreadable key.json in cache entry {name}: {exc}")
        entries.append((entry, mtime, schema_version))
    return entries


def _lease_protected_digests(root: Path) -> set[str]:
    """Digests currently held in use by a lease.

    Final leases protect while their CHILD lives (never the parent — a
    long-running server would otherwise pin every entry it ever used);
    intent leases protect while their parent lives (the child does not
    exist yet). A final lease whose child died while its parent still
    runs is garbage the bridge missed — remove it here; dead leases with
    dead parents stay for the reaper, which also cleans their scratch.
    """
    protected: set[str] = set()
    for path, lease in leases.read_leases(root):
        if lease is None:
            continue  # corrupt: the reaper quarantines
        parent_alive = leases.pid_alive(lease.parent_pid, lease.parent_create_time)
        if lease.phase == "intent":
            if parent_alive:
                protected.add(lease.entry_digest)
            continue
        if leases.pid_alive(lease.child_pid, lease.child_create_time):
            protected.add(lease.entry_digest)
        elif parent_alive:
            # Dead child, live parent: that server's close/lazy-drop missed
            # the lease. Its scratch is still tracked by the live server, so
            # dropping the lease file alone is safe.
            logger.info(f"Removing dead-child lease {path.name}.")
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning(f"Could not remove dead lease {path.name}: {exc}")
    return protected


def _entry_protected(
    entry: Path,
    *,
    now: float,
    min_idle_seconds: float,
    protected_digests: set[str],
    snapshot: "leases.ProcessSnapshot",
) -> bool:
    """Whether ``entry`` must not be evicted right now."""
    try:
        mtime = (entry / _READY_MARKER).stat().st_mtime
    except OSError:
        return True  # vanished or unreadable under us — leave it alone
    if min_idle_seconds > 0 and mtime > now - min_idle_seconds:
        return True
    if entry.name in protected_digests:
        return True
    return snapshot.exe_under(entry)


@contextlib.contextmanager
def _try_build_lock(digest: str, locks_dir: Path) -> Iterator[bool]:
    """Non-blocking variant of :func:`_build_lock`; yields whether both
    the in-process and cross-process locks were acquired."""
    in_proc = _in_process_lock_for(digest)
    got_in_proc = in_proc.acquire(blocking=False)
    file_lock = None
    got_file = False
    if got_in_proc:
        file_lock = FileLock(str(locks_dir / f"{digest}.lock"))
        try:
            file_lock.acquire(timeout=0.01)
            got_file = True
        except Timeout:
            pass
    try:
        yield got_in_proc and got_file
    finally:
        if got_file and file_lock is not None:
            file_lock.release()
        if got_in_proc:
            in_proc.release()


def evict_run_cache(
    cache_root: Path | None = None,
    *,
    max_entries: int | None = None,
    max_age_days: float | None = None,
    min_idle_seconds: float | None = None,
    evict_all: bool = False,
    dry_run: bool = False,
) -> list[str]:
    """One eviction pass; returns the names of the entries it evicted.

    Candidate policy, in order: schema-stale entries (always), entries
    older than the age cap, then oldest-first LRU down to the entry cap.
    ``evict_all`` (the CLI's ``--all``) makes every entry a candidate and
    drops the hot floor — the in-use protections (leases, live-process
    scan) still apply unconditionally. ``dry_run`` reports what the pass
    would evict without renaming, deleting, or sweeping anything.
    Reported by name/count only — under uv's hardlink mode byte counts
    would overstate reclaimed space, and the deleter's post-batch
    ``uv cache prune`` is what actually releases blob storage.
    """
    root = resolve_cache_root(cache_root)
    if not root.is_dir():
        return []
    leases.adopt_root_if_composer_shaped(root)
    if not leases.root_guard_ok(root):
        return []
    max_entries, max_age_days, min_idle_seconds = _resolved_eviction_policy(
        max_entries, max_age_days, min_idle_seconds
    )
    if evict_all:
        min_idle_seconds = 0.0
    locks_dir = root / _LOCKS_DIRNAME
    locks_dir.mkdir(exist_ok=True)
    now = time.time()

    if not dry_run:
        _sweep_failed_dirs(root, now)

    entries = _ready_entries(root)
    victims: dict[Path, str] = {}
    if evict_all:
        for entry, _mtime, _schema in entries:
            victims[entry] = "explicit"
    else:
        for entry, _mtime, schema_version in entries:
            if schema_version is not None and schema_version < COMPOSER_SCHEMA_VERSION:
                victims[entry] = "schema-stale"
        if max_age_days > 0:
            age_cutoff = now - max_age_days * 86_400
            for entry, mtime, _schema in entries:
                if entry not in victims and mtime < age_cutoff:
                    victims[entry] = "age"
        if max_entries > 0:
            survivors = sorted(
                (e for e in entries if e[0] not in victims), key=lambda item: item[1]
            )
            overflow = len(survivors) - max_entries
            for entry, _mtime, _schema in survivors[: max(overflow, 0)]:
                victims[entry] = "count"

    if not victims:
        return []

    snapshot = leases.take_process_snapshot()
    protected_digests = _lease_protected_digests(root)
    evicted: list[str] = []
    skipped: list[str] = []
    for entry, reason in victims.items():
        if _entry_protected(
            entry,
            now=now,
            min_idle_seconds=min_idle_seconds,
            protected_digests=protected_digests,
            snapshot=snapshot,
        ):
            skipped.append(entry.name)
            continue
        if dry_run:
            evicted.append(f"{entry.name} ({reason})")
            continue
        with _try_build_lock(entry.name, locks_dir) as locked:
            if not locked:
                skipped.append(entry.name)  # a builder or evictor owns it
                continue
            # Re-verify under the lock: a cache hit touches .ready and a
            # bridge writes its intent lease under this same digest lock,
            # so a fresh check here closes the scan-to-lock window.
            if _entry_protected(
                entry,
                now=now,
                min_idle_seconds=min_idle_seconds,
                protected_digests=_lease_protected_digests(root),
                snapshot=snapshot,
            ):
                skipped.append(entry.name)
                continue
            target = root / (
                f"{entry.name}{_EVICTING_TAG}{int(now)}.{secrets.token_hex(3)}"
            )
            try:
                os.replace(entry, target)
            except OSError as exc:
                # Windows refuses to rename a directory whose files are
                # open — an extra in-use guard beyond the leases.
                logger.info(f"Skipping eviction of {entry.name} (in use): {exc}")
                skipped.append(entry.name)
                continue
            _ensure_deleter()
            _deleter_queue.put(("rmtree", target))
            evicted.append(f"{entry.name} ({reason})")

    if evicted:
        if not dry_run:
            _deleter_queue.put(("prune", None))
        verb = "would remove" if dry_run else "removed"
        logger.info(
            f"Cache eviction: {verb} {len(evicted)} "
            f"entr{'y' if len(evicted) == 1 else 'ies'} [{', '.join(evicted)}]"
            + (f"; skipped in use: {', '.join(skipped)}" if skipped else "")
        )
    elif skipped:
        logger.info(f"Cache eviction: nothing removed; in use: {', '.join(skipped)}")
    return [name.split(" ")[0] for name in evicted]


def _sweep_failed_dirs(root: Path, now: float) -> None:
    """Queue stale ``.broken.*`` / ``.evicting.*`` remnants for deletion.

    These are crash leftovers (a failed publish, or a deleter that died
    mid-rmtree). Deletion goes through the deleter queue — this runs on
    the eviction pass, never inline on the compose path.
    """
    cutoff = now - _STALE_PARTIAL_AGE_SECONDS
    for entry in root.iterdir():
        name = entry.name
        if _BROKEN_TAG not in name and _EVICTING_TAG not in name:
            continue
        if not entry.is_dir():
            continue
        try:
            mtime = entry.stat().st_mtime
        except OSError:
            continue
        if mtime < cutoff:
            logger.warning(f"Sweeping stale cache remnant {name}")
            _ensure_deleter()
            _deleter_queue.put(("rmtree", entry))
