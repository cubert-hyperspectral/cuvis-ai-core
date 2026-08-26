"""In-use leases for composed child venvs, plus the orphan reaper.

A lease marks a cache entry as "a child process is (about to be) running
out of this venv — do not evict it". Leases are owned by the gRPC bridge
(the spawner ABC stays untouched) and live as one JSON file per session
under ``<cache_root>/.leases/``::

    bridge: compose -> INTENT lease (parent-keyed) -> spawn -> FINAL lease
    close_session | lazy dead-child drop | reaper   -> remove lease

    evictor protection (composer.evict_run_cache):
        final lease w/ live child   -> entry protected
        intent lease w/ live parent -> entry protected
        live-process exe under entry-> entry protected (covers children
                                       that predate leases, mid-spawn
                                       children, and torn leases)

    reaper (hourly, post-bind daemon thread in the production server):
        parent alive              -> another live server's lease: skip
        parent dead, child alive  -> verify (pid, create_time) + cmdline
                                     -> terminate/kill the PROCESS TREE
                                     -> only after confirmed dead:
                                        preserve child logs, rm lease,
                                        rmtree session root (guarded)
        parent dead, child dead   -> preserve logs + cleanup
        stale intent (parent dead)-> remove lease file
        catch-all                 -> leaseless run_runtime processes with
                                     a dead parent, running from the
                                     cache root, are reaped too

All liveness decisions use psutil (pid + create_time identity): RPC
unreachability or ``Popen.poll()`` alone never count as process death,
and killing is always gated on the cmdline containing the child-runtime
module name. Everything is same-user — ``AccessDenied`` (another user's
or an elevated process) always means "not ours: skip, never kill".

Constraint: pid-keyed leases assume a **machine-local** cache root. Two
machines sharing ``CUVIS_RUN_CACHE_DIR`` over a network share would GC
each other's live leases; :func:`root_guard_ok` refuses UNC roots and
documentation covers NFS.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import tempfile
import time
from collections.abc import Container
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil
from loguru import logger

from cuvis_ai_core.orchestrator.crash_logs import preserve_session_logs

# Directory / marker names under the cache root. The marker is written by
# the composer when it creates the root; destructive maintenance
# (eviction, reaping) refuses to run on roots that lack it, so a
# mispointed CUVIS_RUN_CACHE_DIR can never turn the janitor loose on a
# directory the composer does not own.
LEASES_DIRNAME = ".leases"
ROOT_MARKER_NAME = ".cuvis_run_cache"

# The child runtime's module path — every kill decision requires this
# marker in the target's cmdline, so a reused pid belonging to some other
# program is never touched.
_RUNTIME_CMDLINE_MARKER = "cuvis_ai_core.run_runtime"

# Name of the per-session scratch parent directory (under the system temp
# dir). The guarded rmtree refuses to delete anything not below a
# directory with this name.
SCRATCH_ROOT_NAME = "cuvis_runtime_sessions"

# Lease-less scratch dirs (children that predate leases have none) are
# swept only past this age AND only when no live process has its cwd
# inside — a live legacy training session must never lose its scratch.
_LEASELESS_SCRATCH_AGE_SECONDS = 7 * 24 * 60 * 60

# psutil create_time round-trips through JSON as a float; allow a small
# tolerance when matching so serialization never breaks pid identity.
_CREATE_TIME_TOLERANCE_S = 1.0

# Grace given to terminate() before escalating to kill(), per wait stage.
_KILL_WAIT_SECONDS = 5.0

_INTENT_PHASE = "intent"
_FINAL_PHASE = "final"


@dataclass(frozen=True)
class Lease:
    """One session's in-use record for a cache entry.

    ``phase`` is ``"intent"`` between compose and spawn (no child pid
    yet; protection keys on the parent staying alive) and ``"final"``
    once the child runs (protection keys on the child alone — parent
    liveness deliberately does NOT protect, or a long-running server
    would pin every entry it ever touched).
    """

    phase: str
    entry_digest: str
    session_id: str
    parent_pid: int
    parent_create_time: float
    child_pid: int | None = None
    child_create_time: float | None = None
    session_root: str | None = None
    created_at: float = 0.0


def leases_dir(cache_root: Path) -> Path:
    """The ``.leases`` directory under ``cache_root`` (not created)."""
    return cache_root / LEASES_DIRNAME


def _lease_path(cache_root: Path, session_id: str) -> Path:
    """Path of the lease file for ``session_id`` (one lease per session)."""
    return leases_dir(cache_root) / f"{session_id}.json"


def _write_lease(cache_root: Path, lease: Lease) -> None:
    """Atomically write ``lease`` (tmp file + ``os.replace``).

    Atomic same-directory replace makes a torn lease impossible by
    construction — a crash leaves either the old file or the new one,
    never a half-written record. Raises ``OSError`` on failure; callers
    treat that as fail-fast (a child without a lease is evictable).
    """
    directory = leases_dir(cache_root)
    directory.mkdir(parents=True, exist_ok=True)
    final = _lease_path(cache_root, lease.session_id)
    tmp = directory / f".{lease.session_id}.{secrets.token_hex(3)}.tmp"
    tmp.write_text(json.dumps(asdict(lease), indent=2), encoding="utf-8")
    os.replace(tmp, final)


def write_intent_lease(cache_root: Path, session_id: str, entry_digest: str) -> None:
    """Record that this server is about to spawn a child from ``entry_digest``.

    Written immediately after ``compose_env`` returns, before the spawn —
    the endpoint/health polling window can last minutes and the entry
    must already be protected. Raises ``OSError`` on write failure.
    """
    me = psutil.Process()
    _write_lease(
        cache_root,
        Lease(
            phase=_INTENT_PHASE,
            entry_digest=entry_digest,
            session_id=session_id,
            parent_pid=me.pid,
            parent_create_time=me.create_time(),
            created_at=time.time(),
        ),
    )


def finalize_lease(
    cache_root: Path,
    session_id: str,
    entry_digest: str,
    *,
    child_pid: int,
    session_root: Path,
) -> None:
    """Replace the intent lease with the final child-keyed lease.

    ``session_root`` is the per-session scratch parent (the unit
    ``close_session`` deletes — recording only the scratch subdir would
    leak its ``output`` sibling). Raises ``OSError`` on write failure and
    ``psutil.NoSuchProcess`` when the child died before it could be
    recorded; callers fail the spawn in both cases.
    """
    me = psutil.Process()
    child_create_time = psutil.Process(child_pid).create_time()
    _write_lease(
        cache_root,
        Lease(
            phase=_FINAL_PHASE,
            entry_digest=entry_digest,
            session_id=session_id,
            parent_pid=me.pid,
            parent_create_time=me.create_time(),
            child_pid=child_pid,
            child_create_time=child_create_time,
            session_root=str(session_root),
            created_at=time.time(),
        ),
    )


def remove_lease(cache_root: Path, session_id: str) -> None:
    """Delete the session's lease file (missing is fine)."""
    try:
        _lease_path(cache_root, session_id).unlink(missing_ok=True)
    except OSError as exc:
        logger.warning(f"Could not remove lease for session {session_id}: {exc}")


def read_leases(cache_root: Path) -> list[tuple[Path, Lease | None]]:
    """All lease files under the root; ``None`` payload marks a corrupt file.

    Corrupt entries are returned (not silently dropped) so callers can
    quarantine them — but with atomic writes they only appear as crash
    artifacts, and a corrupt lease is NEVER grounds for a kill.
    """
    directory = leases_dir(cache_root)
    if not directory.is_dir():
        return []
    results: list[tuple[Path, Lease | None]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            results.append((path, Lease(**payload)))
        except (OSError, ValueError, TypeError) as exc:
            logger.warning(f"Unreadable lease {path.name}: {exc}")
            results.append((path, None))
    return results


def quarantine_lease(path: Path) -> None:
    """Move a corrupt lease aside so it is not re-parsed every pass."""
    try:
        os.replace(path, path.with_suffix(".corrupt"))
    except OSError as exc:
        logger.warning(f"Could not quarantine corrupt lease {path.name}: {exc}")


def pid_alive(pid: int | None, create_time: float | None) -> bool:
    """True when ``pid`` exists AND matches ``create_time`` (no pid reuse).

    A recycled pid (same number, different process) fails the create_time
    match and counts as dead — the recorded process is gone.
    """
    if pid is None or create_time is None:
        return False
    try:
        actual = psutil.Process(pid).create_time()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        # AccessDenied: not our process (leases only record same-user
        # processes), so the recorded one is gone.
        return False
    return abs(actual - create_time) < _CREATE_TIME_TOLERANCE_S


def ensure_root_marker(cache_root: Path) -> None:
    """Write the root marker file that authorizes maintenance on this root."""
    marker = cache_root / ROOT_MARKER_NAME
    try:
        if not marker.exists():
            marker.write_text(
                "Composed child-env cache root managed by cuvis-ai-core.\n",
                encoding="utf-8",
            )
    except OSError as exc:
        logger.warning(f"Could not write cache-root marker {marker}: {exc}")


def adopt_root_if_composer_shaped(cache_root: Path) -> None:
    """Mark a pre-marker cache root as managed when it is clearly ours.

    Roots created by older releases have entries and a ``.locks`` dir but
    no marker file; without adoption, the first post-upgrade maintenance
    pass would refuse to clean the old backlog until some compose runs.
    A ``.locks`` directory is created by every compose and is not a
    plausible feature of a mispointed directory, so its presence is the
    adoption criterion.
    """
    marker = cache_root / ROOT_MARKER_NAME
    if marker.exists():
        return
    if (cache_root / ".locks").is_dir():
        logger.info(f"Adopting pre-existing composer cache root {cache_root}.")
        ensure_root_marker(cache_root)


def root_guard_ok(cache_root: Path) -> bool:
    """Whether destructive maintenance may run on ``cache_root``.

    Refuses roots without the composer's marker file (a mispointed
    ``CUVIS_RUN_CACHE_DIR`` must never be recursively cleaned) and UNC
    roots (pid-keyed leases are meaningless across machines: one machine
    would treat the other's live pids as dead and evict in-use envs).
    """
    if str(cache_root).startswith("\\\\"):
        logger.error(
            f"Refusing cache maintenance on network share {cache_root}: "
            f"pid-keyed leases are machine-local."
        )
        return False
    if not (cache_root / ROOT_MARKER_NAME).is_file():
        if root_never_composed(cache_root):
            # A fresh install: the root appears (with its marker) at the
            # first compose. Nothing to protect and nothing to clean.
            logger.debug(f"No cache maintenance on {cache_root}: nothing composed yet.")
            return False
        logger.error(
            f"Refusing cache maintenance on {cache_root}: no {ROOT_MARKER_NAME} "
            f"marker — not a composer-managed cache root."
        )
        return False
    return True


def root_never_composed(cache_root: Path) -> bool:
    """A missing or empty root has never seen a compose."""
    try:
        return not cache_root.exists() or not any(cache_root.iterdir())
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Process snapshot — one same-user scan shared by the evictor and reaper.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProcessInfo:
    """The per-process facts the evictor and reaper decide on."""

    pid: int
    create_time: float
    ppid: int
    exe: str | None
    cwd: str | None
    is_run_runtime: bool


@dataclass(frozen=True)
class ProcessSnapshot:
    """A point-in-time view of the processes this user can see.

    Everything here is same-user and unprivileged: fields another user's
    (or an elevated) process refuses to expose are recorded as ``None``
    and only ever cause a skip, never an action.
    """

    infos: tuple[ProcessInfo, ...]

    def exe_under(self, directory: Path) -> bool:
        """True when any live process executes from inside ``directory``."""
        return any(_is_under(info.exe, directory) for info in self.infos)

    def cwd_under(self, directory: Path) -> bool:
        """True when any live process has its cwd inside ``directory``."""
        return any(_is_under(info.cwd, directory) for info in self.infos)


def _is_under(candidate: str | None, directory: Path) -> bool:
    """Case-normalized ``candidate is inside directory`` path test."""
    if not candidate:
        return False
    a = os.path.normcase(os.path.abspath(candidate))
    d = os.path.normcase(str(directory))
    return a == d or a.startswith(d + os.sep)


def take_process_snapshot() -> ProcessSnapshot:
    """Scan live processes once (AccessDenied-tolerant, same-user scope)."""
    infos: list[ProcessInfo] = []
    for proc in psutil.process_iter():
        try:
            with proc.oneshot():
                pid = proc.pid
                create_time = proc.create_time()
                ppid = proc.ppid()
                exe = _safe_proc_field(proc, "exe")
                cwd = _safe_proc_field(proc, "cwd")
                cmdline = _safe_cmdline(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        is_runtime = any(_RUNTIME_CMDLINE_MARKER in part for part in cmdline)
        infos.append(
            ProcessInfo(
                pid=pid,
                create_time=create_time,
                ppid=ppid,
                exe=exe,
                cwd=cwd,
                is_run_runtime=is_runtime,
            )
        )
    return ProcessSnapshot(infos=tuple(infos))


def _safe_proc_field(proc: psutil.Process, field_name: str) -> str | None:
    """Read ``proc.<field_name>()``, mapping any psutil denial to ``None``."""
    try:
        value = getattr(proc, field_name)()
        return str(value) if value else None
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, OSError):
        return None


def _safe_cmdline(proc: psutil.Process) -> list[str]:
    """Read ``proc.cmdline()``, mapping any psutil denial to an empty list."""
    try:
        return proc.cmdline() or []
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, OSError):
        return []


# ---------------------------------------------------------------------------
# Reaper — kill confirmed orphans, then clean their leases and scratch.
# ---------------------------------------------------------------------------


def reap_orphans(cache_root: Path) -> None:
    """One reap pass: orphaned children killed, dead leases and scratch cleaned.

    Safe to run concurrently with other servers sharing the root: leases
    whose parent is alive belong to them and are skipped; every kill is
    gated on (pid, create_time) identity plus the run_runtime cmdline
    marker, and cleanup happens only after death is confirmed. A legacy
    root (entries but no marker) is adopted first, as the evictor does,
    so the first post-upgrade pass reaps instead of waiting a cycle.
    """
    adopt_root_if_composer_shaped(cache_root)
    if not root_guard_ok(cache_root):
        return
    snapshot = take_process_snapshot()

    surviving_session_roots: list[Path] = []
    handled_pids: set[int] = set()
    for path, lease in read_leases(cache_root):
        if lease is None:
            quarantine_lease(path)
            continue
        if pid_alive(lease.parent_pid, lease.parent_create_time):
            # A live server owns this lease (possibly us). Leave it alone.
            if lease.session_root:
                surviving_session_roots.append(Path(lease.session_root))
            continue
        if lease.phase == _INTENT_PHASE:
            # The composing server died before the spawn finished; any
            # actually-spawned process is caught by the catch-all scan.
            logger.info(f"Removing stale intent lease {path.name} (parent dead).")
            _unlink_quietly(path)
            continue
        if pid_alive(lease.child_pid, lease.child_create_time):
            assert lease.child_pid is not None  # pid_alive checked it
            handled_pids.add(lease.child_pid)
            confirmed_dead = _kill_runtime_tree(
                lease.child_pid, lease.child_create_time
            )
            if not confirmed_dead:
                # Never destroy the protection record of a live process.
                logger.error(
                    f"Orphan child pid={lease.child_pid} (session "
                    f"{lease.session_id}) survived kill; keeping its lease."
                )
                if lease.session_root:
                    surviving_session_roots.append(Path(lease.session_root))
                continue
            logger.info(
                f"Reaped orphan child pid={lease.child_pid} "
                f"(session {lease.session_id}, entry {lease.entry_digest})."
            )
        _unlink_quietly(path)
        if lease.session_root:
            _preserve_and_remove_session_root(
                Path(lease.session_root), lease.session_id
            )

    _reap_leaseless_orphans(cache_root, snapshot, handled_pids)
    _sweep_leaseless_scratch(cache_root, snapshot, surviving_session_roots)


def _unlink_quietly(path: Path) -> None:
    """Best-effort unlink used for lease files during a reap pass."""
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning(f"Could not remove {path.name}: {exc}")


def _kill_runtime_tree(pid: int, create_time: float | None) -> bool:
    """Terminate → kill the child-runtime process tree; True when all died.

    Guards, in order: (pid, create_time) identity (a recycled pid is
    someone else's process), then the run_runtime cmdline marker
    (AccessDenied ⇒ not ours ⇒ never kill). Children are collected
    recursively so plugin worker processes (dataloaders) don't survive
    their root and keep using files we are about to delete.
    """
    try:
        proc = psutil.Process(pid)
        if create_time is not None and (
            abs(proc.create_time() - create_time) >= _CREATE_TIME_TOLERANCE_S
        ):
            logger.warning(f"pid {pid} was recycled by another process; not killing.")
            return False
        if not any(_RUNTIME_CMDLINE_MARKER in part for part in _safe_cmdline(proc)):
            logger.warning(
                f"pid {pid} does not look like a child runtime; not killing."
            )
            return False
        targets = [proc, *proc.children(recursive=True)]
    except psutil.NoSuchProcess:
        return True  # already gone
    except (psutil.AccessDenied, psutil.ZombieProcess):
        return False

    for target in targets:
        try:
            target.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    _gone, alive = psutil.wait_procs(targets, timeout=_KILL_WAIT_SECONDS)
    for target in alive:
        try:
            target.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    _gone, alive = psutil.wait_procs(alive, timeout=_KILL_WAIT_SECONDS)
    return not alive


def _reap_leaseless_orphans(
    cache_root: Path,
    snapshot: ProcessSnapshot,
    handled_pids: Container[int] = frozenset(),
) -> None:
    """Kill run_runtime processes under this root whose parent is dead.

    The catch-all for children no lease describes: v4-era children
    (spawned before leases existed), mid-spawn orphans, and torn-lease
    survivors. The exe-under-root check scopes it to THIS cache root, so
    a runtime started from a different root is never touched. Pids the
    lease pass already dealt with are skipped, and the snapshot predates
    that pass, so liveness is re-checked before anything is reported.
    """
    for info in snapshot.infos:
        if not info.is_run_runtime or info.pid in handled_pids:
            continue
        if not _is_under(info.exe, cache_root):
            continue
        if _parent_alive(info):
            continue
        if not pid_alive(info.pid, info.create_time):
            continue
        logger.info(
            f"Reaping leaseless orphan child runtime pid={info.pid} "
            f"(exe under {cache_root})."
        )
        _kill_runtime_tree(info.pid, info.create_time)


def _parent_alive(info: ProcessInfo) -> bool:
    """Whether the recorded parent of ``info`` is still the live parent.

    POSIX reparents orphans to pid 1 (or 0), which reads as "dead
    parent". A parent pid recycled by a *younger* process than the child
    is pid reuse, not a live parent.
    """
    if info.ppid <= 1:
        return False
    try:
        parent_create = psutil.Process(info.ppid).create_time()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False
    return parent_create <= info.create_time + _CREATE_TIME_TOLERANCE_S


def _preserve_and_remove_session_root(session_root: Path, session_id: str) -> None:
    """Preserve the child's logs, then delete the session scratch tree.

    A reaped orphan is by definition an abnormal exit; its stderr log is
    the only postmortem evidence, so it is copied into the crash-log
    store (``crash_logs.crash_dir_root()``) before the tree goes (field
    lesson: an unconditional rmtree destroyed the one log that explained
    a training crash).
    """
    preserved = preserve_session_logs(session_root, session_id=session_id)
    if preserved is not None:
        logger.info(f"Preserved child logs for session {session_id}: {preserved}")
    _guarded_rmtree(session_root)


def _guarded_rmtree(path: Path) -> None:
    """Recursively delete ``path`` only when it is a per-session scratch dir.

    Refuses anything that is not strictly below a ``cuvis_runtime_sessions``
    directory (and never that directory itself) — the reaper must not be
    able to delete arbitrary trees through a corrupted session_root value.
    """
    parts = [p.lower() for p in path.parts]
    if SCRATCH_ROOT_NAME not in parts or path.name.lower() == SCRATCH_ROOT_NAME:
        logger.error(f"Refusing to delete {path}: not a per-session scratch dir.")
        return
    try:
        shutil.rmtree(path, ignore_errors=False)
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.warning(f"Could not remove session scratch {path}: {exc}")


def _sweep_leaseless_scratch(
    cache_root: Path,
    snapshot: ProcessSnapshot,
    surviving_session_roots: list[Path],
    *,
    min_age_seconds: float | None = None,
    dry_run: bool = False,
) -> list[Path]:
    """Sweep old scratch dirs that no lease references.

    Roots: the current temp dir's ``cuvis_runtime_sessions`` plus the
    parents of every surviving lease's session_root (TEMP can differ
    between user/service contexts, so lease paths are the second source
    of truth). Lease-less dirs (pre-lease-era children have none!) go
    only when older than ``min_age_seconds`` (default 7 days) AND no
    live process has its cwd inside — both guards exist so a live
    legacy session never loses its scratch. Returns the swept (or,
    under ``dry_run``, would-be-swept) directories.
    """
    if min_age_seconds is None:
        min_age_seconds = _LEASELESS_SCRATCH_AGE_SECONDS
    sweep_roots = {Path(tempfile.gettempdir()) / SCRATCH_ROOT_NAME}
    referenced: set[str] = set()
    for session_root in surviving_session_roots:
        sweep_roots.add(session_root.parent)
        referenced.add(os.path.normcase(str(session_root)))

    cutoff = time.time() - min_age_seconds
    swept: list[Path] = []
    for sweep_root in sweep_roots:
        if not sweep_root.is_dir() or sweep_root.name.lower() != SCRATCH_ROOT_NAME:
            continue
        for entry in sweep_root.iterdir():
            if not entry.is_dir():
                continue
            if os.path.normcase(str(entry)) in referenced:
                continue
            try:
                mtime = entry.stat().st_mtime
            except OSError:
                continue
            if mtime > cutoff:
                continue
            if snapshot.cwd_under(entry):
                logger.info(
                    f"Keeping scratch {entry.name}: a live process works in it."
                )
                continue
            if dry_run:
                swept.append(entry)
                continue
            logger.info(f"Sweeping stale lease-less scratch dir {entry.name}.")
            _preserve_and_remove_session_root(entry, entry.name)
            swept.append(entry)
    return swept


def sweep_scratch(
    cache_root: Path,
    *,
    relax_age_floor: bool = False,
    dry_run: bool = False,
) -> list[Path]:
    """Sweep lease-less per-session scratch trees (the CLI's ``--sessions``).

    A sweep-only variant of the reaper's final step: no process is ever
    signalled here. ``relax_age_floor`` drops the 7-day age requirement;
    the liveness guards (live cwd, session roots referenced by a lease
    whose parent or child is still alive) always apply. Returns the
    swept (or, under ``dry_run``, would-be-swept) directories.
    """
    if not root_guard_ok(cache_root):
        return []
    snapshot = take_process_snapshot()
    surviving: list[Path] = []
    for _path, lease in read_leases(cache_root):
        if lease is None or not lease.session_root:
            continue
        if pid_alive(lease.parent_pid, lease.parent_create_time) or pid_alive(
            lease.child_pid, lease.child_create_time
        ):
            surviving.append(Path(lease.session_root))
    return _sweep_leaseless_scratch(
        cache_root,
        snapshot,
        surviving,
        min_age_seconds=0.0 if relax_age_floor else None,
        dry_run=dry_run,
    )
