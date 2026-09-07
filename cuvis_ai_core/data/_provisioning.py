"""Shared plumbing for the ``download-model`` and ``dataset`` provisioning CLIs.

Both tools write into a cache root that a GUI may drive from two processes at
once, both stream progress to a parent that renders it, and both publish their
``--json`` output under a versioned contract. The pieces that must behave
identically live here:

* :func:`root_lock`: one exclusive ``filelock`` per cache root
  (``<root>/.cuvis-cache.lock``) taken by every operation that writes or
  deletes under the root, so two instances serialise instead of interleaving.
* :class:`ProgressEmitter` / :class:`ProgressPoller`: the ``--progress-json``
  protocol, one JSON object per line on stdout, written under one lock so a
  progress line can never tear a terminal ``done`` / ``error`` line.
* :func:`load_schema`: the JSON Schema files under ``cuvis_ai_core/data/schemas``
  that pin every ``--json`` shape (the ``schema`` subcommands print them).

Stdlib plus ``filelock`` only: this module is imported by the registry, which
must stay import-light for the offline child.
"""

from __future__ import annotations

import hashlib
import json
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout

SCHEMA_VERSION = 1
"""Version of every ``--json`` payload and of the ``weights.index.json`` file."""

LOCK_FILENAME = ".cuvis-cache.lock"
"""Lock file name inside a cache root (models) or a datasets root."""

EXIT_ERROR = 1
"""Exit code for an operation that failed (message on stderr)."""

EXIT_USAGE = 2
"""Exit code click uses for a usage error (mutually exclusive flags, bad args)."""

_SHA_READ_CHUNK = 1 << 20  # 1 MiB
_STDOUT_LOCK = threading.Lock()
_SCHEMAS_DIR = Path(__file__).with_name("schemas")


class CacheBusyError(RuntimeError):
    """Raised when the cache root's lock could not be acquired within the timeout."""


def sha256_of(path: Path) -> str:
    """Hex sha256 of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_SHA_READ_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_bytes(count: int | None) -> str:
    """Human-readable IEC size (``3.2 GiB``) for tables; ``-`` for unknown."""
    if count is None:
        return "-"
    if count < 1024:
        return f"{count} B"
    value = float(count)
    for unit in ("KiB", "MiB", "GiB"):
        value /= 1024
        if value < 1024:
            return f"{value:.1f} {unit}"
    return f"{value / 1024:.1f} TiB"


def log(message: str) -> None:
    """Human output goes to stderr; stdout is the machine-readable channel."""
    print(message, file=sys.stderr, flush=True)


def emit_json(payload: Any) -> None:
    """Print one ``--json`` payload (pretty, two-space indent) to stdout."""
    with _STDOUT_LOCK:
        print(json.dumps(payload, indent=2), flush=True)


# ---------------------------------------------------------------------------
# The cache-root lock
# ---------------------------------------------------------------------------


@contextmanager
def root_lock(
    root: Path,
    *,
    timeout: float = 3600.0,
    on_wait: Callable[[], None] | None = None,
) -> Iterator[None]:
    """Hold the exclusive lock of ``root`` for the duration of a writing operation.

    Tries a non-blocking acquire first; when another process holds the lock,
    ``on_wait`` is called once (the CLI tells the user and, in progress mode,
    emits a ``waiting`` event) and the acquire then blocks up to ``timeout``
    seconds. Readers (``status``, ``list``) never take the lock.

    Raises:
        CacheBusyError: the lock stayed busy for ``timeout`` seconds.
    """
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / LOCK_FILENAME
    lock = FileLock(str(lock_path))
    try:
        lock.acquire(timeout=0)
    except Timeout:
        if on_wait is not None:
            on_wait()
        try:
            lock.acquire(timeout=timeout)
        except Timeout as exc:
            raise CacheBusyError(
                f"another operation is using the cache ({lock_path}); gave up after "
                f"{timeout:.0f} s"
            ) from exc
    try:
        yield
    finally:
        lock.release()


# ---------------------------------------------------------------------------
# --progress-json
# ---------------------------------------------------------------------------


class ProgressEmitter:
    """Writes ``--progress-json`` events: one JSON object per line on stdout.

    Every write goes through one process-wide lock and flushes, so a consumer
    reading line by line never sees a torn line and a progress line can never
    follow the terminal ``done`` / ``error`` event of the same item (the
    poller is stopped and joined before the terminal event is written).
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def _write(self, event: dict[str, Any]) -> None:
        if not self.enabled:
            return
        with _STDOUT_LOCK:
            sys.stdout.write(json.dumps(event, separators=(",", ":")) + "\n")
            sys.stdout.flush()

    def waiting(self) -> None:
        """Another process holds the cache lock; the operation is queued."""
        self._write({"event": "waiting"})

    def progress(
        self,
        name: str,
        bytes_done: int | None,
        bytes_total: int | None,
        pct: float | None,
    ) -> None:
        """Bytes landed so far; ``pct`` is ``None`` when the amount is indeterminate."""
        self._write(
            {
                "event": "progress",
                "name": name,
                "bytes_done": bytes_done,
                "bytes_total": bytes_total,
                "pct": pct,
            }
        )

    def files_progress(
        self, name: str, files_done: int, files_total: int | None
    ) -> None:
        """File-count progress (datasets): bytes are indeterminate, files are exact."""
        pct = round(100.0 * files_done / files_total, 1) if files_total else None
        self._write(
            {
                "event": "progress",
                "name": name,
                "bytes_done": None,
                "bytes_total": None,
                "files_done": files_done,
                "files_total": files_total,
                "pct": pct,
            }
        )

    def verifying(self, name: str) -> None:
        """All bytes are on disk; the sha256 pass is running."""
        self._write({"event": "verifying", "name": name})

    def done(self, name: str, path: Path | str) -> None:
        """Terminal success event for one item."""
        self._write({"event": "done", "name": name, "path": str(path)})

    def error(self, name: str, message: str) -> None:
        """Terminal failure event for one item; ``message`` is the mapped cause."""
        self._write({"event": "error", "name": name, "message": message})


class ProgressPoller:
    """Samples a growing file and emits ``progress`` events while its size changes.

    Library-agnostic: ``locate`` returns the file to sample right now (the
    ``.incomplete`` blob ``hf_hub_download`` streams into, or an export/import
    destination), or ``None`` while it does not exist yet. Emits every
    ``interval`` seconds while the size changes. Preallocation guard: when the
    first two samples already equal ``total_bytes`` the writer reserved the
    file up front, so the poller emits ``pct: null`` (indeterminate) instead
    of a false 100 % until it is stopped.
    """

    def __init__(
        self,
        emitter: ProgressEmitter,
        name: str,
        total_bytes: int | None,
        locate: Callable[[], Path | None],
        *,
        interval: float = 0.5,
    ) -> None:
        self._emitter = emitter
        self._name = name
        self._total = total_bytes
        self._locate = locate
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="progress-poller", daemon=True
        )
        self.indeterminate = False

    def start(self) -> ProgressPoller:
        """Start sampling in a background thread; returns ``self`` for chaining."""
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stop sampling and join, so no progress line can follow the caller's next event."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()

    def __enter__(self) -> ProgressPoller:
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    def _sample(self) -> int | None:
        path = self._locate()
        if path is None:
            return None
        try:
            return path.stat().st_size
        except OSError:
            return None

    def _run(self) -> None:
        last: int | None = None
        samples = 0
        while not self._stop.wait(self._interval):
            size = self._sample()
            if size is None:
                continue
            samples += 1
            if (
                self._total
                and samples <= 2
                and size >= self._total
                and not self.indeterminate
            ):
                # Reserved up front: the size says nothing about the bytes landed.
                self.indeterminate = True
                self._emitter.progress(self._name, None, self._total, None)
                continue
            if self.indeterminate or size == last:
                continue
            last = size
            pct = (
                round(min(100.0, 100.0 * size / self._total), 1)
                if self._total
                else None
            )
            self._emitter.progress(self._name, size, self._total, pct)


def newest_incomplete_blob(blobs_dir: Path, not_before: float) -> Path | None:
    """The ``.incomplete`` blob created after ``not_before`` (newest by mtime), or None.

    ``hf_hub_download`` streams a file into ``<repo cache>/blobs/<etag>.incomplete``
    and renames it when complete. Stale ``.incomplete`` files from earlier,
    interrupted runs are ignored by the mtime bound.
    """
    if not blobs_dir.is_dir():
        return None
    newest: tuple[float, Path] | None = None
    for candidate in blobs_dir.glob("*.incomplete"):
        try:
            mtime = candidate.stat().st_mtime
        except OSError:
            continue
        if mtime < not_before:
            continue
        if newest is None or mtime > newest[0]:
            newest = (mtime, candidate)
    return newest[1] if newest else None


def monotonic_start() -> float:
    """Wall-clock stamp taken right before a download starts (for the mtime bound).

    File mtimes are wall-clock, so the bound is ``time.time()``; a one-second
    tolerance covers filesystems with coarse timestamps.
    """
    return time.time() - 1.0


# ---------------------------------------------------------------------------
# JSON Schemas
# ---------------------------------------------------------------------------


def schema_names() -> list[str]:
    """Names of the shipped JSON Schemas (``download-model schema <name>``)."""
    return sorted(
        p.name[: -len(".schema.json")] for p in _SCHEMAS_DIR.glob("*.schema.json")
    )


def load_schema(name: str) -> dict[str, Any]:
    """Load ``cuvis_ai_core/data/schemas/<name>.schema.json``.

    Raises:
        KeyError: no schema of that name ships with this release.
    """
    path = _SCHEMAS_DIR / f"{name}.schema.json"
    if not path.is_file():
        raise KeyError(f"unknown schema '{name}'; known: {', '.join(schema_names())}")
    return json.loads(path.read_text(encoding="utf-8"))
