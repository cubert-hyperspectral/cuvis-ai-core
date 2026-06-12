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

from cuvis_ai_core.orchestrator.cache_key import (
    CacheKey,
    CoreSource,
    compute_cache_key,
    spec_hash_of,
)
from cuvis_ai_core.orchestrator.runtime_project import (
    PluginConfig,
    build_runtime_pyproject,
    resolve_plugin_sources,
)
from cuvis_ai_core.orchestrator.uv_runner import uv_lock, uv_sync

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


class ComposerError(RuntimeError):
    """Raised when a composed env cannot be produced."""


# Per-key in-process mutex layered on top of the cross-process file
# lock so two threads in the same process serialise cheaply rather
# than thrashing the OS lock primitive. A WeakValueDictionary keeps the
# map from growing without bound on a long-lived server: while a build
# holds (or waits on) a key's lock the caller's local reference keeps it
# alive, so concurrent callers share the same object; once no one holds
# it the entry is garbage-collected. Dirty local plugins mint a fresh
# digest per run, so an unevicted dict would otherwise leak one lock per
# run forever.
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
    plugin_configs: Mapping[str, PluginConfig],
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

    root = _resolve_cache_root(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    locks_dir = root / _LOCKS_DIRNAME
    locks_dir.mkdir(exist_ok=True)

    final_dir = root / key.directory_name()
    venv_dir = final_dir / ".venv"

    _sweep_stale_partials(root)

    with _build_lock(key.digest, locks_dir):
        return _build_or_reuse(
            final_dir=final_dir,
            venv_dir=venv_dir,
            root=root,
            key=key,
            pyproject_content=pyproject_content,
        )


def _build_or_reuse(
    *,
    final_dir: Path,
    venv_dir: Path,
    root: Path,
    key: CacheKey,
    pyproject_content: str,
) -> Path:
    ready = final_dir / _READY_MARKER
    if ready.exists():
        logger.debug(f"Cache hit: {final_dir.name}")
        return venv_dir

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
    return venv_dir


def _resolve_cache_root(override: Path | None) -> Path:
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
