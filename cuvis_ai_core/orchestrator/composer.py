"""Per-pipeline child venv composer.

Public entry point: :func:`compose_env` takes a plugin set and returns
the path to a ready ``.venv`` containing exactly that set. Cache hits
are O(filesystem stat); misses run ``uv lock`` + ``uv sync`` once and
publish atomically.
"""

from __future__ import annotations

import json
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Mapping

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
_STALE_PARTIAL_AGE_SECONDS = 6 * 60 * 60  # sweep half-built dirs older than 6h


class ComposerError(RuntimeError):
    """Raised when a composed env cannot be produced."""


# Per-key in-process mutex layered on top of the cross-process file
# lock so two threads in the same process serialise cheaply rather
# than thrashing the OS lock primitive.
_in_process_locks: dict[str, threading.Lock] = {}
_in_process_locks_guard = threading.Lock()


def _in_process_lock_for(digest: str) -> threading.Lock:
    with _in_process_locks_guard:
        if digest not in _in_process_locks:
            _in_process_locks[digest] = threading.Lock()
        return _in_process_locks[digest]


def compose_env(
    plugin_configs: Mapping[str, PluginConfig],
    *,
    core_source: CoreSource,
    cache_root: Path | None = None,
    python_requires: str = ">=3.11,<3.14",
) -> Path:
    """Materialise (or reuse) a cached venv for ``plugin_configs``.

    Returns the path to the ``.venv`` directory inside the published
    cache entry. The caller spawns ``venv_python(...)`` against this
    path.
    """
    resolved = resolve_plugin_sources(plugin_configs)
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
    locks_dir = root / ".locks"
    locks_dir.mkdir(exist_ok=True)

    final_dir = root / key.directory_name()
    venv_dir = final_dir / ".venv"

    _sweep_stale_partials(root)

    in_proc_lock = _in_process_lock_for(key.digest)
    file_lock = FileLock(str(locks_dir / f"{key.digest}.lock"))

    with in_proc_lock:
        try:
            file_lock.acquire(timeout=_LOCK_TIMEOUT_SECONDS)
        except Timeout as exc:
            raise ComposerError(
                f"Timed out after {_LOCK_TIMEOUT_SECONDS}s waiting for build "
                f"lock on cache key {key.digest}."
            ) from exc
        try:
            return _build_or_reuse(
                final_dir=final_dir,
                venv_dir=venv_dir,
                root=root,
                key=key,
                pyproject_content=pyproject_content,
            )
        finally:
            file_lock.release()


def _build_or_reuse(
    *,
    final_dir: Path,
    venv_dir: Path,
    root: Path,
    key: CacheKey,
    pyproject_content: str,
) -> Path:
    ready = final_dir / ".ready"
    if ready.exists():
        logger.debug(f"Cache hit: {final_dir.name}")
        return venv_dir

    # Defense in depth: a published directory without ``.ready`` is
    # broken — rename it aside and rebuild.
    if final_dir.exists():
        broken = root / f"{final_dir.name}.broken.{int(time.time())}"
        logger.warning(
            f"Cache dir {final_dir.name} exists without .ready; moving to "
            f"{broken.name} and rebuilding."
        )
        final_dir.rename(broken)

    build_dir = root / (
        f"{final_dir.name}.building.{os.getpid()}.{secrets.token_hex(3)}"
    )
    build_dir.mkdir(parents=True, exist_ok=False)
    (build_dir / "pyproject.toml").write_text(pyproject_content, encoding="utf-8")
    (build_dir / "key.json").write_text(
        json.dumps(key.serialise(), indent=2), encoding="utf-8"
    )

    try:
        logger.info(f"Building cache entry {key.digest} in {build_dir.name}")
        uv_lock(build_dir)
        uv_sync(build_dir)
        (build_dir / ".ready").write_text("ok", encoding="utf-8")
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
        if ".building." not in entry.name or not entry.is_dir():
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
    import shutil

    try:
        shutil.rmtree(path, ignore_errors=False)
    except OSError as exc:
        logger.warning(f"Failed to remove {path}: {exc}")
