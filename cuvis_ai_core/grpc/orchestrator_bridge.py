"""Glue between the gRPC handlers and the per-run child runtime.

A single helper module so the orchestrator path stays out of the
existing service files except for a one-line ``if orchestrator_enabled()
and ...`` early-return. Default behaviour is unchanged: the flag is
opt-in.

Lifecycle on a request that opens the orchestrator branch:

1. Parent's gRPC handler calls :func:`ensure_child_for_session` with
   the parsed pipeline_config and the session's plugin search paths.
2. Helper resolves plugins, composes a per-key venv, spawns the
   child runtime, hands it the session_id and resolved plugin dict
   via ``InitializeSession``, and stashes the handle on
   ``SessionState.child_handle``.
3. Subsequent requests for the same session forward to
   ``session.child_handle.stub()`` directly — the helper is idempotent
   and short-circuits when a handle already exists.

For builtin-only pipelines (no plugin declared, no catalog match), the
helper returns ``None`` so the caller falls back to the in-process
path even when the flag is on.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from cuvis_ai_schemas.grpc.v1 import cuvis_ai_pb2
from loguru import logger

from cuvis_ai_core.grpc.session_manager import SessionManager, SessionState
from cuvis_ai_core.orchestrator.cache_key import CoreSource
from cuvis_ai_core.orchestrator.composer import compose_env
from cuvis_ai_core.orchestrator.spawner import (
    ChildHandle,
    DeclaredPaths,
    LocalChildRuntimeSpawner,
)
from cuvis_ai_core.utils.plugin_config import GitPluginConfig, LocalPluginConfig
from cuvis_ai_core.utils.plugin_resolver import resolve_pipeline_plugins

PluginConfig = GitPluginConfig | LocalPluginConfig
_ENV_FLAG = "CUVIS_USE_ORCHESTRATOR"
_TRUTHY = ("1", "true", "yes", "on")


def orchestrator_enabled() -> bool:
    """Return True iff the orchestrator path is opt-in for this process."""
    return os.environ.get(_ENV_FLAG, "").strip().lower() in _TRUTHY


def detect_core_source() -> CoreSource:
    """Infer how ``cuvis-ai-core`` is installed in the parent process.

    Strategy:
    - If the parent's package directory sits inside a ``site-packages``
      tree, treat it as a PyPI pin and use its installed version.
    - Otherwise, fall back to a local editable source pinned at the
      package's project root (the directory two levels above the
      installed ``cuvis_ai_core/__init__.py``).
    """
    import cuvis_ai_core

    init_path = Path(cuvis_ai_core.__file__).resolve()
    project_root = init_path.parents[1]
    if "site-packages" in str(init_path).lower():
        try:
            from importlib.metadata import version

            return CoreSource(
                kind="pypi", identity=f"cuvis-ai-core=={version('cuvis-ai-core')}"
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                f"Could not read installed cuvis-ai-core version: {exc}; "
                f"falling back to local-editable source."
            )
    return CoreSource(kind="local", identity=str(project_root))


def ensure_child_for_session(
    session_manager: SessionManager,
    session_id: str,
    pipeline_config: Any,
    plugins_dirs: list[Path],
) -> ChildHandle | None:
    """Return a child runtime handle bound to ``session_id``.

    Spawns one on demand the first time the session hits an
    orchestrator-eligible LoadPipeline. ``None`` means the pipeline
    only uses builtin nodes and the caller should fall through to the
    in-process path.
    """
    session = session_manager.get_session(session_id)
    if session.child_handle is not None:
        return session.child_handle

    plugins = getattr(pipeline_config, "plugins", None)
    if not plugins and not plugins_dirs:
        return None  # builtin-only pipeline; in-process path is fine

    resolved = resolve_pipeline_plugins(pipeline_config, plugins_dirs)
    if not resolved:
        return None

    core_source = detect_core_source()
    logger.info(
        f"Composing child env for session {session_id} "
        f"({len(resolved)} plugins, core source: {core_source.kind})"
    )
    venv = compose_env(resolved, core_source=core_source)

    declared = _default_declared_paths(session_id)
    spawner = LocalChildRuntimeSpawner()
    handle = spawner.spawn(
        venv,
        cwd=declared.output_dir,
        declared_paths=declared,
        request_gpu=_gpu_requested(),
    )

    payload = json.dumps(
        {name: cfg.model_dump() for name, cfg in resolved.items()}
    ).encode("utf-8")
    init_response = handle.stub().InitializeSession(
        cuvis_ai_pb2.InitializeSessionRequest(
            session_id=session_id,
            search_paths=list(session.search_paths),
            resolved_plugins_json=payload,
            output_dir=str(declared.output_dir),
            scratch_dir=str(declared.scratch_dir),
        )
    )
    if not init_response.ok:
        handle.terminate(grace_s=2.0)
        raise RuntimeError("Child runtime rejected InitializeSession")

    session.child_handle = handle
    session.resolved_plugins = dict(resolved)
    return handle


def get_child(session: SessionState) -> ChildHandle | None:
    """Return the session's child runtime handle if attached, else ``None``."""
    return session.child_handle


def _default_declared_paths(session_id: str) -> DeclaredPaths:
    """Build per-session ``output_dir`` / ``scratch_dir``.

    Defaults under the system temp tree; can be overridden later by
    item 06's sandbox layer or by a per-session config knob. The
    fake ``HOME`` the spawner injects sits at
    ``<output_dir>/.home/``.
    """
    base = Path(tempfile.gettempdir()) / "cuvis_runtime_sessions" / session_id
    output_dir = base / "output"
    scratch_dir = base / "scratch"
    output_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / ".home").mkdir(exist_ok=True)
    return DeclaredPaths(output_dir=output_dir, scratch_dir=scratch_dir)


def _gpu_requested() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - torch is a hard dep
        return False
