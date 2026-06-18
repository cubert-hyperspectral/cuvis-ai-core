"""Session lifecycle management for the gRPC API."""

from __future__ import annotations

import gc
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.training.config import (
    DataConfig,
    PipelineConfig,
    TrainingConfig,
    TrainRunConfig,
)
from cuvis_ai_core.utils.node_registry import NodeRegistry


@dataclass
class SessionState:
    """State for a single training session."""

    session_id: str
    node_registry: NodeRegistry  # Instance for plugin isolation
    pipeline: CuvisPipeline | None = None
    _pipeline_config: PipelineConfig | None = field(default=None, repr=False)
    data_config: DataConfig | None = None
    training_config: TrainingConfig | None = None
    trainrun_config: TrainRunConfig | None = None
    search_paths: list[str] = field(
        default_factory=lambda: ["./configs", "./configs/pipeline"]
    )
    is_training: bool = False
    trainer: Any | None = None
    # Plugins registered into this session's catalog. This dict tracks what
    # the session *knows about* (parsed manifest entries), NOT what has been
    # installed/imported. The full config of every known plugin lives in
    # ``node_registry.plugin_catalog``; the loaded class set is in
    # ``node_registry.loaded_plugin_nodes``. Populated by the client's
    # ``LoadPlugin`` calls; each registered name is echoed back in that RPC's
    # ``LoadPluginResponse.registered_plugin`` field.
    registered_plugins: dict[str, dict] = field(default_factory=dict)
    # Orchestrator state, populated once the parent has spawned a child
    # runtime for this session. The handle keeps the child alive;
    # ``resolved_plugins`` is the dict the parent computed via
    # ``resolve_pipeline_plugins`` and forwarded to the child via
    # ``InitializeSession``.
    child_handle: Any | None = None
    resolved_plugins: dict[str, Any] | None = None
    # Per-session scratch root the orchestrator created for the child runtime
    # (HOME / TEMP / output redirect). Removed on close so child logs and
    # HF/torch caches don't accumulate under the system temp dir.
    runtime_base_dir: Path | None = None
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)

    @property
    def pipeline_config(self) -> PipelineConfig:
        """Return pipeline config either from cache or by serializing the pipeline."""
        if self._pipeline_config is not None:
            return self._pipeline_config
        if self.pipeline is None:
            raise ValueError("Pipeline is not initialized for this session")
        return self.pipeline.serialize()

    @pipeline_config.setter
    def pipeline_config(self, value: PipelineConfig | None) -> None:
        self._pipeline_config = value


class SessionManager:
    """Create, track, and retire session resources."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    def create_session(
        self,
        pipeline: CuvisPipeline | None = None,
        pipeline_config: PipelineConfig | None = None,
        data_config: DataConfig | None = None,
        training_config: TrainingConfig | None = None,
        trainrun_config: TrainRunConfig | None = None,
        search_paths: list[str] | None = None,
    ) -> str:
        """Create a new session with optional pipeline and configs.

        Args:
            pipeline: Optional pipeline instance
            pipeline_config: Optional pipeline configuration
            data_config: Optional data configuration captured during training
            training_config: Optional training configuration captured during training
            trainrun_config: Optional trainrun configuration (for sessions created via RestoreTrainRun)
            search_paths: Optional search paths for resolving configs/weights

        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())

        # Create NodeRegistry instance for this session
        node_registry = NodeRegistry()

        state = SessionState(
            session_id=session_id,
            node_registry=node_registry,
            pipeline=pipeline,
            _pipeline_config=pipeline_config,
            data_config=data_config,
            training_config=training_config,
            trainrun_config=trainrun_config,
            search_paths=search_paths or ["./configs"],
        )
        self._sessions[session_id] = state
        logger.info(f"Created session: {session_id}")
        return session_id

    def create_session_with_id(self, session_id: str) -> None:
        """Create a session under a caller-supplied id.

        Used by the child runtime's ``InitializeSession`` so the
        parent and child share the same ``session_id`` across the
        gRPC boundary. The public ``CreateSession`` RPC stays empty
        and server-generated; this method is only reachable via the
        internal ``RunRuntime`` service. The id is supplied by the
        caller, so nothing is returned; reach the session via
        ``get_session(session_id)``.
        """
        if not session_id:
            raise ValueError("session_id must be non-empty")
        if session_id in self._sessions:
            logger.debug(
                f"Session {session_id} already exists; reusing without re-initialising."
            )
            return
        self._sessions[session_id] = SessionState(
            session_id=session_id,
            node_registry=NodeRegistry(),
        )
        logger.info(f"Created session with caller-supplied id: {session_id}")

    def get_session(self, session_id: str) -> SessionState:
        """Return the session state, updating last_accessed."""
        if session_id not in self._sessions:
            raise ValueError(f"Session {session_id} not found")

        state = self._sessions[session_id]
        state.last_accessed = time.time()
        return state

    @staticmethod
    def _cleanup_pipeline(pipeline: CuvisPipeline | None) -> None:
        """Best-effort pipeline teardown for session close or replacement."""
        if pipeline is None:
            return

        try:
            pipeline.cleanup()
        except Exception as exc:
            logger.warning("Pipeline cleanup failed during session teardown: {}", exc)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def set_pipeline(
        self,
        session_id: str,
        pipeline: CuvisPipeline | None,
        pipeline_config: PipelineConfig | None = None,
    ) -> None:
        """Attach a pipeline to a session, cleaning up any previous pipeline."""
        session = self.get_session(session_id)
        old_pipeline = session.pipeline
        if old_pipeline is not None and old_pipeline is not pipeline:
            self._cleanup_pipeline(old_pipeline)

        session.pipeline = pipeline
        session.pipeline_config = pipeline_config

    def set_search_paths(
        self, session_id: str, paths: list[str], append: bool = True
    ) -> tuple[list[str], list[str]]:
        """Set or extend session search paths."""
        session = self.get_session(session_id)

        valid_paths: list[str] = []
        rejected_paths: list[str] = []

        for path in paths:
            resolved = self._validate_search_path(path)
            if resolved:
                valid_paths.append(resolved)
            else:
                rejected_paths.append(path)
                logger.warning(f"Rejected invalid search path: {path}")

        if append:
            for path in valid_paths:
                if path not in session.search_paths:
                    session.search_paths.append(path)
        else:
            session.search_paths = (
                valid_paths if valid_paths else ["./configs", "./configs/pipeline"]
            )

        logger.info(f"Session {session_id} search paths: {session.search_paths}")
        return session.search_paths, rejected_paths

    def _validate_search_path(self, path: str) -> str | None:
        """Validate search path and return resolved path if valid."""
        try:
            resolved = Path(path).resolve()
            if resolved.exists() and resolved.is_dir() and resolved.is_absolute():
                return str(resolved)
        except Exception as exc:
            logger.debug(f"Path validation failed for {path}: {exc}")
        return None

    def close_session(self, session_id: str) -> None:
        """Close a session and drop its resources."""
        if session_id not in self._sessions:
            raise ValueError(f"Session {session_id} not found")

        state = self._sessions.pop(session_id)

        # Cleanup trainer
        trainer = state.trainer
        if trainer is not None and hasattr(trainer, "cleanup"):
            try:
                trainer.cleanup()
            except Exception:
                # Cleanup best-effort; avoid cascading errors
                pass
        state.trainer = None

        pipeline = state.pipeline
        state.pipeline = None
        state.pipeline_config = None
        self._cleanup_pipeline(pipeline)

        # Clear plugin tracking (GC will handle registry cleanup automatically)
        state.registered_plugins.clear()
        state.data_config = None
        state.training_config = None
        state.trainrun_config = None

        # Terminate any child runtime bound to this session (orchestrator path).
        child = state.child_handle
        state.child_handle = None
        state.resolved_plugins = None
        if child is not None:
            try:
                child.terminate(grace_s=5.0)
            except Exception as exc:
                logger.warning(
                    f"Child runtime termination raised during close_session: {exc}"
                )
                try:
                    child.kill()
                except Exception as kill_exc:
                    logger.warning(f"Child runtime kill also raised: {kill_exc}")

        # Drop the child's scratch root now that it has exited (its file
        # handles are released). Best-effort: a failure here must not block
        # session teardown. Done after termination so the child isn't still
        # writing into the tree.
        runtime_base_dir = state.runtime_base_dir
        state.runtime_base_dir = None
        if runtime_base_dir is not None:
            shutil.rmtree(runtime_base_dir, ignore_errors=True)

        logger.info(f"Closed session: {session_id}")

    def list_sessions(self) -> list[str]:
        """List all active session IDs."""
        return list(self._sessions.keys())

    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """Remove sessions that haven't been touched within the age window."""
        cutoff = time.time() - (max_age_hours * 3600)
        expired = [
            sid for sid, state in self._sessions.items() if state.last_accessed < cutoff
        ]

        for sid in expired:
            self.close_session(sid)

        return len(expired)


__all__ = ["SessionManager", "SessionState"]
