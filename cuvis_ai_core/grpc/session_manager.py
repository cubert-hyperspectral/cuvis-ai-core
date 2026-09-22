"""Session lifecycle management for the gRPC API."""

from __future__ import annotations

import gc
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from cuvis_ai_core.orchestrator import leases
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.training.config import (
    DataConfig,
    PipelineConfig,
    TrainingConfig,
    TrainRunConfig,
)
from cuvis_ai_core.utils.node_registry import NodeRegistry


class ChildStillRunning(RuntimeError):
    """The session's child runtime survived terminate and kill.

    Raised by :meth:`SessionManager.retire_child` when the caller needs the
    old process gone before it may continue (a pipeline switch must not spawn
    a second runtime beside one that still holds the GPU). The old handle
    stays attached to the session, marked as stopping, so a later load or
    close can try again; requests that still reach it are answered as a
    replacement, not as a crash.
    """


# How long ``close_session`` waits for a load that holds the session's child
# lock before it stops the attached child to unblock itself. A forwarded
# LoadPipeline has no deadline, so a child that hangs inside it would otherwise
# hold the close, and the server's shutdown, for as long as it hangs.
CLOSE_LOCK_GRACE_SECONDS = 5.0


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
    # Cooperative-cancel flag for the session's training run. Set by StopTrain
    # (or a dropped Train stream); checked per batch / between statistical
    # nodes / at Train-stream entry. Cleared ONLY at SetTrainRunConfig (the run
    # boundary), never at Train-stream entry, so a stop issued between trainer
    # phases still cancels the not-yet-started phase of the same run.
    stop_event: threading.Event = field(default_factory=threading.Event)
    # Last TrainResponse yielded on this session's Train stream; what
    # GetTrainStatus reports.
    latest_train_response: Any | None = None
    # Plugins registered into this session's catalog. This dict tracks what
    # the session *knows about* (parsed manifest entries), NOT what has been
    # installed/imported. The full config of every known plugin lives in
    # ``node_registry.plugin_catalog``; the loaded class set is in
    # ``node_registry.loaded_plugin_nodes``. Populated by the client's
    # ``LoadPlugin`` calls; each registered name is echoed back in that RPC's
    # ``LoadPluginResponse.registered_plugin`` field.
    registered_plugins: dict[str, dict] = field(default_factory=dict)
    # Orchestrator state, populated by the parent each time it spawns a child
    # runtime for this session. The handle keeps the child alive;
    # ``resolved_plugins`` and ``child_data_module`` record what the child's
    # env was composed for (the plugin manifests forwarded through
    # ``InitializeSession`` and the data module whose pip extras were
    # installed). The bridge compares a later pipeline against them: a
    # pipeline the child can serve reuses it, any other one replaces it.
    child_handle: Any | None = None
    resolved_plugins: dict[str, Any] | None = None
    child_data_module: str | None = None
    # Serialises everything that decides over or replaces the child: a
    # LoadPipeline / RestoreTrainRun holds it from the reuse-or-replace
    # decision through the forwarded call, close_session holds it while it
    # tears the child down. Re-entrant because a failing restore closes the
    # session it created from inside that scope. Inference never takes it.
    child_lock: threading.RLock = field(
        default_factory=threading.RLock, repr=False, compare=False
    )
    # Per-session scratch root the orchestrator created for the child runtime
    # (HOME / TEMP / output redirect). Removed on close so child logs and
    # HF/torch caches don't accumulate under the system temp dir.
    runtime_base_dir: Path | None = None
    # Where this session's crashed child had its logs preserved. Set at the
    # moment the failure is reported so the same directory can be named in the
    # error detail, in the RPC trailers, and again at teardown.
    crash_log_dir: Path | None = None
    # Cache root holding this session's venv lease (the in-use marker that
    # protects the composed env from eviction). Set by the orchestrator
    # bridge when it writes the lease; close_session removes the lease.
    lease_cache_root: Path | None = None
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

    def has_session(self, session_id: str) -> bool:
        """Whether ``session_id`` is still registered (no last_accessed update)."""
        return session_id in self._sessions

    def close_session(self, session_id: str) -> None:
        """Close a session and drop its resources.

        Takes the session's ``child_lock`` first: a LoadPipeline in flight on
        another thread finishes (and attaches its child) before the teardown
        runs, so no child or lease outlives its session. The load holds that
        lock across its forwarded call, which has no deadline, so a child
        that hangs inside LoadPipeline would hold the close (and the server's
        shutdown) with it. After ``CLOSE_LOCK_GRACE_SECONDS`` the attached
        child is therefore stopped without the lock: its forwarded call fails,
        the load unwinds and releases the lock, and the teardown proceeds.
        """
        state = self._sessions.get(session_id)
        if state is None:
            raise ValueError(f"Session {session_id} not found")

        if not state.child_lock.acquire(timeout=CLOSE_LOCK_GRACE_SECONDS):
            child = state.child_handle
            if child is not None:
                logger.warning(
                    f"Session {session_id}: a load has held the child lock for "
                    f"{CLOSE_LOCK_GRACE_SECONDS:g}s; stopping the child so the "
                    f"close can proceed."
                )
                try:
                    child.retired_by_parent = True
                except AttributeError:  # pragma: no cover - foreign handle types
                    pass
                try:
                    child.terminate(grace_s=2.0)
                except Exception as exc:  # pragma: no cover - best effort
                    logger.warning(f"Child terminate while unblocking a close: {exc}")
            state.child_lock.acquire()
        try:
            self._close_locked(session_id, state)
        finally:
            state.child_lock.release()

    def _close_locked(self, session_id: str, state: SessionState) -> None:
        """The teardown of ``close_session``; the caller holds ``state.child_lock``."""
        if self._sessions.get(session_id) is not state:
            # A racing close won while this one waited for the lock.
            return
        self._sessions.pop(session_id)

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

        # Terminate any child runtime bound to this session (orchestrator
        # path). The session is gone either way, so a child that survives
        # the kill is logged, not raised.
        self.retire_child(state, reason="session close", require_exit=False)

        # Drop the child's scratch root now that it has exited (its file
        # handles are released). Best-effort: a failure here must not block
        # session teardown. Done after termination so the child isn't still
        # writing into the tree.
        runtime_base_dir = state.runtime_base_dir
        state.runtime_base_dir = None
        if runtime_base_dir is not None:
            shutil.rmtree(runtime_base_dir, ignore_errors=True)

        logger.info(f"Closed session: {session_id}")

    @staticmethod
    def retire_child(state: SessionState, *, reason: str, require_exit: bool) -> None:
        """Stop the session's child runtime and forget the env it was composed for.

        The one teardown path for a child the parent no longer wants: session
        close, the dead-child recovery in the bridge, and the pipeline switch
        that replaces a child whose env cannot serve the new pipeline. Marks the
        handle ``retired_by_parent`` before stopping it, so a request that still
        reaches the old child is answered as a replacement rather than a crash,
        then terminates (graceful, then kill), preserves the logs of a child that
        had already died on its own, releases the venv lease and clears
        ``child_handle`` / ``resolved_plugins`` / ``child_data_module``.

        ``require_exit`` is the switch's contract: a child that is still alive
        after the kill keeps the GPU, so spawning a replacement beside it is
        wrong. The handle and the plugin set are put back, the handle stays
        marked as stopping, and :class:`ChildStillRunning` is raised; the
        caller reports it and the next load or close stops the child again.
        ``close_session`` passes ``False`` and only logs the survivor. Callers
        hold ``state.child_lock``.
        """
        child = state.child_handle
        if child is None:
            state.resolved_plugins = None
            state.child_data_module = None
            return

        session_id = state.session_id
        previous = (state.resolved_plugins, state.child_data_module)
        state.child_handle = None
        state.resolved_plugins = None
        state.child_data_module = None

        # Poll BEFORE terminate(): parent-initiated termination exits nonzero
        # too (TerminateProcess reports 1), so only a child that was already
        # gone counts as a crash worth preserving. A child the parent had
        # already told to stop (an earlier retire that hit a survivor) and that
        # exited since is the parent's doing as well, not a crash.
        already_stopping = getattr(child, "retired_by_parent", False) is True
        alive = getattr(child, "returncode", None) is None
        died_on_its_own = not alive and not already_stopping
        if alive:
            # Only a child the parent stops on purpose carries the marker: a
            # request that still reaches a child that died on its own must keep
            # its crash status, trailers and preserved logs.
            try:
                child.retired_by_parent = True
            except AttributeError:  # pragma: no cover - foreign handle types
                pass
        exit_code: int | None = None
        try:
            exit_code = child.terminate(grace_s=5.0)
        except Exception as exc:
            logger.warning(f"Child runtime termination raised ({reason}): {exc}")
            try:
                exit_code = child.kill()
            except Exception as kill_exc:
                logger.warning(f"Child runtime kill also raised: {kill_exc}")

        if getattr(child, "returncode", None) is None:
            # kill() gives up after its wait; the process may still be alive
            # and still hold its GPU memory.
            message = (
                f"The previous child runtime for session {session_id} could not "
                f"be stopped ({reason}); it is still running."
            )
            if require_exit:
                # Keep the handle attached so a later load or close retries the
                # stop, and keep it MARKED: StopRun, terminate and kill were
                # delivered and kill() closed the channel, so the process is
                # stopping and no longer serves. child_can_serve refuses a marked
                # handle, so the next load retires it again instead of forwarding
                # into it, and a request that still reaches it is answered as a
                # replacement rather than as a crash.
                state.child_handle = child
                state.resolved_plugins, state.child_data_module = previous
                raise ChildStillRunning(message)
            logger.error(message)

        if died_on_its_own and isinstance(exit_code, int) and exit_code != 0:
            # The child's logs live under runtime_base_dir, which close_session
            # deletes right after this — copy them aside first so the crash
            # stays diagnosable. Lazy imports: crash_logs is parent-side only.
            from cuvis_ai_core.orchestrator.crash_logs import preserve_child_logs
            from cuvis_ai_core.orchestrator.spawner import format_exit_code

            # Idempotent per session: when the failing RPC already preserved
            # this child's logs, the same directory comes back and no second
            # copy is made.
            crash_dir = preserve_child_logs(
                (
                    getattr(child, "stdout_log", None),
                    getattr(child, "stderr_log", None),
                ),
                session_id=session_id,
                exit_code=exit_code,
                endpoint=getattr(child, "endpoint", None),
            )
            if crash_dir is not None:
                state.crash_log_dir = crash_dir
            location = f"; logs preserved at {crash_dir}" if crash_dir else ""
            logger.warning(
                f"Child runtime for session {session_id} exited on its own "
                f"with code {format_exit_code(exit_code)}{location}."
            )

        # The venv is no longer in use by this session: release its lease so
        # cache eviction may reclaim the entry.
        if state.lease_cache_root is not None:
            leases.remove_lease(state.lease_cache_root, session_id)
            state.lease_cache_root = None

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
