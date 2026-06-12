"""Glue between the gRPC handlers and the per-run child runtime.

The orchestrator is the **only** code path: every LoadPipeline /
Inference / Train / RestoreTrainRun call composes a per-pipeline venv
and runs pipeline materialisation + execution inside a child runtime.
The server process itself never imports plugin modules.

Test seam: the spawner and the composer are module-level injectables.
:func:`set_composer` and :func:`set_spawner` let tests substitute
in-memory implementations so the suite doesn't actually run
``uv lock`` / ``uv sync`` or spawn subprocesses for every pipeline
test. Production never calls those setters — the defaults are the
real implementations.

Lifecycle per session:

1. First request that needs a runtime calls
   :func:`ensure_child_for_session`.
2. Helper resolves plugins from the pipeline yaml, composes / reuses
   a cached venv via the registered composer, spawns the child via
   the registered spawner, hands the child the session_id and
   resolved plugin dict via ``InitializeSession``, and stashes the
   handle on ``SessionState.child_handle``.
3. Subsequent requests for the same session forward to
   ``session.child_handle.stub()`` directly — the helper short-
   circuits when a handle already exists.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

import grpc
from cuvis_ai_schemas.grpc.v1 import cuvis_ai_pb2
from loguru import logger

from cuvis_ai_core.grpc.session_manager import SessionManager, SessionState
from cuvis_ai_core.orchestrator.cache_key import CoreSource
from cuvis_ai_core.orchestrator.composer import compose_env as _real_compose_env
from cuvis_ai_core.orchestrator.spawner import (
    ChildHandle,
    ChildRuntimeSpawner,
    DeclaredPaths,
    LocalChildRuntimeSpawner,
)
from cuvis_ai_schemas.plugin import GitPluginConfig, LocalPluginConfig
from cuvis_ai_core.utils.plugin_resolver import resolve_pipeline_plugins

PluginConfig = GitPluginConfig | LocalPluginConfig

_NO_CHILD_DETAIL = (
    "No child runtime is attached to this session. "
    "Call LoadPipeline or RestoreTrainRun first."
)

# Type aliases for the injectable seams.
ComposerFn = Callable[..., Path]
SpawnerCtor = Callable[[], ChildRuntimeSpawner]

# Default implementations. Tests override via set_composer / set_spawner;
# production code never touches these globals after import.
_composer: ComposerFn = _real_compose_env
_spawner: ChildRuntimeSpawner | None = None


def get_composer() -> ComposerFn:
    return _composer


def set_composer(fn: ComposerFn) -> None:
    """Override the env composer (test-only)."""
    global _composer
    _composer = fn


def reset_composer() -> None:
    """Restore the production composer."""
    global _composer
    _composer = _real_compose_env


def get_spawner() -> ChildRuntimeSpawner:
    global _spawner
    if _spawner is None:
        _spawner = LocalChildRuntimeSpawner()
    return _spawner


def set_spawner(spawner: ChildRuntimeSpawner) -> None:
    """Override the child spawner (test-only)."""
    global _spawner
    _spawner = spawner


def reset_spawner() -> None:
    """Restore the production spawner."""
    global _spawner
    _spawner = None


def detect_core_source() -> CoreSource:
    """Infer how ``cuvis-ai-core`` is installed in the parent process."""
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
    data_module: str | None = None,
) -> ChildHandle:
    """Return a child runtime handle bound to ``session_id``.

    Idempotent: returns the existing handle if one is already attached.
    Otherwise resolves plugins (may be empty — builtin-only pipelines
    still get their own child), composes the venv, spawns, and runs
    ``InitializeSession`` before handing back the handle.
    """
    session = session_manager.get_session(session_id)
    existing = session.child_handle
    if existing is not None:
        # Reuse the attached child only while it is still alive. A child that
        # has exited (crash, OOM-kill) leaves a dead handle behind; without
        # this check the session could never recover, since every later call
        # would forward to a dead stub. Drop the stale handle and re-spawn.
        if existing.returncode is None:
            return existing
        logger.warning(
            f"Child runtime for session {session_id} has exited "
            f"(returncode={existing.returncode}); re-spawning a fresh child."
        )
        session.child_handle = None

    resolved = _resolve_plugins(pipeline_config, plugins_dirs, data_module)
    core_source = detect_core_source()
    logger.info(
        f"Composing child env for session {session_id} "
        f"({len(resolved)} plugins, core source: {core_source.kind}, "
        f"data_module: {data_module or '-'})"
    )
    venv = _composer(resolved, core_source=core_source, active_data_module=data_module)

    declared = _default_declared_paths(session_id)
    handle = get_spawner().spawn(
        venv,
        # Run the child with the server's own working directory so a
        # config's relative data/output paths resolve exactly as they did
        # under the in-process server. declared_paths still drives HOME/TEMP
        # redirection and the future sandbox bind-mount set — it is
        # intentionally not the cwd.
        cwd=Path(os.getcwd()),
        declared_paths=declared,
        request_gpu=_gpu_requested(),
    )

    _initialize_child_session(handle, session_id, session, resolved, declared)

    session.child_handle = handle
    session.resolved_plugins = dict(resolved)
    # Record the child's scratch root (output/scratch share this parent) so
    # close_session can remove it once the child exits.
    session.runtime_base_dir = declared.output_dir.parent
    _mirror_plugin_catalog(session, resolved)
    return handle


def _initialize_child_session(
    handle: ChildHandle,
    session_id: str,
    session: SessionState,
    resolved: Mapping[str, PluginConfig],
    declared: DeclaredPaths,
) -> None:
    """Hand the freshly-spawned child its session context via InitializeSession.

    Terminates the child and raises if it rejects the init handshake.
    """
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
        raise RuntimeError(
            f"Child runtime rejected InitializeSession for session "
            f"{session_id!r} ({len(resolved)} plugins: {sorted(resolved)})."
        )


def _mirror_plugin_catalog(
    session: SessionState, resolved: Mapping[str, PluginConfig]
) -> None:
    """Record resolved plugin metadata on the parent session.

    ListLoadedPlugins / GetPluginInfo / external inspection report what
    the orchestrator materialised, even though the actual class registry
    lives inside the child.
    """
    for name, cfg in resolved.items():
        session.registered_plugins[name] = cfg.model_dump()


def get_child(session: SessionState) -> ChildHandle | None:
    """Return the session's child runtime handle if attached, else ``None``."""
    return session.child_handle


# ---------------------------------------------------------------------------
# Forwarding helpers — the parent-side gRPC handlers call these instead of
# the in-process service methods so the orchestrator is the only path.
# ---------------------------------------------------------------------------


def forward_load_pipeline(
    session_manager: SessionManager,
    request: cuvis_ai_pb2.LoadPipelineRequest,
    context: grpc.ServicerContext,
) -> cuvis_ai_pb2.LoadPipelineResponse:
    """Parent's LoadPipeline path: ensure_child + forward unmodified."""
    from cuvis_ai_core.grpc.error_handling import get_session_or_error
    from cuvis_ai_core.training.config import PipelineConfig

    session = get_session_or_error(session_manager, request.session_id, context)
    if session is None:
        return cuvis_ai_pb2.LoadPipelineResponse(success=False)
    if not request.pipeline or not request.pipeline.config_bytes:
        context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
        context.set_details("pipeline.config_bytes is required")
        return cuvis_ai_pb2.LoadPipelineResponse(success=False)

    try:
        config_dict = json.loads(request.pipeline.config_bytes)
        if not isinstance(config_dict, dict):
            raise ValueError("pipeline config must decode to a JSON object")
        config_dict.pop("version", None)
        pipeline_config = PipelineConfig(**config_dict)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
        context.set_details(
            f"pipeline.config_bytes is not a valid pipeline config: {exc}"
        )
        return cuvis_ai_pb2.LoadPipelineResponse(success=False)

    # Optional data selection: lets the composer resolve the data-module
    # plugin's pip extras at compose time (the child env is frozen here, before
    # DataConfig would otherwise arrive at Train).
    data_module = None
    if request.HasField("data") and request.data.config_bytes:
        from cuvis_ai_core.training.config import DataConfig

        try:
            data_module = DataConfig.from_proto(request.data).data_module
        except Exception:  # noqa: BLE001 - data selection is best-effort here
            data_module = None

    plugins_dirs = _plugins_dirs_for_session(session)
    try:
        child = ensure_child_for_session(
            session_manager,
            request.session_id,
            pipeline_config,
            plugins_dirs,
            data_module=data_module,
        )
    except ValueError as exc:
        # resolve_pipeline_plugins's contract: missing plugins block,
        # ambiguous class, coverage gap, duplicate with diverging refs
        # all raise ValueError. Surface as INVALID_ARGUMENT.
        context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
        context.set_details(str(exc))
        return cuvis_ai_pb2.LoadPipelineResponse(success=False)
    return _call_child_with_error_propagation(
        child.stub(),
        "LoadPipeline",
        request,
        context,
        lambda: cuvis_ai_pb2.LoadPipelineResponse(success=False),
    )


def forward_inference(
    session_manager: SessionManager,
    request: cuvis_ai_pb2.InferenceRequest,
    context: grpc.ServicerContext,
) -> cuvis_ai_pb2.InferenceResponse:
    """Parent's Inference path: route to the session's child runtime."""
    from cuvis_ai_core.grpc.error_handling import get_session_or_error

    session = get_session_or_error(session_manager, request.session_id, context)
    if session is None:
        return cuvis_ai_pb2.InferenceResponse()

    child = get_child(session)
    if child is None:
        context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
        context.set_details(_NO_CHILD_DETAIL)
        return cuvis_ai_pb2.InferenceResponse()
    return _call_child_with_error_propagation(
        child.stub(),
        "Inference",
        request,
        context,
        cuvis_ai_pb2.InferenceResponse,
    )


def forward_train(
    session_manager: SessionManager,
    request: cuvis_ai_pb2.TrainRequest,
    context: grpc.ServicerContext,
) -> Iterator[cuvis_ai_pb2.TrainResponse]:
    """Parent's Train path: re-yield the child stub's server-streaming responses."""
    from cuvis_ai_core.grpc.error_handling import get_session_or_error

    session = get_session_or_error(session_manager, request.session_id, context)
    if session is None:
        return iter([])

    child = get_child(session)
    if child is None:
        context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
        context.set_details(_NO_CHILD_DETAIL)
        return iter([])

    def _proxy():
        try:
            yield from child.stub().Train(request)
        except grpc.RpcError as exc:
            _propagate_rpc_error(exc, context)

    return _proxy()


def forward_restore_train_run(
    session_manager: SessionManager,
    request: cuvis_ai_pb2.RestoreTrainRunRequest,
    context: grpc.ServicerContext,
) -> cuvis_ai_pb2.RestoreTrainRunResponse:
    """Parent's RestoreTrainRun path.

    Allocates a fresh parent session, parses the trainrun yaml far
    enough to learn which plugins the pipeline needs, composes the
    venv, spawns the child via :func:`ensure_child_for_session`, then
    forwards the request. The child's ``RestoreTrainRun`` attaches the
    rebuilt pipeline to the same session_id, so the response we hand
    back to the public caller stays the parent's session id.
    """
    from cuvis_ai_core.grpc.trainrun_service import TrainRunService

    trainrun_path = Path(request.trainrun_path)
    try:
        trainrun_config, _ = TrainRunService.parse_trainrun_yaml(trainrun_path)
    except FileNotFoundError as exc:
        context.set_code(grpc.StatusCode.NOT_FOUND)
        context.set_details(str(exc))
        return cuvis_ai_pb2.RestoreTrainRunResponse()
    except ValueError as exc:
        context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
        context.set_details(str(exc))
        return cuvis_ai_pb2.RestoreTrainRunResponse()
    pipeline_config = trainrun_config.pipeline

    # Allocate the parent's public session first so InitializeSession can
    # pin the child to that id.
    parent_session_id = session_manager.create_session()
    parent_session = session_manager.get_session(parent_session_id)
    plugins_dirs = _plugins_dirs_for_session(parent_session)

    trainrun_data_module = getattr(
        getattr(trainrun_config, "data", None), "data_module", None
    )
    try:
        ensure_child_for_session(
            session_manager,
            parent_session_id,
            pipeline_config,
            plugins_dirs,
            data_module=trainrun_data_module,
        )
    except ValueError as exc:
        # Drop the empty session; surface resolver errors as INVALID_ARGUMENT
        # so callers don't have to dig through "UNKNOWN" wrapping.
        try:
            session_manager.close_session(parent_session_id)
        except Exception:  # pragma: no cover - defensive
            pass
        context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
        context.set_details(str(exc))
        return cuvis_ai_pb2.RestoreTrainRunResponse()
    except Exception:
        # Compose / spawn failed for some other reason; drop the empty
        # session and re-raise so the grpc handler decorator surfaces it.
        try:
            session_manager.close_session(parent_session_id)
        except Exception:  # pragma: no cover - defensive
            pass
        raise

    child = parent_session.child_handle
    response = _call_child_with_error_propagation(
        child.stub(),
        "RestoreTrainRun",
        request,
        context,
        cuvis_ai_pb2.RestoreTrainRunResponse,
    )
    # Contract: the child reuses the session_id we pinned via
    # InitializeSession, so an empty session_id in its response means
    # "same as the parent's". Fill in the parent id for the public client.
    if not response.session_id:
        response.session_id = parent_session_id
    return response


def _forward_pipeline_op(
    session_manager: SessionManager,
    request,
    context: grpc.ServicerContext,
    *,
    stub_method: str,
    empty_response_factory,
):
    """Shared body for handlers that need a live pipeline (which lives in the child).

    Returns the empty response from ``empty_response_factory()`` when
    the session or its child handle is missing, after setting the
    appropriate gRPC status code on the context. Errors raised by the
    child (via the in-memory or real RPC channel) are translated back
    onto the parent's context so the caller sees the original status
    code instead of a generic UNKNOWN.
    """
    from cuvis_ai_core.grpc.error_handling import get_session_or_error

    session = get_session_or_error(session_manager, request.session_id, context)
    if session is None:
        return empty_response_factory()
    child = get_child(session)
    if child is None:
        context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
        context.set_details(_NO_CHILD_DETAIL)
        return empty_response_factory()
    return _call_child_with_error_propagation(
        child.stub(), stub_method, request, context, empty_response_factory
    )


def _propagate_rpc_error(exc: grpc.RpcError, context: grpc.ServicerContext) -> None:
    """Copy a child ``RpcError``'s status code + details onto the parent context."""
    code = exc.code() if hasattr(exc, "code") else grpc.StatusCode.UNKNOWN
    details = exc.details() if hasattr(exc, "details") else str(exc)
    context.set_code(code or grpc.StatusCode.UNKNOWN)
    context.set_details(details or "")


def _call_child_with_error_propagation(
    stub,
    stub_method: str,
    request,
    context: grpc.ServicerContext,
    empty_response_factory,
):
    """Invoke a child stub method and surface its status code on the parent's context."""
    try:
        return getattr(stub, stub_method)(request)
    except grpc.RpcError as exc:
        _propagate_rpc_error(exc, context)
        return empty_response_factory()


def forward_load_pipeline_weights(session_manager, request, context):
    return _forward_pipeline_op(
        session_manager,
        request,
        context,
        stub_method="LoadPipelineWeights",
        empty_response_factory=cuvis_ai_pb2.LoadPipelineWeightsResponse,
    )


def forward_save_pipeline(session_manager, request, context):
    return _forward_pipeline_op(
        session_manager,
        request,
        context,
        stub_method="SavePipeline",
        empty_response_factory=cuvis_ai_pb2.SavePipelineResponse,
    )


def forward_save_train_run(session_manager, request, context):
    return _forward_pipeline_op(
        session_manager,
        request,
        context,
        stub_method="SaveTrainRun",
        empty_response_factory=cuvis_ai_pb2.SaveTrainRunResponse,
    )


def forward_get_pipeline_inputs(session_manager, request, context):
    return _forward_pipeline_op(
        session_manager,
        request,
        context,
        stub_method="GetPipelineInputs",
        empty_response_factory=cuvis_ai_pb2.GetPipelineInputsResponse,
    )


def forward_get_pipeline_outputs(session_manager, request, context):
    return _forward_pipeline_op(
        session_manager,
        request,
        context,
        stub_method="GetPipelineOutputs",
        empty_response_factory=cuvis_ai_pb2.GetPipelineOutputsResponse,
    )


def forward_get_pipeline_visualization(session_manager, request, context):
    return _forward_pipeline_op(
        session_manager,
        request,
        context,
        stub_method="GetPipelineVisualization",
        empty_response_factory=cuvis_ai_pb2.GetPipelineVisualizationResponse,
    )


def forward_set_train_run_config(session_manager, request, context):
    """Parent's SetTrainRunConfig path — forwards to the existing child.

    Pipeline creation is the job of LoadPipeline / RestoreTrainRun. If
    the session has no child runtime yet, the call is rejected with
    FAILED_PRECONDITION; the child's in-process body additionally
    rejects any embedded ``pipeline:`` section in the trainrun config
    so there is only one entry point for pipeline construction.
    """
    from cuvis_ai_core.grpc.error_handling import get_session_or_error

    session = get_session_or_error(session_manager, request.session_id, context)
    if session is None:
        return cuvis_ai_pb2.SetTrainRunConfigResponse(success=False)
    if not request.config.config_bytes:
        context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
        context.set_details("trainrun config_bytes is required")
        return cuvis_ai_pb2.SetTrainRunConfigResponse(success=False)

    # Deliberate early guard (the shared _forward_pipeline_op below also
    # rejects a missing child): this RPC returns a message tailored to
    # its "build the pipeline first" contract rather than the generic
    # no-child message.
    if get_child(session) is None:
        context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
        context.set_details(
            "No pipeline attached to the session. Call LoadPipeline "
            "(or RestoreTrainRun) before SetTrainRunConfig."
        )
        return cuvis_ai_pb2.SetTrainRunConfigResponse(success=False)

    return _forward_pipeline_op(
        session_manager,
        request,
        context,
        stub_method="SetTrainRunConfig",
        empty_response_factory=cuvis_ai_pb2.SetTrainRunConfigResponse,
    )


def forward_get_train_status(session_manager, request, context):
    return _forward_pipeline_op(
        session_manager,
        request,
        context,
        stub_method="GetTrainStatus",
        empty_response_factory=cuvis_ai_pb2.GetTrainStatusResponse,
    )


def _plugins_dirs_for_session(session: SessionState) -> list[Path]:
    plugins_dirs: list[Path] = []
    for search_path in session.search_paths:
        candidate = Path(search_path) / "plugins"
        if candidate.is_dir():
            plugins_dirs.append(candidate)
    return plugins_dirs


# ---------------------------------------------------------------------------
# In-memory test seam — production code never instantiates these.
# ---------------------------------------------------------------------------


class _InMemoryContext:
    """Minimal ``grpc.ServicerContext`` stand-in used by the in-memory stub."""

    def __init__(self) -> None:
        self._code: grpc.StatusCode | None = None
        self._details: str = ""

    def set_code(self, code: grpc.StatusCode) -> None:
        self._code = code

    def set_details(self, details: str) -> None:
        self._details = details

    def code(self) -> grpc.StatusCode | None:
        return self._code

    def details(self) -> str:
        return self._details

    def is_active(self) -> bool:  # pragma: no cover - trivial
        return True


class _InMemoryRpcError(grpc.RpcError):
    """``grpc.RpcError`` analogue raised by the in-memory stub on non-OK codes."""

    def __init__(self, code: grpc.StatusCode, details: str) -> None:
        self._code = code
        self._details = details

    def code(self) -> grpc.StatusCode:
        return self._code

    def details(self) -> str:
        return self._details

    def __str__(self) -> str:
        return f"<_InMemoryRpcError code={self._code} details={self._details!r}>"


class _InMemoryStub:
    """Stand-in for ``RunRuntimeStub`` that calls a local servicer directly."""

    def __init__(self, servicer) -> None:
        self._servicer = servicer

    def _call(self, method_name: str, request, timeout=None):
        ctx = _InMemoryContext()
        method = getattr(self._servicer, method_name)
        result = method(request, ctx)
        code = ctx.code()
        if code is not None and code is not grpc.StatusCode.OK:
            raise _InMemoryRpcError(code, ctx.details())
        return result

    def InitializeSession(self, request, timeout=None):
        return self._call("InitializeSession", request, timeout)

    def LoadPipeline(self, request, timeout=None):
        return self._call("LoadPipeline", request, timeout)

    def LoadPipelineWeights(self, request, timeout=None):
        return self._call("LoadPipelineWeights", request, timeout)

    def SavePipeline(self, request, timeout=None):
        return self._call("SavePipeline", request, timeout)

    def SaveTrainRun(self, request, timeout=None):
        return self._call("SaveTrainRun", request, timeout)

    def GetPipelineInputs(self, request, timeout=None):
        return self._call("GetPipelineInputs", request, timeout)

    def GetPipelineOutputs(self, request, timeout=None):
        return self._call("GetPipelineOutputs", request, timeout)

    def GetPipelineVisualization(self, request, timeout=None):
        return self._call("GetPipelineVisualization", request, timeout)

    def SetTrainRunConfig(self, request, timeout=None):
        return self._call("SetTrainRunConfig", request, timeout)

    def GetTrainStatus(self, request, timeout=None):
        return self._call("GetTrainStatus", request, timeout)

    def RestoreTrainRun(self, request, timeout=None):
        return self._call("RestoreTrainRun", request, timeout)

    def Inference(self, request, timeout=None):
        return self._call("Inference", request, timeout)

    def Train(self, request, timeout=None):
        ctx = _InMemoryContext()
        gen = self._servicer.Train(request, ctx)

        def _iter():
            yield from gen
            code = ctx.code()
            if code is not None and code is not grpc.StatusCode.OK:
                raise _InMemoryRpcError(code, ctx.details())

        return _iter()

    def CloseSession(self, request, timeout=None):
        return self._call("CloseSession", request, timeout)

    def StopRun(self, request, timeout=None):
        return self._call("StopRun", request, timeout)

    def HealthCheck(self, request, timeout=None):
        return self._call("HealthCheck", request, timeout)


class _InMemoryChildHandle:
    """``ChildHandle`` stand-in for in-memory mode (no subprocess)."""

    def __init__(self, servicer) -> None:
        self._servicer = servicer
        self.endpoint = "in-memory"
        # None while "alive", set to 0 on terminate/kill — mirrors the real
        # ChildHandle.returncode (process.poll()) so the orchestrator's
        # liveness check behaves identically in the in-memory seam.
        self._returncode: int | None = None

    def stub(self) -> _InMemoryStub:
        return _InMemoryStub(self._servicer)

    def terminate(self, grace_s: float = 5.0) -> int:
        try:
            self._servicer.shutdown_event.set()
        except Exception:  # pragma: no cover
            pass
        self._returncode = 0
        return 0

    def kill(self) -> int:
        return self.terminate(grace_s=0)

    @property
    def returncode(self) -> int | None:
        return self._returncode


class _InMemorySpawner(ChildRuntimeSpawner):
    """Instantiates a ``RunRuntimeServicer`` in-process. Test-only."""

    def spawn(
        self,
        venv_path: Path,
        *,
        cwd: Path,
        declared_paths: DeclaredPaths,
        request_gpu: bool = False,
    ) -> ChildHandle:
        from cuvis_ai_core.run_runtime.service import RunRuntimeServicer

        servicer = RunRuntimeServicer()
        return _InMemoryChildHandle(servicer)  # type: ignore[return-value]


def _noop_composer(
    plugin_configs: Mapping[str, PluginConfig],
    *,
    core_source: CoreSource,
    **kwargs,
) -> Path:
    """Test-only composer: returns a dummy path the in-memory spawner ignores."""
    return Path("in-memory-venv")


def install_in_memory_orchestrator() -> None:
    """Activate the in-memory orchestrator for tests.

    Replaces the production composer + spawner with stand-ins that
    instantiate a :class:`RunRuntimeServicer` directly in the test
    process. The child still runs through the full RunRuntime servicer
    surface — no shortcuts — but no subprocess is spawned and no
    ``uv lock`` / ``uv sync`` runs.
    """
    set_composer(_noop_composer)
    set_spawner(_InMemorySpawner())


def reset_orchestrator() -> None:
    """Restore the production composer + spawner (test-only)."""
    reset_composer()
    reset_spawner()


def _resolve_plugins(
    pipeline_config: Any, plugins_dirs: list[Path], data_module: str | None = None
) -> Mapping[str, PluginConfig]:
    """Resolve the pipeline's declared plugins against the catalog.

    Always runs the resolver: ``plugins:`` is mandatory, so a pipeline
    that omits it (or names a plugin with no manifest) hard-fails with a
    ``ValueError`` instead of silently loading with an empty plugin set.
    ``data_module`` (from the run's DataConfig) unions the providing data
    plugin into the set, since it ships no node classes to resolve by coverage.
    """
    return resolve_pipeline_plugins(
        pipeline_config, plugins_dirs, data_module=data_module
    )


def _default_declared_paths(session_id: str) -> DeclaredPaths:
    """Build per-session ``output_dir`` / ``scratch_dir``."""
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
