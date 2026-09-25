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
   handle plus what it was composed for on ``SessionState``.
3. A later LoadPipeline / RestoreTrainRun on the same session resolves
   its plugins again and asks :func:`child_can_serve`: a pipeline whose
   plugins (and data module) the child's env already holds reuses the
   warm child; any other one replaces it — the new env is composed
   first, then the old child is retired with a confirmed exit, then the
   replacement is spawned. A compose failure leaves the old child and
   its pipeline untouched.
4. Inference and the other pipeline ops forward to
   ``session.child_handle.stub()`` directly. Loads on one session
   serialise on ``SessionState.child_lock`` for the whole operation;
   a request that still reaches a retired child, or finds no child while
   a load is replacing it, is answered ABORTED (repeat the request), not
   as a crash and not as a precondition failure. A close that finds the
   lock held gives up on it after a bounded wait; the load re-checks
   ``SessionState.closing`` after the compose, after the retire, after the
   spawn and before forwarding, answers NOT_FOUND whatever else went wrong
   meanwhile, and disposes of what it created.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

import grpc
import psutil
from cuvis_ai_schemas.grpc.v1 import cuvis_ai_pb2
from loguru import logger

from cuvis_ai_core.grpc.session_manager import (
    CHILD_STOP_GRACE_SECONDS,
    ChildStillRunning,
    SessionManager,
    SessionState,
)
from cuvis_ai_core.orchestrator import leases
from cuvis_ai_core.orchestrator.cache_key import CoreSource, pyproject_sha256_of
from cuvis_ai_core.orchestrator.composer import ComposerError
from cuvis_ai_core.orchestrator.composer import compose_env as _real_compose_env
from cuvis_ai_core.orchestrator.runtime_project import RuntimeProjectError
from cuvis_ai_core.orchestrator.uv_runner import UvRunnerError
from cuvis_ai_core.orchestrator.spawner import (
    ChildHandle,
    ChildRuntimeSpawner,
    DeclaredPaths,
    LocalChildRuntimeSpawner,
    SpawnError,
    dead_child_details,
    format_exit_code,
)
from cuvis_ai_schemas.plugin import PluginManifest, parse_plugin_manifest
from cuvis_ai_core.utils.plugin_resolver import resolve_against_catalog

_NO_CHILD_DETAIL = (
    "No child runtime is attached to this session. "
    "Call LoadPipeline or RestoreTrainRun first."
)

# Answer for a request that reached a child the parent itself had stopped:
# a pipeline switch replaced it, the session closed, or an earlier stop is
# still in progress. Deliberately not a crash status: the exit code is ours,
# there is nothing to postmortem, and the client's next request lands on the
# session's current runtime (or on NOT_FOUND once the session is gone).
_REPLACED_DETAIL = (
    "The session's child runtime was stopped by the server (replaced by a "
    "pipeline switch, or the session closed); this request went to the "
    "previous runtime. Repeat the request."
)

# Answer for a request that found no child while a load owns the session:
# the child is being replaced right now. ABORTED, gRPC's code for a
# concurrency conflict the caller resolves by repeating the request, so a
# client that shows FAILED_PRECONDITION once and never retries it can tell
# the two apart without reading the text.
_LOAD_IN_FLIGHT_DETAIL = (
    "A pipeline load is replacing this session's child runtime; "
    "repeat the request once it has finished."
)

# Compose and spawn failures the parent answers as FAILED_PRECONDITION: the
# server answered, and a fresh session would run into the same failure.
# Listed by class, not as RuntimeError: NotImplementedError and RecursionError
# are RuntimeErrors too, and a bug in the resolver or the composer has to
# reach the servicer as a bug, not dressed up as a setup problem.
_RUNTIME_SETUP_ERRORS = (SpawnError, UvRunnerError, ComposerError, RuntimeProjectError)


def _answer_no_child(session: SessionState, context: grpc.ServicerContext) -> None:
    """Status for a request that found no child on its session.

    While a load owns the session the child is being replaced: ABORTED, repeat
    the request. Otherwise no pipeline was ever loaded, a precondition the
    caller has to meet first.
    """
    if session.load_in_flight:
        context.set_code(grpc.StatusCode.ABORTED)
        context.set_details(_LOAD_IN_FLIGHT_DETAIL)
    else:
        context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
        context.set_details(_NO_CHILD_DETAIL)


@contextmanager
def _owning_child(
    session_manager: SessionManager, session_id: str, session: SessionState
) -> Iterator[None]:
    """Hold the session's child lock for a whole load and mark the load in flight.

    The flag is what :func:`_answer_no_child` reads: for the duration of the
    reuse-or-replace decision, the compose, the spawn and the forwarded call,
    a request that finds no child is racing this load.

    On the way out, a session that a close popped without the lock (it gave
    up after its grace while this load was composing, retiring or spawning)
    has left its scratch tree to this load: it goes here, whatever path the
    load took, so a compose or spawn that failed after the close leaves no
    tree behind either.
    """
    with session.child_lock:
        session.load_in_flight = True
        try:
            yield
        finally:
            session.load_in_flight = False
            if not session_manager.has_session(session_id):
                _discard_runtime_tree(session)


def _discard_runtime_tree(session: SessionState) -> None:
    """Drop the session's scratch tree; the child that wrote it is gone."""
    tree = session.runtime_base_dir
    session.runtime_base_dir = None
    if tree is not None:
        shutil.rmtree(tree, ignore_errors=True)


def _abandon_if_closing(
    session_manager: SessionManager, session_id: str, session: SessionState
) -> None:
    """Raise :class:`SessionClosedDuringLoad` when a close overtook this load.

    Called between the load's slow steps, before anything new is created. A
    close still waiting for the lock tears the attached child and its tree
    down as soon as the load releases it; a close that gave up after its grace
    has already stopped the attached child and left its tree to this thread.
    """
    if not session_manager.has_session(session_id):
        _discard_runtime_tree(session)
        raise SessionClosedDuringLoad(session_id)
    if session.closing.is_set():
        raise SessionClosedDuringLoad(session_id)


# gRPC trailing-metadata keys carrying a dead child's postmortem. These three
# names are the wire contract with the desktop client, which reads them off the
# call's trailing metadata to build a typed error, so they are literals here and
# there: renaming one silently degrades the client to a message-only error.
TRAILER_CHILD_EXIT_CODE = "cuvis-child-exit-code"
TRAILER_CHILD_EXIT_TEXT = "cuvis-child-exit-text"
TRAILER_CRASH_LOG_DIR = "cuvis-crash-log-dir"

# Statuses a forwarded RPC can fail with when the child process died under it.
# UNAVAILABLE is the endpoint refusing connections; UNKNOWN, CANCELLED and
# INTERNAL are what a stream that was mid-flight reports instead. Each is also
# a status a live child can legitimately return, so a dead-child answer is only
# given once the process is confirmed gone.
_CRASH_PROBE_CODES = frozenset(
    {
        grpc.StatusCode.UNAVAILABLE,
        grpc.StatusCode.UNKNOWN,
        grpc.StatusCode.CANCELLED,
        grpc.StatusCode.INTERNAL,
    }
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
    """Infer how ``cuvis-ai-core`` is installed in the parent process.

    ``uv`` records a VCS install in the distribution's ``direct_url.json``.
    Preserve its resolved commit when composing the child project; falling back
    to the package version would otherwise make a parent pinned to a local
    branch silently produce a child using the PyPI release with that version.
    """
    import cuvis_ai_core

    init_path = Path(cuvis_ai_core.__file__).resolve()
    project_root = init_path.parents[1]
    if "site-packages" in str(init_path).lower():
        try:
            from importlib.metadata import distribution, version

            direct_url = distribution("cuvis-ai-core").read_text("direct_url.json")
            if direct_url:
                source = json.loads(direct_url)
                vcs_info = source.get("vcs_info") or {}
                repo = source.get("url")
                revision = vcs_info.get("commit_id")
                if vcs_info.get("vcs") == "git" and repo and revision:
                    return CoreSource(kind="git", identity=f"{repo}@{revision}")

            return CoreSource(
                kind="pypi", identity=f"cuvis-ai-core=={version('cuvis-ai-core')}"
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                f"Could not read installed cuvis-ai-core version: {exc}; "
                f"falling back to local-editable source."
            )
    # A local core is installed editable into the child, so its own
    # pyproject.toml is the only tree content that shapes the venv —
    # hash it into the identity or a dependency edit would silently
    # reuse a stale child env.
    return CoreSource(
        kind="local",
        identity=str(project_root),
        pyproject_sha256=pyproject_sha256_of(project_root),
    )


class PluginsNotRegisteredError(Exception):
    """A pipeline names plugins that were never registered via LoadPlugin.

    Raised before composing the child env so the gRPC layer can surface a
    FAILED_PRECONDITION telling the caller to LoadPlugin each missing plugin
    first (distinct from a malformed-request ValueError).
    """

    def __init__(self, missing: list[str], registered: list[str]) -> None:
        self.missing = missing
        self.registered = registered
        super().__init__(
            f"Pipeline requires plugin(s) {missing} that are not registered in "
            f"this session. Call LoadPlugin for each before LoadPipeline. "
            f"Registered plugins: {registered or '[]'}."
        )


class SessionClosedDuringLoad(Exception):
    """The session was closed while its child was being composed or spawned.

    The spawned child has been stopped and its lease removed; the caller
    answers NOT_FOUND, as it would for any request on a closed session.
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(
            f"Session {session_id} was closed while its runtime was being prepared."
        )


def _load_failure_status(
    exc: Exception,
    *,
    subject: str,
    session_manager: SessionManager,
    session_id: str,
    session: SessionState,
) -> tuple[grpc.StatusCode, str] | None:
    """Status and details for an exception out of :func:`ensure_child_for_session`.

    Returns ``None`` for anything that is not a load failure: a bug, which the
    caller re-raises so the undecorated servicer reports it as UNKNOWN.

    * :class:`SessionClosedDuringLoad`: NOT_FOUND, as for any request on a
      closed session, so the client replays on a fresh one.
    * :class:`PluginsNotRegisteredError`: FAILED_PRECONDITION, not a malformed
      request; the fix is to call LoadPlugin for each plugin first.
    * ``ValueError``: INVALID_ARGUMENT, ``resolve_against_catalog``'s contract
      (missing plugins block, ambiguous class, coverage gap, duplicate with
      diverging refs).
    * :class:`ChildStillRunning`: FAILED_PRECONDITION, not INTERNAL. The old
      child would not die and the pipeline was not replaced; the old handle
      stays attached and marked, so the next load retries the stop. A client
      that treats INTERNAL as a transport fault would close the session and
      spawn a fresh runtime beside the survivor, the very thing the confirmed
      exit refused.
    * The compose and spawn failures in ``_RUNTIME_SETUP_ERRORS`` (uv, git, the
      plugin source resolver, the scratch tree, the lease, the child's
      endpoint, health check or InitializeSession): FAILED_PRECONDITION with
      the cause. The server answered, and a fresh session would run into the
      same failure, so the client shows it once instead of retrying on a new
      session. A compose failure leaves the old child and its pipeline intact;
      a spawn failure after the retire leaves the session childless until the
      next load recovers.

    A failure on a session that a close overtook meanwhile (the flag is set or
    the session is gone) is NOT_FOUND regardless: the client closed it and must
    not read a verdict about it.
    """
    if isinstance(exc, SessionClosedDuringLoad):
        return grpc.StatusCode.NOT_FOUND, str(exc)
    if isinstance(exc, PluginsNotRegisteredError):
        answer = (grpc.StatusCode.FAILED_PRECONDITION, str(exc))
    elif isinstance(exc, ValueError):
        answer = (grpc.StatusCode.INVALID_ARGUMENT, str(exc))
    elif isinstance(exc, ChildStillRunning):
        answer = (grpc.StatusCode.FAILED_PRECONDITION, str(exc))
    elif isinstance(exc, _RUNTIME_SETUP_ERRORS):
        answer = (
            grpc.StatusCode.FAILED_PRECONDITION,
            f"Preparing the {subject}'s runtime failed: {exc}",
        )
    else:
        return None
    if session.closing.is_set() or not session_manager.has_session(session_id):
        return grpc.StatusCode.NOT_FOUND, f"Session {session_id} was closed"
    return answer


def child_can_serve(
    session: SessionState,
    resolved: Mapping[str, PluginManifest],
    data_module: str | None,
) -> bool:
    """Whether the session's current child env holds everything ``resolved`` needs.

    The child's venv was composed for ``session.resolved_plugins`` (with the
    pip extras of ``session.child_data_module``). It serves a pipeline whose
    plugins are a subset of that set, name for name and source for source
    (a re-registered plugin pointing at another tag or path is another
    plugin), and whose data module is either none or the one the env was
    composed with. Anything else needs a new env: a plugin family the child
    never installed fails inside it with a module import error. A child the
    parent has already told to stop (a retire that hit a survivor) serves
    nothing any more, whatever it was composed for.
    """
    if getattr(session.child_handle, "retired_by_parent", False) is True:
        return False
    current = session.child_install or {}
    for name, manifest in resolved.items():
        if current.get(name) != _install_identity(manifest):
            return False
    return data_module is None or data_module == session.child_data_module


def _install_identity(manifest: PluginManifest) -> tuple:
    """What decides a plugin's installation in the child's venv.

    The source (repo and tag, or path plus the hash of the local project's
    ``pyproject.toml``, the one file of a local plugin whose content shapes the
    venv), the installable name and the pip extras of its data-module
    capabilities. Capability lists, tags, icons and port specs are metadata the
    client may regenerate between two LoadPlugin calls (``emit_metadata``,
    another capability order); they do not change the venv, so they must not
    replace a warm child. A git tag is taken as immutable, as the composer's
    cache key takes it: a tag moved to another commit is not detected here.
    """
    extras = tuple(
        sorted(
            (cap.data_module_name, tuple(sorted(cap.extras)))
            for cap in manifest.capabilities
            if cap.kind == "data_module"
        )
    )
    path = getattr(manifest, "path", None)
    return (
        type(manifest).__name__,
        manifest.package_name,
        getattr(manifest, "repo", None),
        getattr(manifest, "tag", None),
        path,
        pyproject_sha256_of(Path(path)) if path else None,
        extras,
    )


def ensure_child_for_session(
    session_manager: SessionManager,
    session_id: str,
    pipeline_config: Any,
    data_module: str | None = None,
) -> ChildHandle:
    """Return a child runtime handle bound to ``session_id`` that can serve the pipeline.

    Resolves the pipeline's plugins against the session catalog (may be
    empty — builtin-only pipelines still get their own child). A live child
    whose env can serve them (:func:`child_can_serve`) is returned as is. Any
    other case composes the venv, retires the old child if there was one,
    spawns, and runs ``InitializeSession`` before handing back the handle.

    Order on a replacement: compose first, so a compose failure leaves the
    old child and its working pipeline untouched; then retire the old child
    with a confirmed exit (:class:`ChildStillRunning` otherwise, with the old
    child still attached); then spawn. Callers hold ``session.child_lock``.
    Raises :class:`SessionClosedDuringLoad` when the session was closed
    meanwhile; the fresh child is stopped and its lease removed first.
    """
    session = session_manager.get_session(session_id)
    resolved = _resolve_plugins(pipeline_config, session, data_module)

    existing = session.child_handle
    replacing = False
    if existing is not None:
        if existing.returncode is None:
            if child_can_serve(session, resolved, data_module):
                return existing
            replacing = True
            logger.info(
                f"Child runtime for session {session_id} was composed for plugins "
                f"{sorted(session.resolved_plugins or {})} (data module "
                f"{session.child_data_module or '-'}); the pipeline needs "
                f"{sorted(resolved)} (data module {data_module or '-'}). "
                f"Replacing the child."
            )
        else:
            # A child that has exited (crash, OOM-kill) leaves a dead handle
            # behind; without this the session could never recover, since every
            # later call would forward to a dead stub. Drop it (lease included:
            # this path never reaches close_session) and re-spawn.
            logger.warning(
                f"Child runtime for session {session_id} has exited "
                f"(returncode={existing.returncode}); re-spawning a fresh child."
            )
            session_manager.retire_child(
                session, reason="child exited", require_exit=False
            )

    core_source = detect_core_source()
    logger.info(
        f"Composing child env for session {session_id} "
        f"({len(resolved)} plugins, core source: {core_source.kind}, "
        f"data_module: {data_module or '-'})"
    )
    # A failure here (uv, git, network) raises before the old child is touched.
    venv = _composer(resolved, core_source=core_source, active_data_module=data_module)

    # The session may have closed while the env was composing. Nothing was
    # spawned; the composed entry stays in the cache for the next session that
    # needs it. The old child, if any, is the close's to stop: it is either
    # waiting for this lock or has stopped it already.
    _abandon_if_closing(session_manager, session_id, session)

    if replacing:
        # The new env exists; only now stop the old child. terminate() waits
        # for the exit, so the driver can hand its GPU memory back before the
        # replacement starts loading.
        try:
            session_manager.retire_child(
                session, reason="pipeline switch", require_exit=True
            )
        except ChildStillRunning:
            if session.closing.is_set() or not session_manager.has_session(session_id):
                # The session went away while the old child refused to stop.
                # Nothing may put the survivor back on it: log it, drop its
                # lease, and answer as for any load a close overtook.
                session_manager.retire_child(
                    session, reason="session close", require_exit=False
                )
                _abandon_if_closing(session_manager, session_id, session)
            raise
        # A close that arrived while the old child was being stopped found no
        # handle to stop (the retire detaches it first) and tore down without
        # the lock. Spawning now would give a closed session a child.
        _abandon_if_closing(session_manager, session_id, session)

    # The previous child is gone (retired just now, or the dead one retired
    # before the compose) and its scratch tree goes with it, so what it left in
    # its TEMP and HOME does not accumulate switch after switch. The
    # replacement gets the same path, freshly created, and the tree's age
    # restarts with it.
    _discard_runtime_tree(session)
    try:
        declared = _default_declared_paths(session_id)
    except OSError as exc:
        raise SpawnError(
            f"Could not create the runtime scratch tree for session {session_id!r}: "
            f"{exc}"
        ) from exc
    # Record the child's scratch root (output/scratch share this parent) right
    # away, so close_session removes it whether or not the spawn below
    # succeeds: the spawner writes the child's logs into it before anything
    # else can fail.
    session.runtime_base_dir = declared.output_dir.parent
    cache_root = venv.parent.parent
    entry_digest = venv.parent.name
    # Lease lifecycle applies only to real composed cache entries — the
    # root carries the composer's marker file. Test seams (the in-memory
    # composer) return dummy paths with no cache behind them to protect.
    lease_root: Path | None = (
        cache_root if (cache_root / leases.ROOT_MARKER_NAME).is_file() else None
    )
    if lease_root is not None:
        # Intent lease BEFORE the spawn: endpoint/health polling can take
        # minutes and the freshly composed entry must already be protected
        # from eviction. It cannot be written earlier: one lease file per
        # session, and until the retire above returned it still had to name
        # the old child's digest. Between compose and this line the new entry
        # is protected by the composer's hot floor alone. A lease-write
        # failure fails the whole spawn — an unprotected child is worse than a
        # clean error.
        try:
            leases.write_intent_lease(
                lease_root,
                session_id,
                entry_digest,
                session_root=declared.output_dir.parent,
            )
        except OSError as exc:
            raise SpawnError(
                f"Could not write the runtime lease for session {session_id!r}: {exc}"
            ) from exc
        session.lease_cache_root = lease_root
    try:
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
    except Exception as exc:
        if lease_root is not None:
            leases.remove_lease(lease_root, session_id)
            session.lease_cache_root = None
        if isinstance(exc, OSError):
            # Popen, mkdtemp or a log file open failed below the spawner's own
            # SpawnError: still a spawn failure the caller can be told about.
            raise SpawnError(
                f"Could not start the child runtime for session {session_id!r}: {exc}"
            ) from exc
        raise

    try:
        _initialize_child_session(handle, session_id, session, resolved, declared)
        if lease_root is not None:
            try:
                leases.finalize_lease(
                    lease_root,
                    session_id,
                    entry_digest,
                    child_pid=handle.process.pid,
                    session_root=declared.output_dir.parent,
                )
            except (OSError, psutil.Error) as exc:
                # The child died between its health check and this line (psutil
                # finds no process), or the lease could not be rewritten: a
                # spawn failure like any other, told to the caller as such.
                raise SpawnError(
                    f"Could not finalize the runtime lease for session {session_id!r} "
                    f"(the child may have exited): {exc}"
                ) from exc
    except Exception:
        # The handshake helper already terminated the child on rejection;
        # a finalize failure leaves it running — stop it before dropping
        # the lease so no unprotected child survives (terminate is
        # idempotent on a dead process).
        try:
            handle.terminate(grace_s=CHILD_STOP_GRACE_SECONDS)
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning(f"Child terminate during spawn-failure cleanup: {exc}")
        if lease_root is not None:
            leases.remove_lease(lease_root, session_id)
            session.lease_cache_root = None
        raise

    if session.closing.is_set() or not session_manager.has_session(session_id):
        # The session was closed while the child was being prepared: from
        # inside this load's own lock scope (the lock is re-entrant), or by a
        # close that gave up on the lock after its grace while the spawn or the
        # health poll ran. Nothing may own the fresh child: stop it, drop its
        # lease and the scratch tree recreated for it.
        try:
            handle.terminate(grace_s=CHILD_STOP_GRACE_SECONDS)
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning(
                f"Child terminate after a session close raced the spawn: {exc}"
            )
        if getattr(handle, "returncode", None) is None:
            # kill() gave up after its wait. No session owns this child any
            # more; nothing retries the stop, so say so where a support bundle
            # shows it.
            logger.error(
                f"Child runtime spawned for the closed session {session_id} did not "
                f"stop (pid {getattr(getattr(handle, 'process', None), 'pid', '?')}); "
                f"it may still hold its GPU memory."
            )
        if lease_root is not None:
            leases.remove_lease(lease_root, session_id)
            session.lease_cache_root = None
        _discard_runtime_tree(session)
        raise SessionClosedDuringLoad(session_id)

    session.child_handle = handle
    session.resolved_plugins = dict(resolved)
    session.child_data_module = data_module
    session.child_install = {
        name: _install_identity(manifest) for name, manifest in resolved.items()
    }
    return handle


def _initialize_child_session(
    handle: ChildHandle,
    session_id: str,
    session: SessionState,
    resolved: Mapping[str, PluginManifest],
    declared: DeclaredPaths,
) -> None:
    """Hand the freshly-spawned child its session context via InitializeSession.

    Terminates the child and raises :class:`SpawnError` if it rejects the init
    handshake, whether as an ``ok=False`` answer or as an RPC error (the
    child's servicer is undecorated, so a plugin import failure inside
    ``register_plugins_installed`` arrives as UNKNOWN). Either way the server
    has answered and a fresh session would fail the same way, which is what
    the caller's FAILED_PRECONDITION mapping tells the client.
    """
    # resolved_plugins_json is a JSON list of single-plugin manifests; each
    # manifest carries its own `name`, so the list is self-describing.
    payload = json.dumps([cfg.model_dump() for cfg in resolved.values()]).encode(
        "utf-8"
    )
    request = cuvis_ai_pb2.InitializeSessionRequest(
        session_id=session_id,
        search_paths=list(session.search_paths),
        resolved_plugins_json=payload,
        output_dir=str(declared.output_dir),
        scratch_dir=str(declared.scratch_dir),
    )
    try:
        init_response = handle.stub().InitializeSession(request)
    except grpc.RpcError as exc:
        handle.terminate(grace_s=CHILD_STOP_GRACE_SECONDS)
        code = exc.code().name if hasattr(exc, "code") else "UNKNOWN"
        details = exc.details() if hasattr(exc, "details") else str(exc)
        raise SpawnError(
            f"Child runtime failed InitializeSession for session {session_id!r} "
            f"({len(resolved)} plugins: {sorted(resolved)}): {code}: {details}"
        ) from exc
    if not init_response.ok:
        handle.terminate(grace_s=CHILD_STOP_GRACE_SECONDS)
        raise SpawnError(
            f"Child runtime rejected InitializeSession for session "
            f"{session_id!r} ({len(resolved)} plugins: {sorted(resolved)})."
        )


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

    # Optional data-module name: lets the composer resolve the data-module
    # plugin's pip extras at compose time (the child env is frozen here, before
    # the data module would otherwise be needed at Train). It is not part of the
    # pipeline; only a pipeline run needs a data module.
    data_module = request.data_module or None

    # One load at a time per session, for the whole operation: a second load
    # must not retire the child while the first one is still loading into it.
    with _owning_child(session_manager, request.session_id, session):
        if session.closing.is_set() or not session_manager.has_session(
            request.session_id
        ):
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Session {request.session_id} was closed")
            return cuvis_ai_pb2.LoadPipelineResponse(success=False)
        try:
            child = ensure_child_for_session(
                session_manager,
                request.session_id,
                pipeline_config,
                data_module=data_module,
            )
        except Exception as exc:
            answer = _load_failure_status(
                exc,
                subject="pipeline",
                session_manager=session_manager,
                session_id=request.session_id,
                session=session,
            )
            if answer is None:
                # A bug, not a load failure: the undecorated servicer reports
                # it as UNKNOWN.
                raise
            context.set_code(answer[0])
            context.set_details(answer[1])
            return cuvis_ai_pb2.LoadPipelineResponse(success=False)
        if session.closing.is_set():
            # A close gave up on the lock between the child's return and this
            # forward; the child it found attached is stopped or about to be.
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Session {request.session_id} was closed")
            return cuvis_ai_pb2.LoadPipelineResponse(success=False)
        return _call_child_with_error_propagation(
            child,
            "LoadPipeline",
            request,
            context,
            lambda: cuvis_ai_pb2.LoadPipelineResponse(success=False),
            session=session,
        )


def forward_inference(
    session_manager: SessionManager,
    request: cuvis_ai_pb2.InferenceRequest,
    context: grpc.ServicerContext,
) -> cuvis_ai_pb2.InferenceResponse:
    """Parent's Inference path: route to the session's child runtime."""
    return _forward_pipeline_op(
        session_manager,
        request,
        context,
        stub_method="Inference",
        empty_response_factory=cuvis_ai_pb2.InferenceResponse,
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
        _answer_no_child(session, context)
        return iter([])

    def _proxy():
        try:
            yield from child.stub().Train(request)
        except grpc.RpcError as exc:
            # Trailers are set here, before the generator returns: that is the
            # last point at which a streaming RPC can still attach them.
            _propagate_child_failure(child, exc, context, session=session)

    return _proxy()


def forward_restore_train_run(
    session_manager: SessionManager,
    request: cuvis_ai_pb2.RestoreTrainRunRequest,
    context: grpc.ServicerContext,
) -> cuvis_ai_pb2.RestoreTrainRunResponse:
    """Parent's RestoreTrainRun path.

    Parses the trainrun yaml far enough to learn which plugins the pipeline
    needs, composes the venv, spawns the child via
    :func:`ensure_child_for_session`, then forwards the request. The child's
    ``RestoreTrainRun`` attaches the rebuilt pipeline to the same session_id,
    so the response we hand back to the public caller stays the parent's
    session id.

    With ``request.session_id`` set, the restore targets that existing
    session: plugin resolution then uses the session's client-pushed catalog,
    which is the only way to restore a trainrun whose pipeline declares
    ``plugins:`` (CreateSession → LoadPlugin each manifest → RestoreTrainRun).
    Empty keeps the fresh-session behavior; a server-created session has an
    empty catalog, so only plugin-less pipelines can restore that way.
    """
    from cuvis_ai_core.grpc.error_handling import get_session_or_error
    from cuvis_ai_core.grpc.trainrun_service import TrainRunService

    from cuvis_ai_schemas.pipeline import PipelineConfig

    trainrun_path = Path(request.trainrun_path)
    try:
        trainrun_config, pipeline_config_path = TrainRunService.parse_trainrun_yaml(
            trainrun_path
        )
    except FileNotFoundError as exc:
        context.set_code(grpc.StatusCode.NOT_FOUND)
        context.set_details(str(exc))
        return cuvis_ai_pb2.RestoreTrainRunResponse()
    except ValueError as exc:
        context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
        context.set_details(str(exc))
        return cuvis_ai_pb2.RestoreTrainRunResponse()
    # The trainrun references its pipeline by path; load it so the composer can
    # learn the plugin set the child env needs.
    pipeline_config = PipelineConfig.load_from_file(pipeline_config_path)

    if request.session_id:
        # Restore into the caller-prepared session (its LoadPlugin catalog
        # drives plugin resolution). The caller owns this session's lifecycle:
        # it is never closed on failure here.
        if get_session_or_error(session_manager, request.session_id, context) is None:
            return cuvis_ai_pb2.RestoreTrainRunResponse()
        parent_session_id = request.session_id
        owns_session = False
    else:
        # Allocate the parent's public session first so InitializeSession can
        # pin the child to that id.
        parent_session_id = session_manager.create_session()
        owns_session = True
    parent_session = session_manager.get_session(parent_session_id)

    def _drop_owned_session() -> None:
        """Close the fresh session this call created; never the caller's."""
        if not owns_session:
            return
        try:
            session_manager.close_session(parent_session_id)
        except Exception:  # pragma: no cover - defensive
            pass

    trainrun_data_module = getattr(
        getattr(trainrun_config, "data", None), "data_module", None
    )
    # Same lock scope as forward_load_pipeline: the restore owns the child
    # from the reuse-or-replace decision through the forwarded call. The lock
    # is re-entrant, so _drop_owned_session may close the session from here.
    with _owning_child(session_manager, parent_session_id, parent_session):
        if parent_session.closing.is_set() or not session_manager.has_session(
            parent_session_id
        ):
            # A close held the lock first and popped the caller's session:
            # NOT_FOUND, as forward_load_pipeline answers the same race, so
            # the client replays on a fresh session instead of reading a
            # refusal of the trainrun. (An owned session cannot be in this
            # state; nothing else knows its id yet.)
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Session {parent_session_id} was closed")
            return cuvis_ai_pb2.RestoreTrainRunResponse()
        try:
            ensure_child_for_session(
                session_manager,
                parent_session_id,
                pipeline_config,
                data_module=trainrun_data_module,
            )
        except Exception as exc:
            # The status is decided before the owned session is dropped: the
            # drop closes it, and a closed session must not turn every
            # failure into NOT_FOUND.
            answer = _load_failure_status(
                exc,
                subject="trainrun",
                session_manager=session_manager,
                session_id=parent_session_id,
                session=parent_session,
            )
            if not isinstance(exc, SessionClosedDuringLoad):
                _drop_owned_session()
            if answer is None:
                # A bug, not a load failure: the undecorated servicer reports
                # it as UNKNOWN.
                raise
            context.set_code(answer[0])
            context.set_details(answer[1])
            return cuvis_ai_pb2.RestoreTrainRunResponse()

        if parent_session.closing.is_set():
            # See forward_load_pipeline: a close gave up on the lock meanwhile.
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Session {parent_session_id} was closed")
            return cuvis_ai_pb2.RestoreTrainRunResponse()
        child = parent_session.child_handle
        response = _call_child_with_error_propagation(
            child,
            "RestoreTrainRun",
            request,
            context,
            cuvis_ai_pb2.RestoreTrainRunResponse,
            session=parent_session,
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
        _answer_no_child(session, context)
        return empty_response_factory()
    return _call_child_with_error_propagation(
        child, stub_method, request, context, empty_response_factory, session=session
    )


def _propagate_rpc_error(exc: grpc.RpcError, context: grpc.ServicerContext) -> None:
    """Copy a child ``RpcError``'s status code + details onto the parent context."""
    code = exc.code() if hasattr(exc, "code") else grpc.StatusCode.UNKNOWN
    details = exc.details() if hasattr(exc, "details") else str(exc)
    context.set_code(code or grpc.StatusCode.UNKNOWN)
    context.set_details(details or "")


def _propagate_child_failure(
    child,
    exc: grpc.RpcError,
    context: grpc.ServicerContext,
    *,
    session: SessionState | None = None,
) -> None:
    """Turn a transport failure into a child-crash status when the child is gone.

    A dead child fails every forwarded RPC with a status that tells the
    caller nothing: UNAVAILABLE while the endpoint refuses connections,
    but also UNKNOWN, CANCELLED or INTERNAL depending on how far the
    stream had got when the process died. When :func:`dead_child_details`
    confirms the child process exited, the parent answers ``INTERNAL``
    naming the exit code and the cause found in the child's stderr log,
    preserves the child's logs right there (so the client learns the
    location while the failure is being reported, not at teardown), and
    attaches the postmortem as trailing metadata.

    Only UNAVAILABLE pays the reap wait. The other three codes are
    equally plausible from a live child answering a business error, so
    their probe is poll-only and adds no latency to that path. A live
    child's own status is copied through unchanged.

    A child the parent retired itself (a pipeline switch replaced it, or the
    session closed) is answered as a replacement before any probe: ABORTED,
    the code for a conflict the caller resolves by repeating the request,
    not FAILED_PRECONDITION, which the desktop client shows once and never
    retries. Its exit is ours, there are no logs worth preserving and no
    crash trailers to attach, and the caller's next request lands on the
    replacement.
    """
    # `is True`, not truthiness: test doubles built from MagicMock answer every
    # attribute with a truthy mock, and a real handle carries a plain bool.
    if getattr(child, "retired_by_parent", False) is True:
        context.set_code(grpc.StatusCode.ABORTED)
        context.set_details(_REPLACED_DETAIL)
        return
    code = exc.code() if hasattr(exc, "code") else None
    if code not in _CRASH_PROBE_CODES:
        _propagate_rpc_error(exc, context)
        return
    # None lets the spawner resolve its own (env-overridable) reap grace.
    wait_s = None if code == grpc.StatusCode.UNAVAILABLE else 0.0
    details = dead_child_details(child, wait_s=wait_s)
    if details is None:
        _propagate_rpc_error(exc, context)
        return
    crash_dir = _preserve_crash_logs(child, session)
    if crash_dir is not None:
        details = f"{details}; logs preserved at {crash_dir}"
    logger.warning(f"Forwarded RPC failed: {details}")
    context.set_code(grpc.StatusCode.INTERNAL)
    context.set_details(details)
    _set_crash_trailers(context, getattr(child, "returncode", None), crash_dir)


def _preserve_crash_logs(child, session: SessionState | None) -> Path | None:
    """Copy the dead child's logs aside and record the directory on the session.

    Returns the crash-log directory, or ``None`` when there is no session
    to attribute the crash to or nothing could be preserved.
    ``preserve_child_logs`` is idempotent per child (session id plus
    endpoint), so the later retire or ``close_session`` teardown reports this
    same directory.
    """
    if session is None:
        return None
    # Lazy import: crash_logs pulls the composer's cache-root resolution.
    from cuvis_ai_core.orchestrator.crash_logs import preserve_child_logs

    crash_dir = preserve_child_logs(
        (getattr(child, "stdout_log", None), getattr(child, "stderr_log", None)),
        session_id=session.session_id,
        exit_code=getattr(child, "returncode", None),
        endpoint=getattr(child, "endpoint", None),
    )
    if crash_dir is not None:
        session.crash_log_dir = crash_dir
    return crash_dir


def _set_crash_trailers(
    context: grpc.ServicerContext, exit_code: Any, crash_dir: Path | None
) -> None:
    """Attach the child postmortem to the RPC's trailing metadata.

    Set on every forwarded RPC that a dead child failed, the streaming
    ``Train`` included (before its generator returns, which is the last
    moment trailers can still be set). Absent trailers are the older
    server's behaviour, so a client must treat them as optional.
    """
    trailers = [
        (TRAILER_CHILD_EXIT_CODE, str(exit_code) if exit_code is not None else ""),
        (
            TRAILER_CHILD_EXIT_TEXT,
            format_exit_code(exit_code) if isinstance(exit_code, int) else "",
        ),
        (TRAILER_CRASH_LOG_DIR, str(crash_dir) if crash_dir is not None else ""),
    ]
    try:
        context.set_trailing_metadata(tuple(trailers))
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"Could not set crash trailers: {exc}")


def _call_child_with_error_propagation(
    child,
    stub_method: str,
    request,
    context: grpc.ServicerContext,
    empty_response_factory,
    *,
    session: SessionState | None = None,
):
    """Invoke a child stub method and surface its status code on the parent's context."""
    try:
        return getattr(child.stub(), stub_method)(request)
    except grpc.RpcError as exc:
        _propagate_child_failure(child, exc, context, session=session)
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
        if session.load_in_flight:
            _answer_no_child(session, context)
            return cuvis_ai_pb2.SetTrainRunConfigResponse(success=False)
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


def forward_stop_train(session_manager, request, context):
    """Parent's StopTrain path: the stop flag lives in the child's session state."""
    return _forward_pipeline_op(
        session_manager,
        request,
        context,
        stub_method="StopTrain",
        empty_response_factory=cuvis_ai_pb2.StopTrainResponse,
    )


def forward_set_profiling(session_manager, request, context):
    """Parent's SetProfiling path: profiling state lives on the child's pipeline."""
    return _forward_pipeline_op(
        session_manager,
        request,
        context,
        stub_method="SetProfiling",
        empty_response_factory=cuvis_ai_pb2.SetProfilingResponse,
    )


def forward_get_profiling_summary(session_manager, request, context):
    return _forward_pipeline_op(
        session_manager,
        request,
        context,
        stub_method="GetProfilingSummary",
        empty_response_factory=cuvis_ai_pb2.GetProfilingSummaryResponse,
    )


# ---------------------------------------------------------------------------
# In-memory test seam — production code never instantiates these.
# ---------------------------------------------------------------------------


class _InMemoryContext:
    """Minimal ``grpc.ServicerContext`` stand-in used by the in-memory stub."""

    def __init__(self) -> None:
        self._code: grpc.StatusCode | None = None
        self._details: str = ""
        self._trailing_metadata: tuple[tuple[str, str], ...] = ()
        # Registered termination callbacks. The in-memory transport is
        # synchronous, so nothing fires them automatically; tests invoke them
        # to simulate a client-dropped stream.
        self.callbacks: list[Callable[[], None]] = []

    def add_callback(self, callback: Callable[[], None]) -> bool:
        """Record an RPC-termination callback (mirrors grpc.ServicerContext)."""
        self.callbacks.append(callback)
        return True

    def set_trailing_metadata(self, metadata) -> None:
        """Record trailing metadata (mirrors grpc.ServicerContext)."""
        self._trailing_metadata = tuple(metadata)

    def trailing_metadata(self) -> tuple[tuple[str, str], ...]:
        """Return the trailing metadata set on this call, empty when none."""
        return self._trailing_metadata

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

    def StopTrain(self, request, timeout=None):
        return self._call("StopTrain", request, timeout)

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

    def SetProfiling(self, request, timeout=None):
        return self._call("SetProfiling", request, timeout)

    def GetProfilingSummary(self, request, timeout=None):
        return self._call("GetProfilingSummary", request, timeout)

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
        # Mirrors ChildHandle.retired_by_parent (set by SessionManager.retire_child).
        self.retired_by_parent = False

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
    plugin_configs: Mapping[str, PluginManifest],
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
    pipeline_config: Any, session: SessionState, data_module: str | None = None
) -> Mapping[str, PluginManifest]:
    """Resolve the pipeline's declared plugins against the session catalog.

    The catalog is the set of plugins the client registered via ``LoadPlugin``
    (``session.registered_plugins``); the server never scans a plugins
    directory. Every plugin a pipeline names in its ``plugins:`` list must
    already be registered so the child env can be composed — a missing one
    raises :class:`PluginsNotRegisteredError`. ``plugins:`` is mandatory, so a
    pipeline that omits it still hard-fails in the resolver. ``data_module``
    (from the run's DataConfig) unions the providing data plugin into the set,
    since it ships no node classes to resolve by coverage.
    """
    catalog: dict[str, PluginManifest] = {
        name: parse_plugin_manifest(dump)
        for name, dump in session.registered_plugins.items()
    }
    declared = list(getattr(pipeline_config, "plugins", None) or [])
    missing = [name for name in declared if name not in catalog]
    if missing:
        raise PluginsNotRegisteredError(missing, registered=sorted(catalog))
    return resolve_against_catalog(pipeline_config, catalog, data_module=data_module)


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
