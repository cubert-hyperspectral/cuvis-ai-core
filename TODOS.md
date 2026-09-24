# TODOs

## A run-cache eviction whose delete fails keeps a whole venv on disk for at least six hours

- **What:** `evict_run_cache` renames the entry to `<digest>.evicting.<ts>.<tag>` and deletes it
  on the deleter thread; when a file inside is locked (a child that has not exited, an
  antivirus scan) the delete stops and the renamed directory stays, `.venv` and all.
  `_sweep_failed_dirs` does queue such remnants again, but only from a later eviction pass and
  only once they are older than `_STALE_PARTIAL_AGE_SECONDS` (6 h). Seen 2026-09-22 on a
  developer machine: `5503c850aa175e15.evicting.1790064749.32e21b`, a complete `[sam3]` venv of
  several GB renamed at 10:12, swept by the next server's eviction pass at 19:59.
- **Why:** For those hours the bound the eviction exists for (`CUVIS_RUN_CACHE_MAX_ENTRIES`) is
  exceeded by a directory nothing uses, on the disk the cache is meant to protect.
- **Pros:** The cache returns to its bound as soon as the lock is gone, not hours later.
- **Cons:** An earlier retry has to tell an abandoned remnant from one another live server's
  deleter is still working on (the `<tag>` names the renaming process; the process snapshot
  the orphan reaper already takes can settle it).
- **Context:** `orchestrator/composer.py` (`evict_run_cache`, `_sweep_failed_dirs`, the deleter
  thread, `_STALE_PARTIAL_AGE_SECONDS`) and the `clean-run-cache` CLI, which sweeps sessions and
  evicts entries but does not touch `.evicting.*` remnants at all. A reasonable rule: retry the
  `rmtree` of a remnant whose renaming process is gone at server start and in `clean-run-cache`,
  logging the file that could not be removed.
- **Depends on / blocked by:** nothing.

## Bound the wait for a session's child lock on the load path

- **What:** `_owning_child` acquires `SessionState.child_lock` without a timeout. A `LoadPipeline`
  whose forwarded call hangs inside the child holds the lock until a `CloseSession` breaks it;
  every further `LoadPipeline` / `RestoreTrainRun` on that session parks a gRPC worker thread
  (default pool: 10) for as long as that takes.
- **Why:** One hung child plus a client that keeps sending loads can starve the whole server,
  other sessions and the health check included, until the client closes the session.
- **Pros:** A bounded acquire that answers `ABORTED` ("a load is in progress, repeat later") keeps
  the pool free; a deadline on the forwarded `LoadPipeline` / `RestoreTrainRun` stub call would
  make a hung child release the lock on its own.
- **Cons:** The second of two competing loads on one session today waits and succeeds
  (`test_competing_loads_on_one_session_serialise_whole_operation`); a bound turns that into a
  retry the client has to implement, and a load deadline has to leave room for a cold weights
  download.
- **Context:** `grpc/orchestrator_bridge.py` (`_owning_child`, `_call_child_with_error_propagation`),
  `grpc/production_server.py` (`max_workers`); the CuvisNEXT classifier puts `ABORTED` in its
  transport bucket until it gains a retry-in-place branch.
- **Depends on / blocked by:** a client-side `ABORTED` branch in CuvisNEXT `classify_error`.

## Resolve a plugin's git tag once per process

- **What:** Every compose runs `git ls-remote --tags` once per git-sourced plugin
  (`orchestrator/runtime_project.py`, `resolve_git_tag`, 60 s timeout each) before it can compute
  the cache key, so even a switch back to a family this server already composed pays a network
  round trip per git plugin while holding the session's child lock; an unreachable remote stalls
  the switch, and any close of that session, for up to 60 s per plugin.
- **Why:** Pipeline switches now compose routinely (one per family change); the tag lookup is the
  only network step on a warm-cache switch.
- **Pros:** A per-process `(repo, tag) -> sha` memo (a dict under a lock) makes a warm switch
  offline-capable and instant.
- **Cons:** A tag moved to another commit goes unnoticed until the server restarts; the composer's
  cache key already treats tags as immutable, so this is consistent, but the docs have to say it.
- **Context:** `orchestrator/runtime_project.py` (`resolve_git_tag`, `resolve_plugin_sources`),
  `orchestrator/composer.py` (`_build_or_reuse` resolves before the `.ready` check).
- **Depends on / blocked by:** nothing.

## Make a compose or spawn in flight abortable on close

- **What:** `close_session` no longer waits for a compose or a spawn, but the load thread does:
  `uv sync` (minutes) and the child's endpoint / health poll (up to 120 s) run to completion, then
  the load notices the close and disposes of what it created. Server shutdown returns from its
  RPC layer within the close bound, yet the process exits only once those worker threads finish
  (the executor joins them at interpreter exit), so a service manager, or the CuvisNEXT app that
  kills the server 8 s after Ctrl+Break, ends up killing a server that already said goodbye.
- **Why:** A bounded shutdown at the process level, and no throwaway spawn for a session that is
  gone.
- **Pros:** Hand the composer and the spawner the session's `closing` event: the composer kills
  its `uv` subprocess and raises `SessionClosedDuringLoad`, the spawner stops polling and
  terminates the child.
- **Cons:** A killed `uv sync` leaves a partial cache entry (the composer's `_sweep_failed_dirs`
  handles those, but only later); the composer is shared between sessions, so the cancel must be
  per call.
- **Context:** `orchestrator/composer.py` (`compose_env`), `orchestrator/spawner.py`
  (`LocalChildRuntimeSpawner.spawn`, the endpoint and health polls),
  `grpc/orchestrator_bridge.py` (`ensure_child_for_session`), `grpc/production_server.py`
  (`_close_all_sessions`, `serve`).
- **Depends on / blocked by:** nothing.

## One answer for requests that race a pipeline switch

- **What:** While a load owns a session, a racing `Inference` / `GetPipelineOutputs` is answered
  `ABORTED` when it finds no child, but the fresh child is attached before the forwarded
  `LoadPipeline` runs, so during the pipeline build and weight load the same request reaches a
  child with no pipeline and gets the child's own `FAILED_PRECONDITION` ("Build pipeline first").
  One transient state shows two codes depending on timing; and a request that reaches a survivor
  the parent could not stop (`ChildStillRunning`) is forwarded into it instead of being refused up
  front.
- **Why:** The desktop client shows `FAILED_PRECONDITION` once and never retries it, while
  `ABORTED` means "repeat the request"; a switch under a live view should look like one transient
  state.
- **Pros:** A `child_ready` flag on `SessionState` (cleared when a fresh child is attached, set
  once its `LoadPipeline` returned) lets every non-load forwarder answer `ABORTED` for the whole
  switch window and refuse a marked survivor with `FAILED_PRECONDITION`.
- **Cons:** A warm reuse load must keep serving the old pipeline meanwhile (the child handles that
  today), so the flag must cover a replacement only, not every load.
- **Context:** `grpc/orchestrator_bridge.py` (`_answer_no_child`, `forward_inference`,
  `_forward_pipeline_op`, `_propagate_child_failure`), `grpc/error_handling.py`
  (`require_pipeline`).
- **Depends on / blocked by:** the CuvisNEXT `ABORTED` branch, for the retry in place to happen.
