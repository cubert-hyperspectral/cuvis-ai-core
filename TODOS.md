# TODOs

## A run-cache eviction whose delete fails leaves a whole venv on disk

- **What:** `evict_run_cache` renames the entry to `<digest>.evicting.<ts>.<tag>` and deletes it
  on the deleter thread; when a file inside is locked (a child that has not exited, an
  antivirus scan) the delete stops and the renamed directory stays, `.venv` and all. Seen
  2026-09-22 on a developer machine: `5503c850aa175e15.evicting.1790064749.32e21b` with a
  complete venv of a `[sam3]` env, several GB.
- **Why:** The bound the eviction exists for (`CUVIS_RUN_CACHE_MAX_ENTRIES`) is silently
  exceeded by the leftovers, which no later pass touches.
- **Pros:** The cache stays within its bound without operator clean-up.
- **Cons:** A retry has to decide when an `.evicting` directory is abandoned (the deleter of
  another live server may still be working on it).
- **Context:** `orchestrator/composer.py` (`evict_run_cache`, `_sweep_stale_partials`, the
  deleter thread) and the `clean-run-cache` CLI. A reasonable rule: sweep `.evicting.*`
  directories older than the eviction pass's own age floor at server start and in
  `clean-run-cache`, with a retrying `rmtree` that logs the file it could not remove.
- **Depends on / blocked by:** nothing.
