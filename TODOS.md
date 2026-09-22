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
