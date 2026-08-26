"""Explicit run-cache cleanup CLI (``clean-run-cache``).

Manual counterpart to the server's hourly maintenance pass: reap orphaned
child runtimes, evict cache entries, optionally sweep per-session scratch
trees, and finish with ``uv cache prune`` so blob storage is actually
released. Every maintenance guard applies here exactly as it does in the
background pass — the composer root marker, the machine-local (non-UNC)
root requirement, leases, and the live-process scan — so a cache entry or
scratch tree that is in use is never touched.
"""

from __future__ import annotations

from pathlib import Path

import click

from cuvis_ai_core.orchestrator import leases
from cuvis_ai_core.orchestrator.composer import (
    evict_run_cache,
    resolve_cache_root,
    wait_for_deleter,
)
from cuvis_ai_core.orchestrator.uv_runner import (
    UvCacheBusyError,
    UvRunnerError,
    uv_cache_prune,
)


@click.command()
@click.option(
    "--root",
    type=click.Path(path_type=Path),
    default=None,
    help="Cache root (default: $CUVIS_RUN_CACHE_DIR or ~/.cuvis_runs).",
)
@click.option(
    "--all",
    "evict_all",
    is_flag=True,
    help=(
        "Evict every entry not currently in use (ignores the age/count/"
        "hot-floor caps; leases and live processes still protect)."
    ),
)
@click.option(
    "--sessions",
    "sweep_sessions",
    is_flag=True,
    help=(
        "Also sweep lease-less per-session scratch trees (with --all the "
        "7-day age floor is dropped; liveness guards always apply)."
    ),
)
@click.option(
    "--dry-run",
    is_flag=True,
    help=(
        "List what would be removed and remove nothing (also skips orphan "
        "reaping, which signals processes, and the final uv cache prune)."
    ),
)
def main(
    root: Path | None, evict_all: bool, sweep_sessions: bool, dry_run: bool
) -> None:
    """One explicit cleanup pass over the composed-env run cache."""
    cache_root = resolve_cache_root(root)
    if not cache_root.is_dir():
        click.echo(f"Cache root {cache_root} does not exist; nothing to do.")
        return
    leases.adopt_root_if_composer_shaped(cache_root)
    if not leases.root_guard_ok(cache_root):
        if leases.root_never_composed(cache_root):
            click.echo(f"Cache root {cache_root} is empty; nothing composed yet.")
            return
        raise click.ClickException(
            f"{cache_root} is not a cuvis run-cache root "
            "(composer marker missing, or a UNC path)."
        )

    # The first full process scan in a fresh process is slow on Windows
    # (tens of seconds for a few hundred processes); say so up front.
    click.echo("Scanning live processes to protect in-use entries...")
    if dry_run:
        click.echo("Dry run: orphan reaping skipped (it would signal processes).")
    else:
        leases.reap_orphans(cache_root)

    evicted = evict_run_cache(cache_root, evict_all=evict_all, dry_run=dry_run)
    plural = "y" if len(evicted) == 1 else "ies"
    verb = "Would evict" if dry_run else "Evicted"
    click.echo(
        f"{verb} {len(evicted)} cache entr{plural}"
        + (f": {', '.join(evicted)}" if evicted else ".")
    )

    if sweep_sessions:
        swept = leases.sweep_scratch(
            cache_root, relax_age_floor=evict_all, dry_run=dry_run
        )
        verb = "Would sweep" if dry_run else "Swept"
        click.echo(
            f"{verb} {len(swept)} session scratch dir(s)"
            + (f": {', '.join(path.name for path in swept)}" if swept else ".")
        )

    if dry_run:
        click.echo("Dry run: uv cache prune skipped.")
        return
    if not wait_for_deleter(timeout=600.0):
        click.echo("Warning: deletions are still draining in the background.")
    if evicted:
        # evict_run_cache queued one prune behind its deletion batch and
        # the drain above already ran it; a second walk adds nothing.
        return
    click.echo(
        "Running uv cache prune (skipped if another uv process holds the cache)..."
    )
    try:
        uv_cache_prune()
        click.echo("uv cache prune completed.")
    except UvCacheBusyError as exc:
        click.echo(f"uv cache prune skipped: {exc}")
    except UvRunnerError as exc:
        click.echo(f"uv cache prune failed (non-fatal): {exc}")


if __name__ == "__main__":  # pragma: no cover - console-script entry
    main()
