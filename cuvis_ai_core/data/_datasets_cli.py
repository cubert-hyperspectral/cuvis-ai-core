"""The ``dataset`` command line (``cuvis_ai_core.data.public_datasets`` does the work).

Same contract as ``download-model``: stdout is machine-readable (a table or JSON,
one path per line for ``download``, JSON lines for ``--progress-json``), humans
read stderr, exit 0 / 1 (``error: ...``) / 2 (usage).
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import click

from cuvis_ai_core.data import public_datasets as ds
from cuvis_ai_core.data._provisioning import (
    EXIT_ERROR,
    SCHEMA_VERSION,
    ProgressEmitter,
    emit_json,
    format_bytes,
    load_schema,
    schema_names,
)

_F = Callable[..., Any]


def _data_dir_option(fn: _F) -> _F:
    return click.option(
        "--data-dir",
        type=click.Path(path_type=Path, file_okay=False),
        default=Path.cwd() / "data",
        show_default="./data",
        help="Datasets folder; each dataset lands in its own sub-directory.",
    )(fn)


def _json_option(fn: _F) -> _F:
    return click.option(
        "--json",
        "as_json",
        is_flag=True,
        help="Emit JSON (schema_version 1) instead of text.",
    )(fn)


def _fail(exc: BaseException) -> None:
    click.echo(f"error: {exc}", err=True)
    sys.exit(EXIT_ERROR)


def _rows(statuses: list[ds.DatasetStatus], **extra: Any) -> list[dict[str, Any]]:
    return [{**s.spec.to_json_dict(), **s.to_json_dict(), **extra} for s in statuses]


def _payload(data_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "data_dir": str(data_dir),
        "datasets": rows,
    }


def build_cli() -> click.Group:
    """Build the ``dataset`` click group."""

    @click.group()
    def cli() -> None:
        """Cuvis.AI public dataset management."""

    @cli.command("list")
    @_json_option
    @click.option(
        "--verbose", "-v", is_flag=True, help="Show repo, revision and file counts."
    )
    def list_cmd(as_json: bool, verbose: bool) -> None:
        """List the public datasets with their pinned revision, size and task labels."""
        if as_json:
            emit_json(ds.PublicDatasets.list_payload())
            return
        ds.PublicDatasets.list_datasets(verbose=verbose)

    @cli.command()
    @click.argument("names", nargs=-1)
    @_data_dir_option
    @_json_option
    def status(names: tuple[str, ...], data_dir: Path, as_json: bool) -> None:
        """Report what the datasets folder holds (present, incomplete, outdated, damaged, foreign, absent)."""
        try:
            statuses = ds.PublicDatasets.status_all(data_dir, list(names) or None)
        except ds.DatasetError as exc:
            _fail(exc)
            return
        if as_json:
            emit_json(_payload(data_dir, _rows(statuses)))
            return
        click.echo(f"data dir: {data_dir}")
        click.echo(f"{'Name':<30s} {'State':<11s} {'On disk':>10s}  Path")
        click.echo("-" * 110)
        for s in statuses:
            click.echo(
                f"  {s.spec.name:<28s} {s.state:<11s} {format_bytes(s.bytes_on_disk):>10s}  {s.path}"
            )

    @cli.command()
    @click.argument("names", nargs=-1, required=True)
    @_data_dir_option
    @click.option("--force", is_flag=True, help="Re-download even if present.")
    @click.option(
        "--adopt",
        is_flag=True,
        help="Take over an existing folder without a marker (Remove will then delete it).",
    )
    @click.option(
        "--prune-stale",
        is_flag=True,
        help="When refreshing an outdated dataset, delete files the new revision dropped.",
    )
    @_json_option
    @click.option(
        "--progress-json",
        is_flag=True,
        help="One JSON event per line on stdout while downloading (excludes --json).",
    )
    def download(
        names: tuple[str, ...],
        data_dir: Path,
        force: bool,
        adopt: bool,
        prune_stale: bool,
        as_json: bool,
        progress_json: bool,
    ) -> None:
        """Download (or resume, or refresh) datasets by NAME into --data-dir."""
        if as_json and progress_json:
            raise click.UsageError("--json and --progress-json are mutually exclusive.")
        progress = ProgressEmitter() if progress_json else None
        try:
            specs = [ds.PublicDatasets.get_spec(n) for n in names]
        except ds.DatasetError as exc:
            _fail(exc)
            return
        data_dir.mkdir(parents=True, exist_ok=True)
        stale_by_name: dict[str, list[str]] = {}
        for spec in specs:
            try:
                result = ds.PublicDatasets.download(
                    spec.name,
                    data_dir,
                    force=force,
                    adopt=adopt,
                    prune_stale=prune_stale,
                    progress=progress,
                )
            except ds.DatasetError as exc:
                if progress is not None:
                    progress.error(spec.name, str(exc))
                _fail(exc)
                return
            stale_by_name[spec.name] = list(result.stale_files)
            if result.stale_files:
                click.echo(
                    f"{spec.name}: {len(result.stale_files)} file(s) from the previous revision "
                    "are no longer part of this dataset (kept; --prune-stale removes them): "
                    + ", ".join(result.stale_files[:5]),
                    err=True,
                )
            if progress is None and not as_json:
                click.echo(str(result.path))
        if as_json:
            statuses = ds.PublicDatasets.status_all(data_dir, [s.name for s in specs])
            rows = _rows(statuses, downloaded=True)
            for row in rows:
                row["stale_files"] = stale_by_name.get(row["name"], [])
            emit_json(_payload(data_dir, rows))

    @cli.command()
    @click.argument("names", nargs=-1, required=True)
    @_data_dir_option
    @_json_option
    def remove(names: tuple[str, ...], data_dir: Path, as_json: bool) -> None:
        """Delete downloaded datasets (only folders carrying a Cubert marker)."""
        removed: list[dict[str, Any]] = []
        try:
            for name in names:
                spec = ds.PublicDatasets.get_spec(name)
                freed = ds.PublicDatasets.remove(spec.name, data_dir)
                removed.append({"name": spec.name, "freed_bytes": freed})
        except ds.DatasetError as exc:
            _fail(exc)
            return
        if as_json:
            emit_json(
                {
                    "schema_version": SCHEMA_VERSION,
                    "data_dir": str(data_dir),
                    "removed": removed,
                }
            )
        else:
            for item in removed:
                click.echo(f"{item['name']}: freed {format_bytes(item['freed_bytes'])}")

    @cli.command()
    @click.argument(
        "name", type=click.Choice(schema_names()) if schema_names() else str
    )
    def schema(name: str) -> None:
        """Print one of the shipped JSON Schemas (the --json contracts)."""
        click.echo(json.dumps(load_schema(name), indent=2))

    return cli
