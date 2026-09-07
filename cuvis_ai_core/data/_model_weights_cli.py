"""The ``download-model`` command line (``cuvis_ai_core.data.model_weights`` does the work).

stdout is the machine-readable channel: a table or JSON for ``list`` / ``status``,
one absolute path per line for ``download`` / ``import``, JSON lines for
``--progress-json``. Everything human goes to stderr. Exit codes: 0 ok,
1 an operation failed (``error: ...`` on stderr), 2 a usage error.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import click

from cuvis_ai_core.data import model_weights as mw
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


def _plugins_dir_option(fn: _F) -> _F:
    return click.option(
        "--plugins-dir",
        "plugins_dirs",
        multiple=True,
        type=click.Path(path_type=Path, file_okay=False),
        help=(
            "Directory of plugin manifests whose weights: blocks join the registry "
            "(repeatable, earlier wins; default: cuvis-ai's packaged configs/plugins when "
            "cuvis-ai is installed)."
        ),
    )(fn)


def _cache_dir_option(fn: _F) -> _F:
    return click.option(
        "--cache-dir",
        type=click.Path(path_type=Path, file_okay=False),
        default=None,
        help="Hugging Face cache root (default: the shared model cache the runtime reads).",
    )(fn)


def _json_option(fn: _F) -> _F:
    return click.option(
        "--json",
        "as_json",
        is_flag=True,
        help="Emit JSON (schema_version 1) instead of text.",
    )(fn)


def _progress_option(fn: _F) -> _F:
    return click.option(
        "--progress-json",
        is_flag=True,
        help="One JSON event per line on stdout while transferring (excludes --json).",
    )(fn)


def _load_plugins(plugins_dirs: tuple[Path, ...]) -> None:
    dirs = list(plugins_dirs) or mw.default_plugins_dirs()
    try:
        mw.ModelWeights.load_manifests(dirs)
    except (mw.ModelRegistryConflict, ValueError, OSError) as exc:
        _fail(exc)


def _exclusive(as_json: bool, progress_json: bool) -> None:
    if as_json and progress_json:
        raise click.UsageError("--json and --progress-json are mutually exclusive.")


def _fail(exc: BaseException) -> None:
    click.echo(f"error: {exc}", err=True)
    sys.exit(EXIT_ERROR)


def _payload(
    cache_dir: Path, models: list[dict[str, Any]], key: str = "cache_dir"
) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, key: str(cache_dir), "models": models}


def _status_rows(statuses: list[mw.ModelStatus], **extra: Any) -> list[dict[str, Any]]:
    return [{**s.weight.to_json_dict(), **s.to_json_dict(), **extra} for s in statuses]


def build_cli() -> click.Group:
    """Build the ``download-model`` click group."""

    @click.group()
    def cli() -> None:
        """Cuvis.AI model-weight provisioning."""

    @cli.command("list")
    @_json_option
    @_plugins_dir_option
    def list_cmd(as_json: bool, plugins_dirs: tuple[Path, ...]) -> None:
        """List the registry: every weight a plugin needs, with pins, sizes and licences."""
        _load_plugins(plugins_dirs)
        if as_json:
            emit_json(mw.ModelWeights.list_payload())
            return
        click.echo(
            f"{'Name':<26s} {'Plugin':<10s} {'Size':>10s}  {'Repo / file':<58s} Description"
        )
        click.echo("-" * 130)
        for row in mw.ModelWeights.rows():
            e = row.entry
            click.echo(
                f"  {e.name:<24s} {row.plugin:<10s} {format_bytes(row.total_bytes):>10s}  "
                f"{e.repo_id + '/' + e.filename:<58s} {e.summary or e.description}"
            )

    @cli.command()
    @click.argument("names", nargs=-1)
    @_json_option
    @click.option(
        "--verify", is_flag=True, help="Also compute and compare sha256 (slow)."
    )
    @_cache_dir_option
    @_plugins_dir_option
    def status(
        names: tuple[str, ...],
        as_json: bool,
        verify: bool,
        cache_dir: Path | None,
        plugins_dirs: tuple[Path, ...],
    ) -> None:
        """Report what the cache holds (present, damaged, partial, absent); never downloads."""
        _load_plugins(plugins_dirs)
        try:
            statuses = mw.ModelWeights.status_all(
                list(names) or None, cache_dir, verify=verify
            )
        except mw.ModelDownloadError as exc:
            _fail(exc)
            return
        root = mw.ModelWeights.resolve_cache_dir(cache_dir)
        if as_json:
            emit_json(_payload(root, _status_rows(statuses)))
            return
        click.echo(f"cache: {root}")
        click.echo(f"{'Name':<26s} {'State':<8s} {'On disk':>10s}  Path")
        click.echo("-" * 110)
        for s in statuses:
            click.echo(
                f"  {s.weight.name:<24s} {s.state:<8s} {format_bytes(s.size_bytes_on_disk):>10s}  {s.path}"
            )

    @cli.command()
    @click.argument("names", nargs=-1)
    @_json_option
    @_progress_option
    @click.option("--force", is_flag=True, help="Re-download even if present.")
    @_cache_dir_option
    @_plugins_dir_option
    @click.option(
        "--repo-id", default=None, help="Explicit HF repo id (custom or private repo)."
    )
    @click.option("--filename", default=None, help="File in the explicit repo.")
    @click.option(
        "--revision", default=None, help="Revision to pin for the explicit repo."
    )
    @click.option(
        "--token", default=None, help="HF token for the explicit repo (else $HF_TOKEN)."
    )
    @click.option(
        "--out",
        type=click.Path(path_type=Path),
        default=None,
        help="Also copy the resolved file here (single download only).",
    )
    def download(
        names: tuple[str, ...],
        as_json: bool,
        progress_json: bool,
        force: bool,
        cache_dir: Path | None,
        plugins_dirs: tuple[Path, ...],
        repo_id: str | None,
        filename: str | None,
        revision: str | None,
        token: str | None,
        out: Path | None,
    ) -> None:
        """Download registry weights by NAME (or one explicit --repo-id/--filename)."""
        _exclusive(as_json, progress_json)
        _load_plugins(plugins_dirs)
        progress = ProgressEmitter() if progress_json else None
        root = mw.ModelWeights.resolve_cache_dir(cache_dir)
        if repo_id or filename:
            if names or not (repo_id and filename):
                raise click.UsageError(
                    "Pass either registry NAMEs or both --repo-id and --filename."
                )
            label = f"{repo_id}/{filename}"
            try:
                path = mw.ModelWeights.download_model(
                    None,
                    repo_id=repo_id,
                    filename=filename,
                    revision=revision,
                    token=token,
                    cache_dir=cache_dir,
                    out=out,
                    force=force,
                    progress=progress,
                )
            except mw.ModelDownloadError as exc:
                if progress is not None:
                    progress.error(label, str(exc))
                _fail(exc)
                return
            if progress is not None:
                progress.done(label, path)
            else:
                # The explicit-repo form has no registry row, so there is no SPEC to
                # report: the path line is its whole contract, --json or not.
                click.echo(str(path))
            return
        if not names:
            raise click.UsageError(
                "Give at least one registry NAME (see download-model list)."
            )
        if out is not None and len(names) != 1:
            raise click.UsageError("--out applies to a single download.")
        try:
            weights = [mw.ModelWeights.get(n) for n in names]
        except mw.ModelDownloadError as exc:
            _fail(exc)
            return
        paths: list[Path] = []
        for weight in weights:
            try:
                path = mw.ModelWeights.download_model(
                    weight.name,
                    cache_dir=cache_dir,
                    out=out,
                    force=force,
                    progress=progress,
                )
            except mw.ModelDownloadError as exc:
                if progress is not None:
                    progress.error(weight.name, str(exc))
                _fail(exc)
                return
            paths.append(path)
            if progress is not None:
                progress.done(weight.name, path)
            elif not as_json:
                click.echo(str(path))
        if as_json:
            statuses = mw.ModelWeights.status_all([w.name for w in weights], cache_dir)
            emit_json(_payload(root, _status_rows(statuses, downloaded=True)))

    @cli.command()
    @click.option(
        "--to",
        "export_dir",
        required=True,
        type=click.Path(path_type=Path, file_okay=False),
    )
    @click.argument("names", nargs=-1)
    @_json_option
    @_progress_option
    @_cache_dir_option
    @_plugins_dir_option
    def export(
        export_dir: Path,
        names: tuple[str, ...],
        as_json: bool,
        progress_json: bool,
        cache_dir: Path | None,
        plugins_dirs: tuple[Path, ...],
    ) -> None:
        """Copy present weights into a folder that is itself a valid cache (air-gapped sites)."""
        _exclusive(as_json, progress_json)
        _load_plugins(plugins_dirs)
        progress = ProgressEmitter() if progress_json else None
        try:
            results = mw.ModelWeights.export_to(
                export_dir, list(names) or None, cache_dir, progress=progress
            )
        except mw.ModelDownloadError as exc:
            if progress is not None:
                progress.error("export", str(exc))
            _fail(exc)
            return
        if as_json:
            emit_json(_payload(export_dir.resolve(), results, key="export_dir"))
        elif progress is None:
            for r in results:
                click.echo(
                    f"{r['name']}: {'exported to ' + r['path'] if r['exported'] else 'absent, skipped'}"
                )
        if names and any(not r["exported"] for r in results):
            sys.exit(EXIT_ERROR)

    @cli.command("import")
    @click.argument(
        "source_dir", type=click.Path(path_type=Path, exists=True, file_okay=False)
    )
    @click.argument("names", nargs=-1)
    @_json_option
    @_progress_option
    @_cache_dir_option
    @_plugins_dir_option
    def import_cmd(
        source_dir: Path,
        names: tuple[str, ...],
        as_json: bool,
        progress_json: bool,
        cache_dir: Path | None,
        plugins_dirs: tuple[Path, ...],
    ) -> None:
        """Import weights from an exported folder, verified, all or nothing."""
        _exclusive(as_json, progress_json)
        _load_plugins(plugins_dirs)
        progress = ProgressEmitter() if progress_json else None
        root = mw.ModelWeights.resolve_cache_dir(cache_dir)
        try:
            results = mw.ModelWeights.import_from(
                source_dir, list(names) or None, cache_dir, progress=progress
            )
        except mw.ModelDownloadError as exc:
            if progress is not None:
                progress.error("import", str(exc))
            _fail(exc)
            return
        if as_json:
            imported_names = [r["name"] for r in results if "cache_dir_name" in r]
            statuses = (
                mw.ModelWeights.status_all(imported_names, cache_dir)
                if imported_names
                else []
            )
            by_name = {s.weight.name: s for s in statuses}
            rows = []
            for r in results:
                s = by_name.get(str(r.get("name", "")))
                rows.append({**r, **(s.to_json_dict() if s else {})})
            emit_json(_payload(root, rows))
        elif progress is None:
            for r in results:
                if r.get("imported"):
                    click.echo(r["path"])
                else:
                    click.echo(
                        f"{r.get('name')}: skipped ({r.get('reason', 'not requested')})",
                        err=True,
                    )

    @cli.command()
    @click.argument("names", nargs=-1)
    @click.option(
        "--dir",
        "dirname",
        default=None,
        help="Remove a whole models--* folder (orphans).",
    )
    @_json_option
    @_cache_dir_option
    @_plugins_dir_option
    def remove(
        names: tuple[str, ...],
        dirname: str | None,
        as_json: bool,
        cache_dir: Path | None,
        plugins_dirs: tuple[Path, ...],
    ) -> None:
        """Delete a registry weight's files (or, with --dir, an orphaned models--* folder)."""
        _load_plugins(plugins_dirs)
        if bool(names) == bool(dirname):
            raise click.UsageError(
                "Give registry NAMEs or --dir, not both and not neither."
            )
        root = mw.ModelWeights.resolve_cache_dir(cache_dir)
        removed: list[dict[str, Any]] = []
        try:
            if dirname:
                removed.append(
                    {
                        "dir": dirname,
                        "freed_bytes": mw.ModelWeights.remove_dir(dirname, cache_dir),
                    }
                )
            else:
                for name in names:
                    removed.append(
                        {
                            "name": name,
                            "freed_bytes": mw.ModelWeights.remove(name, cache_dir),
                        }
                    )
        except mw.ModelDownloadError as exc:
            _fail(exc)
            return
        if as_json:
            emit_json(
                {
                    "schema_version": SCHEMA_VERSION,
                    "cache_dir": str(root),
                    "removed": removed,
                }
            )
        else:
            for item in removed:
                click.echo(
                    f"{item.get('name') or item.get('dir')}: freed {format_bytes(item['freed_bytes'])}"
                )

    @cli.command()
    @click.argument(
        "name", type=click.Choice(schema_names()) if schema_names() else str
    )
    def schema(name: str) -> None:
        """Print one of the shipped JSON Schemas (the --json contracts)."""
        click.echo(json.dumps(load_schema(name), indent=2))

    return cli
