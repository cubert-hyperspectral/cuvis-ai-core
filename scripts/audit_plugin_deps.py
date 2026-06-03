"""Audit dependency-constraint drift across the cuvis-ai ecosystem.

Two independent checks, selectable with ``--check``:

* ``host`` — a repository's own ``pyproject.toml`` dependency floors against its
  ``uv.lock``. Flags direct dependencies whose lower bound lags the locked (and
  therefore tested) version, and direct dependencies that declare no lower bound
  at all. Dependencies sourced from a local path / workspace / VCS (editable
  siblings) are skipped, as are dependencies whose environment marker does not
  apply to the current platform.
* ``plugins`` — each plugin manifest's requirements against the version that
  cuvis-ai-core locks. Flags a plugin dependency whose specifier excludes the
  version core has locked, which would make the plugin uninstallable alongside
  that core release.

The audit only reads files on disk; it imports nothing from the audited project,
so it runs in an isolated environment against any repository.
"""

from __future__ import annotations

import json
import sys
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

import click
from packaging.requirements import Requirement
from packaging.version import InvalidVersion, Version

# A dependency listed under ``[tool.uv.sources]`` with any of these keys is an
# editable sibling resolved from outside PyPI, so its lock version is irrelevant.
_LOCAL_SOURCE_KEYS = ("path", "workspace", "git", "url")

# Specifier operators that establish a lower bound on a requirement.
_FLOOR_OPERATORS = (">=", ">", "==", "~=")


def normalize(name: str) -> str:
    """Return the PEP 503 normalised form of a distribution name."""
    return name.lower().replace("_", "-")


def load_pyproject(path: Path) -> dict:
    """Parse a ``pyproject.toml`` file into a dict."""
    return tomllib.loads(path.read_text(encoding="utf-8"))


def load_lock(path: Path) -> dict[str, Version]:
    """Map every package in a ``uv.lock`` to its locked version."""
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    locked: dict[str, Version] = {}
    for pkg in data.get("package", []):
        try:
            locked[normalize(pkg["name"])] = Version(pkg["version"])
        except (InvalidVersion, KeyError):
            continue
    return locked


def load_installed_versions() -> dict[str, Version]:
    """Map every distribution installed in the current env to its version.

    Used by per-plugin-repo CI: install the target ``cuvis-ai-core`` release
    into the job's venv, then audit the plugin against the versions actually
    resolved there. This sidesteps the fact that a published sdist/wheel does
    not ship ``uv.lock``.
    """
    from importlib import metadata

    locked: dict[str, Version] = {}
    for dist in metadata.distributions():
        name = dist.metadata["Name"]
        if not name:
            continue
        try:
            locked[normalize(name)] = Version(dist.version)
        except InvalidVersion:
            continue
    return locked


def resolve_core_lock(
    against_core: str | None, project_dir: Path
) -> dict[str, Version]:
    """Resolve the cuvis-ai-core locked-version set for the plugin check.

    ``--against-core`` accepts:

    * unset → the lock in ``--project-dir`` (the repo being audited).
    * ``installed`` → versions of the distributions installed in this env
      (``importlib.metadata``), i.e. the core release CI just pip-installed.
    * any other value → a path to a cuvis-ai-core checkout whose ``uv.lock``
      is read.
    """
    if against_core is None:
        return load_lock(project_dir / "uv.lock")
    if against_core == "installed":
        return load_installed_versions()
    return load_lock(Path(against_core) / "uv.lock")


def project_dependencies(pyproject: dict) -> list[Requirement]:
    """Parse ``[project].dependencies`` whose markers apply to this environment."""
    reqs: list[Requirement] = []
    for raw in pyproject.get("project", {}).get("dependencies") or []:
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        try:
            req = Requirement(text)
        except Exception:  # noqa: BLE001 - skip anything unparseable
            continue
        if req.marker is not None and not req.marker.evaluate():
            continue
        reqs.append(req)
    return reqs


def local_source_names(pyproject: dict) -> set[str]:
    """Names resolved from a local/VCS source via ``[tool.uv.sources]``."""
    sources = pyproject.get("tool", {}).get("uv", {}).get("sources", {}) or {}
    local: set[str] = set()
    for name, spec in sources.items():
        if isinstance(spec, list) or (
            isinstance(spec, dict) and any(k in spec for k in _LOCAL_SOURCE_KEYS)
        ):
            local.add(normalize(name))
    return local


def floor_of(req: Requirement) -> Version | None:
    """Lowest version a requirement admits, or ``None`` if it has no lower bound."""
    floors: list[Version] = []
    for spec in req.specifier:
        if spec.operator in _FLOOR_OPERATORS:
            try:
                floors.append(Version(spec.version.replace(".*", "")))
            except InvalidVersion:
                continue
    return min(floors) if floors else None


def floor_lags(floor: Version, locked: Version) -> bool:
    """True if ``floor`` is below the locked *public* version.

    The locked version's local segment is ignored so that, e.g.,
    ``torch==2.11.0+cu128`` satisfies a ``torch>=2.11.0`` floor and is not
    reported as stale.
    """
    return floor < Version(locked.public)


@dataclass
class HostFinding:
    """A single host-check problem with one direct dependency."""

    name: str
    specifier: str
    locked: str
    kind: str  # "stale" | "missing-floor"


def check_host(
    project_dir: Path,
    pyproject_path: Path | None = None,
    lock_path: Path | None = None,
) -> tuple[list[HostFinding], list[str]]:
    """Compare a repo's dependency floors against its own lock.

    Returns ``(findings, warnings)`` where findings are strict failures and
    warnings (e.g. a declared dependency missing from the lock) are not.
    """
    pp_path = pyproject_path or (project_dir / "pyproject.toml")
    lk_path = lock_path or (project_dir / "uv.lock")
    pyproject = load_pyproject(pp_path)
    locked = load_lock(lk_path)
    siblings = local_source_names(pyproject)

    findings: list[HostFinding] = []
    warnings: list[str] = []
    for req in project_dependencies(pyproject):
        name = normalize(req.name)
        if name in siblings:
            continue
        locked_ver = locked.get(name)
        if locked_ver is None:
            warnings.append(f"{req.name}{req.specifier or ''} (not in lock)")
            continue
        floor = floor_of(req)
        if floor is None:
            findings.append(
                HostFinding(
                    req.name, str(req.specifier), str(locked_ver), "missing-floor"
                )
            )
        elif floor_lags(floor, locked_ver):
            findings.append(
                HostFinding(req.name, str(req.specifier), str(locked_ver), "stale")
            )
    return findings, warnings


@dataclass
class PluginFinding:
    """A plugin dependency whose specifier excludes the locked core version."""

    plugin: str
    name: str
    specifier: str
    core_locked: str


def _resolve_manifest_pyproject(
    name: str, cfg: dict, plugins_dir: Path, cache_dir: Path
) -> Path | None:
    """Locate a plugin's ``pyproject.toml`` from its manifest entry."""
    if "path" in cfg:
        candidate = (plugins_dir / cfg["path"]).resolve() / "pyproject.toml"
        return candidate if candidate.exists() else None
    tag = cfg.get("tag")
    if tag:
        candidate = cache_dir / f"{name}@{tag}" / "pyproject.toml"
        return candidate if candidate.exists() else None
    return None


def _check_one_pyproject(
    plugin_name: str, pp_path: Path, core_lock: dict[str, Version]
) -> list[PluginFinding]:
    """Flag deps in one plugin pyproject whose specifier excludes the core lock."""
    findings: list[PluginFinding] = []
    for req in project_dependencies(load_pyproject(pp_path)):
        key = normalize(req.name)
        if key in {"cuvis-ai-core", "cuvis-ai-schemas"}:
            continue
        locked_ver = core_lock.get(key)
        if locked_ver is None or not req.specifier:
            continue
        if not req.specifier.contains(locked_ver, prereleases=True):
            findings.append(
                PluginFinding(
                    plugin_name, req.name, str(req.specifier), str(locked_ver)
                )
            )
    return findings


def check_plugins(
    plugins_dir: Path, core_lock: dict[str, Version]
) -> tuple[list[PluginFinding], list[str]]:
    """Compare each manifest's plugin requirements against the core lock."""
    import yaml  # local import: only the plugin check needs PyYAML

    cache_dir = Path.home() / ".cuvis_plugins"
    findings: list[PluginFinding] = []
    warnings: list[str] = []
    for manifest in sorted(plugins_dir.glob("*.yaml")):
        data = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
        for name, cfg in (data.get("plugins") or {}).items():
            # Local-path entries are dev checkouts or the built-in catalog
            # (e.g. ``cuvis_ai_builtin`` → the host repo). They are host-checked
            # in their own repo's --check host run, not against core's lock, so
            # the registry check skips them and audits only tag-pinned externals.
            if "path" in cfg:
                warnings.append(
                    f"{name}: local-path entry - skipped (host-checked separately)"
                )
                continue
            pp_path = _resolve_manifest_pyproject(name, cfg, plugins_dir, cache_dir)
            if pp_path is None:
                warnings.append(f"{name}: pyproject not available locally - skipped")
                continue
            findings.extend(_check_one_pyproject(name, pp_path, core_lock))
    return findings, warnings


def check_plugin_pyproject(
    pyproject_path: Path, core_lock: dict[str, Version]
) -> tuple[list[PluginFinding], list[str]]:
    """Compare a single plugin's ``pyproject.toml`` against the core lock.

    Used by per-plugin-repo CI (``--plugin-pyproject ./pyproject.toml``), where
    the plugin under test is the repo being checked out rather than a manifest
    in a catalog directory.
    """
    if not pyproject_path.exists():
        return [], [f"{pyproject_path}: not found - skipped"]
    pyproject = load_pyproject(pyproject_path)
    name = pyproject.get("project", {}).get("name", pyproject_path.parent.name)
    return _check_one_pyproject(name, pyproject_path, core_lock), []


def _print_host(findings: list[HostFinding], warnings: list[str]) -> None:
    click.echo("Host check - pyproject floors vs uv.lock")
    for f in findings:
        label = "STALE" if f.kind == "stale" else "NO FLOOR"
        click.echo(f"  {label:9} {f.name}{f.specifier}  (locked {f.locked})")
    for w in warnings:
        click.echo(f"  note      {w}")
    click.echo(f"  -> {len(findings)} finding(s), {len(warnings)} note(s)")


def _print_plugins(findings: list[PluginFinding], warnings: list[str]) -> None:
    click.echo("Plugin check - plugin requirements vs core lock")
    for f in findings:
        click.echo(
            f"  MISMATCH  {f.plugin}: {f.name}{f.specifier}  (core locks {f.core_locked})"
        )
    for w in warnings:
        click.echo(f"  note      {w}")
    click.echo(f"  -> {len(findings)} mismatch(es), {len(warnings)} note(s)")


@click.command()
@click.option(
    "--check",
    "check",
    type=click.Choice(["host", "plugins", "all"]),
    default="all",
    show_default=True,
    help="Which audit(s) to run.",
)
@click.option(
    "--project-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("."),
    show_default=True,
    help="Repository to host-check (reads its pyproject.toml + uv.lock).",
)
@click.option(
    "--pyproject",
    "pyproject_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Override the pyproject.toml path for the host check.",
)
@click.option(
    "--lock",
    "lock_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Override the uv.lock path for the host check.",
)
@click.option(
    "--plugins-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Directory of per-plugin manifest YAMLs for the plugin check.",
)
@click.option(
    "--plugin-pyproject",
    "plugin_pyproject",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="A single plugin's pyproject.toml to audit against the core lock "
    "(per-plugin-repo CI; use instead of --plugins-dir).",
)
@click.option(
    "--against-core",
    "against_core",
    default=None,
    help="cuvis-ai-core lock source for the plugin check: 'installed' (read the "
    "versions pip-installed in this env), a path to a cuvis-ai-core checkout "
    "(reads its uv.lock), or unset (defaults to --project-dir's lock).",
)
@click.option(
    "--strict",
    is_flag=True,
    default=False,
    help="Exit non-zero when any finding is reported.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
    help="Report format.",
)
def main(
    check: str,
    project_dir: Path,
    pyproject_path: Path | None,
    lock_path: Path | None,
    plugins_dir: Path | None,
    plugin_pyproject: Path | None,
    against_core: str | None,
    strict: bool,
    output_format: str,
) -> None:
    """Report dependency-constraint drift; with --strict, fail on any finding."""
    report: dict = {}
    has_findings = False

    if check in ("host", "all"):
        findings, warnings = check_host(project_dir, pyproject_path, lock_path)
        has_findings = has_findings or bool(findings)
        report["host"] = {
            "findings": [asdict(f) for f in findings],
            "warnings": warnings,
        }

    if check in ("plugins", "all"):
        if plugins_dir is None and plugin_pyproject is None:
            if check == "plugins":
                raise click.UsageError(
                    "the plugin check needs --plugins-dir (a manifest catalog) "
                    "or --plugin-pyproject (a single plugin)"
                )
            report["plugins"] = {"skipped": "no --plugins-dir / --plugin-pyproject"}
        else:
            core_lock = resolve_core_lock(against_core, project_dir)
            if plugin_pyproject is not None:
                findings_p, warnings_p = check_plugin_pyproject(
                    plugin_pyproject, core_lock
                )
            else:
                findings_p, warnings_p = check_plugins(plugins_dir, core_lock)
            has_findings = has_findings or bool(findings_p)
            report["plugins"] = {
                "findings": [asdict(f) for f in findings_p],
                "warnings": warnings_p,
            }

    if output_format == "json":
        click.echo(json.dumps(report, indent=2))
    else:
        if "host" in report:
            host = report["host"]
            _print_host(
                [HostFinding(**f) for f in host.get("findings", [])],
                host.get("warnings", []),
            )
        if "plugins" in report and "skipped" not in report["plugins"]:
            plug = report["plugins"]
            _print_plugins(
                [PluginFinding(**f) for f in plug.get("findings", [])],
                plug.get("warnings", []),
            )

    sys.exit(1 if (strict and has_findings) else 0)


if __name__ == "__main__":
    main()
