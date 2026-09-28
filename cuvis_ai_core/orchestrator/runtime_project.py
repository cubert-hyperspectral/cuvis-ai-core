"""Generate the per-key runtime ``pyproject.toml``.

Translates the resolved plugin set into a single uv-resolvable
project file. Git plugins go through ``git ls-remote --tags`` so the
user-supplied tag becomes a commit sha at composer time — the cache
key is then immutable even if the upstream tag is force-pushed.
Manifests that install one package fold into one requirement
(:func:`merge_by_package`), and the extras a manifest requests are
checked against the lock before anything is installed
(:func:`check_locked_extras`).
"""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from collections.abc import Mapping
from dataclasses import replace as dataclass_replace
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import tomli_w
from loguru import logger
from packaging.utils import canonicalize_name

from cuvis_ai_core.orchestrator.cache_key import (
    CoreSource,
    ResolvedGitPlugin,
    ResolvedLocalPlugin,
    ResolvedPlugin,
    local_plugin_provenance,
)
from cuvis_ai_schemas.plugin import (
    GitPluginSource,
    LocalPluginSource,
    PluginManifest,
)

# Identity-bearing names for the generated runtime project. The source
# key and the dependency string must use the SAME core name or uv
# resolution silently breaks, so it lives in one constant.
CORE_PACKAGE_NAME = "cuvis-ai-core"
RUNTIME_PROJECT_NAME = "cuvis-ai-runtime-project"
RUNTIME_PROJECT_VERSION = "0.0.0"
# git ls-remote lists an annotated tag's commit under the peeled ref with this
# suffix, and only when that ref is asked for by name.
_PEELED_TAG_SUFFIX = "^{}"

_TORCH_PACKAGES = ("torch", "torchvision")
# A PEP 440 local segment that names a real PyTorch wheel index: 2.11.0+cu128 -> cu128.
# Allowlisted rather than interpolated blindly, because it ends up in a URL.
_PYTORCH_INDEX_TAG = re.compile(r"^(cpu|cu\d+|rocm[\d.]+|xpu)$")
_PYTORCH_INDEX_URL = "https://download.pytorch.org/whl/{tag}"


@lru_cache(maxsize=1)
def host_torch_pins() -> tuple[Mapping[str, str], str | None]:
    """Installed torch versions on the composing host, and the wheel-index tag they came from.

    uv honours ``[tool.uv.sources]`` index pins only for DIRECT dependencies. A child
    that picks torch up transitively therefore resolves it from PyPI, whose Windows
    wheels are CPU-only, which is how a CUDA host ends up running a CPU build.
    Declaring the parent's exact versions is what makes the index entry apply at all.

    Mirroring the composing interpreter rather than hardcoding a CUDA version keeps a
    CPU-only or ROCm host correct too: whatever torch the parent env was provisioned
    with is by definition the one this machine is meant to run.
    """
    installed = {}
    for name in _TORCH_PACKAGES:
        try:
            installed[name] = version(name)
        except PackageNotFoundError:
            continue
    if not installed:
        return {}, None

    # A split flavour (torch+cu128 next to torchvision+cpu) is not a coherent host
    # setup; pin the versions but name no index rather than guess which one wins.
    tags = {v.partition("+")[2] for v in installed.values()}
    tag = next(iter(tags)) if len(tags) == 1 else ""
    if _PYTORCH_INDEX_TAG.match(tag):
        return installed, tag
    if any(tags):
        # Without an index the pinned local versions resolve nowhere on PyPI, so
        # the child's lock fails with a bare no-candidates error; name the cause
        # here where the decision is made.
        logger.warning(
            f"Host torch builds {installed} carry local version tags that map to "
            f"no single PyTorch wheel index; pinning versions without an index. "
            f"Child resolution will fail unless PyPI serves these exact versions."
        )
    return installed, None


class RuntimeProjectError(RuntimeError):
    """Raised when the runtime project cannot be generated."""


def resolve_git_tag(repo: str, tag: str) -> str:
    """Resolve a git tag to a 40-char commit sha via ``git ls-remote``.

    Rejects branches and moving refs: only tags listed under
    ``refs/tags/`` count. The check is single-network-round-trip and
    runs once per (repo, tag) at composer time.

    The peeled ref is requested next to the tag: an exact ref pattern does
    not match ``refs/tags/<tag>^{}``, so the tag alone yields an annotated
    tag's object sha, which uv rejects as the ``rev`` of a git source.
    """
    tag_ref = f"refs/tags/{tag}"
    try:
        output = subprocess.check_output(
            ["git", "ls-remote", "--tags", repo, tag_ref, tag_ref + _PEELED_TAG_SUFFIX],
            text=True,
            stderr=subprocess.PIPE,
            timeout=60,
        )
    except FileNotFoundError as exc:
        raise RuntimeProjectError(
            "'git' was not found on PATH — it is required to resolve tag-pinned "
            f"plugin sources (git ls-remote {repo}). Install git or ensure it is "
            "on the server process PATH."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeProjectError(
            f"git ls-remote failed for {repo}: {exc.stderr.strip() or exc}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeProjectError(f"git ls-remote timed out for {repo}") from exc

    lines = [line for line in output.splitlines() if line.strip()]
    if not lines:
        raise RuntimeProjectError(
            f"Tag '{tag}' not found in {repo}. Branches and moving refs "
            "are not accepted — pin to a tag (e.g. v0.1.0)."
        )
    return _sha_from_ls_remote(lines)


def _sha_from_ls_remote(lines: list[str]) -> str:
    """Pick the commit sha from ``git ls-remote --tags`` output lines.

    Annotated tags appear twice — the bare tag and a ``^{}`` peeled form
    pointing at the underlying commit (:func:`resolve_git_tag` asks for
    both). Prefer the peeled form so we get the commit sha, not the
    tag-object sha.
    """
    for line in lines:
        sha, ref = line.split(maxsplit=1)
        if ref.endswith(_PEELED_TAG_SUFFIX):
            return sha
    return lines[0].split(maxsplit=1)[0]


def _plugin_extras(
    cfg: PluginManifest, active_data_module: str | None
) -> tuple[str, ...]:
    """pip extras to install for this plugin: its own plus the activated data module's.

    A manifest's ``extras`` apply whenever the manifest is in the plugin set (an
    optional backend such as TensorRT). A data-module entry's extras apply only
    when that module is the one a run selected, so a tiff_paired run never pulls
    a cu3s module's ``cuvis`` extra. Sorted and de-duplicated, so the dependency
    string, and with it the cache key, is canonical.
    """
    extras = set(cfg.extras)
    if active_data_module:
        for entry in cfg.capabilities:
            if (
                getattr(entry, "kind", "node") == "data_module"
                and getattr(entry, "data_module_name", "") == active_data_module
            ):
                extras.update(getattr(entry, "extras", []) or [])
                break
    return tuple(sorted(extras))


def resolve_plugin_sources(
    plugin_configs: Mapping[str, PluginManifest],
    active_data_module: str | None = None,
) -> tuple[ResolvedPlugin, ...]:
    """Resolve git tags to SHAs and stamp local plugins with content provenance.

    Returns plugins sorted by name so the resulting tuple is
    canonical (cache-key inputs must be order-stable). Each plugin carries the
    extras it requests: its manifest's own plus, when ``active_data_module``
    names one of its data modules, that module's. One entry per manifest;
    :func:`merge_by_package` folds manifests that install one package.
    """
    resolved: list[ResolvedPlugin] = []
    for name in sorted(plugin_configs):
        cfg = plugin_configs[name]
        extras = _plugin_extras(cfg, active_data_module)
        if isinstance(cfg, GitPluginSource):
            sha = resolve_git_tag(cfg.repo, cfg.tag)
            # Prefer the explicit override; otherwise trust the
            # manifest key matches the package name (the case for
            # convention-aligned plugins).
            package_name = cfg.package_name or name
            resolved.append(
                ResolvedGitPlugin(
                    name=name,
                    repo=cfg.repo,
                    sha=sha,
                    tag=cfg.tag,
                    package_name=package_name,
                    extras=extras,
                )
            )
            logger.debug(
                f"Resolved plugin '{name}' tag {cfg.tag} → {sha[:8]} from {cfg.repo}"
            )
        elif isinstance(cfg, LocalPluginSource):
            path = Path(cfg.path).resolve()
            pyproject_sha, head, dirty = local_plugin_provenance(path)
            # Local plugins prefer the explicit override; otherwise
            # read [project] name directly from the plugin's pyproject.
            package_name = cfg.package_name or _read_local_package_name(
                path, manifest_key=name
            )
            resolved.append(
                ResolvedLocalPlugin(
                    name=name,
                    path=path,
                    package_name=package_name,
                    pyproject_sha256=pyproject_sha,
                    git_head=head,
                    dirty=dirty,
                    extras=extras,
                )
            )
        else:  # pragma: no cover - exhaustive
            raise RuntimeProjectError(f"Unknown plugin config type: {type(cfg)!r}")
    return tuple(resolved)


def merge_by_package(plugins: tuple[ResolvedPlugin, ...]) -> tuple[ResolvedPlugin, ...]:
    """Fold manifests that install one package into one resolved plugin.

    Two manifests may name the same package (``rfdetr`` and ``rfdetr_seg_trt``,
    both ``package_name: cuvis-ai-rfdetr``) so a pipeline opts into an optional
    backend by listing the second one. uv takes one requirement per package, so
    they merge into the first manifest in name order with the union of their
    extras. The package must come from one source: the same commit of the same
    repo URL, or the same checkout; anything else is a conflict this raises
    instead of letting the last manifest's source win silently. Package names
    compare canonically (PEP 503), so ``cuvis_ai_rfdetr`` and ``cuvis-ai-rfdetr``
    are one package.
    """
    groups: dict[str, list[ResolvedPlugin]] = {}
    for p in plugins:
        groups.setdefault(canonicalize_name(p.package_name or p.name), []).append(p)
    merged: list[ResolvedPlugin] = []
    for members in groups.values():
        first = members[0]
        for other in members[1:]:
            if _source_identity(other) != _source_identity(first):
                raise RuntimeProjectError(
                    f"Manifests '{first.name}' and '{other.name}' install package "
                    f"'{first.package_name or first.name}' from different sources "
                    f"({_describe_source(first)} vs {_describe_source(other)}). Give "
                    "both manifests the same repo and tag, or the same path."
                )
        extras = tuple(sorted({extra for m in members for extra in m.extras}))
        merged.append(dataclass_replace(first, extras=extras))
    return tuple(merged)


def _source_identity(p: ResolvedPlugin) -> tuple[str, str]:
    """What makes two manifests the same source: the commit of a repo URL, or a path."""
    if isinstance(p, ResolvedGitPlugin):
        return (_ssh_to_url(p.repo), p.sha)
    return ("local", str(p.path))


def _describe_source(p: ResolvedPlugin) -> str:
    """A plugin's source as a manifest author wrote it, for error messages."""
    if isinstance(p, ResolvedGitPlugin):
        return f"{p.repo}@{p.tag}"
    return str(p.path)


def check_locked_extras(lock_path: Path, plugins: tuple[ResolvedPlugin, ...]) -> None:
    """Refuse extras that the locked packages do not declare.

    uv only warns when a requirement names an extra the package lacks, and the
    composed env then silently lacks it: the pipeline fails at the first frame
    with an import error, minutes later. The lock lists, per package, the
    requested extras that exist, so a requested extra missing there is unknown
    to the package (an extra the manifest did not request is not listed either,
    which is why the message names what the lock resolved, not what the package
    declares). Runs after ``uv lock`` and before ``uv sync``, and only when a
    plugin requests extras. A package absent from the lock cannot be checked
    and is logged, not rejected.
    """
    requested = [p for p in plugins if p.extras]
    if not requested:
        return
    if not lock_path.is_file():
        raise RuntimeProjectError(
            f"uv lock produced no {lock_path.name} in {lock_path.parent}; the requested "
            "extras cannot be verified."
        )
    declared: dict[str, set[str]] = {}
    for entry in tomllib.loads(lock_path.read_text(encoding="utf-8")).get(
        "package", []
    ):
        declared.setdefault(canonicalize_name(entry["name"]), set()).update(
            entry.get("optional-dependencies", {})
        )
    for p in requested:
        package = p.package_name or p.name
        known = declared.get(canonicalize_name(package))
        if known is None:
            logger.warning(
                f"Package '{package}' is not in {lock_path.name}; the extras {p.extras} "
                f"of plugin '{p.name}' were not verified."
            )
            continue
        canonical_known = {canonicalize_name(extra) for extra in known}
        for extra in p.extras:
            if canonicalize_name(extra) not in canonical_known:
                resolved = (
                    f"the lock resolved its extras {', '.join(sorted(known))}"
                    if known
                    else "the lock resolved none of its extras"
                )
                raise RuntimeProjectError(
                    f"Plugin '{p.name}' requests extra '{extra}' that package '{package}' "
                    f"({_describe_source(p)}) does not declare ({resolved}). Fix the "
                    "manifest's extras: or add the extra to the plugin's "
                    "[project.optional-dependencies]."
                )


def build_runtime_pyproject(
    *,
    core_source: CoreSource,
    plugins: tuple[ResolvedPlugin, ...],
    python_requires: str,
) -> str:
    """Build the canonical runtime ``pyproject.toml`` content.

    The same inputs on the same host always produce the same bytes — uv
    resolves against this single file and writes ``uv.lock`` next to it.
    The output is host-aware (``required-environments`` names the platform), so
    a Windows cache entry is distinct from a Linux one, which is correct:
    a venv built for one platform can't be reused on the other. ``plugins``
    holds one entry per package (:func:`merge_by_package`): a second entry
    for one package would silently overwrite its source, so it is refused.
    """
    # Core: the pinned PEP 508 spec for PyPI (e.g. "cuvis-ai-core==0.7.3"),
    # else the bare name.
    dependencies: list[str] = [
        core_source.identity if core_source.kind == "pypi" else CORE_PACKAGE_NAME
    ]
    sources: dict[str, dict] = {}

    core_entry = _core_source_entry(core_source)
    if core_entry is not None:
        sources[CORE_PACKAGE_NAME] = core_entry

    for p in plugins:
        dependency_string, source_key, source_entry = _plugin_source_entry(p)
        if source_key in sources:
            raise RuntimeProjectError(
                f"Two plugins install '{source_key}'; fold them with merge_by_package first."
            )
        dependencies.append(dependency_string)
        sources[source_key] = source_entry

    torch_versions, torch_index_tag = host_torch_pins()
    dependencies.extend(f"{name}=={v}" for name, v in torch_versions.items())

    uv_table: dict = {}
    if torch_index_tag:
        index_name = f"pytorch-{torch_index_tag}"
        # explicit: the PyTorch index is a partial mirror of PyPI, so leaving it open
        # would let unrelated dependencies shadow to whatever copy it happens to carry.
        # Only packages that name it may resolve there.
        uv_table["index"] = [
            {
                "name": index_name,
                "url": _PYTORCH_INDEX_URL.format(tag=torch_index_tag),
                "explicit": True,
            }
        ]
        sources.update({name: {"index": index_name} for name in torch_versions})
    if sources:
        uv_table["sources"] = sources
    # The composed venv runs on the host that builds it, so the resolution only
    # has to be installable here. Declaring the host platform makes uv pick a
    # version that ships a wheel for it instead of failing when a dependency's
    # newest release skipped this platform (cuvis-il ships only manylinux
    # wheels for 3.5.3.x: Windows backtracks to 3.5.0, Linux stays current).
    uv_table["required-environments"] = [f"sys_platform == '{sys.platform}'"]

    doc: dict = {
        "project": {
            "name": RUNTIME_PROJECT_NAME,
            "version": RUNTIME_PROJECT_VERSION,
            "requires-python": python_requires,
            "dependencies": dependencies,
        },
        "tool": {"uv": uv_table},
    }
    return tomli_w.dumps(doc)


def _core_source_entry(core_source: CoreSource) -> dict | None:
    """uv ``tool.uv.sources`` entry for core, or None for a plain PyPI pin."""
    if core_source.kind == "git":
        # URLs may themselves contain ``@`` (for example ``ssh://git@...``),
        # while the resolved revision is always the final component.
        repo, _, sha = core_source.identity.rpartition("@")
        return {"git": repo, "rev": sha}
    if core_source.kind == "local":
        return {"path": core_source.identity, "editable": True}
    return None


def _plugin_source_entry(p: ResolvedPlugin, ref: str = "sha") -> tuple[str, str, dict]:
    """Return (dependency_string, source_key, uv source entry) for one plugin.

    ``source_key`` is the bare ``package_name`` (the ``[project].name`` from the
    plugin's pyproject) and keys ``[tool.uv.sources]`` — uv refuses to install a
    dep whose declared name doesn't match the package metadata. The
    ``dependency_string`` is that same name plus any pip ``extras`` it activates
    (``pkg[extra1,extra2]``), and goes into ``[project].dependencies``; uv composes
    extras with a ``tool.uv.sources`` git/path override keyed by the bare name.

    ``ref`` selects how a git plugin is pinned: ``"sha"`` (default) emits the
    resolved commit ``rev`` for a cache-stable, reproducible env (the
    orchestrator composer always uses this); ``"tag"`` emits the manifest tag,
    which the ``provision`` helper uses for a human-readable env file unless the
    caller asks to pin.
    """
    source_key = p.package_name or p.name
    dependency_string = (
        f"{source_key}[{','.join(p.extras)}]" if p.extras else source_key
    )
    if isinstance(p, ResolvedGitPlugin):
        url = _ssh_to_url(p.repo)
        if ref == "tag":
            return dependency_string, source_key, {"git": url, "tag": p.tag}
        return dependency_string, source_key, {"git": url, "rev": p.sha}
    return dependency_string, source_key, {"path": str(p.path), "editable": True}


def _read_local_package_name(path: Path, *, manifest_key: str) -> str:
    """Read the PyPI-style ``[project] name`` from a local plugin's pyproject.toml.

    The manifest key (the YAML map key used in
    ``configs/plugins/<name>.yaml``) is a *logical* identifier for the
    plugin set, not a Python package name. uv refuses to install a
    dependency unless the dep name matches the actual package
    metadata, so the composer must pin against the real name.
    """
    pyproject = path / "pyproject.toml"
    if not pyproject.is_file():
        raise RuntimeProjectError(
            f"Local plugin '{manifest_key}' at {path} has no pyproject.toml; "
            "the composer needs it to learn the package name."
        )
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise RuntimeProjectError(
            f"Local plugin '{manifest_key}' at {pyproject} has malformed TOML: {exc}"
        ) from exc
    name = (data.get("project") or {}).get("name")
    if not isinstance(name, str) or not name:
        raise RuntimeProjectError(
            f"Local plugin '{manifest_key}' at {pyproject} declares no "
            "'[project] name'. Add one matching the importable package."
        )
    return name


def _ssh_to_url(repo: str) -> str:
    """Normalise a git repo URL to a uv-parseable form (no ``git+``, no rev).

    HTTPS / HTTP / explicit ``ssh://`` URLs are returned unchanged;
    ``git@host:path`` shorthand is rewritten to ``ssh://git@host/path``.
    The single source of truth for the accepted URL-scheme set.
    """
    if repo.startswith(("https://", "http://", "ssh://")):
        return repo
    if repo.startswith("git@"):
        host_and_path = repo[len("git@") :]
        if ":" not in host_and_path:
            raise RuntimeProjectError(
                f"Malformed SSH repo URL '{repo}': expected 'git@host:path'."
            )
        host, _, path = host_and_path.partition(":")
        return f"ssh://git@{host}/{path}"
    raise RuntimeProjectError(
        f"Unsupported repo URL scheme '{repo}'. "
        "Expected 'git@', 'https://', 'http://', or 'ssh://'."
    )
