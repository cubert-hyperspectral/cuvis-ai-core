"""Structured cache key for composed child venvs.

A composed venv is reused only when six things match:

1. Python version (exact ``X.Y.Z``)
2. Platform tag (``win_amd64``, ``linux_x86_64``, etc.)
3. Core source identity (PyPI pin, git+sha, or local path)
4. Plugin sources — for git plugins the tag has already been resolved
   to a commit sha; for local plugins a content-provenance triple
   (``pyproject.toml`` hash, ``git HEAD``, ``dirty`` flag) is rolled in.
5. Spec hash — sha256 of the canonical runtime ``pyproject.toml``
   the composer is about to write. Captures any constraint overlay or
   ordering change the structured fields above don't already cover.
6. Schema version — bump this constant when the composer's own logic
   changes so a server upgrade naturally invalidates old envs.

The on-disk directory name is the short hash digest alone, so it stays
well within filesystem name limits no matter how many plugins a
pipeline pulls in. A human-readable ``env_desc.md`` written inside each
env records what the env was composed for. Any local plugin marked
``dirty`` rolls a random value into the key so it hashes uniquely per
run (a fresh env every time); clean keys hash identically across runs.
"""

from __future__ import annotations

import hashlib
import json
import platform
import secrets
import sys
import sysconfig
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

# Bump when the composer's logic changes shape (key fields, hashing
# convention, atomic-publish protocol, etc.) so existing cache entries
# are naturally invalidated.
COMPOSER_SCHEMA_VERSION = 3

# The directory name is the digest alone, so it is the sole on-disk
# identity. 12 hex chars (48 bits) sizes it to make collisions negligible.
_DIR_HASH_LEN = 12

# Cache-protocol kind tags written to key.json and the human manifest.
_KIND_GIT = "git"
_KIND_LOCAL = "local"


@dataclass(frozen=True)
class CoreSource:
    """Identity of the ``cuvis-ai-core`` package the composer is pinning into the runtime project."""

    kind: Literal["pypi", "git", "local"]
    identity: str  # "cuvis-ai-core==0.7.3" / "<repo>@<sha>" / "<absolute-path>"


@dataclass(frozen=True)
class ResolvedGitPlugin:
    """A git-sourced plugin whose tag has already been resolved to a commit sha.

    ``name`` is the plugin manifest key (a logical identifier).
    ``package_name`` is the value that ends up in the runtime
    pyproject's ``dependencies`` list and ``tool.uv.sources`` key — uv
    matches it against the cloned repo's ``[project] name`` during
    locking. Manifest authors who use a short logical key like
    ``sam3`` for a plugin whose actual package is ``cuvis-ai-sam3``
    must set ``package_name`` explicitly in the YAML; when omitted it
    falls back to the manifest key.
    """

    name: str
    repo: str
    sha: str  # 40-char hex, resolved via ``git ls-remote --tags``
    tag: str  # the original user-facing tag, kept for the human prefix
    package_name: str = ""  # populated by resolve_plugin_sources
    # pip extras to install for this plugin (e.g. the activated data module's
    # extras). Lands in the runtime pyproject's dependency string, so spec_hash
    # covers it; intentionally omitted from _plugin_to_dict to avoid double-count.
    extras: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolvedLocalPlugin:
    """A local-path plugin with its content-provenance triple.

    ``name`` is the manifest key (logical). ``package_name`` is the
    actual ``[project] name`` read from the local pyproject.toml — uv
    requires the dep name in the runtime project to match this, not
    the manifest key. Without that read, a manifest entry like
    ``cuvis_ai_builtin -> /path/to/cuvis-ai/`` (whose pyproject is
    named ``cuvis-ai``) trips uv with "Package metadata name does
    not match given name".
    """

    name: str
    path: Path
    package_name: str
    pyproject_sha256: str
    git_head: str | None
    dirty: bool
    # pip extras to install for this plugin (the activated data module's extras).
    extras: tuple[str, ...] = ()


ResolvedPlugin = ResolvedGitPlugin | ResolvedLocalPlugin


@dataclass(frozen=True)
class CacheKey:
    """Structured cache key. ``directory_name()`` is what lands on disk; ``digest`` is the short hash suffix."""

    python_version: str
    platform_tag: str
    core_source: CoreSource
    plugins: tuple[ResolvedPlugin, ...]
    spec_hash: str
    schema_version: int = COMPOSER_SCHEMA_VERSION
    dirty_suffix: str | None = field(default=None)

    @property
    def digest(self) -> str:
        """Short hex hash suffix used in the directory name."""
        h = hashlib.sha256(self._canonical_json().encode("utf-8")).hexdigest()
        return h[:_DIR_HASH_LEN]

    def directory_name(self) -> str:
        """On-disk cache directory name: the digest alone.

        The digest already encodes every key component — python,
        platform, core, plugins, spec hash, schema version, and the
        random ``dirty_suffix`` for dirty local plugins — so it is a
        complete, collision-safe identity. A fixed-width hash name never
        overflows the filesystem name limit, however many plugins a
        pipeline declares. Human-readable detail lives in the
        ``env_desc.md`` file the composer writes inside the env.
        """
        return self.digest

    def human_manifest(self) -> str:
        """Markdown companion to ``key.json``, written inside the env.

        The directory name is an opaque hash; this file tells a person
        which libraries the env was composed for. It records each
        plugin's manifest name *and* its ``package_name`` (the name uv
        actually installs) — the latter is intentionally absent from
        ``key.json`` (see :func:`_plugin_to_dict`), so this is its only
        human-facing record.
        """
        lines = [
            "# Cuvis.AI composed environment",
            "",
            "This directory is a child venv composed by the cuvis-ai-core "
            "orchestrator. Its name is a content hash; this file records what "
            "the env was built for. The machine-readable form is `key.json`.",
            "",
            f"- Digest: `{self.digest}`",
            f"- Python: {self.python_version}",
            f"- Platform: {self.platform_tag}",
            f"- Core: `core@{_short_core_identity(self.core_source)}` "
            f"({self.core_source.kind}: {self.core_source.identity})",
            f"- Schema version: {self.schema_version}",
            f"- Spec hash: `{self.spec_hash}`",
        ]
        if self.dirty_suffix is not None:
            lines.append(
                "- Dev mode: dirty local plugin present "
                f"(unique per run, suffix `{self.dirty_suffix}`)"
            )
        lines += ["", "## Plugins", ""]
        if not self.plugins:
            lines.append("_None._")
        else:
            lines.append("| name | package | source | ref |")
            lines.append("| --- | --- | --- | --- |")
            for p in self.plugins:
                if isinstance(p, ResolvedGitPlugin):
                    pkg = p.package_name or p.name
                    ref = f"{p.tag} -> {p.sha[:8]}"
                    lines.append(f"| {p.name} | {pkg} | {p.repo} | {ref} |")
                else:
                    ref = "local (dirty)" if p.dirty else "local (clean)"
                    lines.append(f"| {p.name} | {p.package_name} | {p.path} | {ref} |")
        return "\n".join(lines) + "\n"

    def serialise(self) -> dict:
        """Full structured key for ``key.json`` forensics."""
        return {
            "python_version": self.python_version,
            "platform_tag": self.platform_tag,
            "core_source": asdict(self.core_source),
            "plugins": [_plugin_to_dict(p) for p in self.plugins],
            "spec_hash": self.spec_hash,
            "schema_version": self.schema_version,
            "dirty_suffix": self.dirty_suffix,
        }

    def _canonical_json(self) -> str:
        return json.dumps(self.serialise(), sort_keys=True, separators=(",", ":"))


def _plugin_to_dict(p: ResolvedPlugin) -> dict:
    # package_name is intentionally absent from the digest: it already
    # lands in the runtime pyproject.toml bytes that spec_hash covers, so
    # listing it here would double-count and needlessly split the cache.
    if isinstance(p, ResolvedGitPlugin):
        return {
            "kind": _KIND_GIT,
            "name": p.name,
            "repo": p.repo,
            "sha": p.sha,
            "tag": p.tag,
        }
    return {
        "kind": _KIND_LOCAL,
        "name": p.name,
        "path": str(p.path),
        "pyproject_sha256": p.pyproject_sha256,
        "git_head": p.git_head,
        "dirty": p.dirty,
    }


def _short_core_identity(core_source: CoreSource) -> str:
    """Short human label for the core source: version (pypi), short sha (git), or 'local'."""
    if core_source.kind == "pypi":
        # "cuvis-ai-core==0.7.3" -> "0.7.3"
        _, _, version = core_source.identity.partition("==")
        return version or core_source.identity
    if core_source.kind == "git":
        # "<repo>@<sha>" -> short sha
        _, _, sha = core_source.identity.rpartition("@")
        return sha[:8] if sha else core_source.identity
    return "local"


def current_python_version() -> str:
    """Exact interpreter version as ``X.Y.Z``."""
    return ".".join(str(v) for v in sys.version_info[:3])


def current_platform_tag() -> str:
    """Wheel-style platform tag for the running interpreter.

    Uses ``sysconfig.get_platform()`` which produces ``win-amd64``,
    ``linux-x86_64``, ``macosx-14.0-arm64``, etc. The tag is kept
    verbatim as a digest input; it no longer appears in the directory
    name, which is the hash alone.
    """
    tag = sysconfig.get_platform()
    machine = platform.machine().lower()
    if machine and machine not in tag.lower():
        tag = f"{tag}-{machine}"
    return tag


def compute_cache_key(
    *,
    core_source: CoreSource,
    plugins: tuple[ResolvedPlugin, ...],
    spec_hash: str,
    python_version: str | None = None,
    platform_tag: str | None = None,
    schema_version: int = COMPOSER_SCHEMA_VERSION,
) -> CacheKey:
    """Build a CacheKey from already-resolved sources.

    ``plugins`` must be ordered deterministically by the caller
    (composer sorts by name) so two callers with the same set produce
    the same key. ``spec_hash`` is the sha256 of the canonical
    ``pyproject.toml`` content the composer will write.

    If any local plugin is dirty, attaches a random ``dirty_suffix``
    so the cache directory is unique per call (dev-mode behaviour).
    """
    py = python_version or current_python_version()
    plat = platform_tag or current_platform_tag()
    dirty_suffix: str | None = None
    if any(isinstance(p, ResolvedLocalPlugin) and p.dirty for p in plugins):
        dirty_suffix = secrets.token_hex(3)
    return CacheKey(
        python_version=py,
        platform_tag=plat,
        core_source=core_source,
        plugins=tuple(plugins),
        spec_hash=spec_hash,
        schema_version=schema_version,
        dirty_suffix=dirty_suffix,
    )


def spec_hash_of(pyproject_toml_content: str) -> str:
    """Canonical content hash for the runtime ``pyproject.toml``."""
    return hashlib.sha256(pyproject_toml_content.encode("utf-8")).hexdigest()


def _git(args: list[str], cwd: Path) -> str | None:
    """Run a read-only ``git`` command, returning stripped stdout or None.

    None means the command failed or git is unavailable; callers treat
    that as "cannot prove provenance". ``subprocess`` is imported lazily
    to keep a process-spawning dependency off this frequently-imported
    module's import path.
    """
    import subprocess

    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def local_plugin_provenance(plugin_dir: Path) -> tuple[str, str | None, bool]:
    """Compute (pyproject_sha256, git_head, dirty) for a local plugin checkout.

    Falls back gracefully when there is no ``pyproject.toml`` (empty
    string hash) or no git repo (None head, True dirty — outside a
    repo we cannot prove the source is stable).
    """
    pyproject = plugin_dir / "pyproject.toml"
    if pyproject.is_file():
        pyproject_sha = hashlib.sha256(pyproject.read_bytes()).hexdigest()
    else:
        pyproject_sha = hashlib.sha256(b"").hexdigest()

    head = _git(["rev-parse", "HEAD"], plugin_dir)
    if head is None:
        return pyproject_sha, None, True

    status = _git(["status", "--porcelain"], plugin_dir)
    # If the status probe itself fails, assume dirty (cannot prove clean).
    dirty = status is None or bool(status.strip())
    return pyproject_sha, head, dirty
