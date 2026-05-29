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

The on-disk directory name keeps a human-readable prefix for ops
inspection and a short hash suffix that encodes the full key. Any
local plugin marked ``dirty`` forces a fresh per-run env via a random
suffix; clean keys hash identically across runs.
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
COMPOSER_SCHEMA_VERSION = 1

_DIR_HASH_LEN = 6
_MAX_NAME_SEGMENT = 32


def _slug(text: str, limit: int = _MAX_NAME_SEGMENT) -> str:
    """Filesystem-safe truncation of an identifier for directory prefixes."""
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in text)
    return safe[:limit]


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
        prefix_parts: list[str] = [
            f"py{self.python_version.rsplit('.', 1)[0]}",
            _slug(self.platform_tag),
            f"core@{_slug(self.core_source.identity)}",
        ]
        for p in self.plugins:
            if isinstance(p, ResolvedGitPlugin):
                prefix_parts.append(f"{_slug(p.name)}@{_slug(p.tag)}")
            else:
                marker = "dirty" if p.dirty else "local"
                prefix_parts.append(f"{_slug(p.name)}@{marker}")
        prefix = "__".join(prefix_parts)
        suffix = self.digest
        if self.dirty_suffix is not None:
            return f"{prefix}__{suffix}__{self.dirty_suffix}"
        return f"{prefix}__{suffix}"

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
    if isinstance(p, ResolvedGitPlugin):
        return {"kind": "git", "name": p.name, "repo": p.repo, "sha": p.sha, "tag": p.tag}
    return {
        "kind": "local",
        "name": p.name,
        "path": str(p.path),
        "pyproject_sha256": p.pyproject_sha256,
        "git_head": p.git_head,
        "dirty": p.dirty,
    }


def current_python_version() -> str:
    """Exact interpreter version as ``X.Y.Z``."""
    return ".".join(str(v) for v in sys.version_info[:3])


def current_platform_tag() -> str:
    """Wheel-style platform tag for the running interpreter.

    Uses ``sysconfig.get_platform()`` which produces ``win-amd64``,
    ``linux-x86_64``, ``macosx-14.0-arm64``, etc. Hyphens are kept in
    the key but the directory name passes through ``_slug`` which
    swaps any non-``[A-Za-z0-9._-]`` to underscore.
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


def local_plugin_provenance(plugin_dir: Path) -> tuple[str, str | None, bool]:
    """Compute (pyproject_sha256, git_head, dirty) for a local plugin checkout.

    Falls back gracefully when there is no ``pyproject.toml`` (empty
    string hash) or no git repo (None head, True dirty — outside a
    repo we cannot prove the source is stable).
    """
    import subprocess

    pyproject = plugin_dir / "pyproject.toml"
    if pyproject.is_file():
        pyproject_sha = hashlib.sha256(pyproject.read_bytes()).hexdigest()
    else:
        pyproject_sha = hashlib.sha256(b"").hexdigest()

    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=plugin_dir,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        head = None

    if head is None:
        return pyproject_sha, None, True

    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=plugin_dir,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        dirty = bool(status.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        dirty = True

    return pyproject_sha, head, dirty
