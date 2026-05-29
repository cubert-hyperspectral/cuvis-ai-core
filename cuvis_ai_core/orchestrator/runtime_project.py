"""Generate the per-key runtime ``pyproject.toml``.

Translates the resolved plugin set into a single uv-resolvable
project file. Git plugins go through ``git ls-remote --tags`` so the
user-supplied tag becomes a commit sha at composer time — the cache
key is then immutable even if the upstream tag is force-pushed.
"""

from __future__ import annotations

import subprocess
from typing import Mapping

import tomli_w
from loguru import logger

from cuvis_ai_core.orchestrator.cache_key import (
    CoreSource,
    ResolvedGitPlugin,
    ResolvedLocalPlugin,
    ResolvedPlugin,
    local_plugin_provenance,
)
from cuvis_ai_core.utils.plugin_config import GitPluginConfig, LocalPluginConfig

PluginConfig = GitPluginConfig | LocalPluginConfig


class RuntimeProjectError(RuntimeError):
    """Raised when the runtime project cannot be generated."""


def resolve_git_tag(repo: str, tag: str) -> str:
    """Resolve a git tag to a 40-char commit sha via ``git ls-remote``.

    Rejects branches and moving refs: only tags listed under
    ``refs/tags/`` count. The check is single-network-round-trip and
    runs once per (repo, tag) at composer time.
    """
    try:
        output = subprocess.check_output(
            ["git", "ls-remote", "--tags", repo, f"refs/tags/{tag}"],
            text=True,
            stderr=subprocess.PIPE,
            timeout=60,
        )
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

    # ``git ls-remote --tags`` lists annotated tags twice — the bare
    # tag and a ``^{}`` peeled form pointing at the underlying commit.
    # Prefer the peeled form so we get the commit sha, not the tag
    # object sha.
    for line in lines:
        sha, ref = line.split(maxsplit=1)
        if ref.endswith("^{}"):
            return sha
    sha, _ = lines[0].split(maxsplit=1)
    return sha


def git_source_url(repo: str, sha: str) -> str:
    """Format a uv-compatible ``git+...`` source URL preserving the original transport.

    - HTTPS / HTTP repos stay HTTPS / HTTP.
    - SSH repos in ``git@host:path`` shorthand are rewritten to
      ``ssh://git@host/path`` so uv can parse them.
    """
    if repo.startswith("https://"):
        return f"git+{repo}@{sha}"
    if repo.startswith("http://"):
        return f"git+{repo}@{sha}"
    if repo.startswith("ssh://"):
        return f"git+{repo}@{sha}"
    if repo.startswith("git@"):
        # git@host:path → ssh://git@host/path
        host_and_path = repo[len("git@") :]
        if ":" not in host_and_path:
            raise RuntimeProjectError(
                f"Malformed SSH repo URL '{repo}': expected 'git@host:path'."
            )
        host, _, path = host_and_path.partition(":")
        return f"git+ssh://git@{host}/{path}@{sha}"
    raise RuntimeProjectError(
        f"Unsupported repo URL scheme '{repo}'. "
        "Expected 'git@', 'https://', 'http://', or 'ssh://'."
    )


def resolve_plugin_sources(
    plugin_configs: Mapping[str, PluginConfig],
) -> tuple[ResolvedPlugin, ...]:
    """Resolve git tags to SHAs and stamp local plugins with content provenance.

    Returns plugins sorted by name so the resulting tuple is
    canonical (cache-key inputs must be order-stable).
    """
    resolved: list[ResolvedPlugin] = []
    for name in sorted(plugin_configs):
        cfg = plugin_configs[name]
        if isinstance(cfg, GitPluginConfig):
            sha = resolve_git_tag(cfg.repo, cfg.tag)
            resolved.append(
                ResolvedGitPlugin(name=name, repo=cfg.repo, sha=sha, tag=cfg.tag)
            )
            logger.debug(
                f"Resolved plugin '{name}' tag {cfg.tag} → {sha[:8]} from {cfg.repo}"
            )
        elif isinstance(cfg, LocalPluginConfig):
            from pathlib import Path

            path = Path(cfg.path).resolve()
            pyproject_sha, head, dirty = local_plugin_provenance(path)
            resolved.append(
                ResolvedLocalPlugin(
                    name=name,
                    path=path,
                    pyproject_sha256=pyproject_sha,
                    git_head=head,
                    dirty=dirty,
                )
            )
        else:  # pragma: no cover - exhaustive
            raise RuntimeProjectError(f"Unknown plugin config type: {type(cfg)!r}")
    return tuple(resolved)


def build_runtime_pyproject(
    *,
    core_source: CoreSource,
    plugins: tuple[ResolvedPlugin, ...],
    python_requires: str,
) -> str:
    """Build the canonical runtime ``pyproject.toml`` content.

    The same inputs always produce the same bytes — uv resolves
    against this single file and writes ``uv.lock`` next to it.
    """
    dependencies: list[str] = [_core_dependency(core_source)]
    sources: dict[str, dict] = {}

    if core_source.kind == "git":
        repo, _, sha = core_source.identity.partition("@")
        sources["cuvis-ai-core"] = {"git": repo, "rev": sha}
    elif core_source.kind == "local":
        sources["cuvis-ai-core"] = {"path": core_source.identity, "editable": True}

    for p in plugins:
        if isinstance(p, ResolvedGitPlugin):
            dependencies.append(p.name)
            repo_for_uv = _ssh_to_url(p.repo)
            sources[p.name] = {"git": repo_for_uv, "rev": p.sha}
        else:
            dependencies.append(p.name)
            sources[p.name] = {"path": str(p.path), "editable": True}

    doc: dict = {
        "project": {
            "name": "cuvis-ai-runtime-project",
            "version": "0.0.0",
            "requires-python": python_requires,
            "dependencies": dependencies,
        },
        "tool": {"uv": {"sources": sources}} if sources else {"uv": {}},
    }
    return tomli_w.dumps(doc)


def _core_dependency(core_source: CoreSource) -> str:
    if core_source.kind == "pypi":
        # identity is the full PEP-508 string, e.g. "cuvis-ai-core==0.7.3"
        return core_source.identity
    return "cuvis-ai-core"


def _ssh_to_url(repo: str) -> str:
    """Normalise ``git@host:path`` → ``ssh://git@host/path`` for uv source URLs.

    HTTPS / HTTP / explicit ssh:// URLs are returned unchanged.
    """
    if repo.startswith(("https://", "http://", "ssh://")):
        return repo
    if repo.startswith("git@") and ":" in repo:
        host_and_path = repo[len("git@") :]
        host, _, path = host_and_path.partition(":")
        return f"ssh://git@{host}/{path}"
    raise RuntimeProjectError(f"Unsupported repo URL scheme '{repo}'.")
