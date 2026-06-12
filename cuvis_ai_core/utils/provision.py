"""Provision a plugin environment for a pipeline.

The import-only registration path (``NodeRegistry.register_plugins``) expects
every plugin to be installed in the active environment already. This module is
its companion: it reuses the orchestrator's plugin resolver to turn a pipeline
plus its plugins manifests into pip-install specs, then prints them, installs
them into the active venv, writes an env file, or installs them into a Jupyter
kernel. It replaces the convenience the old in-process clone/install gave, with
no clone/install machinery of its own.

Reuse map (everything heavy already exists in the orchestrator):
- :func:`resolve_pipeline_plugins` - pipeline + manifests -> resolved configs.
- :func:`resolve_plugin_sources` - tag->sha (when pinning) + extras stamping.
- :func:`_plugin_source_entry` - per-plugin git-vs-local dispatch (``ref``).
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from loguru import logger

from cuvis_ai_core.orchestrator.cache_key import ResolvedGitPlugin
from cuvis_ai_core.orchestrator.runtime_project import (
    _plugin_source_entry,
    resolve_plugin_sources,
)
from cuvis_ai_core.utils.plugin_resolver import resolve_pipeline_plugins
from cuvis_ai_schemas.pipeline.config import PipelineConfig


def _top_import_module(cfg) -> str | None:
    """Top-level import package of a plugin (from its first provides FQCN)."""
    for entry in cfg.provides:
        return entry.class_name.split(".", 1)[0]
    return None


def _is_satisfied(cfg) -> bool:
    """True when the plugin's package is already importable in this environment."""
    mod = _top_import_module(cfg)
    if not mod:
        return False
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def _spec_for(p, ref: str) -> str:
    """One pip-installable spec for a resolved plugin.

    Git -> ``name[extras] @ git+<url>@<tag-or-sha>``; local -> a ``file://``
    direct reference so the spec is a valid requirement line on its own.
    """
    dep_str, _key, entry = _plugin_source_entry(p, ref=ref)
    if isinstance(p, ResolvedGitPlugin):
        rev = entry.get("tag") or entry.get("rev")
        return f"{dep_str} @ git+{entry['git']}@{rev}"
    return f"{dep_str} @ {Path(entry['path']).resolve().as_uri()}"


def resolve_install_specs(
    pipeline_path: str | Path,
    plugins_dirs: Sequence[str | Path],
    data_module: str | None = None,
    *,
    pin: bool = False,
    include_satisfied: bool = False,
) -> list[str]:
    """Return pip-install specs for the plugins a pipeline needs.

    Git plugins become ``name[extras] @ git+<url>@<tag>`` (or ``@<sha>`` when
    ``pin`` is set); local plugins become ``name[extras] @ file://<path>``.
    Plugins already importable in the active environment are skipped unless
    ``include_satisfied`` is set (so a dev checkout with editable installs
    yields an empty list - nothing to provision).
    """
    pipeline_path = Path(pipeline_path)
    cfg = PipelineConfig.load_from_file(pipeline_path)
    dirs = [Path(d) for d in plugins_dirs]
    resolved_cfgs = resolve_pipeline_plugins(cfg, dirs, data_module)

    satisfied: set[str] = (
        set()
        if include_satisfied
        else {name for name, c in resolved_cfgs.items() if _is_satisfied(c)}
    )
    plugins = resolve_plugin_sources(resolved_cfgs, active_data_module=data_module)
    ref = "sha" if pin else "tag"
    return [_spec_for(p, ref) for p in plugins if p.name not in satisfied]


def format_install_command(specs: Sequence[str], *, magic: bool = False) -> str:
    """Format specs into a single install command line.

    ``magic=False`` -> ``uv pip install '...'`` (terminal / CLI).
    ``magic=True``  -> ``%pip install '...'`` (the Jupyter in-kernel magic).
    """
    if not specs:
        return "# all plugins already provisioned; nothing to install"
    quoted = " ".join(f"'{s}'" for s in specs)
    return f"%pip install {quoted}" if magic else f"uv pip install {quoted}"


def _env_file_text(specs: Sequence[str], pipeline_name: str) -> str:
    """A pyproject-shaped env file consumable via ``uv pip install -r``."""
    import tomli_w

    doc = {
        "project": {
            "name": f"{pipeline_name}-env",
            "version": "0.0.0",
            "requires-python": ">=3.11,<3.12",
            "dependencies": list(specs),
        },
        "tool": {
            "uv": {"required-environments": [f"sys_platform == '{sys.platform}'"]}
        },
    }
    return tomli_w.dumps(doc)


def _provision_notebook(specs: Sequence[str], *, apply: bool) -> str | None:
    """Emit (and optionally run) the in-kernel ``%pip install`` line."""
    line = format_install_command(specs, magic=True)
    if not apply:
        print(line)
        if specs:
            print("# Run the line above, then restart the kernel (Kernel > Restart).")
        return line

    try:
        from IPython import get_ipython
    except ImportError as exc:  # pragma: no cover - IPython always present in a kernel
        raise RuntimeError(
            "provision_environment(notebook=True, apply=True) requires IPython."
        ) from exc
    ip = get_ipython()
    if ip is None:
        raise RuntimeError(
            "provision_environment(notebook=True, apply=True) requires a running "
            "Jupyter/IPython kernel."
        )
    if specs:
        ip.run_line_magic("pip", "install " + " ".join(f"'{s}'" for s in specs))
        print(
            f"Installed {len(specs)} plugin(s) into this kernel.\n"
            "Restart the kernel (Kernel > Restart) and re-run from the top so the "
            "new packages import cleanly."
        )
    else:
        print("All plugins already provisioned; nothing to install.")
    return None


def provision_environment(
    specs: Sequence[str],
    *,
    apply: bool = False,
    env_file: str | Path | None = None,
    sync: bool = False,
    notebook: bool = False,
    pipeline_name: str = "pipeline",
) -> str | None:
    """Print, apply, write, or notebook-install a set of install specs.

    Modes (mutually exclusive aside from ``env_file`` + ``sync``):
    - default: print the ``uv pip install`` command and return it (no side
      effects).
    - ``apply=True``: ``uv pip install`` the specs into the active venv.
    - ``env_file=...``: write an env file (``.toml`` pyproject-shaped, ``.txt``
      flat requirements); ``sync=True`` then ``uv pip install -r`` it.
    - ``notebook=True``: emit the ``%pip install`` line; ``apply=True`` runs it
      in-kernel and prompts a restart.
    """
    if env_file is not None and (apply or notebook):
        raise ValueError(
            "env_file cannot be combined with apply=True or notebook=True."
        )
    if notebook and apply and env_file is not None:  # defensive, covered above
        raise ValueError("notebook and env_file are mutually exclusive.")

    if notebook:
        return _provision_notebook(specs, apply=apply)

    if env_file is not None:
        env_path = Path(env_file)
        if env_path.suffix == ".txt":
            text = ("\n".join(specs) + "\n") if specs else ""
        else:
            text = _env_file_text(specs, pipeline_name)
        env_path.write_text(text, encoding="utf-8")
        logger.info(f"wrote {env_path} ({len(specs)} specs)")
        if sync and specs:
            subprocess.run(["uv", "pip", "install", "-r", str(env_path)], check=True)
        return None

    cmd = format_install_command(specs)
    if apply:
        if specs:
            subprocess.run(["uv", "pip", "install", *specs], check=True)
        else:
            logger.info("All plugins already provisioned; nothing to install.")
        return None
    print(cmd)
    return cmd
