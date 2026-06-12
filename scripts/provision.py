"""CLI: provision the plugins a pipeline needs into the active environment.

Reads a pipeline's ``plugins:`` list plus ``--data-module``, resolves them
through the same resolver the orchestrator uses, and prints (default),
installs (``--apply``), or writes an env file (``--requirements <file>``) of
the pip-install specs. Git plugins are pinned to their manifest tag unless
``--pin`` resolves them to a commit sha.

This is the import-only world's setup step: provision once, then
``NodeRegistry.register_plugins`` imports the now-installed plugins.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cuvis_ai_core.utils.provision import provision_environment, resolve_install_specs


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="provision",
        description="Provision the plugins a pipeline needs (import-only world).",
    )
    parser.add_argument(
        "--pipeline-path",
        required=True,
        help="Path to the pipeline YAML whose plugins should be provisioned.",
    )
    parser.add_argument(
        "--plugins-dir",
        action="append",
        default=[],
        metavar="DIR",
        help="Plugins directory holding per-plugin manifests (repeatable).",
    )
    parser.add_argument(
        "--data-module",
        default=None,
        help="DataModule name (e.g. cu3s) to also provision its plugin + extras.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Install the resolved specs into the active environment with uv.",
    )
    parser.add_argument(
        "--requirements",
        default=None,
        metavar="FILE",
        help="Write the specs to FILE (.toml pyproject-shaped, .txt flat). "
        "With --apply, also installs from it.",
    )
    parser.add_argument(
        "--pin",
        action="store_true",
        help="Pin git plugins to a resolved commit sha instead of the manifest tag.",
    )
    parser.add_argument(
        "--include-satisfied",
        action="store_true",
        help="Include plugins already importable in the active environment.",
    )
    args = parser.parse_args()

    specs = resolve_install_specs(
        args.pipeline_path,
        args.plugins_dir,
        data_module=args.data_module,
        pin=args.pin,
        include_satisfied=args.include_satisfied,
    )
    pipeline_name = Path(args.pipeline_path).stem

    if args.requirements:
        provision_environment(
            specs,
            env_file=args.requirements,
            sync=args.apply,
            pipeline_name=pipeline_name,
        )
    else:
        provision_environment(specs, apply=args.apply, pipeline_name=pipeline_name)


if __name__ == "__main__":
    main()
