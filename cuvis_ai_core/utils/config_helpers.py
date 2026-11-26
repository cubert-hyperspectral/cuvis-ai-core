"""Configuration helper functions for the cuvis.ai framework."""

from __future__ import annotations

import functools
import os
import re
import threading
from pathlib import Path, PureWindowsPath
from typing import Any

import yaml
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ValidationError

from cuvis_ai_schemas.pipeline.config import PipelineConfig
from cuvis_ai_schemas.training import (
    CallbacksConfig,
    DataConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
    TrainRunConfig,
)

from cuvis_ai_core.utils.path_confinement import (
    PathNotAllowedError,
    allowed_roots,
    check_not_remote,
    confine_resolved,
)

# ------------------------------------------------------------------
# Config registry and validation helpers
# ------------------------------------------------------------------

CONFIG_TYPE_REGISTRY: dict[str, type[BaseModel]] = {
    "training": TrainingConfig,
    "optimizer": OptimizerConfig,
    "scheduler": SchedulerConfig,
    "callbacks": CallbacksConfig,
    "data": DataConfig,
    "pipeline": PipelineConfig,
    "trainrun": TrainRunConfig,
}


def get_config_class(config_type: str) -> type[BaseModel]:
    """Get Pydantic model class for config type."""
    if config_type not in CONFIG_TYPE_REGISTRY:
        raise ValueError(
            f"Unknown config type: {config_type}. "
            f"Available types: {list(CONFIG_TYPE_REGISTRY.keys())}"
        )
    return CONFIG_TYPE_REGISTRY[config_type]


def generate_json_schema(config_type: str) -> dict:
    """Generate JSON Schema for config type."""
    config_class = get_config_class(config_type)
    return config_class.model_json_schema()


def validate_config_dict(config_type: str, config_dict: dict) -> tuple[bool, list[str]]:
    """Validate configuration dictionary."""
    config_class = get_config_class(config_type)

    try:
        config_class.model_validate(config_dict)
        return True, []
    except ValidationError as exc:
        errors = []
        for err in exc.errors():
            field_path = ".".join(str(x) for x in err["loc"])
            errors.append(f"{field_path}: {err['msg']}")
        return False, errors


# ------------------------------------------------------------------
# Hydra-based config resolution and file helpers
# ------------------------------------------------------------------


def apply_config_overrides(
    config: dict[str, Any] | DictConfig,
    overrides: list[str] | dict[str, Any] | None,
) -> dict[str, Any]:
    """Apply overrides (list or dict) to an OmegaConf config and return a plain dict."""

    def _format_override_key(key: str) -> str:
        """Normalize dot notation with numeric parts into OmegaConf bracket notation."""
        if "[" in key:
            return key

        formatted_parts: list[str] = []
        for part in key.split("."):
            if part.isdigit() and formatted_parts:
                formatted_parts[-1] = f"{formatted_parts[-1]}[{part}]"
            elif part.isdigit():
                formatted_parts.append(f"[{part}]")
            else:
                formatted_parts.append(part)
        return ".".join(formatted_parts)

    if overrides in (None, [], {}):
        return OmegaConf.to_container(OmegaConf.create(config), resolve=True)  # type: ignore[return-value]

    if isinstance(config, DictConfig):
        config_omega = config.copy()
    else:
        config_omega = OmegaConf.create(config)

    if isinstance(overrides, list):
        for override in overrides:
            if "=" not in override:
                raise ValueError(
                    f"Invalid override format: {override}. Expected format: key=value"
                )
            key, value = override.split("=", 1)
            formatted_key = _format_override_key(key)
            parsed_value = yaml.safe_load(value)
            OmegaConf.update(config_omega, formatted_key, parsed_value, merge=True)
    elif isinstance(overrides, dict):
        override_omega = OmegaConf.create(overrides)
        config_omega = OmegaConf.merge(config_omega, override_omega)
    else:
        raise TypeError(
            f"overrides must be a list of strings, a dict, or None. Got {type(overrides)}"
        )

    return OmegaConf.to_container(config_omega, resolve=True)  # type: ignore[return-value]


def _refuse_hydra_settings(config_file: Path, overrides: list[str]) -> None:
    """Refuse Hydra's own settings while ``CUVIS_ALLOWED_ROOTS`` is set.

    ``hydra.searchpath`` (from an override or from the primary config) makes Hydra test
    folders the caller names before anything else runs, a network share included.
    """
    for override in overrides:
        key = override.split("=", 1)[0].lstrip("+~").strip()
        if key == "hydra" or key.startswith(("hydra.", "hydra/")):
            raise PathNotAllowedError(
                f"overrides may not change hydra settings while CUVIS_ALLOWED_ROOTS "
                f"is set: {override!r}"
            )
    with config_file.open("r", encoding="utf-8") as f:
        primary = yaml.safe_load(f)
    if isinstance(primary, dict) and "hydra" in primary:
        raise PathNotAllowedError(
            f"{config_file} sets hydra options, which are not allowed while "
            "CUVIS_ALLOWED_ROOTS is set"
        )


_HYDRA_GUARD_LOCK = threading.Lock()


def _guard_hydra_config_sources() -> None:
    """Keep every config Hydra tests or opens inside the allowed roots or its own package.

    ``compose()`` follows a config's ``defaults:`` list and group overrides (``+data=x``) by
    handing their names to each config source in turn: Hydra's own package
    (``pkg://hydra.conf``) first, then the config folder. A package source joins the name to
    its package, so an absolute name or ``..`` would read any YAML on the host; it answers that
    it holds no such config. The folder source checks the joined path against the roots the
    way Hydra builds it and raises ``PathNotAllowedError`` before Hydra touches it. Structured
    configs live in memory. Installed once; while ``CUVIS_ALLOWED_ROOTS`` is unset nothing is
    checked.
    """
    from hydra._internal.sources_registry import SourcesRegistry
    from hydra.core.plugins import Plugins

    with _HYDRA_GUARD_LOCK:
        # Hydra's first plugin scan imports its core plugins again, so a source class imported
        # before it is not the one compose() uses; take the registered classes after it.
        Plugins.instance()
        for scheme, source in SourcesRegistry.instance().types.items():
            if scheme == "structured" or vars(source).get("_cuvis_roots_guard"):
                continue
            if scheme == "file":
                _guard_folder_source(source)
            else:
                _guard_package_source(source)
            source._cuvis_roots_guard = True  # type: ignore[attr-defined]


def _leaves_its_package(name: str) -> bool:
    """Whether a config name points outside the package it is looked up in."""
    try:
        check_not_remote(name, what="config include")
    except PathNotAllowedError:
        return True
    return (
        name.startswith(("/", "\\"))
        or bool(PureWindowsPath(name).drive)
        or ".." in re.split(r"[\\/]", name)
    )


def _guard_package_source(source: type) -> None:
    """A package source looks a name up inside its package only; any other name is not there."""

    def contained(method):
        @functools.wraps(method)
        def wrapper(self, config_path, *args, **kwargs):
            if allowed_roots() is not None and _leaves_its_package(config_path):
                if method.__name__ == "load_config":
                    raise PathNotAllowedError(
                        f"config include: {config_path!r} is not inside {self.path}"
                    )
                return [] if method.__name__ == "list" else False
            return method(self, config_path, *args, **kwargs)

        return wrapper

    for name in ("load_config", "is_config", "is_group", "list"):
        setattr(source, name, contained(getattr(source, name)))


def _guard_folder_source(source: type) -> None:
    """The folder source checks the path it joins from the name against the roots."""

    def guarded(method, normalize: bool):
        @functools.wraps(method)
        def wrapper(self, config_path, *args, **kwargs):
            if allowed_roots() is not None:
                name = (
                    self._normalize_file_name(config_path) if normalize else config_path
                )
                confine_resolved(
                    Path(os.path.join(self.path, name)),
                    what="config include",
                    read=True,
                )
            return method(self, config_path, *args, **kwargs)

        return wrapper

    # load_config and is_config join the name with ".yaml" added; is_group and list as given.
    for name, normalize in (
        ("load_config", True),
        ("is_config", True),
        ("is_group", False),
        ("list", False),
    ):
        setattr(source, name, guarded(getattr(source, name), normalize))


def resolve_config_with_hydra(
    config_type: str,
    config_path: str,
    search_paths: list[str],
    overrides: list[str] | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve configuration using Hydra composition and validate with Pydantic."""
    from hydra import compose, initialize_config_dir

    logger.info(f"Resolving {config_type} config: {config_path}")
    logger.debug(f"Search paths: {search_paths}")
    logger.debug(f"Overrides: {overrides}")

    config_file = _find_config_file(config_path, search_paths).resolve()
    logger.debug(f"Found config file: {config_file}")

    # Determine the Hydra config root and config name relative to that root.
    config_root = config_file.parent
    config_name = config_file.stem
    roots = allowed_roots()

    for search_dir in search_paths:
        resolved_dir = Path(search_dir).resolve()
        try:
            relative = config_file.relative_to(resolved_dir)
        except ValueError:
            continue
        if roots is not None:
            # Hydra probes its config folder itself, so a folder above the roots is refused.
            try:
                confine_resolved(resolved_dir, what="config root", read=True)
            except PathNotAllowedError:
                continue

        config_root = resolved_dir
        config_name = relative.with_suffix("").as_posix()
        break

    hydra_overrides = overrides if isinstance(overrides, list) else []
    if roots is not None:
        _refuse_hydra_settings(config_file, hydra_overrides)
        _guard_hydra_config_sources()

    with initialize_config_dir(config_dir=str(config_root), version_base="1.3"):
        cfg = compose(config_name=config_name, overrides=hydra_overrides)
        config_dict = OmegaConf.to_container(cfg, resolve=True)

    # Handle Hydra packaging for all config types
    # When resolving configs, Hydra may wrap them in a key matching the config_type
    # due to @package directives or config structure. Unwrap if needed.
    if isinstance(config_dict, dict) and config_type in config_dict:
        # Check if the wrapped value looks like the actual config (has expected fields)
        # For pipeline: should have "nodes" or "name"
        # For training: should have "seed" or "optimizer"
        # For data: should have "cu3s_file_path" or "batch_size"
        wrapped_value = config_dict[config_type]
        if isinstance(wrapped_value, dict):
            # Unwrap if the wrapper only contains the config_type key, or if the top level
            # doesn't have the expected structure
            config_class = get_config_class(config_type)
            expected_fields = set(config_class.model_fields.keys())
            top_level_fields = set(config_dict.keys())
            wrapped_fields = set(wrapped_value.keys())

            # If wrapped_value has more expected fields than top level, unwrap it
            if len(expected_fields & wrapped_fields) > len(
                expected_fields & top_level_fields
            ):
                logger.debug(f"Unwrapping {config_type} config from Hydra packaging")
                config_dict = wrapped_value

    # Apply overrides using shared helper to keep behavior consistent across entrypoints
    if isinstance(config_dict, dict):
        config_dict = apply_config_overrides(config_dict, overrides)  # type: ignore[arg-type]
    else:
        # Handle case where config_dict might not be a dict (shouldn't happen in practice)
        logger.warning(f"Config dict is not a dict type, got {type(config_dict)}")
        config_dict = {}

    config_class = get_config_class(config_type)
    config_model = config_class.model_validate(config_dict)

    logger.info(f"Successfully resolved {config_type} config")
    return config_model.to_dict()


def _find_config_file(path: str, search_paths: list[str]) -> Path:
    """Find config file in search paths, supporting relative and absolute paths."""
    path_obj = Path(path)

    if not path_obj.suffix:
        path_obj = path_obj.with_suffix(".yaml")

    if path_obj.is_absolute():
        if path_obj.exists():
            return confine_resolved(path_obj, what="config path", read=True)
        raise FileNotFoundError(f"Config file not found: {path_obj}")

    for search_dir in search_paths:
        candidate = Path(search_dir) / path_obj
        if candidate.exists():
            logger.debug(f"Found config in {search_dir}: {path_obj}")
            return confine_resolved(candidate.resolve(), what="config path", read=True)

    raise FileNotFoundError(
        f"Config file '{path}' not found in search paths: {search_paths}"
    )


__all__ = [
    "apply_config_overrides",
    "resolve_config_with_hydra",
    "validate_config_dict",
    "generate_json_schema",
    "get_config_class",
]
