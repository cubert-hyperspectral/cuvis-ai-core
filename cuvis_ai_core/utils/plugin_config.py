"""Pydantic configuration models for NodeRegistry plugin system."""

from pathlib import Path
from typing import List, Dict, Union, Annotated
from pydantic import BaseModel, ConfigDict, Field, field_validator
import yaml

class _BasePluginConfig(BaseModel):
    """Base plugin configuration with strict validation.

    All plugin types inherit from this base class to ensure
    consistent validation and error handling.
    """
    model_config = ConfigDict(
        extra="forbid",  # Reject unknown fields (catch typos)
        validate_assignment=True,  # Validate on attribute assignment
        populate_by_name=True,  # Allow field aliases
    )

    provides: List[str] = Field(
        description="List of fully-qualified class paths this plugin provides",
        min_length=1,  # At least one class required
    )

    @field_validator("provides")
    @classmethod
    def _validate_class_paths(cls, value: List[str]) -> List[str]:
        """Ensure class paths are well-formed."""
        for class_path in value:
            if not class_path or "." not in class_path:
                raise ValueError(
                    f"Invalid class path '{class_path}'. "
                    "Must be fully-qualified (e.g., 'package.module.ClassName')"
                )
        return value

class GitPluginConfig(_BasePluginConfig):
    """Git repository plugin configuration.

    Supports:
    - SSH URLs: git@gitlab.com:user/repo.git
    - HTTPS URLs: https://github.com/user/repo.git
    - Git refs: tags (v1.2.3), branches (main), commits (abc123)
    """

    repo: str = Field(
        description="Git repository URL (SSH or HTTPS)",
        min_length=1,
    )

    ref: str = Field(
        description="Git reference: tag (v1.2.3), branch (main), or commit hash (abc123)",
        min_length=1,
    )

    @field_validator("repo")
    @classmethod
    def _validate_repo_url(cls, value: str) -> str:
        """Validate Git repository URL format."""
        if not (value.startswith("git@") or
                value.startswith("https://") or
                value.startswith("http://")):
            raise ValueError(
                f"Invalid repo URL '{value}'. "
                "Must start with 'git@', 'https://', or 'http://'"
            )
        return value

    @field_validator("ref")
    @classmethod
    def _validate_ref(cls, value: str) -> str:
        """Validate Git ref is not empty."""
        if not value.strip():
            raise ValueError("Git ref cannot be empty")
        return value.strip()

class LocalPluginConfig(_BasePluginConfig):
    """Local filesystem plugin configuration.

    Supports:
    - Absolute paths: /home/user/my-plugin
    - Relative paths: ../my-plugin (resolved relative to manifest file)
    - Windows paths: C:\\Users\\user\\my-plugin
    """

    path: str = Field(
        description="Absolute or relative path to plugin directory",
        min_length=1,
    )

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        """Validate path is not empty."""
        if not value.strip():
            raise ValueError("Path cannot be empty")
        return value.strip()

    def resolve_path(self, manifest_dir: Path) -> Path:
        """Resolve relative paths to absolute paths.

        Args:
            manifest_dir: Directory containing the manifest file

        Returns:
            Absolute path to plugin directory
        """
        plugin_path = Path(self.path)
        if not plugin_path.is_absolute():
            plugin_path = (manifest_dir / plugin_path).resolve()
        return plugin_path

class PluginManifest(BaseModel):
    """Complete plugin manifest containing all plugin configurations.

    This is the root configuration object validated when loading
    a plugins.yaml file or dictionary.
    """
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    plugins: Dict[str, Annotated[
        Union[GitPluginConfig, LocalPluginConfig],
        Field(discriminator=None)  # Pydantic will auto-detect based on fields
    ]] = Field(
        description="Map of plugin names to their configurations",
        default_factory=dict,
    )

    @field_validator("plugins")
    @classmethod
    def _validate_plugin_names(cls, value: Dict) -> Dict:
        """Ensure plugin names are valid Python identifiers."""
        for name in value.keys():
            if not name.isidentifier():
                raise ValueError(
                    f"Invalid plugin name '{name}'. "
                    "Must be a valid Python identifier"
                )
        return value

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "PluginManifest":
        """Load and validate manifest from YAML file."""
        if not yaml_path.exists():
            raise FileNotFoundError(f"Plugin manifest not found: {yaml_path}")

        with yaml_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data:
            return cls(plugins={})

        return cls.model_validate(data)

    @classmethod
    def from_dict(cls, data: Dict) -> "PluginManifest":
        """Load and validate manifest from dictionary."""
        return cls.model_validate(data)

    def to_yaml(self, yaml_path: Path) -> None:
        """Save manifest to YAML file."""
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with yaml_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(
                self.model_dump(exclude_none=True),
                f,
                sort_keys=False,
                default_flow_style=False,
            )
