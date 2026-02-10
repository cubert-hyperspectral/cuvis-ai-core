# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Changelog validation step in release workflow to prevent publishing without release notes
- Labels and commit-message prefixes to Dependabot configuration for better PR organization

## [0.1.2] - 2026-02-09

### Added
- `input_specs` and `output_specs` fields to NodeInfo for accurate port validation
- `NodeRegistry.get_builtin_class()` method for direct builtin node class access
- New testable helper functions in `git_and_os.py`: `parse_plugin_config`, `extract_package_prefixes`, `clear_package_modules`, `import_plugin_nodes`
- Support for variadic ports and optional port flags in gRPC protocol
- Automatic dtype and shape extraction from node `INPUT_SPECS` and `OUTPUT_SPECS`
- GitHub Actions CI workflow with automated testing, type checking, linting, and security scanning
- Automated PyPI release workflow with TestPyPI and PyPI publishing
- Dependabot configuration for automated dependency updates (excludes PyTorch for CUDA compatibility)
- Codecov integration for code coverage tracking (80% target)
- Security scanning with pip-audit, detect-secrets, and bandit
- Complete PyPI package metadata with classifiers and SPDX license identifier
- Apache License 2.0 file in repository root

### Changed
- Refactored `NodeRegistry.load_plugin` to use testable helper functions for better maintainability
- Improved plugin spec dtype handling for better type safety
- Updated CHANGELOG format to "Keep a Changelog" standard with semantic versioning
- Aligned all Python version references to 3.11 (Black, MyPy, Ruff)
- Updated project URLs to GitHub (from GitLab)

### Fixed
- Plugin installation and dependency management issues
- Restore pipeline functionality with improved error handling
- Path string handling in `coco_labels.py` (raw string prefix for Windows compatibility)

## [0.1.1] - 2026-01-26

### Fixed
- Plugin installation by automatically installing plugin dependencies from pyproject.toml
- Improved error messaging for plugin-related issues with usage examples

### Added
- New visualization-extension input parameter for restore pipeline (supports PNG and Markdown formats)
- Enhanced plugin support in pipeline loading with optional node registry parameter

## [0.1.0] - 2026-01-23

### Added
- Initial repository creation with framework-only architecture extracted from cuvis-ai monolith
- Core framework components: Base Node class, Pipeline infrastructure, Training framework, gRPC services, NodeRegistry
- Plugin system with Git and local filesystem support via extended NodeRegistry
- Pydantic-validated plugin configuration (GitPluginConfig, LocalPluginConfig, PluginManifest)
- Session-scoped plugin isolation with gRPC plugin management interface
- New plugin management RPCs: `LoadPlugins`, `ListLoadedPlugins`, `GetPluginInfo`, `ClearPluginCache`
- Cache management for Git repositories with version pinning (`~/.cuvis_plugins/`)
- 422 tests migrated from cuvis-ai with comprehensive fixture library
- Modular service architecture: SessionService, ConfigService, PipelineService, TrainingService, InferenceService, IntrospectionService, DiscoveryService, PluginService

### Fixed
- DataLoader access violation (num_workers=0)
- Single-threaded gRPC compatibility

---

[unreleased]: https://github.com/cubert-hyperspectral/cuvis-ai-core/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/cubert-hyperspectral/cuvis-ai-core/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/cubert-hyperspectral/cuvis-ai-core/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/cubert-hyperspectral/cuvis-ai-core/releases/tag/v0.1.0
