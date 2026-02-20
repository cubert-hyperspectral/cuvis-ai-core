# Changelog

## [Unreleased]

- Added `@grpc_handler` decorator for unified exception-to-gRPC-status mapping across all services
- Added `get_session_or_error()` and `require_pipeline()` gRPC validation helpers in `error_handling.py`
- Added session `fps` metadata property to `CuvisDataset`
- Added spectral radiance processing mode with dark-reference validation in `CuvisDataset`
- Added `@grpc_handler` to 7 service methods that were missing it (close_session, get_parameter_schema, validate_config, list_loaded_plugins, get_plugin_info, list_available_nodes, get_pipeline_visualization)
- Fixed `Pipelinees` typo — renamed to `Pipelines` across gRPC stubs, services, helpers, and tests (upstream fix in cuvis-ai-schemas PR #8)
- Changed all 8 gRPC service files to use `@grpc_handler` decorator, replacing inline try/except blocks
- Changed training configs to use `cuvis-ai-schemas` as single source of truth
- Changed `NodeConfig` usage from `params` to `hparams` across core
- Removed duplicate proto definitions now owned by `cuvis-ai-schemas`
- Removed `VALIDATE` compatibility code from execution stage handling
- Removed dead code and fixed stale docstrings

## 0.1.3 - 2026-02-11

- Added changelog validation step in release workflow
- Added labels and commit-message prefixes to Dependabot configuration

## 0.1.2 - 2026-02-09

- Added input_specs and output_specs fields to NodeInfo for port validation
- Added NodeRegistry.get_builtin_class() for direct builtin node class access
- Added testable helper functions in git_and_os.py for plugin loading
- Added variadic port and optional port flag support in gRPC protocol
- Added automatic dtype and shape extraction from node port specs
- Added GitHub Actions CI workflow with testing, type checking, linting, and security
- Added automated PyPI release workflow with TestPyPI and production publishing
- Added Dependabot configuration (excludes PyTorch for CUDA compatibility)
- Added Codecov integration with 80% coverage target
- Added security scanning with pip-audit, detect-secrets, and bandit
- Added PyPI package metadata with classifiers and SPDX license identifier
- Added Apache License 2.0
- Changed NodeRegistry.load_plugin to use testable helper functions
- Improved plugin spec dtype handling for better type safety
- Changed Python version references to 3.11 across tooling
- Changed project URLs from GitLab to GitHub
- Fixed plugin installation and dependency management
- Fixed restore pipeline with improved error handling
- Fixed path string handling in coco_labels.py for Windows compatibility

## 0.1.1 - 2026-01-26

- Fixed plugin installation by auto-installing dependencies from pyproject.toml
- Improved error messaging for plugin-related issues
- Added visualization-extension input parameter for restore pipeline
- Added plugin support in pipeline loading with optional node registry parameter

## 0.1.0 - 2026-01-23

- Added initial repository with framework extracted from cuvis-ai monolith
- Added core components: Base Node, Pipeline, Training framework, gRPC services, NodeRegistry
- Added plugin system with Git and local filesystem support
- Added Pydantic-validated plugin configuration
- Added session-scoped plugin isolation with gRPC plugin management
- Added plugin management RPCs: LoadPlugins, ListLoadedPlugins, GetPluginInfo, ClearPluginCache
- Added cache management for Git repositories with version pinning
- Added 422 tests migrated from cuvis-ai with comprehensive fixtures
- Added modular service architecture with 8 service modules
- Fixed DataLoader access violation with num_workers=0
- Fixed gRPC single-threaded compatibility
