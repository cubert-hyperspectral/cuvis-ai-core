# Changelog

## [Unreleased]

## 0.4.1 - 2026-04-22

- Removed the SHA-256 hash from the `torchvision==0.26.0+cu128` manylinux_x86_64 wheel entry in `uv.lock`, matching the existing unhashed precedent for the aarch64 and win_amd64 torchvision wheels. PyTorch's R2 CDN periodically re-publishes wheels under the same URL with slightly different build metadata, which drifts the sha256 and breaks every CI job at `uv sync`; this normalization eliminates the recurring hash-mismatch failure for torchvision.
- Updated README logo reference to the `github.com/.../blob/...?raw=true` form for the cuvis.sdk banner image.
- Retitled the README to "Cuvis.AI Core" and added a Cuvis.AI umbrella-framework introduction; reframed the cuvis-ai-core description as the infrastructure layer beneath Cuvis.AI. Capability bullets unchanged.

## 0.4.0 - 2026-04-22

- Fixed `_convert_port_spec_to_proto` so `PortSpec(dtype=torch.Tensor, …)` (the generic-tensor marker) resolves to `D_TYPE_UNSPECIFIED` instead of raising "Unsupported numpy dtype"; the dedicated branch was previously shadowed by a `hasattr(spec.dtype, "dtype")` check because `torch.Tensor` exposes a class-level `dtype` descriptor. Nodes declaring generic-tensor ports (e.g. `CU3SDataNode.INPUT_SPECS["cube"]`) now return populated port specs from `ListAvailableNodes`.
- Added unit tests for `_convert_port_spec_to_proto` covering torch dtypes, the `torch.Tensor` marker, numpy scalar classes, Python builtins, unsupported inputs, and symbolic shape dimensions.
- Consolidated the dtype → proto dispatch into a single public helper `cuvis_ai_core.grpc.helpers.dtype_to_proto`. `_convert_port_spec_to_proto` now delegates to it so the port-spec converter and future tensor serializers cannot drift. Added direct unit tests for `dtype_to_proto` covering every dispatch branch.

## 0.3.4 - 2026-04-10

- Added `Node.cleanup()` virtual method for releasing runtime resources held by individual nodes.
- Added `CuvisPipeline.cleanup()` that tears down nodes in reverse order, clears validation and profiling caches, and resets the graph.
- Added `SessionManager.set_pipeline()` that cleans up the previous pipeline before attaching a replacement, preventing GPU memory leaks on pipeline swap.
- Fixed gRPC session teardown to eagerly release GPU-backed resources via `gc.collect()` and `torch.cuda.empty_cache()` instead of waiting for deferred garbage collection.
- Fixed `close_session()` to explicitly nullify trainer, pipeline, and config references so they become eligible for immediate collection.
- Changed `SetupPipeline` and `RestorePipeline` gRPC handlers to route through `SessionManager.set_pipeline()` instead of directly assigning `session.pipeline`.
- Added tests for session pipeline cleanup, pipeline replacement cleanup, and pipeline introspection teardown.

## 0.3.3 - 2026-04-09

- Switched from `opencv-python` to `opencv-python-headless` to avoid file-locking conflicts when plugins install headless variant at runtime on Windows.
- Increased default gRPC max message size from 200 MB to 300 MB to match the client default.

## 0.3.2 - 2026-04-09

- Fixed torchvision wheel hash mismatch caused by inconsistent PyTorch CDN content across regions.

## 0.3.0 - 2026-04-01

- Added `cuvis_ai_core.training.Predictor` with predict-mode datamodule inference support and progress reporting that disables cleanly in non-interactive environments.
- Added automatic per-node runtime profiling APIs, summary formatting utilities, and gRPC profiling service endpoints for pipeline execution.
- Added public dataset download helpers and CLI-facing exports in `cuvis_ai_core.data`.
- Added RLE and video data utilities, hardened COCO mask decoding against swapped or missing canvas sizes, and guarded video datamodule setup against predict-time reinitialization.
- Changed gRPC pipeline discovery and lookup to use `pipeline_path` values, with path validation and alignment to the released schema field names.
- Added inference parsing support for `rgb_image`, `frame_id`, `mesu_index`, and optional `BoundingBox.object_id` values in batch inputs.
- Improved pipeline node-state serialization behavior, extra pipeline input handling, and restore utilities for long-running inference workflows.
- Switched core schema resolution to the published `cuvis-ai-schemas>=0.3.0` package and removed the local editable override from dependency wiring.
- Updated CI and release workflows to use `codecov/codecov-action@v6`, `actions/upload-artifact@v7`, and `actions/download-artifact@v8` while preserving container git safe-directory handling.

## 0.2.0 - 2026-02-27

- Added recursive pipeline discovery to find configs in subdirectories (e.g., `anomaly/adaclip/baseline`)
- Added pipeline names as relative paths from base directory instead of bare stem
- Added `TRAINABLE_BUFFERS: tuple[str, ...]` class attribute on `Node` base class
- Added `__init_subclass__` hook that validates `TRAINABLE_BUFFERS` is a tuple of strings at class definition time
- Changed `freeze()` and `unfreeze()` to iterate `TRAINABLE_BUFFERS` for automatic buffer↔parameter conversion
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
- Renamed `Node.freezed` to private `_frozen` attribute with read-only `frozen` property
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
