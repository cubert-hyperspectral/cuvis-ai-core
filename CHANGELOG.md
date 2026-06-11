# Changelog

## [Unreleased]

- **Pluggable DataModules; the cuvis SDK leaves core.** Added the SDK-free
  `cuvis_ai_core.data.datamodule.BaseHyperspectralDataModule` (split-to-stage mapping + the four
  `*_dataloader()` methods + `validate_params`) and `create_data_module(registry, data_config)`. A
  concrete DataModule now ships from a plugin (see `cuvis-ai-dataloader`), registered as a
  `kind: data_module` manifest entry; core ships no concrete DataModules.
- **`NodeRegistry` gains a `data_modules` registry + kind routing.** `register_preinstalled` /
  `load_plugin` route each provides entry by its static `kind`: `data_module` entries register into
  `data_modules[DATA_MODULE_NAME]` (globally unique, asserts the class `DATA_MODULE_NAME` matches the
  manifest), `node` entries into `loaded_plugin_nodes` as before. The node palette
  (`ListAvailableNodes`) and the pipeline-plugin resolver coverage now filter to `kind == "node"`, so
  a loader-only plugin emits no palette nodes and does not trip the "provides no nodes" warning.
- **Registry dispatch at the two construction sites.** `training_service` and `restore.py` no longer
  hardcode `SingleCu3sDataModule`; both dispatch via `registry.data_modules[data_config.data_module]`.
  `restore-pipeline` inference is now generic over any DataModule (`--data-module` + repeatable
  `--data-arg key=value`; the legacy `--cu3s-file-path` / `--processing-mode` /
  `--annotation-json-path` / `--measurement-indices` flags are removed). Deleted
  `data/datasets.py` + `data/coco_labels.py` (moved to the `cuvis-ai-dataloader` plugin, refactored
  onto the base); `data/rle.py` stays (sibling consumers).
- **`grpc/helpers.py` drops `import cuvis`.** `PROCESSING_MODE_MAP` values are plain strings and
  `proto_to_processing_mode` returns a `str`; the cu3s reader coerces to the SDK enum at its own
  boundary. Core now imports no `cuvis` SDK (`pyproject` drops `cuvis`, `scikit-image`,
  `dataclass-wizard`; `pycocotools` stays for `rle.py`).
- **Orchestrator dep-isolation for data plugins.** The child-env composer emits `pkg[extras]` in
  `[project].dependencies` (keyed off the activated data module's extras) while keeping the bare
  package name for `[tool.uv.sources]`; the resolver unions the data-module plugin into the compose
  set, and `LoadPipeline` / `RestoreTrainRun` carry the selected `data_module` so extras resolve at
  compose time. A `tiff_paired` run never pulls a cu3s module's `cuvis`. `COMPOSER_SCHEMA_VERSION`
  bumped to invalidate stale caches.
- Consumes `cuvis-ai-schemas`' new `DataConfig {data_module, splits, batch_size, num_workers, params}`
  + `DataSplitConfig` + `CatalogNodeEntry.kind/data_module_name/extras` + additive
  `LoadPipelineRequest.data` field.

## 0.7.1 - 2026-06-10

- **Security:** raised dependency floors for transitively-pulled packages so downstream plugins inherit the fix instead of pinning each individually: `gitpython>=3.1.50` (CVE-2026-42215 / 42284 / 44244, GHSA-mv93-w799-cj2w), plus new direct floors `idna>=3.18` (CVE-2026-45409), `urllib3>=2.7.0` (PYSEC-2026-141 / 142), and `aiohttp>=3.14.1` (CVE-2026-34993 / 47265). Re-locked `uv.lock`.

## 0.7.0 - 2026-06-09

- Added the **child-env-per-run orchestrator** (new `cuvis_ai_core/orchestrator/` package): `cache_key` (a structured per-run venv key), `runtime_project` (generates the child `pyproject.toml`), `composer` (atomic build+publish via `filelock`, `uv lock` / `uv sync`, cache hits and broken-dir recovery), `spawner`, `venv_paths`, `uv_runner`, and `catalog`; plus `grpc/orchestrator_bridge.py`. The child runtime becomes the server's **sole** execution path — `LoadPipeline` / `Inference` / `Train` / `RestoreTrainRun` route through the orchestrator unconditionally.
- Added the **child runtime** (`cuvis_ai_core/run_runtime/`, run via `python -m cuvis_ai_core.run_runtime`); its `InitializeSession` registers the resolved plugin set via `NodeRegistry.register_preinstalled`, which imports already-installed plugin packages without re-installing, cloning, or mutating `sys.path`.
- **Breaking:** `LoadPlugins` is now **register-only** — it registers catalog entries instead of installing/importing; `SessionState.loaded_plugins` → `registered_plugins`. A pipeline that omits the mandatory `plugins:` field hard-fails with a fix-it pointer to the new `suggest-plugins-fix` CLI (`plugin_fixer.py`).
- Added **pipeline-driven plugin resolution**: the pure `plugin_resolver.resolve_pipeline_plugins(...)` materialises only the bare names declared in the pipeline's `plugins:` field.
- Changed `list_available_nodes` to read the inline node catalog from each manifest entry (no plugin imports); dropped the import-based fallback. New `scripts/emit_metadata.py` regenerates a manifest's inline catalog.
- **Breaking (CLI):** removed the legacy `--plugins-path` flag; `--plugins-dir` (the plugins directory + the pipeline's `plugins:` field) is the sole loader flag.
- **Breaking:** `SetTrainRunConfig` now requires an existing pipeline (no embedded pipeline section).
- **Breaking:** ports are one `PortSpec` each. `Node.INPUT_SPECS` / `OUTPUT_SPECS` are typed `dict[str, PortSpec]`; node init rejects list-form specs and a `variadic=True` output spec with a clear migration error. The pipeline accumulates a fan-in list only for `variadic` input ports.
- Changed `NodeInfo` construction and `scripts/emit_metadata.py` to the single-spec form (`map<string, PortSpec>` / `dict[str, CatalogPortSpec]`), carrying `variadic`. Dropped the `_unwrap_spec` list-normalization from the pipeline visualizer.
- Realigned plugin loading to the bare-name manifest flow (resolver, plugin config, restore) and removed the standalone gRPC plugin-management example.
- Reworked `node_registry` to two instance dicts: `plugin_catalog` (every known plugin's config, the single config source) and `loaded_plugin_nodes` (`{class_name -> node class}`, renamed from `plugin_registry`). Dropped the redundant `plugin_configs` dict, the unused `cache_dir` / `plugin_class_map` fields, and the standalone `pipeline/restore_preinstalled.py` module (folded into `NodeRegistry.register_preinstalled`). A plugin is "loaded" iff its classes are in `loaded_plugin_nodes`, so `clear_plugins` now clears both dicts and a loaded plugin's catalog config can't be silently replaced. The rename reaches `cuvis_ai_schemas.is_plugin`, so the `cuvis-ai-schemas[proto]` floor is bumped to `>=0.5.1`. Re-export the `cuvis_ai_core_pb2` type stub from cuvis-ai-schemas so the proto types have a single source of truth.
- Orchestrator robustness: child-runtime spawn timeouts are env-configurable; child stdout/stderr is routed to files (not `subprocess.PIPE`); relative config paths resolve against the server cwd; local plugins are pinned by `[project]` name rather than manifest key; the child env keeps `LD_LIBRARY_PATH` (dropping it as a CUDA var made the child interpreter exit 127 before Python ran on runners that resolve libs via it).
- Extended `scripts/audit_plugin_deps.py` for per-plugin CI and bumped stale dependency floors; added the `.github/workflows/dep_compat.yml` host-floor check and synced `uv.lock`. Added a `ruamel.yaml` dev dependency for the comment-preserving metadata emitter.
- Pipeline serialization now carries the declared `plugins:` field, so a saved pipeline keeps the mandatory field and reloads under the new contract: the builder records the declared plugin names and `serialize()` re-emits them.
- The per-run composer resolves for the composing host: the generated runtime `pyproject.toml` declares the host as a uv `required-environments`, so the child venv installs a wheel that exists for it. On Windows this backtracks `cuvis-il` to 3.5.0 (3.5.3.x ship manylinux-only wheels, no `win_amd64`); on Linux it keeps the latest.
- Orchestrator lifecycle hardening: server shutdown closes every open session so its child runtime is terminated instead of leaking as an orphaned subprocess; each session's per-run scratch root is removed on close; the in-process composer lock map is a `WeakValueDictionary` (no per-build leak on a long-lived server); the child-env deny-list covers more credential carriers.
- Re-registered the `Lentils` public dataset (`cubert-gmbh/XMR_Lentils`, `target_dir: Lentils`) so `uv run dataset download Lentils` lands the single-session test cube at `data/Lentils/Lentils_000.cu3s`, the path the training-backed tests read.

## 0.6.0 - 2026-05-11

- **Breaking**: Renamed the lentils public-dataset entry. `PublicDatasets.download_dataset("Lentils_Anomaly" | "lentils")` no longer resolves; use `Demo_Industrial_FOD_Lentils` / `demo_industrial_fod_lentils`, which points at the superseding HuggingFace repo [`cubert-gmbh/XMR_Demo_Industrial_Foreign_Object_Detection_Lentils`](https://huggingface.co/datasets/cubert-gmbh/XMR_Demo_Industrial_Foreign_Object_Detection_Lentils) (~6 GB; 69-frame XMR CU3S session with Dinomaly companion pipeline featured in the AdaClip tutorial and the docs dataset catalog). The legacy `cubert-gmbh/XMR_Lentils` repo is no longer referenced from the catalog and ships no companion pipeline; users still needing it can `snapshot_download(repo_id="cubert-gmbh/XMR_Lentils", repo_type="dataset", ...)` directly.
- Removed the post-download `lentils -> Lentils` symlink / `copytree` fallback in `download_data_cli`. It existed only to bridge case-insensitive access to the old `XMR_Lentils` target_dir; the new entry's `target_dir` matches its alias, so no remapping is needed.
- Exposed `download_data_cli` as the `dataset` console script via `[project.scripts]`. The function's docstring has long advertised a `uv run dataset` entry point, but the script was never registered — `uv run dataset list` / `uv run dataset download <name>` now work after `uv sync`. Promoted `click` from a transitively-assumed import to a declared dependency (`click>=8.0`) so the CLI works in minimal envs that don't pick `click` up via `jupyterlab` or similar.

## 0.5.3 - 2026-05-11

- Updated `PublicDatasets` entry for `Blood_Perfusion` to point at the renamed HuggingFace repo `cubert-gmbh/XMR_Demo_Blood_Perfusion` (was `cubert-gmbh/XMR_Blood_Perfusion`); `download_dataset("blood_perfusion", ...)` against `0.5.2` 404s on HuggingFace. Local `target_dir` renamed to `XMR_Demo_Blood_Perfusion` to match the `Demo_Object_Tracking` peer entry. Users with an existing `<download_path>/XMR_Blood_Perfusion/` folder can rename it in place to avoid re-downloading ~7 GB.

## 0.5.2 - 2026-05-05

- Fixed `PipelineVisualizer` crash (`AttributeError: 'list' object has no attribute 'dtype'`) when rendering pipelines that contain nodes with variadic input ports declared as `list[PortSpec]` (e.g. `TensorBoardMonitorNode.INPUT_SPECS["artifacts"]`). The visualizer now mirrors the `isinstance(spec, list): spec = spec[0]` normalization that `cuvis_ai_core/pipeline/pipeline.py` applies in seven other call sites, via a new `_unwrap_spec` helper threaded through `_resolve_port_spec`, `_port_dots_html` (the failing site), and `_format_port_spec`.
- Added regression tests that render a pipeline with a variadic-port node via `to_graphviz` / `to_graphviz(show_port_types=True)` / `to_mermaid` and direct-test the unwrap helper.

## 0.5.1 - 2026-05-05

- Bumped stale `pyproject.toml` dependency floors to align with `uv.lock`, preventing the plugin loader's in-process `uv pip install` from upgrading the live venv on Windows where packages with loaded native extensions (Pillow `_imaging.pyd`, lxml `etree.pyd`) fail with "Access is denied".

## 0.5.0 - 2026-04-30

- Added `NodeCategory`, `NodeTag`, and `_icon_name` ClassVars on `Node` base, with `get_category()` / `get_tags()` / `get_icon_name()` accessors. Defaults to `NodeCategory.UNSPECIFIED` and an empty tag set so unannotated subclasses still compile.
- Added `cuvis_ai_core/utils/icon_helpers.py` with a per-node SVG → schemas-default icon resolution chain backing `NodeInfo.icon_svg`.
- Added `MissingNodeMetadataWarning`: pipelines emit a once-per-class warning when a node lacks explicit `_category` / `_tags`, including a copy-pasteable remediation hint with concrete `NodeCategory` / `NodeTag` examples. `frozenset({NodeTag.UNSPECIFIED})` is treated as missing.
- Changed `list_available_nodes` gRPC to populate `NodeInfo.category`, `tags`, and `icon_svg`. Class lookup runs in its own try-block so a lookup failure no longer crashes port-spec extraction; metadata extraction is independently safe and falls back to `(UNSPECIFIED, [], unspecified.svg)` on any per-node error.
- Added `style="card"` mode to `PipelineVisualizer.to_graphviz`: rounded category-coloured cards via Graphviz HTML-table labels, with per-port colored dots wired through HTML `PORT` anchors so each edge attaches to a specific port (no bundled multi-edges) and per-edge dtype colors from `cuvis_ai_schemas.extensions.ui.port_display`.
- **Breaking**: Removed the legacy `style="classic"` visualizer mode along with its `node_shape`, `node_colors`, and `node_type_resolver` knobs. Card layout is now the only output.
- Repointed `resolve_display` and `is_plugin` from the deleted local `cuvis_ai_core/pipeline/node_display.py` to the canonical `cuvis_ai_schemas.extensions.ui.node_display`. The schemas-side `is_plugin` reads only from `registry.plugin_registry` — the local copy's `__display__["plugin"]` opt-in is gone (no production consumers).
- Bumped `cuvis-ai-schemas` floor to `>=0.4.0` from PyPI; removed the editable `schemas-icons` worktree path source.
- Added `Demo_Object_Tracking` public dataset entry (XMR multi-person tracking demo) with `demo_object_tracking` snake-case alias.
- Added hyphen-form normalisation at dataset-name lookup so every dataset accepts both underscore and hyphen forms (e.g. `Demo-Object-Tracking`, `blood-perfusion`); a single `PublicDatasets._normalize` helper applied in `download_dataset` and `get_target_dir`.
- Fixed `_convert_port_spec_to_proto` so `PortSpec(dtype=torch.Tensor, …)` resolves to `D_TYPE_UNSPECIFIED` instead of raising; previously the dedicated branch was shadowed by a `hasattr(spec.dtype, "dtype")` check because `torch.Tensor` exposes a class-level `dtype` descriptor.
- Consolidated the dtype → proto dispatch into `cuvis_ai_core.grpc.helpers.dtype_to_proto`; the port-spec converter now delegates so the two paths cannot drift.
- Fixed `Predictor.predict()` tqdm progress bar inside Jupyter kernels by switching to `from tqdm.auto import tqdm` and short-circuiting `_should_disable_progress_bar()` when `IPython.get_ipython()` is not None, before the TTY check.
- Widened `requires-python` to `>=3.11, <3.14` so the package installs under Python 3.12 and 3.13 in addition to 3.11.
- Updated package description to "Cuvis.AI Core Framework" to align with the umbrella-framework branding.
- Bumped `softprops/action-gh-release` from v2 to v3 in the PyPI release workflow (folded in from #29).
- Refreshed README badges to flat-square style with reference-link form.

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
