# Changelog
## V0.1.2 In Progress
- Extended gRPC NodeInfo with port specifications (PortSpec, PortSpecList messages)
- Added input_specs and output_specs fields to NodeInfo for accurate port validation
- Automatic dtype and shape extraction from node INPUT_SPECS and OUTPUT_SPECS
- Support for variadic ports and optional port flags in gRPC protocol
- Refactored NodeRegistry.load_plugin to use testable helper functions in git_and_os.py
- New helpers: parse_plugin_config, extract_package_prefixes, clear_package_modules, import_plugin_nodes
- Added NodeRegistry.get_builtin_class() for direct builtin node class access

## V0.1.1
- Fix for plugin installation by automatically installing plugin dependencies from pyproject.toml
- New visualization-extension input parameter for restore pipeline (supports PNG and Markdown formats)
- Enhanced plugin support in pipeline loading with optional node registry parameter
- Improved error messaging for plugin-related issues with usage examples

## V0.1.0
- Initial repository creation with framework-only architecture extracted from cuvis-ai monolith
- Core framework components: Base Node class, Pipeline infrastructure, Training framework, gRPC services, NodeRegistry
- Plugin system with Git and local filesystem support via extended NodeRegistry
- Pydantic-validated plugin configuration (GitPluginConfig, LocalPluginConfig, PluginManifest)
- Session-scoped plugin isolation with gRPC plugin management interface
- New plugin management RPCs: `LoadPlugins`, `ListLoadedPlugins`, `GetPluginInfo`, `ClearPluginCache`
- Cache management for Git repositories with version pinning (`~/.cuvis_plugins/`)
- 422 tests migrated from cuvis-ai with comprehensive fixture library
- Bug fixes: DataLoader access violation (num_workers=0), single-threaded gRPC compatibility
- Modular service architecture: SessionService, ConfigService, PipelineService, TrainingService, InferenceService, IntrospectionService, DiscoveryService, PluginService
