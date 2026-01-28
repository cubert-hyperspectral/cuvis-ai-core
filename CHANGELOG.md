# Changelog
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
