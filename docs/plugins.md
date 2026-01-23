# NodeRegistry Plugin System

The NodeRegistry plugin system enables dynamic loading of external nodes from Git repositories and local filesystem paths, allowing teams to extend the cuvis-ai framework without modifying the core or catalog repositories.

## Overview

The plugin system integrates directly into the existing `NodeRegistry` class, providing a unified API for all node lookups while maintaining backward compatibility with existing pipelines.

### Key Features

- **Dynamic Plugin Loading**: Load nodes from external Git repositories
- **Version Pinning**: Support Git tags, branches, and commit hashes via `ref` field
- **Local Development**: Support filesystem paths for development workflows
- **Intelligent Caching**: Cache Git repos and verify versions before re-cloning
- **Pydantic Validation**: Strict validation for all plugin configurations
- **Backward Compatible**: Existing pipelines work without modification

## Installation

The plugin system is included in cuvis-ai-core. For Git-based plugins, ensure GitPython is installed:

```bash
uv add "cuvis-ai-core[plugins]"
```

## Usage

### Basic Usage Pattern

```python
from cuvis_ai_core.utils.node_registry import NodeRegistry
from cuvis_ai_core.pipeline.factory import PipelineBuilder

# Step 1: Load plugins into NodeRegistry BEFORE building pipeline
NodeRegistry.load_plugins("plugins.yaml")

# Step 2: Build pipeline (unchanged - uses NodeRegistry.get() internally)
builder = PipelineBuilder()
pipeline = builder.build_from_config("pipeline.yaml")

# Run inference
results = pipeline.run(input_data)
```

### Loading Plugins

#### From YAML Manifest

Create a `plugins.yaml` file:

```yaml
plugins:
  adaclip:
    repo: "git@gitlab.cubert.local:cubert/cuvis-ai-adaclip.git"
    ref: "v1.2.3"
    provides:
      - cuvis_ai_adaclip.node.adaclip_node.AdaCLIPDetector

  custom_models:
    repo: "https://github.com/company/custom-models.git"
    ref: "v2.0.0"
    provides:
      - custom_models.segmentation.SegmentationNode
      - custom_models.classification.ClassifierNode
```

Load plugins:

```python
NodeRegistry.load_plugins("plugins.yaml")
```

#### Programmatic Loading

```python
# Git plugin
NodeRegistry.load_plugin("adaclip", {
    "repo": "git@gitlab.cubert.local:cubert/cuvis-ai-adaclip.git",
    "ref": "v1.2.3",
    "provides": ["cuvis_ai_adaclip.node.adaclip_node.AdaCLIPDetector"]
})

# Local plugin
NodeRegistry.load_plugin("local_dev", {
    "path": "../my-plugin",
    "provides": ["my_plugin.MyNode"]
})
```

### Plugin Management

```python
# List loaded plugins
plugins = NodeRegistry.list_plugins()
print(f"Loaded plugins: {plugins}")

# List plugin nodes
plugin_nodes = NodeRegistry.list_plugin_nodes()
print(f"Plugin nodes: {plugin_nodes}")

# Unload a plugin
NodeRegistry.unload_plugin("adaclip")

# Clear all plugins
NodeRegistry.clear_plugins()

# Clear Git cache
NodeRegistry.clear_plugin_cache()  # Clear all cached plugins
NodeRegistry.clear_plugin_cache("adaclip")  # Clear specific plugin
```

## Plugin Configuration

### Git Plugin Configuration

```python
from cuvis_ai_core.utils.plugin_config import GitPluginConfig

config = GitPluginConfig(
    repo="git@gitlab.cubert.local:cubert/test-plugin.git",
    ref="v1.2.3",  # Can be tag, branch, or commit hash
    provides=["test_plugin.TestNode"]
)
```

### Local Plugin Configuration

```python
from cuvis_ai_core.utils.plugin_config import LocalPluginConfig

config = LocalPluginConfig(
    path="/path/to/plugin",  # Absolute or relative path
    provides=["local_plugin.LocalNode"]
)
```

### Plugin Manifest

```python
from cuvis_ai_core.utils.plugin_config import PluginManifest

# Load from YAML
manifest = PluginManifest.from_yaml("plugins.yaml")

# Load from dict
manifest = PluginManifest.from_dict({
    "plugins": {
        "plugin1": {...},
        "plugin2": {...}
    }
})

# Save to YAML
manifest.to_yaml("plugins.yaml")
```

## Plugin Development

### Creating a Plugin

A plugin is simply a Python package that contains Node classes. The package should have:

1. A proper Python package structure with `__init__.py`
2. Node classes that inherit from `cuvis_ai_core.node.node.Node`
3. All required dependencies declared in `pyproject.toml`

### Example Plugin Structure

```
my_plugin/
├── __init__.py
├── pyproject.toml
└── nodes/
    ├── __init__.py
    └── my_node.py
```

### Example Node Implementation

```python
from cuvis_ai_core.node.node import Node

class MyCustomNode(Node):
    def __init__(self, name="MyCustomNode"):
        super().__init__(name)
        self.custom_param = "default_value"

    def forward(self, input_data):
        # Your node logic here
        result = process_data(input_data, self.custom_param)
        return {"output": result}
```

## Cache Management

The plugin system caches Git repositories to improve performance. Cache behavior:

- **Cache Location**: `~/.cuvis_plugins/` by default
- **Cache Structure**: `~/.cuvis_plugins/{plugin_name}@{ref}/`
- **Cache Verification**: Automatically verifies cached repos match expected ref
- **Cache Reuse**: Reuses cached repos when ref matches

### Custom Cache Directory

```python
NodeRegistry.set_cache_dir("/custom/cache/path")
```

### Cache Performance

| Plugin Size | First Load | Cached Load |
|-------------|------------|-------------|
| Small (< 10 MB) | ~2-5s | < 0.1s |
| Medium (10-50 MB) | ~5-15s | < 0.1s |
| Large (> 50 MB) | ~15-60s | < 0.1s |

## Error Handling

The plugin system provides clear error messages for common issues:

```python
# Invalid configuration
try:
    NodeRegistry.load_plugin("invalid", {
        "repo": "invalid-url",
        "ref": "v1.0.0",
        "provides": ["test.Node"]
    })
except ValidationError as e:
    print(f"Configuration error: {e}")

# Missing plugin
try:
    NodeRegistry.get("NonExistentNode")
except KeyError as e:
    print(f"Node not found: {e}")
```

## Best Practices

1. **Use Tagged Releases**: More reliable than branches for production
2. **Pin Versions**: Avoid unexpected changes from floating branches
3. **Pre-warm Cache**: Clone plugins before production deployment
4. **Monitor Cache Size**: Clear old plugin versions periodically
5. **Use Local Paths for Development**: Faster iteration during development

## API Reference

### NodeRegistry Plugin Methods

#### `load_plugins(manifest_path: Union[str, Path]) -> int`
Load multiple plugins from a YAML manifest file.

#### `load_plugin(name: str, config: dict) -> None`
Load a single plugin from a configuration dict.

#### `unload_plugin(name: str) -> None`
Unload a plugin and remove its nodes from the registry.

#### `list_plugins() -> list[str]`
List all loaded plugin names.

#### `list_plugin_nodes() -> list[str]`
List all nodes provided by plugins.

#### `clear_plugins() -> None`
Unload all plugins and clear plugin registries.

#### `clear_plugin_cache(plugin_name: Optional[str] = None) -> None`
Clear cached Git repositories.

#### `set_cache_dir(path: Union[str, Path]) -> None`
Set the cache directory for Git plugins.

### Pydantic Models

#### `GitPluginConfig`
Git repository plugin configuration.

#### `LocalPluginConfig`
Local filesystem plugin configuration.

#### `PluginManifest`
Complete plugin manifest containing all plugin configurations.

## Migration Guide

### From Hard Dependencies to Plugins

If you're currently using hard dependencies like `cuvis-ai-adaclip`, you can migrate to plugins:

1. **Remove hard dependency** from your `pyproject.toml`
2. **Add plugin configuration** to your `plugins.yaml`
3. **Load plugins** before building pipelines

### Example Migration

**Before:**
```toml
[project]
dependencies = [
    "cuvis-ai-core>=0.1.0",
    "cuvis-ai-adaclip>=1.0.0",  # Hard dependency
]
```

**After:**
```toml
[project]
dependencies = [
    "cuvis-ai-core>=0.1.0",
    # adaclip is now loaded as a plugin
]
```

**plugins.yaml:**
```yaml
plugins:
  adaclip:
    repo: "git@gitlab.cubert.local:cubert/cuvis-ai-adaclip.git"
    ref: "v1.2.3"
    provides:
      - cuvis_ai_adaclip.node.adaclip_node.AdaCLIPDetector
```

**Python code:**
```python
# Load plugins before building pipeline
NodeRegistry.load_plugins("plugins.yaml")

# Build pipeline (unchanged)
builder = PipelineBuilder()
pipeline = builder.build_from_config("pipeline.yaml")
```

## Troubleshooting

### Common Issues

**GitPython not installed:**
```bash
uv add gitpython>=3.1.40
```

**Plugin not found:**
- Verify the plugin name is correct
- Check that the plugin was loaded successfully
- Ensure the class path in `provides` is correct

**Cache verification failed:**
- The cached repository may be corrupted
- Try clearing the cache: `NodeRegistry.clear_plugin_cache()`
- The plugin will be re-cloned on next load

**Node instantiation failed:**
- Verify the node class implements all required methods
- Check that all dependencies are installed
- Ensure the node follows the cuvis-ai Node API

## Examples

### Production Setup with Git Plugins

**plugins.yaml:**
```yaml
plugins:
  adaclip:
    repo: "git@gitlab.cubert.local:cubert/cuvis-ai-adaclip.git"
    ref: "v1.2.3"
    provides:
      - cuvis_ai_adaclip.node.adaclip_node.AdaCLIPDetector

  custom_models:
    repo: "https://github.com/company/custom-models.git"
    ref: "v2.0.0"
    provides:
      - custom_models.segmentation.SegmentationNode
      - custom_models.classification.ClassifierNode
```

### Development Setup with Local Plugins

**plugins.yaml:**
```yaml
plugins:
  dev_plugin:
    path: "../my-plugin-dev"  # Relative path
    provides:
      - my_plugin.experimental.ExperimentalNode
```

### Multiple Plugin Versions

```yaml
plugins:
  adaclip_stable:
    repo: "git@gitlab.cubert.local:cubert/cuvis-ai-adaclip.git"
    ref: "v1.2.3"
    provides:
      - cuvis_ai_adaclip.node.adaclip_node.AdaCLIPDetector

  adaclip_experimental:
    repo: "git@gitlab.cubert.local:cubert/cuvis-ai-adaclip.git"
    ref: "develop"
    provides:
      - cuvis_ai_adaclip.experimental.ExperimentalDetector
```

## Support

For issues or questions about the plugin system:

1. Check the [troubleshooting section](#troubleshooting)
2. Review the [API reference](#api-reference)
3. Consult the [implementation documentation](ALL_4976_cuvis_ai_core_phase2.md)
4. Contact the cuvis-ai development team
