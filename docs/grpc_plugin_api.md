# gRPC Plugin Management API

This document describes the plugin management capabilities exposed via the cuvis-ai-core gRPC API.

## Overview

The plugin management API allows clients to:
- Load custom nodes dynamically from Git repositories or local paths
- Query loaded plugins and available nodes
- Manage plugin cache
- Isolate plugins per session for multi-tenant safety

## Key Concepts

### Session-Scoped Plugins

Plugins are loaded **per-session**, ensuring isolation between concurrent clients:

```
Session A                    Session B
  ├─ Plugin X                  ├─ Plugin Y
  │   └─ NodeX                 │   └─ NodeY
  └─ Builtin Nodes             └─ Builtin Nodes
```

- Session A can use NodeX (not visible to Session B)
- Session B can use NodeY (not visible to Session A)
- Both sessions can use builtin nodes
- Plugins are auto-cleaned when session closes

### Plugin Types

**Git Plugin**: Load from Git repository
```python
GitPluginConfig(
    repo="git@gitlab.example.com:org/plugin.git",
    ref="v1.2.3",  # Tag, branch, or commit
    provides=["plugin.nodes.CustomNode"]
)
```

**Local Plugin**: Load from filesystem
```python
LocalPluginConfig(
    path="/path/to/plugin",
    provides=["plugin.nodes.CustomNode"]
)
```

## RPCs

### LoadPlugins

Load plugins from a manifest into a session.

**Request**: `LoadPluginsRequest`
```protobuf
message LoadPluginsRequest {
  string session_id = 1;
  PluginManifest manifest = 2;  // JSON-serialized Pydantic manifest
}
```

**Response**: `LoadPluginsResponse`
```protobuf
message LoadPluginsResponse {
  repeated string loaded_plugins = 1;        // Successfully loaded
  map<string, string> failed_plugins = 2;    // name → error message
}
```

**Python Example**:
```python
from cuvis_ai_schemas.plugin import PluginManifest, GitPluginConfig

manifest = PluginManifest(
    plugins={
        "custom_detector": GitPluginConfig(
            repo="git@example.com:org/detector-plugin.git",
            ref="v2.0.0",
            provides=["detector_plugin.CustomDetector"]
        )
    }
)

response = client.LoadPlugins(
    cuvis_ai_core_pb2.LoadPluginsRequest(
        session_id=session_id,
        manifest=cuvis_ai_core_pb2.PluginManifest(
            config_bytes=manifest.model_dump_json().encode()
        )
    )
)

print(f"Loaded: {response.loaded_plugins}")
print(f"Failed: {dict(response.failed_plugins)}")
```

**Behavior**:
- Loads each plugin in the manifest
- Returns success/failure per plugin (partial failures allowed)
- Plugins registered in session-scoped registry
- Tracked in `SessionState.loaded_plugins`

---

### ListLoadedPlugins

List all plugins loaded in a session.

**Request**: `ListLoadedPluginsRequest`
```protobuf
message ListLoadedPluginsRequest {
  string session_id = 1;
}
```

**Response**: `ListLoadedPluginsResponse`
```protobuf
message ListLoadedPluginsResponse {
  repeated PluginInfo plugins = 1;
}

message PluginInfo {
  string name = 1;
  string type = 2;        // "git" or "local"
  string source = 3;      // Repo URL or filesystem path
  string ref = 4;         // Git ref (if applicable)
  repeated string provides = 5;  // Class paths provided
}
```

**Python Example**:
```python
response = client.ListLoadedPlugins(
    cuvis_ai_core_pb2.ListLoadedPluginsRequest(session_id=session_id)
)

for plugin in response.plugins:
    print(f"Plugin: {plugin.name}")
    print(f"  Type: {plugin.type}")
    print(f"  Source: {plugin.source}")
    print(f"  Provides: {list(plugin.provides)}")
```

---

### GetPluginInfo

Get information about a specific loaded plugin.

**Request**: `GetPluginInfoRequest`
```protobuf
message GetPluginInfoRequest {
  string session_id = 1;
  string plugin_name = 2;
}
```

**Response**: `GetPluginInfoResponse`
```protobuf
message GetPluginInfoResponse {
  PluginInfo plugin = 1;
}
```

**Python Example**:
```python
response = client.GetPluginInfo(
    cuvis_ai_core_pb2.GetPluginInfoRequest(
        session_id=session_id,
        plugin_name="custom_detector"
    )
)

print(f"Type: {response.plugin.type}")
print(f"Source: {response.plugin.source}")
print(f"Provides: {list(response.plugin.provides)}")
```

**Error**: Raises error if plugin not found in session.

---

### ListAvailableNodes

List all available nodes (builtin + session plugins).

**Request**: `ListAvailableNodesRequest`
```protobuf
message ListAvailableNodesRequest {
  string session_id = 1;
}
```

**Response**: `ListAvailableNodesResponse`
```protobuf
message ListAvailableNodesResponse {
  repeated NodeInfo nodes = 1;
}

message NodeInfo {
  string class_name = 1;    // Short name (e.g., "CustomDetector")
  string full_path = 2;     // Full import path
  string source = 3;        // "builtin" or "plugin"
  string plugin_name = 4;   // Plugin name (if source="plugin")
}
```

**Python Example**:
```python
response = client.ListAvailableNodes(
    cuvis_ai_core_pb2.ListAvailableNodesRequest(session_id=session_id)
)

print(f"Available nodes ({len(response.nodes)}):")
for node in response.nodes:
    if node.source == "builtin":
        print(f"  • {node.class_name} (builtin)")
    else:
        print(f"  • {node.class_name} (plugin: {node.plugin_name})")
```

---

### ClearPluginCache

Clear Git plugin cache (cloned repositories).

**Request**: `ClearPluginCacheRequest`
```protobuf
message ClearPluginCacheRequest {
  string plugin_name = 1;  // Empty string = clear all
}
```

**Response**: `ClearPluginCacheResponse`
```protobuf
message ClearPluginCacheResponse {
  int32 cleared_count = 1;  // Number of repos cleared
}
```

**Python Example**:
```python
# Clear specific plugin cache
response = client.ClearPluginCache(
    cuvis_ai_core_pb2.ClearPluginCacheRequest(plugin_name="custom_detector")
)
print(f"Cleared {response.cleared_count} repo(s)")

# Clear all plugin caches
response = client.ClearPluginCache(
    cuvis_ai_core_pb2.ClearPluginCacheRequest(plugin_name="")
)
print(f"Cleared all caches: {response.cleared_count} repo(s)")
```

**Note**: Does not affect loaded plugins in active sessions.

---

## Complete Workflow Example

```python
import grpc
from cuvis_ai_core.grpc.v1 import cuvis_ai_core_pb2, cuvis_ai_core_pb2_grpc
from cuvis_ai_schemas.plugin import PluginManifest, GitPluginConfig

# 1. Connect to server
channel = grpc.insecure_channel("localhost:50051")
client = cuvis_ai_core_pb2_grpc.CuvisAIServiceStub(channel)

# 2. Create session
session_resp = client.CreateSession(cuvis_ai_core_pb2.CreateSessionRequest())
session_id = session_resp.session_id
print(f"Session: {session_id}")

# 3. Load plugins
manifest = PluginManifest(
    plugins={
        "detector": GitPluginConfig(
            repo="git@example.com:ml/detector.git",
            ref="v1.0.0",
            provides=["detector.CustomDetector"]
        )
    }
)

plugin_resp = client.LoadPlugins(
    cuvis_ai_core_pb2.LoadPluginsRequest(
        session_id=session_id,
        manifest=cuvis_ai_core_pb2.PluginManifest(
            config_bytes=manifest.model_dump_json().encode()
        )
    )
)
print(f"Loaded: {plugin_resp.loaded_plugins}")

# 4. List available nodes
nodes_resp = client.ListAvailableNodes(
    cuvis_ai_core_pb2.ListAvailableNodesRequest(session_id=session_id)
)
print(f"Available nodes: {len(nodes_resp.nodes)}")

# 5. Load pipeline using plugin node
pipeline_yaml = """
metadata:
  name: Detection Pipeline
nodes:
  - name: detector
    class: CustomDetector
    params:
      threshold: 0.8
connections: []
"""

pipeline_resp = client.LoadPipeline(
    cuvis_ai_core_pb2.LoadPipelineRequest(
        session_id=session_id,
        config=cuvis_ai_core_pb2.PipelineConfig(
            config_bytes=pipeline_yaml.encode()
        )
    )
)
print(f"Pipeline loaded: {pipeline_resp.pipeline_id}")

# 6. Run inference
# ... (existing inference API)

# 7. Close session (auto-cleanup plugins)
client.CloseSession(cuvis_ai_core_pb2.CloseSessionRequest(session_id=session_id))
print("Session closed")
```

---

## Session Isolation

Plugins loaded in one session **do not affect** other sessions:

```python
# Session A loads plugin X
session_a = client.CreateSession(...)
manifest_a = PluginManifest(plugins={"plugin_x": ...})
client.LoadPlugins(session_id=session_a.session_id, manifest=manifest_a)

# Session B loads plugin Y
session_b = client.CreateSession(...)
manifest_b = PluginManifest(plugins={"plugin_y": ...})
client.LoadPlugins(session_id=session_b.session_id, manifest=manifest_b)

# Session A sees only plugin_x
nodes_a = client.ListAvailableNodes(session_id=session_a.session_id)
# → Contains plugin_x nodes, NOT plugin_y nodes

# Session B sees only plugin_y
nodes_b = client.ListAvailableNodes(session_id=session_b.session_id)
# → Contains plugin_y nodes, NOT plugin_x nodes
```

---

## Error Handling

### Partial Failures

`LoadPlugins` reports per-plugin success/failure:

```python
response = client.LoadPlugins(...)

if response.failed_plugins:
    for name, error in response.failed_plugins.items():
        print(f"Failed to load {name}: {error}")
        
if response.loaded_plugins:
    print(f"Successfully loaded: {response.loaded_plugins}")
```

### Common Errors

**Invalid Repository**:
```
failed_plugins: {"bad_plugin": "Repository not found: git@invalid.git"}
```

**Missing Dependencies**:
```
failed_plugins: {"plugin_a": "ModuleNotFoundError: No module named 'torch'"}
```

**Invalid Provides Path**:
```
failed_plugins: {"plugin_b": "Cannot import 'plugin.nonexistent.Node'"}
```

---

## Best Practices

### 1. Version Pin Git Plugins

Use specific tags/commits instead of branch names:

✅ **Good**:
```python
GitPluginConfig(repo="...", ref="v1.2.3")  # Reproducible
```

❌ **Avoid**:
```python
GitPluginConfig(repo="...", ref="main")  # Can change over time
```

### 2. Handle Partial Failures

Check both success and failure lists:

```python
response = client.LoadPlugins(...)

if response.failed_plugins:
    # Log failures but continue with loaded plugins
    logger.warning(f"Some plugins failed: {response.failed_plugins}")

if not response.loaded_plugins:
    # All plugins failed
    raise RuntimeError("No plugins loaded successfully")
```

### 3. Query Available Nodes

Verify plugin nodes are available before building pipeline:

```python
# Load plugins
client.LoadPlugins(...)

# Verify node available
nodes = client.ListAvailableNodes(session_id=session_id)
available_classes = {node.class_name for node in nodes.nodes}

if "CustomDetector" not in available_classes:
    raise RuntimeError("CustomDetector not available")

# Now safe to use in pipeline
client.LoadPipeline(...)
```

### 4. Close Sessions

Always close sessions to cleanup plugins:

```python
try:
    session_id = client.CreateSession(...).session_id
    client.LoadPlugins(...)
    # ... work ...
finally:
    client.CloseSession(session_id=session_id)
```

### 5. Cache Management

Clear cache after plugin updates:

```python
# Clear specific plugin cache after update
client.ClearPluginCache(plugin_name="my_plugin")

# Reload with new version
manifest = PluginManifest(
    plugins={"my_plugin": GitPluginConfig(..., ref="v2.0.0")}
)
client.LoadPlugins(...)
```

---

## Advanced Topics

### Custom Plugin Development

See `docs/plugins.md` for creating custom node plugins.

### Local Development

Use `LocalPluginConfig` during development:

```python
manifest = PluginManifest(
    plugins={
        "dev_plugin": LocalPluginConfig(
            path="/workspace/my-plugin",
            provides=["dev_plugin.ExperimentalNode"]
        )
    }
)
```

### Multi-Plugin Pipelines

Load multiple plugins and combine their nodes:

```python
manifest = PluginManifest(
    plugins={
        "detector": GitPluginConfig(...),
        "classifier": GitPluginConfig(...),
        "visualizer": LocalPluginConfig(...)
    }
)

client.LoadPlugins(...)

# Pipeline using nodes from all plugins
pipeline_yaml = """
nodes:
  - name: detect
    class: Detector  # from detector plugin
  - name: classify
    class: Classifier  # from classifier plugin
  - name: viz
    class: Visualizer  # from visualizer plugin
connections:
  - from: detect.outputs.detections
    to: classify.inputs.objects
  - from: classify.outputs.classes
    to: viz.inputs.labels
"""
```

---

## API Reference Summary

| RPC | Purpose | Session-Scoped |
|-----|---------|----------------|
| `LoadPlugins` | Load plugins from manifest | ✓ |
| `ListLoadedPlugins` | List plugins in session | ✓ |
| `GetPluginInfo` | Get specific plugin details | ✓ |
| `ListAvailableNodes` | List all available nodes | ✓ |
| `ClearPluginCache` | Clear Git repo cache | ✗ (global) |

---

## See Also

- [Plugin Development Guide](plugins.md)
- [gRPC API Overview](../README.md#grpc-api)
- [Example Client](../examples/grpc_plugin_management_client.py)
