# gRPC Plugin Management API

This document describes the plugin-management RPCs exposed by the cuvis-ai-core gRPC server
(`CuvisAIService`). It covers how a client registers plugins, inspects the available node
catalog, and runs a pipeline that uses plugin nodes.

## Overview

The plugin API lets a client:

- **Register** plugins (Git or local) into a session as catalog metadata.
- **Inspect** the node palette — built-in nodes plus the static catalog of every registered plugin.
- **Run** a pipeline that references those plugins; the server materialises (clones, installs,
  imports) each plugin lazily, in an isolated per-pipeline environment.

### Register, then materialise

This is the central concept and the biggest difference from older versions:

1. `LoadPlugins` is **register-only**. It parses the manifest, validates each entry, and records
   its catalog metadata in the session. It does **not** clone, install, or import anything. The
   response field is `registered_plugins` (names registered as metadata), not "loaded".
2. A plugin is **materialised lazily** when `LoadPipeline` loads a pipeline whose top-level
   `plugins:` list references that plugin. At that point the server composes an isolated
   environment containing exactly the plugins the pipeline declares, installs them, and runs the
   pipeline there. **Install/import failures surface at `LoadPipeline`, not `LoadPlugins`.**

This split keeps the server process clean: a plugin's dependencies are never installed into the
server's own environment, so conflicting plugin dependencies cannot corrupt the server.

### The inline node catalog

A plugin manifest entry's `provides:` list **is** the plugin's node catalog. Each entry is one
node — a fully-qualified `class_name` plus optional palette metadata (category, tags, icon,
per-port specs, doc summary). `ListAvailableNodes` serves this catalog **without importing any
plugin code**, so the palette is available before a plugin is ever installed.

### Session isolation

Plugins are registered **per session**. A plugin registered in session A is not visible in
session B, and a session's registrations are dropped when it closes.

```
Session A                    Session B
  ├─ registers plugin X        ├─ registers plugin Y
  └─ builtin nodes             └─ builtin nodes
```

## Plugin configuration

Manifests are the Pydantic models in `cuvis_ai_schemas.plugin`. A manifest is a mapping of
logical plugin name → plugin config.

**Git plugin** — pinned to a Git **tag** (branches and commit hashes are intentionally not
supported, for reproducibility):

```python
from cuvis_ai_schemas.plugin import GitPluginConfig

GitPluginConfig(
    repo="git@github.com:cubert-hyperspectral/cuvis-ai-adaclip.git",  # git@ or https://
    tag="v1.2.3",                                                      # tag only
    provides=[{"class_name": "cuvis_ai_adaclip.node.AdaCLIPDetector"}],
    package_name="cuvis-ai-adaclip",  # optional: real [project].name if it differs from the key
)
```

**Local plugin** — a filesystem path (absolute, or relative to the manifest file):

```python
from cuvis_ai_schemas.plugin import LocalPluginConfig

LocalPluginConfig(
    path="../my-plugin",
    provides=[{"class_name": "my_plugin.node.MyNode"}],
)
```

**`provides` entries** are `CatalogNodeEntry` objects, not bare strings. `class_name` (an FQCN)
is required; the rest are optional palette metadata:

```python
{
    "class_name": "my_plugin.node.MyNode",   # required, fully-qualified
    "category": "model",                      # NodeCategory value
    "tags": ["torch", "inference"],
    "icon_svg": "<svg ...></svg>",
    "input_specs":  {"data":   {"dtype": "float32", "shape": [-1, -1, -1, -1]}},
    "output_specs": {"scores": {"dtype": "float32", "shape": [-1, -1, -1, 1]}},
    "doc_summary": "One-line description for the palette.",
}
```

## RPCs

All plugin RPCs live on `CuvisAIService`.

### LoadPlugins

Register the manifest's plugins as catalog metadata in a session. **No install, no import.**

**Request / Response**

```protobuf
message LoadPluginsRequest {
  string session_id = 1;
  PluginManifest manifest = 2;     // PluginManifest{ bytes config_bytes }
}

message LoadPluginsResponse {
  repeated string registered_plugins = 1;   // registered as catalog metadata (NOT installed)
  map<string, string> failed_plugins = 2;   // name → Pydantic validation error
}
```

`PluginManifest.config_bytes` is the JSON of the Pydantic `PluginManifest`
(`manifest.model_dump_json().encode()`).

**Python example**

```python
from cuvis_ai_schemas.plugin import PluginManifest, GitPluginConfig

manifest = PluginManifest(
    plugins={
        "adaclip": GitPluginConfig(
            repo="git@github.com:cubert-hyperspectral/cuvis-ai-adaclip.git",
            tag="v1.2.3",
            provides=[{"class_name": "cuvis_ai_adaclip.node.AdaCLIPDetector"}],
        )
    }
)

response = client.LoadPlugins(
    cuvis_ai_pb2.LoadPluginsRequest(
        session_id=session_id,
        manifest=cuvis_ai_pb2.PluginManifest(
            config_bytes=manifest.model_dump_json().encode()
        ),
    )
)

print(f"Registered: {response.registered_plugins}")
print(f"Failed:     {dict(response.failed_plugins)}")
```

**Behaviour**

- Validates each manifest entry (Pydantic). Per-entry validation errors go to `failed_plugins`;
  the other entries still register (partial success).
- Records each plugin's catalog metadata in the session; tracked in `SessionState.registered_plugins`.
- Does **not** clone/install/import — that happens at `LoadPipeline` (see below).

---

### ListLoadedPlugins

List the plugins registered in a session.

```protobuf
message ListLoadedPluginsResponse { repeated PluginInfo plugins = 1; }

message PluginInfo {
  string name = 1;
  string type = 2;               // "git" or "local"
  string source = 3;             // repo URL or filesystem path
  string tag = 4;                // Git tag (git plugins only)
  repeated string provides = 5;  // fully-qualified class names
}
```

```python
resp = client.ListLoadedPlugins(
    cuvis_ai_pb2.ListLoadedPluginsRequest(session_id=session_id)
)
for p in resp.plugins:
    print(f"{p.name} [{p.type}] {p.source} {p.tag} → {list(p.provides)}")
```

---

### GetPluginInfo

Get one registered plugin's info. Returns `NOT_FOUND` if it isn't registered in the session.

```protobuf
message GetPluginInfoRequest { string session_id = 1; string plugin_name = 2; }
message GetPluginInfoResponse { PluginInfo plugin = 1; }
```

---

### ListAvailableNodes

List every available node: built-in nodes plus each registered plugin's **inline catalog**. The
server never imports plugin code to answer this — plugin nodes come purely from the manifest's
`provides:` metadata. A plugin whose entry provides no nodes contributes nothing to the palette
(and is logged).

```protobuf
message ListAvailableNodesResponse { repeated NodeInfo nodes = 1; }

message NodeInfo {
  string class_name = 1;                    // short display name
  string full_path  = 2;                    // fully-qualified class path
  string source     = 3;                    // "builtin", "plugin", or "custom"
  string plugin_name = 4;                   // set when source = "plugin"
  map<string, PortSpec> input_specs  = 5;   // one spec per input port
  map<string, PortSpec> output_specs = 6;   // one spec per output port
  bytes        icon_svg = 7;                // SVG payload (empty when none)
  NodeCategory category = 8;
  repeated NodeTag tags = 9;
}

message PortSpec {
  string name = 1;
  DType  dtype = 2;             // D_TYPE_UNSPECIFIED for generic / non-tensor ports
  repeated int64 shape = 3;     // -1 for dynamic dimensions
  bool   optional = 4;
  string description = 5;
  bool   variadic = 6;          // inputs only: accepts fan-in from multiple connections
}
```

```python
resp = client.ListAvailableNodes(
    cuvis_ai_pb2.ListAvailableNodesRequest(session_id=session_id)
)
for node in resp.nodes:
    where = "builtin" if node.source == "builtin" else f"plugin: {node.plugin_name}"
    print(f"  • {node.class_name} ({where})")
```

---

### ClearPluginCache

Clear cloned Git plugin repositories from the `NodeRegistry` clone cache.

```protobuf
message ClearPluginCacheRequest  { string plugin_name = 1; }  // empty = clear all
message ClearPluginCacheResponse { int32 cleared_count = 1; }
```

> Scope: this clears the Git **clone** cache only. The isolated per-pipeline environments that
> the orchestrator composes for `LoadPipeline` are managed separately and are not affected by
> this RPC.

---

## Complete workflow

```python
import grpc
from cuvis_ai_core.grpc.v1 import cuvis_ai_pb2, cuvis_ai_pb2_grpc
from cuvis_ai_schemas.plugin import PluginManifest, GitPluginConfig

# 1. Connect
channel = grpc.insecure_channel("localhost:50051")
client = cuvis_ai_pb2_grpc.CuvisAIServiceStub(channel)

# 2. Create a session
session_id = client.CreateSession(cuvis_ai_pb2.CreateSessionRequest()).session_id

# 3. Register plugins (metadata only — nothing is installed yet)
manifest = PluginManifest(
    plugins={
        "adaclip": GitPluginConfig(
            repo="git@github.com:cubert-hyperspectral/cuvis-ai-adaclip.git",
            tag="v1.2.3",
            provides=[{"class_name": "cuvis_ai_adaclip.node.AdaCLIPDetector"}],
        )
    }
)
reg = client.LoadPlugins(
    cuvis_ai_pb2.LoadPluginsRequest(
        session_id=session_id,
        manifest=cuvis_ai_pb2.PluginManifest(config_bytes=manifest.model_dump_json().encode()),
    )
)
print(f"Registered: {reg.registered_plugins}")

# 4. (Optional) inspect the palette — served from the inline catalog, no plugin import
nodes = client.ListAvailableNodes(
    cuvis_ai_pb2.ListAvailableNodesRequest(session_id=session_id)
).nodes

# 5. Load a pipeline that declares the plugin by bare name.
#    The plugin is materialised HERE (compose env → install → import).
#    Install/import errors surface as a LoadPipeline failure.
pipeline_yaml = """
metadata:
  name: Detection Pipeline
plugins:
  - adaclip          # bare name → resolves to the registered "adaclip" plugin
nodes:
  - name: detector
    class_name: cuvis_ai_adaclip.node.AdaCLIPDetector
    hparams: {}
connections: []
"""
loaded = client.LoadPipeline(
    cuvis_ai_pb2.LoadPipelineRequest(
        session_id=session_id,
        config=cuvis_ai_pb2.PipelineConfig(config_bytes=pipeline_yaml.encode()),
    )
)

# 6. Run inference … (see the inference API)

# 7. Close the session (drops the session's plugin registrations)
client.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))
```

## Error handling

`LoadPlugins` reports only **manifest/validation** failures (per entry, in `failed_plugins`):

| Failure | Example |
|---|---|
| Bad repo URL | `Invalid repo URL '…'. Must start with 'git@', 'https://', or 'http://'` |
| Branch/commit instead of tag | git plugins accept a **tag** only |
| Non-FQCN class path | `Invalid class path 'MyNode'. Must be fully-qualified` |
| Empty `provides` | a plugin must provide at least one node |

**Install/import failures happen later, at `LoadPipeline`** — when the declaring pipeline triggers
materialisation. Examples: clone failure, dependency resolution failure, `ModuleNotFoundError`
inside the composed environment. They surface as a `LoadPipeline` error, not in
`LoadPlugins.failed_plugins`.

## Best practices

- **Pin Git plugins by tag.** Branches/commits are not supported; a tag is reproducible.
- **Keep the manifest `provides:` catalog accurate.** It drives both the palette
  (`ListAvailableNodes`) and the install/import target — a stale catalog yields a wrong palette.
- **Reference plugins by bare name** in the pipeline's top-level `plugins:` list; each name must
  match a registered plugin (or a manifest in the server's plugins directory).
- **Close sessions** to release their plugin registrations.

## See also

- [gRPC API Overview](../README.md#grpc-api)
- Manifest/config schemas: `cuvis_ai_schemas.plugin` (`GitPluginConfig`, `LocalPluginConfig`,
  `PluginManifest`) and `cuvis_ai_schemas.catalog` (`CatalogNodeEntry`).
