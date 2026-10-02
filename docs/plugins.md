# NodeRegistry Plugin System

The plugin system lets teams extend cuvis-ai with external nodes (and data modules) shipped in
their own packages, without modifying the core or node libraries. A plugin is a pip-installable
Python package plus a **bare manifest** that declares what it provides.

## Concepts

### One file, one plugin

A plugin manifest is a single YAML file describing exactly one plugin: an explicit `name`, a source
(a local `path` or a git `repo` + `tag`), and a `capabilities` list. There is no multi-plugin
wrapper. A directory of these files (the "plugins directory") is the catalog.

```yaml
# configs/plugins/adaclip.yaml  (git source)
name: adaclip                       # required, explicit; never derived from the filename
repo: "git@github.com:cubert-hyperspectral/cuvis-ai-adaclip.git"   # git@ or https://
tag: "v1.2.3"                       # tag only — branches and commit hashes are not supported
package_name: "cuvis-ai-adaclip"    # optional: real [project].name when it differs from `name`
capabilities:
  - class_name: cuvis_ai_adaclip.node.AdaCLIPDetector   # FQCN; optional palette metadata may follow
```

```yaml
# configs/plugins/my_plugin.yaml  (local source, for development)
name: my_plugin
path: "../my-plugin"                # absolute, or relative to this manifest file
capabilities:
  - class_name: my_plugin.node.MyNode
```

Each `capabilities` entry is a `PluginCapabilityEntry`: `class_name` (an FQCN) is required; the rest
is optional palette metadata (`category`, `tags`, `icon_svg`, `input_specs`, `output_specs`,
`doc_summary`). A `kind: data_module` entry instead registers a data module (it carries
`data_module_name` + pip `extras` and never appears in the node palette). A manifest may also
carry its own top-level `extras` (see [Optional heavy dependencies](#optional-heavy-dependencies)).

The `capabilities` list **is** the plugin's node catalog: the server enumerates it for the node
palette without importing any plugin code.

### Optional heavy dependencies

A node plugin with an optional backend keeps the heavy package behind a pip extra of its own
`pyproject.toml` (`[project.optional-dependencies]`) and lets a manifest request it:

```yaml
# configs/plugins/rfdetr_seg_trt.yaml  (a second manifest of the same package)
name: rfdetr_seg_trt
repo: "https://github.com/cubert-hyperspectral/cuvis-ai-rfdetr.git"
tag: "v0.5.1"
package_name: "cuvis-ai-rfdetr"       # the same package as the plain `rfdetr` manifest
extras: [tensorrt]                    # installed whenever this manifest is in a pipeline's plugin set
capabilities:
  - class_name: cuvis_ai_rfdetr.node.rfdetr_segmenter.RFDETRSegmenter
```

The rules:

- A manifest's `extras` are installed whenever that manifest is in the pipeline's plugin set, united
  with the selected data module's extras (a `kind: data_module` entry's `extras` apply only to the run
  that selects that module). Names are PEP 508 extras, normalised per PEP 685 (`Tensor_RT` is
  `tensor-rt`); a duplicate after normalisation is rejected by the schema.
- Manifests that install one package (the same canonical `package_name`, the same repo and tag or the
  same path) merge into one requirement, `cuvis-ai-rfdetr[tensorrt]`. The same package from two
  different sources fails the compose with `FAILED_PRECONDITION` naming both manifests.
- A pipeline that needs the backend lists **both** manifests, `plugins: [rfdetr, rfdetr_seg_trt]`.
  Warm-child reuse compares manifest names: a child composed for both serves a plain `[rfdetr]`
  pipeline, a child composed for the variant alone does not. Keep the variant manifest minimal (only
  the node that needs the backend), so the plain manifest stays the one every pipeline lists.
- An extra the package does not declare fails the compose right after `uv lock` (uv itself only warns),
  with the extras the package declares:
  `Plugin 'rfdetr_seg_trt' requests extra 'tensort' that package 'cuvis-ai-rfdetr' (...) does not declare (the lock resolved its extras tensorrt, train)`.
- A git manifest that names a package other than its `name` needs `package_name`, or uv's metadata
  check fails with `Package metadata name ... does not match given name`.

To see what a pipeline will install before any compose, run the dry run: `uv run provision
--pipeline-path <pipeline.yaml> --plugins-dir configs/plugins` prints one spec per package, extras
included (`cuvis-ai-rfdetr[tensorrt] @ git+https://github.com/cubert-hyperspectral/cuvis-ai-rfdetr.git@v0.5.1`).
The server logs the same merged requirement when it composes (`Composing for cuvis-ai-rfdetr[tensorrt]`),
and the composed entry's `env_desc.md` records the plugin set.

### Register, then materialise

Registering a plugin records its manifest as catalog metadata. It does **not** clone, install, or
import anything. A plugin is materialised lazily — cloned/installed/imported into an isolated
per-pipeline environment — only when a pipeline that references it is loaded. This keeps the server
process clean: plugin dependencies never land in the server's own environment.

## Manifest schemas

The Pydantic models live in `cuvis_ai_schemas.plugin`:

```python
from cuvis_ai_schemas.plugin import (
    GitPluginSource,         # a git-sourced plugin manifest (repo + tag)
    LocalPluginSource,       # a local-path-sourced plugin manifest
    PluginManifest,          # the union: GitPluginSource | LocalPluginSource
    PluginCapabilityEntry,   # one capability (a node or a data module)
    PluginCapabilities,      # install-stripped capability set for the palette
    NodePortSpec,            # serialized port spec
    load_plugin_manifest,    # load one bare manifest file (resolves a local path)
    parse_plugin_manifest,   # validate an in-memory dict
    write_plugin_manifest,   # write a bare manifest file
)
```

The git/local variants are named for the plugin **source** (where it comes from), not a
kind of manifest. The directory scan + cross-directory duplicate-name guard lives in
`cuvis-ai-core` (`plugin_resolver._build_catalog`), not in schemas.

## In-process use (CLI, notebooks)

For an in-process pipeline, register plugins into a `NodeRegistry` instance. The plugin package must
already be importable in the active environment (an editable `[tool.uv.sources]` entry in dev, the
`provision` helper, or `uv pip install`). This path never clones or installs.

```python
from cuvis_ai_core.utils.node_registry import NodeRegistry

registry = NodeRegistry()

# Register one bare manifest (one plugin) by path.
registry.register_plugin("configs/plugins/adaclip.yaml")
```

A plugin whose package is not installed raises a guided `ModuleNotFoundError` pointing at the
`provision` command:

```bash
uv run provision --pipeline-path <pipeline.yaml> --plugins-dir configs/plugins --apply
```

## Pipelines reference plugins by name

A pipeline declares the plugins it needs by **bare name** in a top-level `plugins:` list; each name
resolves to a manifest in the plugins directory.

```yaml
plugins:
  - adaclip
  - cuvis_ai_builtin
nodes:
  - name: detector
    class_name: cuvis_ai_adaclip.node.AdaCLIPDetector
    hparams: {}
connections: []
```

Load it against a plugins directory:

```bash
uv run restore-pipeline --pipeline-path pipeline.yaml --plugins-dir configs/plugins
```

If a pipeline omits the mandatory `plugins:` field, `restore-pipeline` fails with a fix-it hint that
points at `suggest-plugins-fix`, which proposes the field from the plugins directory.

## gRPC

Over gRPC the same model applies, with the client owning the catalog: `LoadPlugin` registers **one**
manifest as session catalog metadata (its `config_bytes` is a single bare manifest), so the client
loops to register each plugin its pipeline needs. `LoadPipeline` then resolves the pipeline's
`plugins:` against that client-pushed catalog (the server never scans a directory) and materialises
them — a pipeline naming an unregistered plugin fails with `FAILED_PRECONDITION`. See
[gRPC Plugin Management API](grpc_plugin_api.md) for the RPCs, message shapes, and a complete client
workflow.

## Keeping the catalog accurate

A node entry's palette metadata (category, tags, icon, port specs) is regenerated from the live node
classes by `scripts/emit_metadata.py`:

```bash
uv run python -m scripts.emit_metadata --manifest configs/plugins/adaclip.yaml
uv run python -m scripts.emit_metadata --manifest configs/plugins/adaclip.yaml --check   # CI drift guard
```

`--check` exits non-zero when the committed metadata has drifted from the live specs. `data_module`
entries carry no palette metadata and are left untouched.

## See also

- [gRPC Plugin Management API](grpc_plugin_api.md)
- Manifest/capability schemas: `cuvis_ai_schemas.plugin`
