# Model weights: registry, cache contract and provisioning

Every pretrained weight a Cuvis.AI plugin loads is served from a public, ungated mirror under the
`cubert-gmbh` Hugging Face organisation, pinned to a mirror commit and a sha256. This page is the
contract between the three parties that touch those files: the plugins that declare and load them,
`cuvis-ai-core`'s registry and `download-model` CLI that provision them, and the consumers
(CuvisNEXT, the installer helper, notebooks) that read the registry as JSON.

## Where the rows come from

`cuvis_ai_core.data.model_weights.ModelWeights` is a process-wide registry of
`cuvis_ai_schemas.plugin.PluginWeightEntry` rows plus the owning plugin name. A row has one of three
`source`s:

| source | who adds it | when |
|---|---|---|
| `plugin` | the plugin's package `__init__` calls `ModelWeights.register("<name>", WEIGHTS)` | at import, so `resolve()` inside the plugin and inside the offline child needs no manifest on disk |
| `manifest` | `ModelWeights.load_manifests([dirs])` reads the `weights:` block of every plugin manifest | in environments that hold cuvis-ai but not the plugins (the CuvisNEXT venv, the installer helper); every `download-model` subcommand does this for `--plugins-dir` (default: cuvis-ai's packaged `configs/plugins` when cuvis-ai is installed) |
| `dict` | core itself, at import | the Cubert-trained Dinomaly pipelines no plugin owns (`TRAINED_PIPELINES`) |

Precedence: an imported plugin's declaration wins over its manifest row. When the two differ, core
prints one warning on stderr naming both revisions and sets `pin_mismatch: true` on the row (the
manifest is stale: re-run `emit_metadata`). Two manifests declaring one name raise
`ModelRegistryConflict`, and so does any name or alias collision across plugins: names and aliases
share one namespace.

### Declaring weights in a plugin

`cuvis_ai_<plugin>/weights.py` is a side-effect-free module:

```python
from cuvis_ai_schemas.plugin import AuxFile, PluginWeightEntry

WEIGHTS = (
    PluginWeightEntry(
        name="efficienttam_s",                 # registry key == the model_type value
        display_name="RTSAM (EfficientTAM small)",
        summary="Point expansion, propagation (fastest RTSAM variant)",   # <= 60 chars
        used_for=["Point expansion", "Propagation"],                       # USED_FOR_LABELS
        repo_id="cubert-gmbh/efficient-track-anything",
        filename="efficienttam_s.pt",
        revision="3dfd0228d7774b94c24116cf729e03c209ff448a",
        sha256="2b572be30d9e96ee29c8d785fe157c6b079ede7d56fbc8a3671d4120e63c89cd",
        size_bytes=136_375_868,
        license="Apache-2.0",
        license_file="LICENSE",
        aliases=["efficienttam"],              # other hparam values that pick this row
        selected_by="model_type",              # the node hparam whose value picks a row
        default=True,                          # the row a pipeline gets without model_type
        explicit_path_hparams=["model_dir"],   # hparams that bypass the cache entirely
    ),
)
```

and the package `__init__` registers it:

```python
from cuvis_ai_core.data.model_weights import ModelWeights
from cuvis_ai_rtsam2.weights import WEIGHTS

ModelWeights.register("rtsam2", WEIGHTS)
```

Rules the schema enforces: `revision` is 40 lowercase hex, `sha256` 64, `size_bytes > 0`,
`summary` at most 60 characters, `default` requires `selected_by`, a `trained_pipeline` row carries
neither, `license_file` is a bare filename that exists in the mirror repo or `None` when upstream
states no licence for the weights. Core additionally requires every `used_for` label to be one of
`USED_FOR_LABELS` (`Point expansion`, `Propagation`, `Text prompts`, `Segment everything`, `Anomaly
detection`, `Zero-shot`, `Backbone`, `Trained pipeline`), in that display order.

The pins come from `tools/mirror_weights.py plan|upload`, which prints ready-made
`PluginWeightEntry(...)` rows (repo id, revision, sha256, sizes, aux files) for the plugin author to
complete. `tools/mirror_weights.py check` audits every registry row against the Hub.

### Projecting the declaration into the manifest

`uv run python -m scripts.emit_metadata --manifest configs/plugins/<plugin>.yaml` imports
`<package>.weights` (the import root of the first capability; `--weights-module pkg.mod` overrides
it), checks that every `selected_by` and `explicit_path_hparams` value is a constructor parameter of
one of the plugin's node classes, and writes the `weights:` block right after `capabilities:`,
dropping fields at their default. `--check` compares the committed block with the live tuple and
names the drifted rows and fields. A plugin without a `weights` module leaves the block untouched.

Fields core derives for every row (they appear in `list --json`, never in a manifest):

| field | meaning |
|---|---|
| `plugin_default` | `kind == "weights"` and (`default` or `selected_by is None`): the rows a plugin needs out of the box; installers and the pipeline gate filter on it |
| `total_bytes` | `size_bytes` plus every `aux_files[].size_bytes`: what a disk check reserves |
| `family` | the repo-name part of `repo_id`, for grouping |
| `cache_dir_name` | `models--<org>--<repo>`, the repo's folder in a Hugging Face cache |
| `source`, `pin_mismatch` | see above |

## The cache contract

One rule, shared by core, the plugins and CuvisNEXT: a weight is **present** iff

```
<hf_cache>/<cache_dir_name>/snapshots/<pinned revision>/<filename>
```

exists with the registered `size_bytes`, and every aux file exists beside it with its size. Nothing
reads `refs/main`: core writes it after a download so `hf cache ls` and loaders that ask for the
default revision keep working, but a stray newer mirror commit can never shadow the pinned bytes.
`status --verify` adds a sha256 pass on demand; a cache hit downloads nothing and hashes nothing.

```
<hf_cache>/
  models--cubert-gmbh--sam3/
    blobs/<etag>                        the bytes (a download streams into <etag>.incomplete)
    snapshots/<revision>/sam3.pt        the pinned file (link or copy of the blob)
    snapshots/<revision>/config.json    an aux file, same revision
    refs/main                           written, never read
  .cuvis-cache.lock                     held by every writing operation
```

`<hf_cache>` is `hf_cache_dir(os.environ)` from `cuvis_ai_core.orchestrator.model_cache`:
`$HF_HUB_CACHE` → `$HUGGINGFACE_HUB_CACHE` → `$HF_HOME/hub` → `<CUVIS_MODEL_CACHE_DIR or
<cache root>/model_cache>/hf`. The spawner exports exactly that path as `HF_HUB_CACHE` to every
child together with `HF_HUB_OFFLINE=1`, so the provisioner and the offline child always agree.

States `status` reports: `present`, `damaged` (file there, wrong size or an aux file missing),
`partial` (absent, but an `.incomplete` blob from an interrupted download exists; its size is
`partial_bytes`), `absent`.

## Consumption inside a plugin

```python
from cuvis_ai_core.data.model_weights import ModelWeights, ModelWeightsMissingError

path = ModelWeights.resolve("efficienttam_s")              # alias "efficienttam" works too
path = ModelWeights.resolve("sam3", download=False)        # never touches the network
path = ModelWeights.materialize("dinov2_vitb14_reg4", dir) # a fixed path for loaders that need one
```

`resolve` returns the present path, downloads on a miss when allowed (`download=None` means
"unless `HF_HUB_OFFLINE` is set", the child's situation), else raises `ModelWeightsMissingError`
whose message names the provisioning command: `'<name>' (...) is not in the model cache (<dir>).
Provision it with: uv run download-model download <name> (CuvisNEXT: Settings > Cuvis.AI > Model
weights), or pass an explicit checkpoint path.` Registry downloads are anonymous (`token=False`); a
stored Hugging Face login is never sent to a mirror. Library methods never print to stdout.

## The `download-model` CLI

| command | what it does | stdout |
|---|---|---|
| `list [--json]` | the registry | table, or the `model_list` object |
| `status [NAME...] [--json] [--verify] [--cache-dir]` | what the cache holds; never downloads | table, or the `status` object |
| `download NAME... [--json \| --progress-json] [--force] [--cache-dir] [--out]` | provision, verified | one absolute path per line, `status` rows with `downloaded`, or progress events |
| `download --repo-id R --filename F [--revision] [--token]` | the escape hatch for private or custom repos; forwards the token | the path |
| `export --to DIR [NAME...]` | copy present weights into `DIR` in the cache layout plus `cuvis-model-weights.json` (`DIR` is itself a valid `HF_HUB_CACHE`); merges with an earlier export in the same folder | `export` object or lines |
| `import DIR [NAME...]` | copy from an export folder: staged under `<cache>/.import-<uuid>/`, every file size- and sha-checked, then renamed into place; any mismatch leaves the cache unchanged | paths, `status` rows with `imported`, or progress events |
| `remove NAME... \| --dir models--*` | delete a row's files (other rows sharing the repo stay) or an orphaned folder that is a direct child of the cache | `remove` object or lines |
| `schema NAME` | print a shipped JSON Schema | the schema |

Every subcommand takes `--plugins-dir DIR` (repeatable, earlier wins). Exit codes: 0 ok, 1 an
operation failed (`error: <cause>` is the last stderr line), 2 a usage error. Everything human goes
to stderr. Writing operations hold `<root>/.cuvis-cache.lock`; a second instance waits ("another
operation is using the cache; waiting...", and a `waiting` event in progress mode).

`--progress-json` writes one JSON object per line on stdout: `{"event":"waiting"}`,
`{"event":"progress","name":...,"bytes_done":N,"bytes_total":M,"pct":P}` every 500 ms while bytes
change (`pct` and `bytes_done` are `null` when the writer preallocated the file and the amount is
indeterminate), `{"event":"verifying","name":...}` once the bytes are complete and the sha256 pass
runs, then exactly one `{"event":"done","name":...,"path":...}` or `{"event":"error","name":...,
"message":...}` per item; no progress line follows an item's terminal event. `--json` and
`--progress-json` are mutually exclusive.

The `--json` shapes are pinned by the JSON Schemas in `cuvis_ai_core/data/schemas/` (`model_list`,
`status`, `export`, `remove`, `progress_event`, `dataset_list`, `dataset_status`); the tests
validate every payload against them and a consumer vendors copies. `schema_version` is 1; any field
removal or retyping bumps it.

Error sentences a consumer can show verbatim: `'<repo>' was not found or is not public (registry
mis-pin)`, `'<repo>' is gated on Hugging Face; a Cubert mirror must be public and ungated`, `'<repo>'
has no file '<f>' at revision <r>`, `'<name>' is not cached and the network is unavailable
(HF_HUB_OFFLINE=...)`, `Hugging Face is rate-limiting or unavailable (HTTP 429); retry in a few
minutes`, `sha256 mismatch for <path> ...; re-run with --force to re-download`, `another operation
is using the cache (<lock>)`.

## `weights.index.json`

`download-model list --json --plugins-dir cuvis_ai/configs/plugins`, serialised with
`index_json()` (two-space indent, declaration order, rows sorted by name, one trailing newline), is
committed in cuvis-ai as `cuvis_ai/configs/plugins/weights.index.json`. Non-Python consumers (the
CuvisNEXT installer generator, via CMake `string(JSON)`) read it; cuvis-ai CI regenerates and diffs
it, so a stale pin cannot ship silently.

## Datasets

`cuvis_ai_core.data.public_datasets.PublicDatasets` mirrors the same design for the six public
`cubert-gmbh` datasets: typed `DatasetSpec` rows pinned to a dataset revision with exact
`size_bytes`, `file_count`, task `tags` (`Anomaly detection`, `Segmentation`, `Tracking`,
`Statistical`) and `camera` (`XMR`, `X4 SWIR`). The only fact `status` reads is the marker
`<data_dir>/<target_dir>/.cuvis-dataset.json`:

| marker | state |
|---|---|
| directory missing, or holding only hub bookkeeping | `absent` |
| files but no marker | `foreign` (not a Cubert download: never deleted, never written into without `--adopt`) |
| `{"state": "downloading", ...}` or unparsable | `incomplete` (the next download resumes) |
| `{"state": "complete", "revision": <other>, ...}` | `outdated` (`download` refreshes in place, reports files the new revision dropped, deletes them only with `--prune-stale`) |
| `complete` at the pinned revision, every listed file present at its size | `present`, else `damaged` |

`download` writes the `downloading` marker before the first byte and the `complete` marker (with
`files: [{path, size_bytes}]`, the repo's file list at the pinned revision) only after
`snapshot_download` returned every file; markers are written through a `.tmp` rename. `remove`
deletes a folder only when its marker names the expected repo. The `dataset` CLI (`list`, `status`,
`download`, `remove`, `schema`) follows the `download-model` conventions; its progress events count
files (`files_done` / `files_total`), bytes are `null`.

## Security notes

Registry downloads never carry a token, and the child runs with `HF_HUB_OFFLINE=1` and the token
file denied, so a user's Hugging Face login is never sent to a mirror and never reaches untrusted
plugin code. `remove --dir` accepts only a `models--*` folder that is a direct child of the cache;
`import` verifies every byte before anything is renamed into place; `export`/`import` folders and
cache roots are locked while written.
