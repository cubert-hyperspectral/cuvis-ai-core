"""Model-weight registry and provisioning for Cuvis.AI.

Every pretrained weight a Cuvis.AI plugin loads is served from a Cubert-controlled
mirror under the ``cubert-gmbh`` Hugging Face organisation: public, ungated,
byte-identical to upstream, commit-pinned and sha256-verified. The registry is
the single source of truth for where a weight lives and what it is for; plugins
ask :meth:`ModelWeights.resolve` instead of hardcoding an upstream repo id, so
the provisioner and the offline runtime always look in the same cache folder
(``models--cubert-gmbh--<repo>``).

Where the rows come from (:class:`RegisteredWeight.source`):

* ``plugin``: a plugin declares its weights as a tuple of
  ``cuvis_ai_schemas.plugin.PluginWeightEntry`` in a side-effect-free
  ``weights`` module and calls :meth:`ModelWeights.register` from its package
  ``__init__``, so ``resolve()`` inside the plugin (and inside the offline
  child) needs no manifest on disk.
* ``manifest``: :meth:`ModelWeights.load_manifests` reads the ``weights:`` block
  of every plugin manifest in a directory, for environments that hold cuvis-ai
  but not the plugins (the CuvisNEXT venv, the installer helper). An imported
  plugin's declaration wins over its manifest row; a pin difference is reported
  once on stderr and flagged as ``pin_mismatch``.
* ``dict``: the Cubert-trained pipelines below, which no plugin owns.

Two roles share one cache:

* provisioning (trusted, online): ``download-model download <name>`` fetches the
  pinned file(s) into the shared Hugging Face cache and validates sha256;
* consumption (in-process, or in the sandboxed child that runs with
  ``HF_HUB_OFFLINE=1`` and no token): :meth:`ModelWeights.resolve` returns the
  cached path, downloads when online, or raises
  :class:`ModelWeightsMissingError` naming the provisioning command;
  :meth:`ModelWeights.materialize` additionally places a hardlink or copy at a
  fixed path for loaders that cannot read the Hugging Face cache layout.

Cache contract (one rule, three consumers): a weight is present iff
``<hf_cache>/<cache_dir_name>/snapshots/<pinned revision>/<filename>`` exists with
the registered size, and every aux file beside it does too. ``refs/main`` is
written after a download so ``hf cache ls`` and loaders that ask for the default
revision keep working, but nothing here reads it: a stray newer mirror commit
can never shadow the pinned bytes. Cache layout::

    <hf_cache>/
      models--cubert-gmbh--sam3/
        blobs/<etag>                       the bytes (a download streams into <etag>.incomplete)
        snapshots/<revision>/sam3.pt       the pinned file (link or copy of the blob)
        snapshots/<revision>/config.json   an aux file, same revision
        refs/main                          written, never read

Import-light on purpose: the module imports stdlib, ``filelock`` and the schemas
package; ``huggingface_hub`` is imported lazily so a missing optional dependency
yields a clear ``pip install cuvis-ai-core[hf]`` message at call time.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import shutil
import time
import uuid
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from cuvis_ai_schemas.plugin import AuxFile, PluginWeightEntry

from cuvis_ai_core.data._provisioning import (
    SCHEMA_VERSION,
    CacheBusyError,
    ProgressEmitter,
    ProgressPoller,
    log,
    monotonic_start,
    newest_incomplete_blob,
    root_lock,
    sha256_of,
)

_HF_EXTRA_HINT = "pip install cuvis-ai-core[hf]"
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})

HF_ORG = "cubert-gmbh"
"""Hugging Face organisation that hosts every registry entry."""

USED_FOR_LABELS: tuple[str, ...] = (
    "Point expansion",
    "Propagation",
    "Text prompts",
    "Segment everything",
    "Anomaly detection",
    "Zero-shot",
    "Backbone",
    "Trained pipeline",
)
"""The fixed ``used_for`` vocabulary, in display order (CuvisNEXT renders the facet in it)."""

WEIGHTS_HOSTS: tuple[str, ...] = ("huggingface.co", "us.aws.cdn.hf.co")
"""Hosts a download talks to: the hub, and the CDN its ``resolve`` redirects large files to.

Observed on 2026-09-07 for the cubert-gmbh mirrors (``us.aws.cdn.hf.co``, the Xet bridge);
small files come straight from ``huggingface.co``. An installer preflight probes these.
"""

EXPORT_MANIFEST_NAME = "cuvis-model-weights.json"
"""Manifest ``download-model export`` writes beside the exported cache layout."""

_IMPORT_STAGING_PREFIX = ".import-"
_STALE_STAGING_SECONDS = 3600
_INCOMPLETE_SUFFIX = ".incomplete"

Source = Literal["plugin", "manifest", "dict"]


class ModelDownloadError(RuntimeError):
    """Raised when a model download, import, export or removal fails."""


class ModelWeightsMissingError(ModelDownloadError):
    """Raised by :meth:`ModelWeights.resolve` when a weight is not cached and
    downloading is not allowed (offline child, or ``download=False``)."""


class ModelRegistryConflict(ValueError):
    """Two declarations claim one weight name or alias, or one plugin re-registers different pins."""


# ---------------------------------------------------------------------------
# Rows and statuses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RegisteredWeight:
    """One registry row: a manifest weight entry plus the plugin it belongs to.

    ``plugin`` is the plugin's logical (manifest) name, the same name a pipeline
    lists under ``plugins:``. ``source`` says where the row came from and
    ``pin_mismatch`` is set when an imported plugin's declaration differs from
    the manifest row a consumer also loaded (the plugin's pins are in force).
    """

    entry: PluginWeightEntry
    plugin: str
    source: Source
    pin_mismatch: bool = False

    @property
    def name(self) -> str:
        """Registry key."""
        return self.entry.name

    @property
    def cache_dir_name(self) -> str:
        """Folder name of the mirror repo inside a Hugging Face hub cache."""
        return ModelWeights.cache_dir_token(self.entry.repo_id)

    @property
    def family(self) -> str:
        """Mirror repo family (the repo name part of ``repo_id``), for grouping."""
        return self.entry.repo_id.split("/", 1)[1]

    @property
    def total_bytes(self) -> int:
        """Bytes a disk check reserves: the primary file plus every aux file."""
        return self.entry.size_bytes + sum(
            aux.size_bytes for aux in self.entry.aux_files
        )

    @property
    def plugin_default(self) -> bool:
        """The rows a plugin needs out of the box.

        A ``weights`` row that is either the ``default`` of its selector or not
        selected by any hyper-parameter at all; a trained pipeline never is.
        """
        entry = self.entry
        return entry.kind == "weights" and (entry.default or entry.selected_by is None)

    def to_json_dict(self) -> dict[str, Any]:
        """The ``SPEC`` object of the ``--json`` contract (``model_list.schema.json``)."""
        entry = self.entry
        return {
            "name": entry.name,
            "display_name": entry.display_name,
            "summary": entry.summary,
            "used_for": list(entry.used_for),
            "plugin": self.plugin,
            "family": self.family,
            "kind": entry.kind,
            "aux_files": [aux.model_dump(mode="json") for aux in entry.aux_files],
            "repo_id": entry.repo_id,
            "filename": entry.filename,
            "revision": entry.revision,
            "sha256": entry.sha256,
            "size_bytes": entry.size_bytes,
            "total_bytes": self.total_bytes,
            "license": entry.license,
            "license_file": entry.license_file,
            "aliases": list(entry.aliases),
            "selected_by": entry.selected_by,
            "default": entry.default,
            "plugin_default": self.plugin_default,
            "explicit_path_hparams": list(entry.explicit_path_hparams),
            "cache_dir_name": self.cache_dir_name,
            "description": entry.description,
            "source": self.source,
            "pin_mismatch": self.pin_mismatch,
        }


@dataclass(frozen=True, slots=True)
class ModelStatus:
    """What the cache holds for one registry row, without touching the network."""

    weight: RegisteredWeight
    path: Path
    exists: bool
    size_bytes_on_disk: int | None
    size_ok: bool | None
    sha256_ok: bool | None
    partial_bytes: int | None
    aux_ok: bool | None

    @property
    def present(self) -> bool:
        """The pinned snapshot is complete: right size, every aux file in place."""
        return self.exists and bool(self.size_ok) and self.aux_ok is not False

    @property
    def state(self) -> str:
        """``present`` | ``damaged`` | ``partial`` | ``absent``."""
        if self.present:
            return "present"
        if self.exists:
            return "damaged"
        if self.partial_bytes is not None:
            return "partial"
        return "absent"

    def to_json_dict(self) -> dict[str, Any]:
        """The ``STATUS`` object of the ``--json`` contract (``status.schema.json``)."""
        return {
            "state": self.state,
            "present": self.present,
            "path": str(self.path),
            "size_bytes_on_disk": self.size_bytes_on_disk,
            "size_ok": self.size_ok,
            "sha256_ok": self.sha256_ok,
            "aux_ok": self.aux_ok,
            "partial_bytes": self.partial_bytes,
        }


# ---------------------------------------------------------------------------
# Built-in rows: the Cubert-trained pipelines (no plugin owns them)
# ---------------------------------------------------------------------------


def _trained_pipeline(
    name: str,
    display_name: str,
    summary: str,
    *,
    repo_id: str,
    revision: str,
    filename: str,
    sha256: str,
    size_bytes: int,
    yaml_path: str,
    yaml_sha256: str,
    yaml_size: int,
    license: str,
    description: str,
) -> PluginWeightEntry:
    return PluginWeightEntry(
        name=name,
        display_name=display_name,
        summary=summary,
        used_for=["Anomaly detection", "Trained pipeline"],
        kind="trained_pipeline",
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        sha256=sha256,
        size_bytes=size_bytes,
        aux_files=[AuxFile(path=yaml_path, size_bytes=yaml_size, sha256=yaml_sha256)],
        license=license,
        license_file=None,
        description=description,
    )


_LENTILS_REPO = f"{HF_ORG}/XMR_Demo_Industrial_Foreign_Object_Detection_Lentils"
_LENTILS_REVISION = "2e2c592b616d4b82de97ca3a26a061874e002303"

TRAINED_PIPELINES: tuple[PluginWeightEntry, ...] = (
    _trained_pipeline(
        "dinomaly_bedding_all6",
        "Dinomaly, bedding (all six classes)",
        "Trained anomaly detector for the bedding dataset",
        repo_id=f"{HF_ORG}/dinomaly-bedding-all6",
        revision="271c06017bbe3fb7f629c30c5ca197ac01466b6f",
        filename="dinomaly_bedding_all6.pt",
        sha256="f023ceeb629a95138826323fa58ee360a277a1790db3169a31183d3e30145e4f",
        size_bytes=593_813_967,
        yaml_path="dinomaly_bedding_all6.yaml",
        yaml_sha256="90e45074be10d0f1638a195c803e7b3658fa448c2d6a071ad4fb9148ceb35ab3",
        yaml_size=3392,
        license="unspecified",
        description=(
            "Dinomaly pipeline trained on the X4 SWIR bedding foreign-object dataset; the "
            ".pt plus its pipeline yaml, loaded through the pipeline picker's trained-weights "
            "path (cubert-gmbh/dinomaly-bedding-all6)."
        ),
    ),
    _trained_pipeline(
        "dinomaly_lentils_cir",
        "Dinomaly, lentils demo (CIR input)",
        "Trained anomaly detector for the lentils demo, CIR input",
        repo_id=_LENTILS_REPO,
        revision=_LENTILS_REVISION,
        filename="dinomaly_cir_full_pipeline/dinomaly_cir.pt",
        sha256="bf39bd966ed0079f591082e18388648441a21f9ccb4819801154d070597fd601",
        size_bytes=592_005_300,
        yaml_path="dinomaly_cir_full_pipeline/dinomaly_cir.yaml",
        yaml_sha256="e9a8ebbea078bbe1fef8445caaed5b7f4c9030899fdb82ab4143e6197e0ba280",
        yaml_size=2744,
        license="Apache-2.0",
        description=(
            "Dinomaly pipeline trained on the lentils foreign-object demo with a CIR channel "
            "selector in front of the encoder (cubert-gmbh/XMR_Demo_Industrial_Foreign_Object_"
            "Detection_Lentils, dinomaly_cir_full_pipeline)."
        ),
    ),
    _trained_pipeline(
        "dinomaly_lentils_custom",
        "Dinomaly, lentils demo (custom channel selector)",
        "Trained anomaly detector for the lentils demo, custom bands",
        repo_id=_LENTILS_REPO,
        revision=_LENTILS_REVISION,
        filename="dinomaly_custom_selector_full_pipeline/dinomaly_custom.pt",
        sha256="d25f873dbaab0f7d9a80c0fad9a38f646f991f8cd0b41039949cfba95c2da2ef",
        size_bytes=592_011_077,
        yaml_path="dinomaly_custom_selector_full_pipeline/dinomaly_custom.yaml",
        yaml_sha256="9b4a92a38d4e599e1709450314b8a883c9a24a0f7d38c83d8f6b0f8b2185080d",
        yaml_size=2734,
        license="Apache-2.0",
        description=(
            "Dinomaly pipeline trained on the lentils foreign-object demo with a custom channel "
            "selector (cubert-gmbh/XMR_Demo_Industrial_Foreign_Object_Detection_Lentils, "
            "dinomaly_custom_selector_full_pipeline)."
        ),
    ),
    _trained_pipeline(
        "dinomaly_lentils_rgb",
        "Dinomaly, lentils demo (RGB input)",
        "Trained anomaly detector for the lentils demo, RGB input",
        repo_id=_LENTILS_REPO,
        revision=_LENTILS_REVISION,
        filename="dinomaly_rgb_full_pipeline/dinomaly_rgb.pt",
        sha256="94e9d292e86b1527710d8b007ef9380a425161feea02b04d5e94037b097f2bd3",
        size_bytes=592_005_300,
        yaml_path="dinomaly_rgb_full_pipeline/dinomaly_rgb.yaml",
        yaml_sha256="35a5df5589f4c232374d03f92c407455f74fe47b182b47d956fdbc8611c038d5",
        yaml_size=2770,
        license="Apache-2.0",
        description=(
            "Dinomaly pipeline trained on the lentils foreign-object demo with a false-RGB "
            "front end (cubert-gmbh/XMR_Demo_Industrial_Foreign_Object_Detection_Lentils, "
            "dinomaly_rgb_full_pipeline)."
        ),
    ),
)
"""Cubert-trained pipelines: registered under the ``dinomaly`` plugin with source ``dict``."""

_TRAINED_PIPELINES_PLUGIN = "dinomaly"


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


class ModelWeights:
    """Registry, resolver and provisioning downloader for Cuvis.AI model weights.

    Class-level state on purpose: plugins register at import, the CLI loads
    manifests, and every consumer in the process sees one registry. Library
    methods never print to stdout (progress goes to stderr or to a
    :class:`ProgressEmitter`), so they are safe inside node constructors.
    """

    _registry: dict[str, RegisteredWeight] = {}
    _keys: dict[str, str] = {}  # every name and alias -> the owning name
    _warned_mismatch: set[str] = set()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    @classmethod
    def register(
        cls,
        plugin: str,
        entries: Iterable[PluginWeightEntry],
        *,
        source: Source = "plugin",
    ) -> None:
        """Register a plugin's weight declarations (called from the plugin's ``__init__``).

        Idempotent on full entry equality, so a package imported twice is fine.
        Names and aliases share one namespace across every plugin: a key another
        row already claims raises :class:`ModelRegistryConflict`, as does one
        plugin re-registering a name with different content. A plugin's
        declaration wins over a manifest row loaded earlier for the same name;
        differing pins are reported once on stderr and flagged ``pin_mismatch``.
        """
        for entry in entries:
            cls._check_vocabulary(entry)
            existing = cls._registry.get(entry.name)
            if existing is not None and existing.source == "manifest":
                mismatch = existing.entry != entry or existing.plugin != plugin
                if mismatch:
                    cls._warn_pin_mismatch(
                        entry.name, plugin, entry.revision, existing.entry.revision
                    )
                cls._unindex(existing)
                cls._index(
                    RegisteredWeight(entry, plugin, source, pin_mismatch=mismatch)
                )
                continue
            if existing is not None:
                if existing.entry == entry and existing.plugin == plugin:
                    continue
                raise ModelRegistryConflict(
                    f"weight '{entry.name}' is already registered by plugin "
                    f"'{existing.plugin}' with different content; plugin '{plugin}' cannot "
                    "register it again"
                )
            cls._check_namespace(entry, plugin)
            cls._index(RegisteredWeight(entry, plugin, source))

    @classmethod
    def load_manifests(cls, plugins_dirs: Iterable[str | Path]) -> int:
        """Register the ``weights:`` blocks of every plugin manifest in ``plugins_dirs``.

        Directories are scanned in argument order, manifests sorted by file name,
        and a plugin name declared twice across the set is an error (the same
        rule the pipeline resolver applies). A row whose plugin is imported in
        this process keeps the plugin's pins; a pin difference is reported once
        on stderr and flagged ``pin_mismatch``. Two manifests declaring one
        weight name raise :class:`ModelRegistryConflict`. Returns the number of
        rows added.
        """
        from cuvis_ai_core.utils.plugin_resolver import _build_catalog

        catalog = _build_catalog([Path(d) for d in plugins_dirs])
        added = 0
        for plugin in sorted(catalog):
            for entry in catalog[plugin].weights:
                cls._check_vocabulary(entry)
                existing = cls._registry.get(entry.name)
                if existing is None:
                    cls._check_namespace(entry, plugin)
                    cls._index(RegisteredWeight(entry, plugin, "manifest"))
                    added += 1
                    continue
                if existing.source == "manifest":
                    if existing.entry == entry and existing.plugin == plugin:
                        continue  # the same manifest read again
                    raise ModelRegistryConflict(
                        f"weight '{entry.name}' is declared by the manifests of both "
                        f"'{existing.plugin}' and '{plugin}'"
                    )
                if existing.entry != entry or existing.plugin != plugin:
                    cls._warn_pin_mismatch(
                        entry.name,
                        existing.plugin,
                        existing.entry.revision,
                        entry.revision,
                    )
                    cls._registry[entry.name] = dataclasses.replace(
                        existing, pin_mismatch=True
                    )
        return added

    @classmethod
    def reset(cls) -> None:
        """Forget every row and re-register the built-in trained pipelines (tests)."""
        cls._registry.clear()
        cls._keys.clear()
        cls._warned_mismatch.clear()
        cls.register(_TRAINED_PIPELINES_PLUGIN, TRAINED_PIPELINES, source="dict")

    @classmethod
    def _check_vocabulary(cls, entry: PluginWeightEntry) -> None:
        unknown = [label for label in entry.used_for if label not in USED_FOR_LABELS]
        if unknown:
            raise ModelRegistryConflict(
                f"weight '{entry.name}': used_for labels {unknown} are not in the shared "
                f"vocabulary {list(USED_FOR_LABELS)}"
            )

    @classmethod
    def _check_namespace(cls, entry: PluginWeightEntry, plugin: str) -> None:
        for key in (entry.name, *entry.aliases):
            owner = cls._keys.get(key)
            if owner is not None:
                other = cls._registry[owner]
                raise ModelRegistryConflict(
                    f"weight key '{key}' of '{entry.name}' (plugin '{plugin}') is already "
                    f"taken by '{owner}' (plugin '{other.plugin}'); names and aliases share "
                    "one namespace across plugins"
                )

    @classmethod
    def _index(cls, weight: RegisteredWeight) -> None:
        cls._registry[weight.name] = weight
        for key in (weight.name, *weight.entry.aliases):
            cls._keys[key] = weight.name

    @classmethod
    def _unindex(cls, weight: RegisteredWeight) -> None:
        cls._registry.pop(weight.name, None)
        for key in (weight.name, *weight.entry.aliases):
            if cls._keys.get(key) == weight.name:
                del cls._keys[key]

    @classmethod
    def _warn_pin_mismatch(
        cls, name: str, plugin: str, plugin_revision: str, manifest_revision: str
    ) -> None:
        if name in cls._warned_mismatch:
            return
        cls._warned_mismatch.add(name)
        log(
            f"warning: weight '{name}': the installed plugin '{plugin}' pins revision "
            f"{plugin_revision[:12]} but its manifest pins {manifest_revision[:12]}; using the "
            "plugin's pins. Re-run emit_metadata on the manifest."
        )

    # ------------------------------------------------------------------
    # Lookup and listing
    # ------------------------------------------------------------------

    @classmethod
    def names(cls) -> list[str]:
        """Registry keys, sorted."""
        return sorted(cls._registry)

    @classmethod
    def get(cls, name_or_alias: str) -> RegisteredWeight:
        """The row for a registry name or one of its aliases.

        Raises:
            ModelDownloadError: unknown key (the message lists the known names).
        """
        owner = cls._keys.get(name_or_alias)
        if owner is None:
            raise ModelDownloadError(
                f"Unknown model '{name_or_alias}'. Known: {', '.join(cls.names())}. "
                "Or pass --repo-id and --filename explicitly."
            )
        return cls._registry[owner]

    @classmethod
    def rows(cls) -> list[RegisteredWeight]:
        """Every row, sorted by name."""
        return [cls._registry[name] for name in cls.names()]

    @classmethod
    def entries(cls) -> list[dict[str, Any]]:
        """Registry as ``SPEC`` dicts sorted by name (the ``models`` array of ``list --json``)."""
        return [row.to_json_dict() for row in cls.rows()]

    @classmethod
    def list_payload(cls) -> dict[str, Any]:
        """The complete ``list --json`` object (also the ``weights.index.json`` content)."""
        return {
            "schema_version": SCHEMA_VERSION,
            "used_for_labels": list(USED_FOR_LABELS),
            "weights_hosts": list(WEIGHTS_HOSTS),
            "models": cls.entries(),
        }

    @staticmethod
    def cache_dir_token(repo_id: str) -> str:
        """Folder name of ``repo_id`` inside a Hugging Face hub cache."""
        return "models--" + repo_id.replace("/", "--")

    @classmethod
    def default_cache_dir(cls) -> Path:
        """The Hugging Face cache the offline child reads (``hf_cache_dir(os.environ)``)."""
        from cuvis_ai_core.orchestrator.model_cache import hf_cache_dir

        return hf_cache_dir(os.environ)

    @classmethod
    def cache_repo_dir(cls, name: str, cache_dir: str | Path | None = None) -> Path:
        """``<cache>/models--<org>--<repo>`` for a registry row."""
        return cls.resolve_cache_dir(cache_dir) / cls.get(name).cache_dir_name

    @classmethod
    def missing_guidance(cls, name: str, cache_dir: Path) -> str:
        """The sentence a consumer sees when a weight is not provisioned."""
        weight = cls._registry.get(cls._keys.get(name, ""), None)
        what = (
            f"'{name}' ({weight.entry.display_name}, {weight.entry.repo_id}/{weight.entry.filename})"
            if weight
            else f"'{name}'"
        )
        return (
            f"{what} is not in the model cache ({cache_dir}). Provision it with: "
            f"uv run download-model download {name} (CuvisNEXT: Settings > Cuvis.AI > "
            "Model weights), or pass an explicit checkpoint path."
        )

    # ------------------------------------------------------------------
    # Status (never touches the network)
    # ------------------------------------------------------------------

    @classmethod
    def snapshot_path(cls, weight: RegisteredWeight, cache_dir: Path) -> Path:
        """``<cache>/<cache_dir_name>/snapshots/<revision>/<filename>``."""
        entry = weight.entry
        return (
            cache_dir
            / weight.cache_dir_name
            / "snapshots"
            / entry.revision
            / entry.filename
        )

    @classmethod
    def status(
        cls, name: str, cache_dir: str | Path | None = None, *, verify: bool = False
    ) -> ModelStatus:
        """What the cache holds for ``name``: presence by pinned path + size, sha on request."""
        weight = cls.get(name)
        root = cls.resolve_cache_dir(cache_dir)
        return cls._status_of(weight, root, verify=verify)

    @classmethod
    def status_all(
        cls,
        names: Sequence[str] | None = None,
        cache_dir: str | Path | None = None,
        *,
        verify: bool = False,
    ) -> list[ModelStatus]:
        """Statuses for ``names`` (default: every row), in registry order."""
        root = cls.resolve_cache_dir(cache_dir)
        cls._sweep_stale_staging(root)
        rows = [cls.get(n) for n in names] if names else cls.rows()
        return [cls._status_of(row, root, verify=verify) for row in rows]

    @classmethod
    def _status_of(
        cls, weight: RegisteredWeight, root: Path, *, verify: bool
    ) -> ModelStatus:
        entry = weight.entry
        path = cls.snapshot_path(weight, root)
        size_on_disk = cls._file_size(path)
        exists = size_on_disk is not None
        size_ok = (size_on_disk == entry.size_bytes) if exists else None
        aux_ok: bool | None = None
        if entry.aux_files:
            aux_ok = all(
                cls._file_size(path.parent / aux.path) == aux.size_bytes
                for aux in entry.aux_files
            )
        sha_ok: bool | None = None
        if verify and exists and size_ok:
            sha_ok = sha256_of(path) == entry.sha256 and all(
                sha256_of(path.parent / aux.path) == aux.sha256
                for aux in entry.aux_files
                if (path.parent / aux.path).is_file()
            )
        partial: int | None = None
        if not exists:
            blob = newest_incomplete_blob(root / weight.cache_dir_name / "blobs", 0.0)
            partial = cls._file_size(blob) if blob is not None else None
        return ModelStatus(
            weight=weight,
            path=path,
            exists=exists,
            size_bytes_on_disk=size_on_disk,
            size_ok=size_ok,
            sha256_ok=sha_ok,
            partial_bytes=partial,
            aux_ok=aux_ok,
        )

    @staticmethod
    def _file_size(path: Path | None) -> int | None:
        if path is None:
            return None
        try:
            return path.stat().st_size if path.is_file() else None
        except OSError:
            return None

    # ------------------------------------------------------------------
    # Consumption
    # ------------------------------------------------------------------

    @classmethod
    def resolve(
        cls,
        name: str,
        *,
        download: bool | None = None,
        cache_dir: str | Path | None = None,
    ) -> Path:
        """Return the local path of a registry weight, fetching it if allowed.

        A present weight (pinned path, registered size, aux files in place) is
        returned as is. On a miss, ``download=True`` fetches it (sha256-verified,
        anonymously); ``download=False`` raises :class:`ModelWeightsMissingError`.
        The default ``download=None`` means "download unless ``HF_HUB_OFFLINE`` is
        set", which is exactly the sandboxed child's situation. Accepts an alias.
        """
        weight = cls.get(name)
        root = cls.resolve_cache_dir(cache_dir)
        current = cls._status_of(weight, root, verify=False)
        if current.present:
            return current.path
        if download is None:
            download = not cls._hf_offline()
        if not download:
            raise ModelWeightsMissingError(cls.missing_guidance(weight.name, root))
        with cls._locked(root):
            return cls._download_weight(weight, root, force=False)

    @classmethod
    def materialize(
        cls,
        name: str,
        dest_dir: str | Path,
        *,
        filename: str | None = None,
        download: bool | None = None,
        cache_dir: str | Path | None = None,
    ) -> Path:
        """Place a registry weight at ``dest_dir/<filename>`` and return that path.

        For loaders that read a fixed path instead of the Hugging Face cache
        (a vendored CLIP loader, anomalib's DINOv2 loader). An existing
        destination is returned untouched, so a seeded directory never triggers
        a cache lookup or a download. Otherwise the weight is resolved (see
        :meth:`resolve`) and hardlinked into place; when a hardlink is not
        possible (different volume) it is copied through a ``.part`` file and
        renamed, so an interrupted copy never leaves a truncated destination.
        """
        weight = cls.get(name)
        dest_dir = Path(dest_dir)
        dst = dest_dir / (filename or Path(weight.entry.filename).name)
        if dst.exists():
            return dst
        src = cls.resolve(name, download=download, cache_dir=cache_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, dst)
        except OSError:
            if dst.exists():  # another process finished first
                return dst
            part = dst.with_suffix(dst.suffix + ".part")
            try:
                shutil.copyfile(src, part)
                os.replace(part, dst)
            except BaseException:
                part.unlink(missing_ok=True)
                raise
        return dst

    # ------------------------------------------------------------------
    # Provisioning
    # ------------------------------------------------------------------

    @classmethod
    def download(
        cls,
        name: str,
        cache_dir: str | Path | None = None,
        *,
        force: bool = False,
        progress: ProgressEmitter | None = None,
    ) -> Path:
        """Provision one registry weight into the shared cache (verified, anonymous)."""
        weight = cls.get(name)
        root = cls.resolve_cache_dir(cache_dir)
        with cls._locked(root, progress):
            return cls._download_weight(weight, root, force=force, progress=progress)

    @classmethod
    def download_many(
        cls,
        names: Sequence[str],
        cache_dir: str | Path | None = None,
        *,
        force: bool = False,
        progress: ProgressEmitter | None = None,
    ) -> list[Path]:
        """Provision several weights in argument order, stopping at the first failure."""
        weights = [
            cls.get(n) for n in names
        ]  # every unknown name fails before any download
        root = cls.resolve_cache_dir(cache_dir)
        paths: list[Path] = []
        with cls._locked(root, progress):
            for weight in weights:
                paths.append(
                    cls._download_weight(weight, root, force=force, progress=progress)
                )
        return paths

    @classmethod
    def download_model(
        cls,
        name: str | None = None,
        *,
        repo_id: str | None = None,
        filename: str | None = None,
        revision: str | None = None,
        sha256: str | None = None,
        token: str | None = None,
        cache_dir: str | Path | None = None,
        out: str | Path | None = None,
        force: bool = False,
        progress: ProgressEmitter | None = None,
    ) -> Path:
        """Download a registry weight, or an explicit ``repo_id`` + ``filename``.

        A registry ``name`` is fetched anonymously from its mirror (``token`` is
        ignored: the mirrors are public and a stored login must never be sent to
        them). The explicit form is the escape hatch for private or custom repos
        and forwards ``token`` (else ``$HF_TOKEN``). ``cache_dir`` defaults to the
        HF cache the child reads; ``out`` additionally copies the resolved file to
        a standalone location (e.g. a node ``checkpoint_path``).

        Raises:
            ModelDownloadError: unknown name, a hub error, or a sha256 mismatch.
        """
        root = cls.resolve_cache_dir(cache_dir)
        if name is not None:
            if repo_id or filename or revision or sha256:
                raise ModelDownloadError(
                    "Pass either a registry NAME or --repo-id/--filename, not both."
                )
            weight = cls.get(name)
            with cls._locked(root, progress):
                resolved = cls._download_weight(
                    weight, root, force=force, progress=progress
                )
        else:
            if not repo_id or not filename:
                raise ModelDownloadError(
                    "Need a registry name or both --repo-id and --filename."
                )
            with cls._locked(root, progress):
                resolved = cls._download_custom(
                    repo_id,
                    filename,
                    revision=revision,
                    sha256=sha256,
                    token=token or os.getenv("HF_TOKEN"),
                    cache_dir=root,
                    force=force,
                    progress=progress,
                )
        if out is not None:
            out = Path(out)
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.resolve() != resolved.resolve():
                shutil.copy2(resolved, out)
            resolved = out
        log(f"Ready: {resolved}")
        return resolved

    @classmethod
    def _download_weight(
        cls,
        weight: RegisteredWeight,
        root: Path,
        *,
        force: bool,
        progress: ProgressEmitter | None = None,
    ) -> Path:
        """Fetch a registry row (primary + aux files) into ``root``; caller holds the lock."""
        entry = weight.entry
        if not force:
            current = cls._status_of(weight, root, verify=False)
            if current.present:
                log(f"{weight.name}: present at {current.path}")
                return current.path
        hf_hub_download, _ = cls._require_hf_hub()
        root.mkdir(parents=True, exist_ok=True)
        blobs_dir = root / weight.cache_dir_name / "blobs"
        files: list[tuple[str, str, int]] = [
            (entry.filename, entry.sha256, entry.size_bytes)
        ]
        files.extend((aux.path, aux.sha256, aux.size_bytes) for aux in entry.aux_files)
        primary: Path | None = None
        for fname, sha, size in files:
            log(f"Fetching {entry.repo_id}/{fname}@{entry.revision[:12]} -> {root}")
            started = monotonic_start()
            poller = (
                ProgressPoller(
                    progress,
                    weight.name,
                    size,
                    lambda: newest_incomplete_blob(blobs_dir, started),
                ).start()
                if progress is not None
                else None
            )
            try:
                fetched = Path(
                    hf_hub_download(
                        repo_id=entry.repo_id,
                        filename=fname,
                        revision=entry.revision,
                        token=False,
                        cache_dir=str(root),
                        force_download=force,
                    )
                )
            except Exception as exc:  # mapped below; hub errors are many classes
                if poller is not None:
                    poller.stop()
                raise cls._map_hf_error(
                    exc, weight.name, entry.repo_id, fname, entry.revision
                ) from exc
            finally:
                if poller is not None:
                    poller.stop()
            if progress is not None:
                progress.verifying(weight.name)
            cls._validate_sha(fetched, expected=sha)
            if primary is None:
                primary = fetched
        assert primary is not None
        cls._alias_default_revision(primary)
        return primary

    @classmethod
    def _download_custom(
        cls,
        repo_id: str,
        filename: str,
        *,
        revision: str | None,
        sha256: str | None,
        token: str | None,
        cache_dir: Path,
        force: bool,
        progress: ProgressEmitter | None,
    ) -> Path:
        """The explicit-repo escape hatch: any repo, any file, optional pins, a token allowed."""
        hf_hub_download, _ = cls._require_hf_hub()
        cache_dir.mkdir(parents=True, exist_ok=True)
        label = f"{repo_id}/{filename}"
        log(
            f"Fetching {label}"
            + (f"@{revision}" if revision else "")
            + f" -> {cache_dir}"
        )
        blobs_dir = cache_dir / cls.cache_dir_token(repo_id) / "blobs"
        started = monotonic_start()
        poller = (
            ProgressPoller(
                progress,
                label,
                None,
                lambda: newest_incomplete_blob(blobs_dir, started),
            ).start()
            if progress is not None
            else None
        )
        try:
            resolved = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    revision=revision,
                    token=token,
                    cache_dir=str(cache_dir),
                    force_download=force,
                )
            )
        except Exception as exc:
            raise cls._map_hf_error(
                exc, label, repo_id, filename, revision, custom=True
            ) from exc
        finally:
            if poller is not None:
                poller.stop()
        if progress is not None:
            progress.verifying(label)
        cls._validate_sha(resolved, expected=sha256)
        cls._alias_default_revision(resolved)
        return resolved

    # ------------------------------------------------------------------
    # Export / import (training rooms, air-gapped sites)
    # ------------------------------------------------------------------

    @classmethod
    def export_to(
        cls,
        export_dir: str | Path,
        names: Sequence[str] | None = None,
        cache_dir: str | Path | None = None,
        *,
        progress: ProgressEmitter | None = None,
    ) -> list[dict[str, Any]]:
        """Copy present weights into ``export_dir`` in the cache layout, plus a manifest.

        The export directory is itself a valid ``HF_HUB_CACHE``: each model lands
        under ``models--<org>--<repo>/snapshots/<revision>/`` with ``refs/main``
        written, and ``cuvis-model-weights.json`` records the rows and the sha256
        of every copied file for the importer. Absent models are reported
        (``exported: false``) and skipped. Returns one result dict per row.
        """
        export_dir = Path(export_dir)
        root = cls.resolve_cache_dir(cache_dir)
        rows = [cls.get(n) for n in names] if names else cls.rows()
        results: list[dict[str, Any]] = []
        exported: list[dict[str, Any]] = []
        with cls._locked(export_dir, progress):
            export_dir.mkdir(parents=True, exist_ok=True)
            for weight in rows:
                current = cls._status_of(weight, root, verify=False)
                if not current.present:
                    log(f"{weight.name}: not present in {root}, skipped")
                    results.append(
                        {**weight.to_json_dict(), "exported": False, "path": None}
                    )
                    continue
                dst_primary = cls.snapshot_path(weight, export_dir)
                copied: list[dict[str, Any]] = []
                for rel, sha, size in cls._files_of(weight):
                    src = current.path.parent / rel
                    dst = dst_primary.parent / rel
                    cls._copy_with_progress(src, dst, weight.name, size, progress)
                    if progress is not None:
                        progress.verifying(weight.name)
                    digest = sha256_of(dst)
                    if digest != sha:
                        dst.unlink(missing_ok=True)
                        raise ModelDownloadError(
                            f"sha256 mismatch after copying {src} to {dst}: expected {sha}, got "
                            f"{digest}. The cached file does not match the pinned weights; re-run "
                            "download-model download --force."
                        )
                    copied.append({"path": rel, "size_bytes": size, "sha256": sha})
                cls._alias_default_revision(dst_primary)
                exported.append({**weight.to_json_dict(), "files": copied})
                results.append(
                    {
                        **weight.to_json_dict(),
                        "exported": True,
                        "path": str(dst_primary),
                    }
                )
                if progress is not None:
                    progress.done(weight.name, dst_primary)
            # Merge with what an earlier export already recorded in this folder, so a
            # folder filled in several runs describes every model it holds.
            manifest_path = export_dir / EXPORT_MANIFEST_NAME
            merged: dict[str, dict[str, Any]] = {}
            if manifest_path.is_file():
                try:
                    previous = json.loads(manifest_path.read_text(encoding="utf-8"))
                    merged = {m["name"]: m for m in previous.get("models", [])}
                except (ValueError, KeyError, TypeError, AttributeError):
                    merged = {}
            merged.update({m["name"]: m for m in exported})
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "models": [merged[name] for name in sorted(merged)],
            }
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return results

    @classmethod
    def import_from(
        cls,
        source_dir: str | Path,
        names: Sequence[str] | None = None,
        cache_dir: str | Path | None = None,
        *,
        progress: ProgressEmitter | None = None,
    ) -> list[dict[str, Any]]:
        """Import weights from an export directory: two passes, all or nothing.

        Every matching model that is not already present is copied into a
        staging folder ``<cache>/.import-<uuid>/`` on the cache's own volume and
        verified there (size and sha256 of every file); only when all pass is
        each snapshot renamed into place (a same-volume rename). Any mismatch or
        error removes the staging folder and leaves the cache exactly as it was.
        Returns one result dict per row of the export manifest that matched.
        """
        source_dir = Path(source_dir)
        root = cls.resolve_cache_dir(cache_dir)
        manifest_path = source_dir / EXPORT_MANIFEST_NAME
        if not manifest_path.is_file():
            raise ModelDownloadError(
                f"{source_dir} is not an exported weights folder (no {EXPORT_MANIFEST_NAME})."
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            exported_rows = list(manifest["models"])
        except (ValueError, KeyError, TypeError) as exc:
            raise ModelDownloadError(
                f"{manifest_path} is not a valid export manifest: {exc}"
            ) from exc
        wanted = set(names) if names else None
        results: list[dict[str, Any]] = []
        with cls._locked(root, progress):
            root.mkdir(parents=True, exist_ok=True)
            staging = root / f"{_IMPORT_STAGING_PREFIX}{uuid.uuid4().hex}"
            staging.mkdir()
            moves: list[tuple[RegisteredWeight, Path]] = []
            try:
                for row in exported_rows:
                    name = row.get("name")
                    if wanted is not None and name not in wanted:
                        continue
                    weight = cls._registry.get(cls._keys.get(str(name), ""))
                    if weight is None:
                        log(f"{name}: not in this version's registry, skipped")
                        results.append(
                            {**row, "imported": False, "reason": "unknown model"}
                        )
                        continue
                    if cls._status_of(weight, root, verify=False).present:
                        results.append(
                            {
                                **weight.to_json_dict(),
                                "imported": False,
                                "reason": "present",
                            }
                        )
                        continue
                    src_primary = cls.snapshot_path(weight, source_dir)
                    if not src_primary.is_file():
                        raise ModelDownloadError(
                            f"{weight.name}: {src_primary} is missing from the export folder"
                        )
                    staged_primary = cls.snapshot_path(weight, staging)
                    for rel, sha, size in cls._files_of(weight):
                        src = src_primary.parent / rel
                        dst = staged_primary.parent / rel
                        cls._copy_with_progress(src, dst, weight.name, size, progress)
                        if progress is not None:
                            progress.verifying(weight.name)
                        if cls._file_size(dst) != size or sha256_of(dst) != sha:
                            raise ModelDownloadError(
                                f"Import of {weight.entry.display_name} failed: checksum mismatch "
                                f"on {rel}. Nothing was changed."
                            )
                    moves.append((weight, staged_primary))
                # Second pass: every staged file verified; rename snapshots into place.
                for weight, staged_primary in moves:
                    final_primary = cls.snapshot_path(weight, root)
                    final_primary.parent.parent.mkdir(parents=True, exist_ok=True)
                    if final_primary.parent.exists():
                        for item in staged_primary.parent.rglob("*"):
                            if item.is_file():
                                target = final_primary.parent / item.relative_to(
                                    staged_primary.parent
                                )
                                target.parent.mkdir(parents=True, exist_ok=True)
                                os.replace(item, target)
                    else:
                        os.replace(staged_primary.parent, final_primary.parent)
                    cls._alias_default_revision(final_primary)
                    results.append(
                        {
                            **weight.to_json_dict(),
                            "imported": True,
                            "path": str(final_primary),
                        }
                    )
                    if progress is not None:
                        progress.done(weight.name, final_primary)
            finally:
                shutil.rmtree(staging, ignore_errors=True)
        return results

    @classmethod
    def _sweep_stale_staging(cls, root: Path) -> None:
        """Remove ``.import-*`` folders older than an hour (a crash mid-import)."""
        if not root.is_dir():
            return
        cutoff = time.time() - _STALE_STAGING_SECONDS
        for entry in root.glob(f"{_IMPORT_STAGING_PREFIX}*"):
            try:
                if entry.is_dir() and entry.stat().st_mtime < cutoff:
                    shutil.rmtree(entry, ignore_errors=True)
            except OSError:
                continue

    # ------------------------------------------------------------------
    # Removal
    # ------------------------------------------------------------------

    @classmethod
    def remove(cls, name: str, cache_dir: str | Path | None = None) -> int:
        """Delete a registry weight's files from the cache; returns the bytes freed.

        Deletes the pinned snapshot files (primary + aux) and the blobs they
        point at when no other snapshot still references them, then the repo
        folder if nothing is left in it. Other rows sharing the repo (the
        EfficientTAM variants) are untouched.
        """
        weight = cls.get(name)
        root = cls.resolve_cache_dir(cache_dir)
        freed = 0
        with cls._locked(root):
            primary = cls.snapshot_path(weight, root)
            repo_dir = root / weight.cache_dir_name
            for rel, _sha, _size in cls._files_of(weight):
                freed += cls._remove_snapshot_file(primary.parent / rel, repo_dir)
            cls._prune_repo_dir(repo_dir)
        return freed

    @classmethod
    def remove_dir(cls, dirname: str, cache_dir: str | Path | None = None) -> int:
        """Delete a whole ``models--*`` folder that is a direct child of the cache (orphans)."""
        root = cls.resolve_cache_dir(cache_dir)
        if (
            "/" in dirname
            or "\\" in dirname
            or not dirname.startswith("models--")
            or ".." in dirname
        ):
            raise ModelDownloadError(
                f"refusing to remove '{dirname}': only a 'models--*' folder directly inside the "
                "cache can be removed"
            )
        target = root / dirname
        if not target.is_dir():
            raise ModelDownloadError(f"{target} is not a directory in the cache")
        with cls._locked(root):
            freed = sum(p.stat().st_size for p in target.rglob("*") if p.is_file())
            shutil.rmtree(target)
        return freed

    @classmethod
    def _remove_snapshot_file(cls, path: Path, repo_dir: Path) -> int:
        """Delete one snapshot file and its blob when unreferenced; returns bytes freed."""
        if not path.is_file() and not path.is_symlink():
            return 0
        size = cls._file_size(path) or 0
        blob: Path | None = None
        if path.is_symlink():
            try:
                blob = path.resolve()
            except OSError:
                blob = None
        path.unlink()
        freed = size
        if blob is not None and blob.is_file() and (repo_dir / "blobs") in blob.parents:
            still_referenced = (
                any(
                    other.is_symlink() and other.resolve() == blob
                    for other in (repo_dir / "snapshots").rglob("*")
                )
                if (repo_dir / "snapshots").is_dir()
                else False
            )
            if not still_referenced:
                freed += cls._file_size(blob) or 0
                blob.unlink(missing_ok=True)
        return freed

    @classmethod
    def _prune_repo_dir(cls, repo_dir: Path) -> None:
        """Drop the repo folder once no snapshot file is left in it."""
        snapshots = repo_dir / "snapshots"
        if snapshots.is_dir() and any(
            p.is_file() or p.is_symlink() for p in snapshots.rglob("*")
        ):
            return
        shutil.rmtree(repo_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @classmethod
    def _files_of(cls, weight: RegisteredWeight) -> list[tuple[str, str, int]]:
        """``(path relative to the snapshot dir, sha256, size)`` for the primary and every aux file."""
        entry = weight.entry
        files = [(entry.filename, entry.sha256, entry.size_bytes)]
        files.extend((aux.path, aux.sha256, aux.size_bytes) for aux in entry.aux_files)
        return files

    @classmethod
    def _copy_with_progress(
        cls,
        src: Path,
        dst: Path,
        name: str,
        size: int,
        progress: ProgressEmitter | None,
    ) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        poller = (
            ProgressPoller(progress, name, size, lambda: dst).start()
            if progress is not None
            else None
        )
        try:
            shutil.copyfile(src, dst)
        finally:
            if poller is not None:
                poller.stop()

    @classmethod
    def resolve_cache_dir(cls, cache_dir: str | Path | None) -> Path:
        """``cache_dir`` as a path, or the default cache when None."""
        return Path(cache_dir) if cache_dir is not None else cls.default_cache_dir()

    @classmethod
    @contextmanager
    def _locked(
        cls, root: Path, progress: ProgressEmitter | None = None
    ) -> Iterator[None]:
        """The cache-root lock every writing operation holds; reports the wait."""

        def _on_wait() -> None:
            log("another operation is using the cache; waiting...")
            if progress is not None:
                progress.waiting()

        try:
            with root_lock(root, on_wait=_on_wait):
                yield
        except CacheBusyError as exc:
            raise ModelDownloadError(str(exc)) from exc

    @staticmethod
    def _hf_offline() -> bool:
        return os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in _TRUE_VALUES

    @staticmethod
    def _require_hf_hub() -> tuple[Callable[..., str], dict[str, type[BaseException]]]:
        try:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.utils import (
                EntryNotFoundError,
                GatedRepoError,
                HfHubHTTPError,
                LocalEntryNotFoundError,
                RepositoryNotFoundError,
                RevisionNotFoundError,
            )
        except ImportError as exc:
            raise ModelDownloadError(
                f"huggingface_hub is not installed. Install with: {_HF_EXTRA_HINT}"
            ) from exc
        return hf_hub_download, {
            "EntryNotFoundError": EntryNotFoundError,
            "GatedRepoError": GatedRepoError,
            "HfHubHTTPError": HfHubHTTPError,
            "LocalEntryNotFoundError": LocalEntryNotFoundError,
            "RepositoryNotFoundError": RepositoryNotFoundError,
            "RevisionNotFoundError": RevisionNotFoundError,
        }

    @classmethod
    def _map_hf_error(
        cls,
        exc: BaseException,
        name: str,
        repo_id: str,
        filename: str,
        revision: str | None,
        *,
        custom: bool = False,
    ) -> ModelDownloadError:
        """Translate a hub exception into the sentence a user (or CuvisNEXT) sees."""
        if isinstance(exc, ModelDownloadError):
            return exc
        _, errors = cls._require_hf_hub()
        where = "registry mis-pin" if not custom else "check --repo-id / --revision"
        if isinstance(exc, errors["LocalEntryNotFoundError"]):
            offline = os.environ.get("HF_HUB_OFFLINE", "")
            return ModelDownloadError(
                f"'{name}' is not cached and the network is unavailable "
                f"(HF_HUB_OFFLINE={offline!r}). ({exc})"
            )
        if isinstance(exc, errors["GatedRepoError"]):
            if custom:
                return ModelDownloadError(
                    f"Access to '{repo_id}' is gated. Accept the model licence at "
                    f"https://huggingface.co/{repo_id} and pass --token. ({exc})"
                )
            return ModelDownloadError(
                f"'{repo_id}' is gated on Hugging Face; a Cubert mirror must be public and "
                f"ungated ({where}). ({exc})"
            )
        if isinstance(exc, errors["RepositoryNotFoundError"]):
            return ModelDownloadError(
                f"'{repo_id}' was not found or is not public ({where}). ({exc})"
            )
        if isinstance(
            exc, (errors["RevisionNotFoundError"], errors["EntryNotFoundError"])
        ):
            return ModelDownloadError(
                f"'{repo_id}' has no file '{filename}' at revision "
                f"{revision or 'main'} ({where}). ({exc})"
            )
        if isinstance(exc, errors["HfHubHTTPError"]):
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 401:
                return ModelDownloadError(
                    "Hugging Face rejected the request (401)."
                    + (
                        " Set a valid $HF_TOKEN or pass --token."
                        if custom
                        else f" ({where})"
                    )
                    + f" ({exc})"
                )
            if status == 429 or (status is not None and status >= 500):
                return ModelDownloadError(
                    f"Hugging Face is rate-limiting or unavailable (HTTP {status}); retry in a "
                    f"few minutes. ({exc})"
                )
            return ModelDownloadError(
                f"Download of '{repo_id}/{filename}' failed"
                + (f" (HTTP {status})" if status is not None else "")
                + f": {exc}"
            )
        return ModelDownloadError(f"Download of '{repo_id}/{filename}' failed: {exc}")

    @classmethod
    def _alias_default_revision(cls, cached_file: Path) -> None:
        """Alias the default revision to the fetched commit so offline loads resolve.

        ``hf_hub_download(revision=<commit>)`` populates ``snapshots/<commit>`` and
        ``blobs/`` but writes no ``refs/main``. A loader that requests the default
        revision offline reads ``refs/main`` and fails with a local-cache miss when
        it is absent, despite the snapshot being present. Write it, pointing at
        the commit actually fetched, parsed from HF's cache layout
        ``<cache>/models--*/snapshots/<commit>/<...>``. Nothing in this module
        reads it back. No-op for a non-standard path (e.g. an ``--out`` copy).
        """
        parts = cached_file.parts
        marker = [
            i
            for i, part in enumerate(parts)
            if part == "snapshots" and i > 0 and parts[i - 1].startswith("models--")
        ]
        if not marker or marker[-1] + 1 >= len(parts):
            return  # not the standard HF cache layout; nothing to alias
        idx = marker[-1]
        repo_dir = Path(*parts[:idx])
        commit = parts[idx + 1]
        try:
            from huggingface_hub.constants import DEFAULT_REVISION
        except Exception:  # pragma: no cover - stable constant; fall back defensively
            DEFAULT_REVISION = "main"
        ref = repo_dir / "refs" / DEFAULT_REVISION
        try:
            ref.parent.mkdir(parents=True, exist_ok=True)
            ref.write_text(commit)
        except OSError as exc:  # non-fatal: the download already succeeded
            log(f"warning: could not write default ref {ref}: {exc}")

    @classmethod
    def _validate_sha(cls, path: Path, *, expected: str | None) -> None:
        digest = sha256_of(path)
        if expected:
            if digest.lower() != expected.lower():
                raise ModelDownloadError(
                    f"sha256 mismatch for {path}: expected {expected}, got {digest}. "
                    "The cached file does not match the pinned weights; re-run with --force "
                    "to re-download."
                )
            log(f"sha256 OK ({digest})")
        else:
            log(f"sha256 {digest} (no pinned value; record it in the registry)")

    @staticmethod
    def _log(message: str) -> None:
        """Kept for callers of the 0.16 name; see :func:`_provisioning.log`."""
        log(message)


ModelWeights.reset()


def default_plugins_dirs() -> list[Path]:
    """The manifests ``download-model`` reads when ``--plugins-dir`` is not given.

    cuvis-ai's packaged ``configs/plugins`` when ``cuvis_ai`` is importable (located
    without importing it), else nothing: an environment with core alone has only
    the built-in rows plus whatever plugins register at import.
    """
    spec = importlib.util.find_spec("cuvis_ai")
    if spec is None or not spec.origin:
        return []
    candidate = Path(spec.origin).parent / "configs" / "plugins"
    return [candidate] if candidate.is_dir() else []


def index_json(payload: dict[str, Any]) -> str:
    """Deterministic serialisation of the ``list --json`` object (``weights.index.json``).

    Two-space indent, keys in declaration order, rows sorted by name (the payload
    already is), one trailing newline: byte-stable so a CI job can regenerate the
    committed file and diff it.
    """
    return json.dumps(payload, indent=2, sort_keys=False) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def download_model_cli() -> None:
    """CLI entry point for model-weight provisioning (``uv run download-model``)."""
    from cuvis_ai_core.data._model_weights_cli import build_cli

    build_cli()()
