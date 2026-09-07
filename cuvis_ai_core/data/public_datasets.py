"""Public dataset registry, downloader and the on-disk marker contract.

The six public Cubert datasets on Hugging Face (``cubert-gmbh``) are registered
here with exact pins: the dataset commit, the byte size and the file count. A
consumer (the ``dataset`` CLI, a notebook, CuvisNEXT's Datasets tab) downloads
them with :meth:`PublicDatasets.download` and learns what is on disk from
:meth:`PublicDatasets.status`, which reads exactly one fact: the marker file
``<data_dir>/<target_dir>/.cuvis-dataset.json``.

Marker contract (two phases, written atomically through a ``.tmp`` rename):

* ``download`` writes ``{"state": "downloading", "repo_id", "revision",
  "started_at"}`` before the first byte and rewrites it as ``{"state":
  "complete", "repo_id", "revision", "file_count", "size_bytes",
  "finished_at", "files": [{"path", "size_bytes"}, ...]}`` when the snapshot
  returned; the ``files`` list is the repo's file list at the pinned revision.
* ``status`` maps: no directory -> ``absent``; a directory without a marker ->
  ``foreign`` (files present, not a Cubert download: never deleted, never
  written into without ``adopt``); an unparsable or ``downloading`` marker ->
  ``incomplete`` (the next download resumes); ``complete`` at another revision
  -> ``outdated``; ``complete`` at the pinned revision with every listed file
  present at its size -> ``present``, otherwise ``damaged``.
* ``remove`` deletes a directory only when its marker names the expected repo,
  so a user's own recordings in a same-named folder are never touched.

Every operation that writes under ``data_dir`` holds the same
``<data_dir>/.cuvis-cache.lock`` the model cache uses on its root, so two
CuvisNEXT instances sharing a datasets folder serialise. Library methods never
print to stdout; human output goes to stderr.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal

from cuvis_ai_core.data._provisioning import (
    SCHEMA_VERSION,
    CacheBusyError,
    ProgressEmitter,
    log,
    root_lock,
)

HF_ORG = "cubert-gmbh"
"""Hugging Face organisation that hosts every dataset."""

MARKER_NAME = ".cuvis-dataset.json"
"""The one file ``status`` reads inside a dataset directory."""

DATASET_TAGS: tuple[str, ...] = (
    "Anomaly detection",
    "Segmentation",
    "Tracking",
    "Statistical",
)
"""Task labels, in display order (CuvisNEXT renders the Task facet in it)."""

CAMERAS: tuple[str, ...] = ("XMR", "X4 SWIR")
"""Camera labels, in display order."""

_HF_EXTRA_HINT = "pip install cuvis-ai-core[hf]"

DatasetState = Literal[
    "present", "damaged", "incomplete", "outdated", "foreign", "absent"
]


class DatasetError(RuntimeError):
    """Raised when a dataset download, removal or lookup fails."""


# ---------------------------------------------------------------------------
# Specs, statuses, the registry table
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """One public dataset: where it lives, how big it is, what it is for."""

    name: str
    display_name: str
    summary: str
    repo_id: str
    target_dir: str
    revision: str
    size_bytes: int
    file_count: int
    license: str
    tags: tuple[str, ...]
    camera: str
    description: str
    aliases: tuple[str, ...] = ()

    def to_json_dict(self) -> dict[str, Any]:
        """The ``DSPEC`` object of the ``--json`` contract (``dataset_list.schema.json``)."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "summary": self.summary,
            "repo_id": self.repo_id,
            "target_dir": self.target_dir,
            "revision": self.revision,
            "size_bytes": self.size_bytes,
            "file_count": self.file_count,
            "license": self.license,
            "tags": list(self.tags),
            "camera": self.camera,
            "description": self.description,
            "aliases": list(self.aliases),
        }


@dataclass(frozen=True, slots=True)
class DatasetStatus:
    """What ``data_dir`` holds for one dataset, read from its marker alone."""

    spec: DatasetSpec
    path: Path
    state: DatasetState
    bytes_on_disk: int | None
    files_ok: bool | None
    marker: dict[str, Any] | None

    @property
    def present(self) -> bool:
        """Complete at the pinned revision with every file in place."""
        return self.state == "present"

    def to_json_dict(self) -> dict[str, Any]:
        """The ``DSTATUS`` object of the ``--json`` contract (``dataset_status.schema.json``)."""
        return {
            "state": self.state,
            "present": self.present,
            "path": str(self.path),
            "bytes_on_disk": self.bytes_on_disk,
            "files_ok": self.files_ok,
        }


@dataclass(frozen=True, slots=True)
class DownloadResult:
    """Outcome of :meth:`PublicDatasets.download`."""

    path: Path
    stale_files: tuple[str, ...]
    """Files a previous revision had that the pinned revision no longer lists (kept unless pruned)."""


DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        name="Lentils",
        display_name="Lentils (single session)",
        summary="One lentil-conveyor session, used by the test suite",
        repo_id=f"{HF_ORG}/XMR_Lentils",
        target_dir="Lentils",
        revision="24ae01a2a4818156640fc65103aac8e5bc7bcdfc",
        size_bytes=921_338_131,
        file_count=4,
        license="Apache-2.0",
        tags=("Statistical",),
        camera="XMR",
        description=(
            "Single lentil-conveyor CU3S session (Lentils_000) used by the cuvis-ai-core "
            "test suite and the statistical-training examples."
        ),
        aliases=("lentils",),
    ),
    DatasetSpec(
        name="Demo_Industrial_FOD_Lentils",
        display_name="Demo: industrial FOD, lentils",
        summary="69-frame lentil conveyor demo with foreign objects",
        repo_id=f"{HF_ORG}/XMR_Demo_Industrial_Foreign_Object_Detection_Lentils",
        target_dir="XMR_Demo_Industrial_Foreign_Object_Detection_Lentils",
        revision="6f0e6f0f4e8345a5b9a5a6e1ff66daa06e05a6c9",
        size_bytes=6_442_039_738,
        file_count=21,
        license="Apache-2.0",
        tags=("Anomaly detection", "Segmentation"),
        camera="XMR",
        description=(
            "Hyperspectral foreign-object detection on a lentil conveyor: a 69-frame XMR CU3S "
            "session with pixel-level annotations and the Dinomaly companion pipeline."
        ),
        aliases=("demo_industrial_fod_lentils",),
    ),
    DatasetSpec(
        name="Industrial_FOD_Lentils",
        display_name="Industrial FOD, lentils",
        summary="15 sessions, 1,136 frames, 7 foreign-object classes",
        repo_id=f"{HF_ORG}/XMR_Industrial_Foreign_Object_Detection_Lentils",
        target_dir="XMR_Industrial_Foreign_Object_Detection_Lentils",
        revision="935d509d3ad8a4c910218b9c9292912e394c807f",
        size_bytes=56_996_444_870,
        file_count=82,
        license="Apache-2.0",
        tags=("Anomaly detection", "Segmentation"),
        camera="XMR",
        description=(
            "Full industrial foreign-object detection dataset on a lentil conveyor: 15 merged "
            "XMR CU3S sessions across three acquisition days (1,136 frames, 696 annotated), "
            "pixel-level COCO masks for 7 foreign-object classes."
        ),
        aliases=("industrial_fod_lentils",),
    ),
    DatasetSpec(
        name="Industrial_FOD_Bedding",
        display_name="Industrial FOD, bedding (X4 SWIR)",
        summary="252 VIS+SWIR frames, 23 foreign-object classes",
        repo_id=f"{HF_ORG}/X4_SWIR_Industrial_Foreign_Object_Detection_Bedding",
        target_dir="X4_SWIR_Industrial_Foreign_Object_Detection_Bedding",
        revision="5c6a22690288b9d8d06552848fd80658e9b57107",
        size_bytes=178_163_344_074,
        file_count=683,
        license="Apache-2.0",
        tags=("Anomaly detection", "Segmentation"),
        camera="X4 SWIR",
        description=(
            "Industrial foreign-object detection in bedding substrate: 6-channel VIS+SWIR X4 "
            "still frames (450/550/625/1050/1200/1450 nm), 252 frames (193 train / 59 val), "
            "pixel masks for 23 foreign-object classes."
        ),
        aliases=("industrial_fod_bedding",),
    ),
    DatasetSpec(
        name="Blood_Perfusion",
        display_name="Demo: blood perfusion",
        summary="Blood perfusion reflectance sessions",
        repo_id=f"{HF_ORG}/XMR_Demo_Blood_Perfusion",
        target_dir="XMR_Demo_Blood_Perfusion",
        revision="a26bc15b84a57143413da066b4df30d5bed4436f",
        size_bytes=11_247_205_611,
        file_count=6,
        license="Apache-2.0",
        tags=("Statistical",),
        camera="XMR",
        description="XMR blood perfusion reflectance dataset (no training annotations).",
        aliases=("blood_perfusion",),
    ),
    DatasetSpec(
        name="Demo_Object_Tracking",
        display_name="Demo: object tracking",
        summary="Multi-person tracking demo, passive and active sessions",
        repo_id=f"{HF_ORG}/XMR_Demo_Object_Tracking",
        target_dir="XMR_Demo_Object_Tracking",
        revision="714498e5134d1df393d9f2056a2dcf40d58b2256",
        size_bytes=24_354_255_238,
        file_count=18,
        license="Apache-2.0",
        tags=("Tracking", "Segmentation"),
        camera="XMR",
        description=(
            "Hyperspectral multi-person tracking demo: passive SAM3 and active spectral-ink "
            "sessions (no training annotations; the tracking nodes do not train)."
        ),
        aliases=("demo_object_tracking",),
    ),
)
"""The registry: exactly the datasets the ``cubert-gmbh`` organisation publishes."""


def _normalize(name: str) -> str:
    return name.replace("-", "_").lower()


_BY_KEY: dict[str, DatasetSpec] = {}
for _spec in DATASETS:
    for _key in (_spec.name, *_spec.aliases):
        _BY_KEY[_normalize(_key)] = _spec


class PublicDatasets:
    """Registry, status reader, downloader and remover for the public datasets."""

    DATASETS: tuple[DatasetSpec, ...] = DATASETS

    # ------------------------------------------------------------------
    # Registry
    # ------------------------------------------------------------------

    @classmethod
    def list_specs(cls) -> list[DatasetSpec]:
        """Every dataset, registry order."""
        return list(cls.DATASETS)

    @classmethod
    def get_spec(cls, name: str) -> DatasetSpec:
        """The spec for a name or alias (hyphens and case are forgiven).

        Raises:
            DatasetError: unknown name; the message lists the known names.
        """
        spec = _BY_KEY.get(_normalize(name))
        if spec is None:
            known = ", ".join(s.name for s in cls.DATASETS)
            raise DatasetError(f"Dataset '{name}' not found. Available: {known}")
        return spec

    @classmethod
    def list_payload(cls) -> dict[str, Any]:
        """The complete ``dataset list --json`` object."""
        return {
            "schema_version": SCHEMA_VERSION,
            "tags": list(DATASET_TAGS),
            "cameras": list(CAMERAS),
            "datasets": [spec.to_json_dict() for spec in cls.DATASETS],
        }

    @classmethod
    def get_target_dir(cls, dataset_name: str) -> str:
        """Directory name a dataset downloads into (raises ``KeyError`` when unknown)."""
        try:
            return cls.get_spec(dataset_name).target_dir
        except DatasetError as exc:
            raise KeyError(dataset_name) from exc

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @staticmethod
    def dataset_dir(spec: DatasetSpec, data_dir: str | Path) -> Path:
        """``<data_dir>/<target_dir>``."""
        return Path(data_dir) / spec.target_dir

    @staticmethod
    def read_marker(path: Path) -> dict[str, Any] | None:
        """The parsed marker of a dataset directory, ``None`` when absent, ``{}`` when unreadable."""
        marker_path = path / MARKER_NAME
        if not marker_path.is_file():
            return None
        try:
            data = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    @classmethod
    def status(cls, name: str, data_dir: str | Path) -> DatasetStatus:
        """What ``data_dir`` holds for ``name``, from the marker alone (see the module docstring)."""
        spec = cls.get_spec(name)
        path = cls.dataset_dir(spec, data_dir)
        if not path.is_dir():
            return DatasetStatus(spec, path, "absent", None, None, None)
        marker = cls.read_marker(path)
        if marker is None:
            # huggingface_hub's own bookkeeping (.cache/huggingface) is not user data: a
            # folder holding only that (a crash before the first marker write) is reused.
            has_files = any(
                p.is_file()
                for p in path.rglob("*")
                if ".cache" not in p.relative_to(path).parts
            )
            state: DatasetState = "foreign" if has_files else "absent"
            return DatasetStatus(
                spec, path, state, cls._du(path) if has_files else None, None, None
            )
        if not marker or marker.get("state") != "complete":
            return DatasetStatus(
                spec, path, "incomplete", cls._du(path), None, marker or {}
            )
        files = marker.get("files") or []
        files_ok = (
            all(
                isinstance(f, dict)
                and (path / str(f.get("path", ""))).is_file()
                and (path / str(f["path"])).stat().st_size == f.get("size_bytes")
                for f in files
            )
            if files
            else False
        )
        on_disk = sum(
            (path / str(f["path"])).stat().st_size
            for f in files
            if isinstance(f, dict) and (path / str(f.get("path", ""))).is_file()
        )
        if marker.get("revision") != spec.revision:
            return DatasetStatus(spec, path, "outdated", on_disk, files_ok, marker)
        return DatasetStatus(
            spec, path, "present" if files_ok else "damaged", on_disk, files_ok, marker
        )

    @classmethod
    def status_all(
        cls, data_dir: str | Path, names: Sequence[str] | None = None
    ) -> list[DatasetStatus]:
        """Statuses for ``names`` (default: every dataset), registry order."""
        specs = [cls.get_spec(n) for n in names] if names else cls.list_specs()
        return [cls.status(spec.name, data_dir) for spec in specs]

    # ------------------------------------------------------------------
    # Download / remove
    # ------------------------------------------------------------------

    @classmethod
    def download(
        cls,
        name: str,
        data_dir: str | Path,
        *,
        force: bool = False,
        adopt: bool = False,
        prune_stale: bool = False,
        progress: ProgressEmitter | None = None,
    ) -> DownloadResult:
        """Download (or resume, or refresh) a dataset into ``<data_dir>/<target_dir>``.

        Present and not ``force``: nothing happens. A foreign directory (files but no
        marker) is refused unless ``adopt`` is set. The marker is written as
        ``downloading`` before the first byte and as ``complete`` (with the file
        manifest) after ``snapshot_download`` returned every file of the pinned
        revision, so a crash mid-way reads ``incomplete`` and the next call resumes.
        Refreshing an ``outdated`` directory keeps files the new revision no longer
        lists and reports them as ``stale_files``; ``prune_stale`` deletes them.
        Progress is per file (``files_done`` / ``files_total``); bytes are not known.

        Raises:
            DatasetError: unknown name, a foreign directory without ``adopt``, a
            failed or incomplete snapshot, or a busy datasets folder.
        """
        spec = cls.get_spec(name)
        root = Path(data_dir)
        target = cls.dataset_dir(spec, root)
        with cls._locked(root, progress):
            current = cls.status(spec.name, root)
            if current.present and not force:
                log(f"{spec.name}: present at {target}")
                return DownloadResult(target, ())
            if current.state == "foreign" and not adopt:
                raise DatasetError(
                    f"{target} exists and is not a Cubert download; pass --adopt to manage it "
                    "(Remove will then delete it)."
                )
            previous_files = (
                {
                    str(f.get("path"))
                    for f in (current.marker or {}).get("files", [])
                    if isinstance(f, dict)
                }
                if current.marker
                else set()
            )
            snapshot_download, hf_api = cls._require_hf_hub()
            target.mkdir(parents=True, exist_ok=True)
            cls._write_marker(
                target,
                {
                    "state": "downloading",
                    "repo_id": spec.repo_id,
                    "revision": spec.revision,
                    "started_at": _now(),
                },
            )
            try:
                repo_files = sorted(
                    hf_api.list_repo_files(
                        spec.repo_id, repo_type="dataset", revision=spec.revision
                    )
                )
                kwargs: dict[str, Any] = {}
                if progress is not None:
                    kwargs["tqdm_class"] = _files_progress_class(progress, spec.name)
                snapshot_download(
                    repo_id=spec.repo_id,
                    repo_type="dataset",
                    revision=spec.revision,
                    local_dir=str(target),
                    token=False,
                    force_download=force,
                    **kwargs,
                )
            except Exception as exc:
                raise DatasetError(
                    f"Download of '{spec.name}' ({spec.repo_id}) failed: {exc}. "
                    f"Manual download: https://huggingface.co/datasets/{spec.repo_id}"
                ) from exc
            missing = [p for p in repo_files if not (target / p).is_file()]
            if missing:
                raise DatasetError(
                    f"'{spec.name}': {len(missing)} file(s) of revision {spec.revision[:12]} are "
                    f"missing after the download (first: {missing[0]}); re-run to resume."
                )
            files = [
                {"path": p, "size_bytes": (target / p).stat().st_size}
                for p in repo_files
            ]
            stale = sorted(previous_files - set(repo_files))
            if prune_stale:
                for rel in stale:
                    (target / rel).unlink(missing_ok=True)
                stale = []
            cls._write_marker(
                target,
                {
                    "state": "complete",
                    "repo_id": spec.repo_id,
                    "revision": spec.revision,
                    "file_count": len(files),
                    "size_bytes": sum(f["size_bytes"] for f in files),
                    "finished_at": _now(),
                    "files": files,
                },
            )
            if progress is not None:
                progress.done(spec.name, target)
            return DownloadResult(target, tuple(stale))

    @classmethod
    def remove(cls, name: str, data_dir: str | Path) -> int:
        """Delete a downloaded dataset directory; returns the bytes freed.

        Refuses a directory without a marker naming the expected repo (a user's
        own data in a same-named folder), whatever the marker's state.
        """
        spec = cls.get_spec(name)
        root = Path(data_dir)
        target = cls.dataset_dir(spec, root)
        with cls._locked(root):
            if not target.is_dir():
                return 0
            marker = cls.read_marker(target)
            if not marker or marker.get("repo_id") != spec.repo_id:
                raise DatasetError(
                    f"{target} is not a downloaded Cubert dataset (no marker for "
                    f"{spec.repo_id}); refusing to delete it."
                )
            freed = cls._du(target)
            shutil.rmtree(target)
        return freed

    # ------------------------------------------------------------------
    # Compatibility helpers (the pre-0.17 API notebooks and skills call)
    # ------------------------------------------------------------------

    @classmethod
    def download_dataset(
        cls, dataset_name: str, *, download_path: str = ".", force: bool = False
    ) -> bool:
        """Download by name into ``download_path``; True on success, False on error.

        Kept for notebooks written against earlier releases; new code calls
        :meth:`download`, which raises instead of returning False.
        """
        try:
            cls.download(dataset_name, download_path, force=force)
        except DatasetError as exc:
            log(str(exc))
            return False
        return True

    @classmethod
    def list_datasets(cls, verbose: bool = False) -> None:
        """Print the registry table to stdout (the ``dataset list`` text output)."""
        print(f"{'Name':<30s} {'Camera':<8s} {'Size':>10s}  Summary")
        print("-" * 100)
        for spec in cls.DATASETS:
            from cuvis_ai_core.data._provisioning import format_bytes

            alias_str = f"  (alias: {', '.join(spec.aliases)})" if spec.aliases else ""
            print(
                f"  {spec.name:<28s} {spec.camera:<8s} {format_bytes(spec.size_bytes):>10s}  "
                f"{spec.summary}{alias_str}"
            )
            if verbose:
                print(f"    repo: {spec.repo_id} @ {spec.revision[:12]}")
                print(
                    f"    dir:  {spec.target_dir}  ({spec.file_count} files, {spec.license})"
                )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _write_marker(target: Path, payload: dict[str, Any]) -> None:
        marker = target / MARKER_NAME
        tmp = marker.with_suffix(marker.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, marker)

    @staticmethod
    def _du(path: Path) -> int:
        total = 0
        for item in path.rglob("*"):
            if item.is_file():
                try:
                    total += item.stat().st_size
                except OSError:
                    continue
        return total

    @classmethod
    @contextmanager
    def _locked(
        cls, root: Path, progress: ProgressEmitter | None = None
    ) -> Iterator[None]:
        def _on_wait() -> None:
            log("another operation is using the datasets folder; waiting...")
            if progress is not None:
                progress.waiting()

        try:
            with root_lock(root, on_wait=_on_wait):
                yield
        except CacheBusyError as exc:
            raise DatasetError(str(exc)) from exc

    @staticmethod
    def _require_hf_hub() -> tuple[Callable[..., str], Any]:
        try:
            from huggingface_hub import HfApi, snapshot_download
        except ImportError as exc:
            raise DatasetError(
                f"huggingface_hub is not installed. Install with: {_HF_EXTRA_HINT}"
            ) from exc
        return snapshot_download, HfApi()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _files_progress_class(emitter: ProgressEmitter, name: str) -> type:
    """A tqdm subclass ``snapshot_download`` drives per file; it emits our progress events."""
    from tqdm.auto import tqdm

    class _FilesProgress(tqdm):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["disable"] = True  # no console bar; the events are the output
            super().__init__(*args, **kwargs)
            self._files_done = 0

        def update(self, n: float | None = 1) -> bool | None:
            self._files_done += int(n or 0)
            emitter.files_progress(name, self._files_done, self.total)
            return True

    return _FilesProgress


def download_data_cli() -> None:
    """CLI entry point for dataset management (``uv run dataset``)."""
    from cuvis_ai_core.data._datasets_cli import build_cli

    build_cli()()
