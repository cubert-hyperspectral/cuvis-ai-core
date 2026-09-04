"""Model-weight registry and provisioning for Cuvis.AI.

Every pretrained weight a Cuvis.AI plugin loads is registered here and served
from a Cubert-controlled mirror under the ``cubert-gmbh`` Hugging Face
organisation: public, ungated, byte-identical to upstream, commit-pinned and
sha256-verified. The registry is the single source of truth for where a weight
lives. Plugins ask :meth:`ModelWeights.resolve` instead of hardcoding an
upstream repo id, so the provisioner and the offline runtime always look in the
same cache folder (``models--cubert-gmbh--<repo>``).

Two roles share one cache:

* provisioning (trusted, online): ``download-model download <name>`` fetches the
  pinned file(s) into the shared Hugging Face cache and validates sha256;
* consumption (in-process, or in the sandboxed child that runs with
  ``HF_HUB_OFFLINE=1`` and no token): :meth:`ModelWeights.resolve` returns the
  cached path, downloads when online, or raises
  :class:`ModelWeightsMissingError` naming the provisioning command;
  :meth:`ModelWeights.materialize` additionally places a hardlink or copy at a
  fixed path for loaders that cannot read the Hugging Face cache layout.

Import-light on purpose: module-level imports are stdlib only; ``huggingface_hub``
is imported lazily so importing this module never drags the HF stack, and a
missing optional dependency yields a clear ``pip install cuvis-ai-core[hf]``
message instead of an ImportError at import time.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

# huggingface_hub lives in the optional ``hf`` extra. Absence is reported with an
# actionable message at call time (see _require_hf_hub), not at import.
_HF_EXTRA_HINT = "pip install cuvis-ai-core[hf]"
_SHA_READ_CHUNK = 1 << 20  # 1 MiB
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})

HF_ORG = "cubert-gmbh"
"""Hugging Face organisation that hosts every registry entry."""


class ModelDownloadError(RuntimeError):
    """Raised when a model download or its post-download validation fails."""


class ModelWeightsMissingError(ModelDownloadError):
    """Raised by :meth:`ModelWeights.resolve` when a weight is not cached and
    downloading is not allowed (offline child, or ``download=False``)."""


class ModelWeights:
    """Registry, resolver and provisioning downloader for Cuvis.AI model weights.

    Each registry entry pins a mirror ``repo_id`` + ``filename`` plus the mirror
    ``revision`` (commit) and the upstream ``sha256`` of the file. ``aux_files``
    maps companion files the model's own loader reads from the same repo to
    their sha256. ``plugin`` and ``license`` are informational and surface in
    ``download-model list --json``.
    """

    # ------------------------------------------------------------------
    # Provisioning
    # ------------------------------------------------------------------

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
    ) -> Path:
        """Download a model-weight file and return its resolved local path.

        Provide a registry ``name`` (e.g. ``"sam3"``) OR an explicit
        ``repo_id`` + ``filename``. Registry values fill in any field left
        unset. ``token`` defaults to ``$HF_TOKEN`` (only needed for private or
        custom repos; the registry mirrors are public). ``cache_dir`` defaults
        to the HF cache the child will read (operator ``HF_HUB_CACHE`` /
        ``HF_HOME``, else the shared model cache) so the child loads it offline;
        pass ``out`` to also copy the resolved file to a standalone location
        (e.g. for shipping or a node ``checkpoint_path``).

        Raises:
            ModelDownloadError: on a missing spec, a gated/auth failure, or a
            sha256 mismatch.
        """
        spec = cls._resolve_spec(
            name, repo_id=repo_id, filename=filename, revision=revision, sha256=sha256
        )
        cache_dir = (
            Path(cache_dir) if cache_dir is not None else cls._default_cache_dir()
        )
        resolved = cls._fetch(
            spec, token=token or os.getenv("HF_TOKEN"), cache_dir=cache_dir, force=force
        )

        if out is not None:
            out = Path(out)
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.resolve() != resolved.resolve():
                shutil.copy2(resolved, out)
            resolved = out

        cls._log(f"Ready: {resolved}")
        # Machine-readable contract: the resolved path is the ONLY thing on
        # stdout (all human output above goes to stderr) so callers can parse it.
        print(str(resolved))
        return resolved

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

        Looks the pinned ``repo_id`` / ``filename`` / ``revision`` up in the
        shared Hugging Face cache (the same directory ``download_model`` fills).
        A cached file is returned as is. On a miss, ``download=True`` fetches it
        through :meth:`download_model`'s code path (sha256-verified);
        ``download=False`` raises :class:`ModelWeightsMissingError`. The default
        ``download=None`` means "download unless ``HF_HUB_OFFLINE`` is set",
        which is exactly the sandboxed child's situation. Never prints to
        stdout, so it is safe inside node constructors.
        """
        spec = cls._resolve_spec(
            name, repo_id=None, filename=None, revision=None, sha256=None
        )
        cache_dir = (
            Path(cache_dir) if cache_dir is not None else cls._default_cache_dir()
        )
        cached = cls._cached_path(spec, cache_dir)
        if cached is not None:
            return cached
        if download is None:
            download = not cls._hf_offline()
        if not download:
            raise ModelWeightsMissingError(
                f"'{name}' is not in the model cache ({cache_dir}). Provision it "
                f"with: uv run download-model download {name}"
            )
        return cls._fetch(
            spec, token=os.getenv("HF_TOKEN"), cache_dir=cache_dir, force=False
        )

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
        spec = cls._resolve_spec(
            name, repo_id=None, filename=None, revision=None, sha256=None
        )
        dest_dir = Path(dest_dir)
        dst = dest_dir / (filename or spec["filename"])
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
    # Listing
    # ------------------------------------------------------------------

    @classmethod
    def entries(cls) -> list[dict]:
        """Registry as a list of plain dicts (the ``list --json`` payload)."""
        rows = []
        for name, m in cls._models.items():
            rows.append(
                {
                    "name": name,
                    "plugin": m.get("plugin"),
                    "repo_id": m["repo_id"],
                    "filename": m["filename"],
                    "revision": m.get("revision"),
                    "sha256": m.get("sha256"),
                    "aux_files": dict(m.get("aux_files") or {}),
                    "license": m.get("license"),
                    "requires_token": bool(m.get("requires_token", False)),
                    "cache_dir_token": cls.cache_dir_token(m["repo_id"]),
                    "description": m.get("description", ""),
                }
            )
        return rows

    @staticmethod
    def cache_dir_token(repo_id: str) -> str:
        """Folder name of ``repo_id`` inside a Hugging Face hub cache."""
        return "models--" + repo_id.replace("/", "--")

    @classmethod
    def list_models(cls, *, as_json: bool = False) -> None:
        """Print the registry to stdout, as a table or as JSON (``--json``)."""
        if as_json:
            print(json.dumps(cls.entries(), indent=2))
            return
        print(f"{'Name':<24s} {'Repo / file':<56s} Description")
        print("-" * 110)
        for row in cls.entries():
            repo_file = f"{row['repo_id']}/{row['filename']}"
            print(f"  {row['name']:<22s} {repo_file:<56s} {row['description']}")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @classmethod
    def _fetch(
        cls, spec: dict, *, token: str | None, cache_dir: Path, force: bool
    ) -> Path:
        """Download ``spec`` (primary + companions) into ``cache_dir``, verified.

        Shared by :meth:`download_model` and :meth:`resolve`. Logs to stderr
        only; the caller decides what, if anything, goes to stdout.
        """
        hf_hub_download, hf_errors = cls._require_hf_hub()
        cache_dir.mkdir(parents=True, exist_ok=True)

        cls._log(
            f"Fetching {spec['repo_id']}/{spec['filename']}"
            + (f"@{spec['revision']}" if spec["revision"] else "")
            + f" -> {cache_dir}"
        )
        try:
            resolved = Path(
                hf_hub_download(
                    repo_id=spec["repo_id"],
                    filename=spec["filename"],
                    revision=spec["revision"],
                    token=token,
                    cache_dir=str(cache_dir),
                    force_download=force,
                )
            )
        except hf_errors["GatedRepoError"] as exc:
            raise ModelDownloadError(
                f"Access to '{spec['repo_id']}' is gated. Your token is recognized but the "
                f"account has not accepted the model license. Accept it at "
                f"https://huggingface.co/{spec['repo_id']} then retry.\n  ({exc})"
            ) from exc
        except hf_errors["HfHubHTTPError"] as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 401:
                raise ModelDownloadError(
                    "Hugging Face rejected the token (401). Set a valid $HF_TOKEN or pass "
                    f"--token.\n  ({exc})"
                ) from exc
            raise ModelDownloadError(
                f"Download of '{spec['repo_id']}/{spec['filename']}' failed: {exc}"
            ) from exc

        # skip-if-present must still validate: a cached file with the wrong hash
        # is a broken state, not a success. Validate against the expected sha
        # (when pinned); always surface the computed sha so it can be recorded.
        cls._validate_sha(resolved, expected=spec["sha256"])

        # Companion files the model's own loader also reads from the same repo
        # (e.g. SAM3's config.json). Pull them into the same cache so an offline
        # child resolves the whole set, not just the checkpoint.
        for aux, aux_sha in (spec.get("aux_files") or {}).items():
            cls._log(f"Fetching companion {spec['repo_id']}/{aux}")
            aux_path = Path(
                hf_hub_download(
                    repo_id=spec["repo_id"],
                    filename=aux,
                    revision=spec["revision"],
                    token=token,
                    cache_dir=str(cache_dir),
                    force_download=force,
                )
            )
            cls._validate_sha(aux_path, expected=aux_sha)

        # Make the provisioned cache resolvable offline by the DEFAULT revision: a
        # loader that calls hf_hub_download without a revision resolves "main" via
        # refs/main, which hf_hub_download does not write for a pinned commit.
        # Alias it to the commit we actually fetched so the cache is
        # offline-complete -- for any model, not only a commit-pinned one.
        cls._alias_default_revision(resolved)
        return resolved

    @classmethod
    def _cached_path(cls, spec: dict, cache_dir: Path) -> Path | None:
        """Pinned file already in ``cache_dir``, or None (a pure lookup)."""
        try:
            from huggingface_hub import try_to_load_from_cache
        except ImportError as exc:
            raise ModelDownloadError(
                f"huggingface_hub is not installed. Install with: {_HF_EXTRA_HINT}"
            ) from exc
        cached = try_to_load_from_cache(
            repo_id=spec["repo_id"],
            filename=spec["filename"],
            cache_dir=str(cache_dir),
            revision=spec["revision"],
        )
        # A str is a hit; None and the _CACHED_NO_EXIST sentinel are both misses.
        # gstack-shortcut(dec-00efa593): primary-file hit without sha, upgrade when
        # download-model --verify exists or on the first half-provisioned or
        # corrupted-cache report
        return Path(cached) if isinstance(cached, str) else None

    @staticmethod
    def _hf_offline() -> bool:
        return os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in _TRUE_VALUES

    @classmethod
    def _resolve_spec(
        cls,
        name: str | None,
        *,
        repo_id: str | None,
        filename: str | None,
        revision: str | None,
        sha256: str | None,
    ) -> dict:
        base: dict = {
            "repo_id": None,
            "filename": None,
            "revision": None,
            "sha256": None,
            "aux_files": {},
        }
        if name is not None:
            try:
                base.update(cls._models[name])
            except KeyError:
                raise ModelDownloadError(
                    f"Unknown model '{name}'. Known: {', '.join(cls._models)}. "
                    "Or pass --repo-id and --filename explicitly."
                ) from None
        # Explicit args override registry values.
        base["repo_id"] = repo_id or base.get("repo_id")
        base["filename"] = filename or base.get("filename")
        base["revision"] = revision or base.get("revision")
        base["sha256"] = sha256 or base.get("sha256")
        if not base["repo_id"] or not base["filename"]:
            raise ModelDownloadError(
                "Need a registry name or both --repo-id and --filename."
            )
        return base

    @staticmethod
    def _require_hf_hub():
        try:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.utils import GatedRepoError, HfHubHTTPError
        except ImportError as exc:
            raise ModelDownloadError(
                f"huggingface_hub is not installed. Install with: {_HF_EXTRA_HINT}"
            ) from exc
        return hf_hub_download, {
            "GatedRepoError": GatedRepoError,
            "HfHubHTTPError": HfHubHTTPError,
        }

    @staticmethod
    def _default_cache_dir() -> Path:
        # Lazy import keeps this module free of the orchestrator import cost.
        # Resolve the SAME HF cache the sandboxed child will read (operator
        # HF_HUB_CACHE / HF_HOME, else the shared model cache) so a provisioned
        # weight lands exactly where the offline child looks for it.
        from cuvis_ai_core.orchestrator.model_cache import hf_cache_dir

        return hf_cache_dir(os.environ)

    @classmethod
    def _alias_default_revision(cls, cached_file: Path) -> None:
        """Alias the default revision to the fetched commit so offline loads resolve.

        ``hf_hub_download(revision=<commit>)`` populates ``snapshots/<commit>`` and
        ``blobs/`` but writes no ``refs/main``. A loader that requests the default
        revision offline (huggingface's ``DEFAULT_REVISION`` -- ``main``) reads
        ``refs/main`` and fails with a local-cache miss when it is absent, despite
        the snapshot being present. Write it, pointing at the commit actually
        fetched, parsed from HF's cache layout
        ``<cache>/models--*/snapshots/<commit>/<file>``. No-op for a non-standard
        path (e.g. an ``--out`` copy).
        """
        snapshot_dir = cached_file.parent  # .../snapshots/<commit>
        snapshots = snapshot_dir.parent  # .../snapshots
        repo_dir = snapshots.parent  # .../models--<org>--<name>
        if snapshots.name != "snapshots" or not repo_dir.name.startswith("models--"):
            return  # not the standard HF cache layout; nothing to alias
        try:
            from huggingface_hub.constants import DEFAULT_REVISION
        except Exception:  # pragma: no cover - stable constant; fall back defensively
            DEFAULT_REVISION = "main"
        ref = repo_dir / "refs" / DEFAULT_REVISION
        try:
            ref.parent.mkdir(parents=True, exist_ok=True)
            ref.write_text(snapshot_dir.name)
        except OSError as exc:  # non-fatal: the snapshot download already succeeded
            cls._log(f"warning: could not write default ref {ref}: {exc}")

    @classmethod
    def _validate_sha(cls, path: Path, *, expected: str | None) -> None:
        digest = cls._sha256(path)
        if expected:
            if digest.lower() != expected.lower():
                raise ModelDownloadError(
                    f"sha256 mismatch for {path}: expected {expected}, got {digest}. "
                    "The cached file does not match the pinned weights; re-run with --force "
                    "to re-download."
                )
            cls._log(f"sha256 OK ({digest})")
        else:
            cls._log(f"sha256 {digest} (no pinned value; record it in the registry)")

    @staticmethod
    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(_SHA_READ_CHUNK), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _log(message: str) -> None:
        # Human/progress output goes to stderr; stdout is reserved for the path.
        import sys

        print(message, file=sys.stderr)

    # ------------------------------------------------------------------
    # Model registry
    # ------------------------------------------------------------------
    #
    # Every entry points at a public, ungated mirror under the cubert-gmbh
    # organisation (see tools/mirror_weights.py, which builds the mirrors and
    # prints these entries). ``revision`` is the mirror commit, ``sha256`` the
    # upstream value, so integrity is verifiable against upstream. The registry
    # names are the contract plugins resolve by; keep them stable.
    _models: dict[str, dict] = {
        # --- cuvis-ai-sam3 -----------------------------------------------------
        "sam3": {
            "repo_id": f"{HF_ORG}/sam3",
            "filename": "sam3.pt",
            "revision": "6d25af14a085ff9d3e1342c35bae7c87de4811f4",
            "sha256": "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e",
            # Companion the SAM3 builder reads from the same repo alongside the .pt.
            "aux_files": {
                "config.json": (
                    "4616385e4b21f2e5e22c875b65679185cbccfa95de42542b9166f7dc3d57160f"
                ),
            },
            "plugin": "cuvis-ai-sam3",
            "license": "SAM License",
            "description": "SAM3 checkpoint (mirror of facebook/sam3)",
        },
        # --- cuvis-ai-rtsam2 (EfficientTAM) -------------------------------------
        "efficienttam_s": {
            "repo_id": f"{HF_ORG}/efficient-track-anything",
            "filename": "efficienttam_s.pt",
            "revision": "3dfd0228d7774b94c24116cf729e03c209ff448a",
            "sha256": "2b572be30d9e96ee29c8d785fe157c6b079ede7d56fbc8a3671d4120e63c89cd",
            "plugin": "cuvis-ai-rtsam2",
            "license": "Apache-2.0",
            "description": "EfficientTAM small checkpoint (RTSAM2 default)",
        },
        "efficienttam_ti": {
            "repo_id": f"{HF_ORG}/efficient-track-anything",
            "filename": "efficienttam_ti.pt",
            "revision": "3dfd0228d7774b94c24116cf729e03c209ff448a",
            "sha256": "acbb17b28cca1f860acee09c9ecb6efdb732080dc7a85a07292c31813175fa7d",
            "plugin": "cuvis-ai-rtsam2",
            "license": "Apache-2.0",
            "description": "EfficientTAM tiny checkpoint (RTSAM2)",
        },
        "efficienttam_s_512x512": {
            "repo_id": f"{HF_ORG}/efficient-track-anything",
            "filename": "efficienttam_s_512x512.pt",
            "revision": "3dfd0228d7774b94c24116cf729e03c209ff448a",
            "sha256": "67b5840012737ed2c94a4cb8787c5c1b27b3a946045d2e602e65ef77230b6085",
            "plugin": "cuvis-ai-rtsam2",
            "license": "Apache-2.0",
            "description": "EfficientTAM small checkpoint, 512x512 input (RTSAM2)",
        },
        "efficienttam_ti_512x512": {
            "repo_id": f"{HF_ORG}/efficient-track-anything",
            "filename": "efficienttam_ti_512x512.pt",
            "revision": "3dfd0228d7774b94c24116cf729e03c209ff448a",
            "sha256": "7d4d652a465f0081391050932f45d8a66768ccc99c8ea393ce8a5927e83f3b9b",
            "plugin": "cuvis-ai-rtsam2",
            "license": "Apache-2.0",
            "description": "EfficientTAM tiny checkpoint, 512x512 input (RTSAM2)",
        },
        # --- cuvis-ai-dinomaly ----------------------------------------------------
        "dinov2_vitb14_reg4": {
            "repo_id": f"{HF_ORG}/dinov2",
            "filename": "dinov2_vitb14_reg4_pretrain.pth",
            "revision": "e2c8060a74112f7537484ed50097b1497d5f032c",
            "sha256": "73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71",
            "plugin": "cuvis-ai-dinomaly",
            "license": "Apache-2.0",
            "description": "DINOv2 ViT-B/14 reg4 backbone (Dinomaly encoder)",
        },
        # --- cuvis-ai-adaclip -----------------------------------------------------
        "clip_vit_l_14_336": {
            "repo_id": f"{HF_ORG}/clip",
            "filename": "ViT-L-14-336px.pt",
            "revision": "a223b3db0b7bd1b55cf8f6421629b30b46c995de",
            "sha256": "3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02",
            "plugin": "cuvis-ai-adaclip",
            "license": "unspecified (code: MIT)",
            "description": "OpenAI CLIP ViT-L/14@336px backbone (AdaCLIP)",
        },
        "adaclip_all": {
            "repo_id": f"{HF_ORG}/adaclip",
            "filename": "pretrained_all.pth",
            "revision": "16153b4ba74c2fe54a99679fc2e1b1e29993dc3f",
            "sha256": "33e8d3db1cb4aab030866b8b70a46e10aa27ebf2c23b5463cb07f2574addd98c",
            "plugin": "cuvis-ai-adaclip",
            "license": "unspecified (code: MIT)",
            "description": "AdaCLIP heads trained on all auxiliary datasets",
        },
        # The two dataset-specific heads keep the upstream Drive filenames; the
        # upstream README row labels disagree with those names (documented in the
        # mirror's model card).
        "adaclip_mvtec_colondb": {
            "repo_id": f"{HF_ORG}/adaclip",
            "filename": "pretrained_mvtec_colondb.pth",
            "revision": "16153b4ba74c2fe54a99679fc2e1b1e29993dc3f",
            "sha256": "be51a42c052bd4cf060e54f503a1f5d0b2a3b899bc8dc2e243042f18b215427e",
            "plugin": "cuvis-ai-adaclip",
            "license": "unspecified (code: MIT)",
            "description": (
                "AdaCLIP heads, upstream file pretrained_mvtec_colondb.pth "
                "(README row: MVTec AD and ClinicDB)"
            ),
        },
        "adaclip_visa_clinicdb": {
            "repo_id": f"{HF_ORG}/adaclip",
            "filename": "pretrained_visa_clinicdb.pth",
            "revision": "16153b4ba74c2fe54a99679fc2e1b1e29993dc3f",
            "sha256": "3deabbbaf1e412cfdfcb42923a500b986f4b9ee96ccbc7a735d89dbc87df44c8",
            "plugin": "cuvis-ai-adaclip",
            "license": "unspecified (code: MIT)",
            "description": (
                "AdaCLIP heads, upstream file pretrained_visa_clinicdb.pth "
                "(README row: VisA and ColonDB)"
            ),
        },
    }


def download_model_cli() -> None:
    """CLI entry point for model-weight provisioning (``uv run download-model``)."""
    import click

    @click.group()
    def cli() -> None:
        """CUVIS.AI model-weight provisioning."""

    @cli.command("list")
    @click.option("--json", "as_json", is_flag=True, help="Emit the registry as JSON.")
    def list_cmd(as_json: bool) -> None:
        """List registry models."""
        ModelWeights.list_models(as_json=as_json)

    @cli.command()
    @click.argument("name", required=False)
    @click.option("--repo-id", default=None, help="HF repo id (overrides registry).")
    @click.option(
        "--filename", default=None, help="File in the repo (overrides registry)."
    )
    @click.option("--revision", default=None, help="HF revision/commit to pin.")
    @click.option(
        "--token",
        default=None,
        help="HF token (else $HF_TOKEN); only needed for private or custom repos.",
    )
    @click.option(
        "--cache-dir",
        type=click.Path(path_type=Path),
        default=None,
        help="Cache target (default: the shared model cache).",
    )
    @click.option(
        "--out",
        type=click.Path(path_type=Path),
        default=None,
        help="Also copy the resolved file here (e.g. for shipping / checkpoint_path).",
    )
    @click.option("--force", is_flag=True, help="Re-download even if cached.")
    def download(
        name: str | None,
        repo_id: str | None,
        filename: str | None,
        revision: str | None,
        token: str | None,
        cache_dir: Path | None,
        out: Path | None,
        force: bool,
    ) -> None:
        """Download a model by registry NAME or by --repo-id/--filename."""
        try:
            ModelWeights.download_model(
                name,
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                token=token,
                cache_dir=cache_dir,
                out=out,
                force=force,
            )
        except ModelDownloadError as exc:
            raise SystemExit(f"error: {exc}") from exc

    cli()
