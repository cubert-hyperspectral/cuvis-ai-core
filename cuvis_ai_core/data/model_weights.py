"""Model-weight registry and provisioning downloader for CUVIS.AI.

Pre-fetches model weights from Hugging Face into the shared model cache (or an
explicit target) so a sandboxed child runtime can load them offline, and so
weights can be shipped / provisioned ahead of an offline install.

Complements the runtime path: the spawner injects ``HF_HUB_CACHE`` / ``TORCH_HOME``
so a child fetches missing weights on first run. This CLI is the *out-of-band*
provisioning tool for the same cache -- run once with an authorized token, then
runs are offline.

Import-light on purpose: module-level imports are stdlib only; ``huggingface_hub``
is imported lazily so importing this module never drags the HF stack, and a
missing optional dependency yields a clear ``pip install cuvis-ai-core[hf]``
message instead of an ImportError at import time.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

# huggingface_hub lives in the optional ``hf`` extra. Absence is reported with an
# actionable message at call time (see _require_hf_hub), not at import.
_HF_EXTRA_HINT = "pip install cuvis-ai-core[hf]"
_SHA_READ_CHUNK = 1 << 20  # 1 MiB


class ModelDownloadError(RuntimeError):
    """Raised when a model download or its post-download validation fails."""


class ModelWeights:
    """Registry and provisioning downloader for CUVIS.AI model weights.

    Each registry entry pins a Hugging Face ``repo_id`` + ``filename`` and,
    for reproducibility / integrity, an optional ``revision`` (commit) and
    ``sha256``. ``revision``/``sha256`` are ``None`` until pinned; the CLI
    prints the resolved commit and computed sha256 after a download so they can
    be recorded here.
    """

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
        unset. ``token`` defaults to ``$HF_TOKEN``. ``cache_dir`` defaults to
        the HF cache the child will read (operator ``HF_HUB_CACHE`` / ``HF_HOME``,
        else the shared model cache) so the child loads it offline; pass ``out``
        to also copy the resolved file to a standalone location (e.g. for
        shipping or a node ``checkpoint_path``).

        Raises:
            ModelDownloadError: on a missing spec, a gated/auth failure, or a
            sha256 mismatch.
        """
        spec = cls._resolve_spec(
            name, repo_id=repo_id, filename=filename, revision=revision, sha256=sha256
        )
        hf_hub_download, hf_errors = cls._require_hf_hub()

        token = token or os.getenv("HF_TOKEN")
        cache_dir = (
            Path(cache_dir) if cache_dir is not None else cls._default_cache_dir()
        )
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

        # Companion files the model's own loader also fetches from the same repo
        # (e.g. SAM3's config.json). Pull them into the same cache so an offline
        # child resolves the whole set, not just the checkpoint.
        for aux in spec.get("aux_files") or []:
            cls._log(f"Fetching companion {spec['repo_id']}/{aux}")
            hf_hub_download(
                repo_id=spec["repo_id"],
                filename=aux,
                revision=spec["revision"],
                token=token,
                cache_dir=str(cache_dir),
                force_download=force,
            )

        # Make the provisioned cache resolvable offline by the DEFAULT revision: a
        # loader that calls hf_hub_download without a revision (e.g. SAM3's builder)
        # resolves "main" via refs/main, which hf_hub_download does not write for a
        # pinned commit. Alias it to the commit we actually fetched so the cache is
        # offline-complete -- for any model, not only a commit-pinned one.
        cls._alias_default_revision(resolved)

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

    @classmethod
    def list_models(cls) -> None:
        """Print the available registry models to stderr-free stdout."""
        print(f"{'Name':<12s} {'Repo / file':<40s} Description")
        print("-" * 78)
        for name, m in cls._models.items():
            repo_file = f"{m['repo_id']}/{m['filename']}"
            print(f"  {name:<10s} {repo_file:<40s} {m['description']}")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

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
        base = {"repo_id": None, "filename": None, "revision": None, "sha256": None}
        if name is not None:
            try:
                base = dict(cls._models[name])
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
        revision offline (huggingface's ``DEFAULT_REVISION`` -- ``main`` -- which is
        what SAM3's builder does by calling ``hf_hub_download`` with no revision)
        reads ``refs/main`` and fails with a local-cache miss when it is absent,
        despite the snapshot being present. Write it, pointing at the commit actually
        fetched, parsed from HF's cache layout
        ``<cache>/models--*/snapshots/<commit>/<file>``. No-op for a non-standard path
        (e.g. an ``--out`` copy).
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
    # revision / sha256 are None until pinned. After the first authorized
    # download the CLI prints both; record them here for reproducibility +
    # integrity validation (D2).
    _models: dict[str, dict] = {
        "sam3": {
            "repo_id": "facebook/sam3",
            "filename": "sam3.pt",
            "revision": "3c879f39826c281e95690f02c7821c4de09afae7",
            "sha256": "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e",
            # Companion files the model's own loader fetches from the same repo;
            # provisioned into the same cache so an offline child resolves the
            # whole set (SAM3's builder pulls config.json alongside the .pt).
            "aux_files": ["config.json"],
            "description": "SAM3 checkpoint (gated; requires an accepted license)",
        },
        # EfficientTAM checkpoints for the RTSAM2 streaming-propagation plugin.
        # Public (Apache-2.0), so no token is needed; the loader reads the config
        # from the installed package, so only the .pt is provisioned here.
        # revision / sha256 stay None until pinned from a first authorized fetch.
        "efficienttam_s": {
            "repo_id": "yunyangx/efficient-track-anything",
            "filename": "efficienttam_s.pt",
            "revision": None,
            "sha256": None,
            "description": "EfficientTAM small checkpoint (RTSAM2 default; public)",
        },
        "efficienttam_ti": {
            "repo_id": "yunyangx/efficient-track-anything",
            "filename": "efficienttam_ti.pt",
            "revision": None,
            "sha256": None,
            "description": "EfficientTAM tiny checkpoint (RTSAM2; public)",
        },
    }


def download_model_cli() -> None:
    """CLI entry point for model-weight provisioning (``uv run download-model``)."""
    import click

    @click.group()
    def cli() -> None:
        """CUVIS.AI model-weight provisioning."""

    @cli.command("list")
    def list_cmd() -> None:
        """List registry models."""
        ModelWeights.list_models()

    @cli.command()
    @click.argument("name", required=False)
    @click.option("--repo-id", default=None, help="HF repo id (overrides registry).")
    @click.option(
        "--filename", default=None, help="File in the repo (overrides registry)."
    )
    @click.option("--revision", default=None, help="HF revision/commit to pin.")
    @click.option("--token", default=None, help="HF token (else $HF_TOKEN).")
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
