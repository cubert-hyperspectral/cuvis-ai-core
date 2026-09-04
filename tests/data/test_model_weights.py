"""Tests for the model-weight registry, resolver and provisioning downloader.

huggingface_hub's network call (``hf_hub_download``) is mocked with an attribute
patch; the cache lookup (``try_to_load_from_cache``) runs for real against
layouts built in ``tmp_path``, so the resolver is exercised against the actual
Hugging Face cache conventions without network access.

# gstack-shortcut(dec-39c56de9): no live mirror test, upgrade when
# mirror_weights.py --check runs scheduled or on the first mirror/pin
# provisioning failure
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path

import pytest

from cuvis_ai_core.data.model_weights import (
    HF_ORG,
    ModelDownloadError,
    ModelWeights,
    ModelWeightsMissingError,
)

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _fake_download_factory(content: bytes = b"weights-bytes"):
    def _fake(
        repo_id,
        filename,
        revision=None,
        token=None,
        cache_dir=None,
        force_download=False,
    ):
        # Emulate HF's cache layout so the downloader's default-revision aliasing
        # (which parses <cache>/models--*/snapshots/<commit>/<file>) sees a real path.
        commit = revision or "0" * 40
        target = (
            Path(cache_dir)
            / ("models--" + repo_id.replace("/", "--"))
            / "snapshots"
            / commit
            / filename
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        _fake.calls.append(
            {
                "repo_id": repo_id,
                "filename": filename,
                "revision": revision,
                "token": token,
                "cache_dir": cache_dir,
                "force_download": force_download,
            }
        )
        return str(target)

    _fake.calls = []
    return _fake


def _seed_cache(
    cache: Path, repo_id: str, revision: str, filename: str, content: bytes
):
    """Build a Hugging Face cache layout for one file at a pinned commit."""
    repo_dir = cache / ("models--" + repo_id.replace("/", "--"))
    target = repo_dir / "snapshots" / revision / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    # A provisioned cache also carries refs/main (core writes it), so a lookup
    # by the default revision resolves the snapshot too.
    ref = repo_dir / "refs" / "main"
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_text(revision)
    return target


def _spec(name: str) -> dict:
    return ModelWeights._models[name]


def _pinned_revision(name: str) -> str:
    return _spec(name)["revision"] or "0" * 40


@pytest.fixture
def no_network(monkeypatch):
    """Fail loudly if anything reaches for hf_hub_download."""

    def _boom(*args, **kwargs):  # pragma: no cover - only hit on regression
        raise AssertionError("hf_hub_download must not be called")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", _boom)


# ----------------------------------------------------------------------------
# Registry invariants
# ----------------------------------------------------------------------------


def test_every_registry_entry_points_at_the_cubert_mirrors():
    """Hard cut-over: no upstream repo id may remain in the registry."""
    for name, spec in ModelWeights._models.items():
        assert spec["repo_id"].startswith(f"{HF_ORG}/"), name


def test_every_registry_entry_is_pinned():
    """Mirror commit and upstream sha256 are recorded for every file."""
    for name, spec in ModelWeights._models.items():
        assert spec["revision"] and _HEX40.match(spec["revision"]), name
        assert _HEX64.match(spec["sha256"]), name
        for aux, sha in (spec.get("aux_files") or {}).items():
            assert _HEX64.match(sha), f"{name}/{aux}"
        assert spec["plugin"].startswith("cuvis-ai-"), name
        assert spec["license"], name


def test_registry_covers_every_plugin_weight():
    expected = {
        "sam3",
        "efficienttam_s",
        "efficienttam_ti",
        "efficienttam_s_512x512",
        "efficienttam_ti_512x512",
        "dinov2_vitb14_reg4",
        "clip_vit_l_14_336",
        "adaclip_all",
        "adaclip_mvtec_colondb",
        "adaclip_visa_clinicdb",
    }
    assert set(ModelWeights._models) == expected


# ----------------------------------------------------------------------------
# download_model (provisioning)
# ----------------------------------------------------------------------------


def test_download_forwards_token_and_records_registry_spec(
    monkeypatch, tmp_path, capsys
):
    fake = _fake_download_factory()
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake)
    # The sam3 entry pins a real sha256; the fake returns placeholder bytes, so
    # skip the content check here (sha validation has its own tests).
    monkeypatch.setattr(
        ModelWeights, "_validate_sha", classmethod(lambda cls, *a, **k: None)
    )
    monkeypatch.setenv("HF_TOKEN", "tok-123")

    resolved = ModelWeights.download_model("sam3", cache_dir=tmp_path / "cache")

    # Main checkpoint call carries the mirror repo / filename / revision / token.
    main = next(c for c in fake.calls if c["filename"] == "sam3.pt")
    assert main["repo_id"] == f"{HF_ORG}/sam3"
    assert main["token"] == "tok-123"
    assert main["revision"] == _spec("sam3")["revision"]
    # Companion config.json is provisioned into the same cache so an offline
    # child resolves the whole SAM3 set, not just the checkpoint.
    assert any(c["filename"] == "config.json" for c in fake.calls)
    assert Path(resolved).exists()
    # Output contract: stdout is the resolved path only (last line).
    out = capsys.readouterr().out.strip().splitlines()
    assert out[-1] == str(resolved)


def test_explicit_token_overrides_env(monkeypatch, tmp_path):
    fake = _fake_download_factory()
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake)
    monkeypatch.setenv("HF_TOKEN", "env-tok")

    ModelWeights.download_model(
        repo_id="acme/model",
        filename="m.pt",
        token="explicit",
        cache_dir=tmp_path / "c",
    )
    assert fake.calls[-1]["token"] == "explicit"
    assert fake.calls[-1]["repo_id"] == "acme/model"


def test_revision_is_pinned(monkeypatch, tmp_path):
    fake = _fake_download_factory()
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake)

    ModelWeights.download_model(
        repo_id="a/b", filename="w.pt", revision="abc123", cache_dir=tmp_path / "c"
    )
    assert fake.calls[-1]["revision"] == "abc123"


def test_download_aliases_default_revision_to_fetched_commit(monkeypatch, tmp_path):
    """hf_hub_download(revision=<commit>) writes no refs/main; the downloader must
    alias the default revision to the fetched commit so an offline loader that
    requests the default resolves the snapshot."""
    fake = _fake_download_factory()
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake)

    commit = "a" * 40
    cache = tmp_path / "cache"
    ModelWeights.download_model(
        repo_id="acme/model", filename="w.pt", revision=commit, cache_dir=cache
    )
    assert (cache / "models--acme--model" / "refs" / "main").read_text() == commit


def test_alias_written_even_without_pinned_revision(monkeypatch, tmp_path):
    """The alias is driven by the resolved snapshot commit, not the revision arg, so
    it is written for any provisioned model -- not only commit-pinned ones."""
    fake = _fake_download_factory()
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake)

    cache = tmp_path / "cache"
    ModelWeights.download_model(repo_id="acme/model", filename="w.pt", cache_dir=cache)
    # The fake resolves an unspecified revision to a placeholder commit.
    assert (cache / "models--acme--model" / "refs" / "main").read_text() == "0" * 40


def test_efficienttam_registry_entries_resolve_mirror_repo(monkeypatch, tmp_path):
    """The four EfficientTAM variants live in one public mirror repo and carry no
    aux files -- the loader reads the config from the installed package, so only
    the .pt is provisioned and no token is required."""
    fake = _fake_download_factory()
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake)
    # The entries pin a real sha256; the fake returns placeholder bytes, so skip
    # the content check here (sha validation has its own tests).
    monkeypatch.setattr(
        ModelWeights, "_validate_sha", classmethod(lambda cls, *a, **k: None)
    )
    monkeypatch.delenv("HF_TOKEN", raising=False)

    for name in (
        "efficienttam_s",
        "efficienttam_ti",
        "efficienttam_s_512x512",
        "efficienttam_ti_512x512",
    ):
        fake.calls.clear()
        ModelWeights.download_model(name, cache_dir=tmp_path / name)
        assert len(fake.calls) == 1  # only the checkpoint; no companion aux files
        call = fake.calls[0]
        assert call["repo_id"] == f"{HF_ORG}/efficient-track-anything"
        assert call["filename"] == f"{name}.pt"
        assert call["revision"] == _spec(name)["revision"]
        assert call["token"] is None


def test_out_copy_places_standalone_file(monkeypatch, tmp_path):
    fake = _fake_download_factory(content=b"payload")
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake)

    out = tmp_path / "ship" / "w.pt"
    resolved = ModelWeights.download_model(
        repo_id="a/b", filename="w.pt", cache_dir=tmp_path / "c", out=out
    )
    assert Path(resolved) == out
    assert out.read_bytes() == b"payload"


def test_sha_mismatch_raises(monkeypatch, tmp_path):
    fake = _fake_download_factory(content=b"actual")
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake)

    with pytest.raises(ModelDownloadError, match="sha256 mismatch"):
        ModelWeights.download_model(
            repo_id="a/b",
            filename="w.pt",
            sha256="deadbeef",
            cache_dir=tmp_path / "c",
        )


def test_sha_match_passes(monkeypatch, tmp_path):
    content = b"exact-bytes"
    expected = hashlib.sha256(content).hexdigest()
    fake = _fake_download_factory(content=content)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake)

    resolved = ModelWeights.download_model(
        repo_id="a/b", filename="w.pt", sha256=expected, cache_dir=tmp_path / "c"
    )
    assert Path(resolved).exists()


def test_fetch_validates_companion_sha(monkeypatch, tmp_path):
    """A companion file with the wrong hash is a broken cache, not a success."""
    content = b"primary"
    fake = _fake_download_factory(content=content)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake)
    spec = {
        "repo_id": "a/b",
        "filename": "w.pt",
        "revision": "c" * 40,
        "sha256": hashlib.sha256(content).hexdigest(),
        "aux_files": {"config.json": "0" * 64},  # fake writes b"primary" here too
    }
    with pytest.raises(ModelDownloadError, match="config.json"):
        ModelWeights._fetch(spec, token=None, cache_dir=tmp_path / "c", force=False)


def test_default_cache_dir_follows_hf_env(monkeypatch, tmp_path):
    """The default provisioning dir tracks the child's HF cache resolution.

    So a weight pulled with no explicit ``--cache-dir`` lands exactly where the
    offline child (which resolves the same way) will look for it.
    """
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    assert ModelWeights._default_cache_dir() == tmp_path / "hub"

    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "home"))
    assert ModelWeights._default_cache_dir() == tmp_path / "home" / "hub"


def test_unknown_registry_name_raises(tmp_path):
    with pytest.raises(ModelDownloadError, match="Unknown model"):
        ModelWeights.download_model("does-not-exist", cache_dir=tmp_path / "c")


def test_missing_repo_or_filename_raises(tmp_path):
    with pytest.raises(ModelDownloadError, match="repo-id and --filename"):
        ModelWeights.download_model(repo_id="a/b", cache_dir=tmp_path / "c")


def test_gated_repo_error_maps_to_license_message(monkeypatch, tmp_path):
    from unittest.mock import MagicMock

    from huggingface_hub.utils import GatedRepoError

    def _raise(*args, **kwargs):
        raise GatedRepoError("gated", response=MagicMock(status_code=403))

    monkeypatch.setattr("huggingface_hub.hf_hub_download", _raise)

    with pytest.raises(ModelDownloadError, match="gated"):
        ModelWeights.download_model(
            repo_id="acme/private", filename="w.pt", cache_dir=tmp_path / "c"
        )


def test_http_401_maps_to_token_message(monkeypatch, tmp_path):
    from unittest.mock import MagicMock

    from huggingface_hub.utils import HfHubHTTPError

    def _raise(*args, **kwargs):
        raise HfHubHTTPError("unauthorized", response=MagicMock(status_code=401))

    monkeypatch.setattr("huggingface_hub.hf_hub_download", _raise)

    with pytest.raises(ModelDownloadError, match="401"):
        ModelWeights.download_model(
            repo_id="a/b", filename="w.pt", cache_dir=tmp_path / "c"
        )


def test_http_error_maps_to_generic_message(monkeypatch, tmp_path):
    from unittest.mock import MagicMock

    from huggingface_hub.utils import HfHubHTTPError

    def _raise(*args, **kwargs):
        raise HfHubHTTPError("boom", response=MagicMock(status_code=500))

    monkeypatch.setattr("huggingface_hub.hf_hub_download", _raise)

    with pytest.raises(ModelDownloadError, match="failed"):
        ModelWeights.download_model(
            repo_id="a/b", filename="w.pt", cache_dir=tmp_path / "c"
        )


# ----------------------------------------------------------------------------
# resolve (consumption)
# ----------------------------------------------------------------------------


def test_resolve_returns_cached_primary_without_downloading(tmp_path, no_network):
    cache = tmp_path / "hub"
    seeded = _seed_cache(
        cache, f"{HF_ORG}/sam3", _pinned_revision("sam3"), "sam3.pt", b"cached"
    )

    assert ModelWeights.resolve("sam3", cache_dir=cache) == seeded


def test_resolve_offline_miss_names_the_provisioning_command(
    monkeypatch, tmp_path, no_network
):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    cache = tmp_path / "hub"

    with pytest.raises(ModelWeightsMissingError) as exc:
        ModelWeights.resolve("sam3", cache_dir=cache)
    message = str(exc.value)
    assert "download-model download sam3" in message
    assert str(cache) in message
    # It is a ModelDownloadError too, so existing handlers keep working.
    assert isinstance(exc.value, ModelDownloadError)


def test_resolve_download_false_never_downloads_even_online(
    monkeypatch, tmp_path, no_network
):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)

    with pytest.raises(ModelWeightsMissingError):
        ModelWeights.resolve("sam3", download=False, cache_dir=tmp_path / "hub")


def test_resolve_online_miss_downloads_from_the_mirror(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    fake = _fake_download_factory()
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake)
    monkeypatch.setattr(
        ModelWeights, "_validate_sha", classmethod(lambda cls, *a, **k: None)
    )
    cache = tmp_path / "hub"

    resolved = ModelWeights.resolve("sam3", cache_dir=cache)

    assert resolved.exists()
    assert fake.calls[0]["repo_id"] == f"{HF_ORG}/sam3"
    assert fake.calls[0]["token"] is None  # public mirror, no token needed
    assert {c["filename"] for c in fake.calls} == {"sam3.pt", "config.json"}
    # Second call is a cache hit: no further downloads.
    fake.calls.clear()
    assert ModelWeights.resolve("sam3", cache_dir=cache) == resolved
    assert fake.calls == []
    # resolve() never writes to stdout (it runs inside node constructors).
    assert capsys.readouterr().out == ""


def test_resolve_treats_no_exist_marker_as_a_miss(monkeypatch, tmp_path, no_network):
    """huggingface_hub returns a sentinel object (not None) for a cached
    "file does not exist" marker; that is a miss, not a path."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    cache = tmp_path / "hub"
    revision = _pinned_revision("sam3")
    marker = cache / f"models--{HF_ORG}--sam3" / ".no_exist" / revision / "sam3.pt"
    marker.parent.mkdir(parents=True)
    marker.touch()

    with pytest.raises(ModelWeightsMissingError):
        ModelWeights.resolve("sam3", cache_dir=cache)


def test_resolve_ignores_a_cache_provisioned_under_the_upstream_repo_id(
    monkeypatch, tmp_path, no_network
):
    """Regression for the lockstep failure: a cache filled under the old
    facebook/sam3 folder must not satisfy the cubert-gmbh/sam3 lookup."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    cache = tmp_path / "hub"
    _seed_cache(cache, "facebook/sam3", _pinned_revision("sam3"), "sam3.pt", b"old")

    with pytest.raises(ModelWeightsMissingError):
        ModelWeights.resolve("sam3", cache_dir=cache)


def test_resolve_unknown_name_raises(tmp_path, no_network):
    with pytest.raises(ModelDownloadError, match="Unknown model"):
        ModelWeights.resolve("nope", cache_dir=tmp_path / "hub")


# ----------------------------------------------------------------------------
# materialize
# ----------------------------------------------------------------------------


def test_materialize_returns_existing_destination_untouched(tmp_path, no_network):
    dest = tmp_path / "vendored"
    dest.mkdir()
    existing = dest / "ViT-L-14-336px.pt"
    existing.write_bytes(b"already here")

    # Empty cache and no download allowed: the existing file short-circuits.
    got = ModelWeights.materialize(
        "clip_vit_l_14_336", dest, download=False, cache_dir=tmp_path / "hub"
    )
    assert got == existing
    assert existing.read_bytes() == b"already here"


def test_materialize_links_the_cached_file(tmp_path, no_network):
    cache = tmp_path / "hub"
    _seed_cache(
        cache,
        f"{HF_ORG}/clip",
        _pinned_revision("clip_vit_l_14_336"),
        "ViT-L-14-336px.pt",
        b"clip-bytes",
    )
    dest = tmp_path / "vendored"

    got = ModelWeights.materialize("clip_vit_l_14_336", dest, cache_dir=cache)

    assert got == dest / "ViT-L-14-336px.pt"
    assert got.read_bytes() == b"clip-bytes"
    assert not list(dest.glob("*.part"))


def test_materialize_honours_a_custom_filename(tmp_path, no_network):
    cache = tmp_path / "hub"
    _seed_cache(
        cache,
        f"{HF_ORG}/dinov2",
        _pinned_revision("dinov2_vitb14_reg4"),
        "dinov2_vitb14_reg4_pretrain.pth",
        b"dino",
    )
    got = ModelWeights.materialize(
        "dinov2_vitb14_reg4", tmp_path / "d", filename="backbone.pth", cache_dir=cache
    )
    assert got == tmp_path / "d" / "backbone.pth"
    assert got.read_bytes() == b"dino"


def test_materialize_copies_when_hardlink_is_impossible(
    monkeypatch, tmp_path, no_network
):
    cache = tmp_path / "hub"
    _seed_cache(
        cache,
        f"{HF_ORG}/clip",
        _pinned_revision("clip_vit_l_14_336"),
        "ViT-L-14-336px.pt",
        b"clip-bytes",
    )

    def _no_link(*args, **kwargs):
        raise OSError("cross-device link")

    monkeypatch.setattr(os, "link", _no_link)

    got = ModelWeights.materialize("clip_vit_l_14_336", tmp_path / "v", cache_dir=cache)
    assert got.read_bytes() == b"clip-bytes"
    assert not list((tmp_path / "v").glob("*.part"))


def test_materialize_interrupted_copy_leaves_no_destination(
    monkeypatch, tmp_path, no_network
):
    cache = tmp_path / "hub"
    _seed_cache(
        cache,
        f"{HF_ORG}/clip",
        _pinned_revision("clip_vit_l_14_336"),
        "ViT-L-14-336px.pt",
        b"clip-bytes",
    )

    def _no_link(*args, **kwargs):
        raise OSError("cross-device link")

    def _disk_full(src, dst, *args, **kwargs):
        Path(dst).write_bytes(b"clip")  # truncated write, then failure
        raise OSError("disk full")

    monkeypatch.setattr(os, "link", _no_link)
    monkeypatch.setattr(shutil, "copyfile", _disk_full)

    with pytest.raises(OSError, match="disk full"):
        ModelWeights.materialize("clip_vit_l_14_336", tmp_path / "v", cache_dir=cache)
    assert not (tmp_path / "v" / "ViT-L-14-336px.pt").exists()
    assert not list((tmp_path / "v").glob("*.part"))


def test_materialize_offline_miss_raises(monkeypatch, tmp_path, no_network):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(ModelWeightsMissingError):
        ModelWeights.materialize(
            "clip_vit_l_14_336", tmp_path / "v", cache_dir=tmp_path / "hub"
        )


# ----------------------------------------------------------------------------
# listing
# ----------------------------------------------------------------------------


def test_list_models_prints_registry(capsys):
    ModelWeights.list_models()

    out = capsys.readouterr().out
    assert "sam3" in out
    assert f"{HF_ORG}/sam3" in out
    assert "facebook/sam3/sam3.pt" not in out  # repo column, not the description


def test_list_models_json_payload(capsys):
    ModelWeights.list_models(as_json=True)

    rows = json.loads(capsys.readouterr().out)
    by_name = {row["name"]: row for row in rows}
    assert set(by_name) == set(ModelWeights._models)
    sam3 = by_name["sam3"]
    assert set(sam3) == {
        "name",
        "plugin",
        "repo_id",
        "filename",
        "revision",
        "sha256",
        "aux_files",
        "license",
        "requires_token",
        "cache_dir_token",
        "description",
    }
    assert sam3["cache_dir_token"] == f"models--{HF_ORG}--sam3"
    assert sam3["requires_token"] is False
    assert sam3["plugin"] == "cuvis-ai-sam3"
    assert "config.json" in sam3["aux_files"]
    assert by_name["efficienttam_s"]["aux_files"] == {}


def test_require_hf_hub_missing_raises(monkeypatch):
    import sys

    # A None entry in sys.modules makes ``import huggingface_hub`` raise ImportError,
    # which the seam turns into an actionable install hint.
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)

    with pytest.raises(ModelDownloadError, match="huggingface_hub is not installed"):
        ModelWeights._require_hf_hub()


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def test_cli_list_prints_registry(monkeypatch, capsys):
    import sys

    from cuvis_ai_core.data.model_weights import download_model_cli

    monkeypatch.setattr(sys, "argv", ["download-model", "list"])

    with pytest.raises(SystemExit) as exc:
        download_model_cli()

    assert exc.value.code == 0
    assert "sam3" in capsys.readouterr().out


def test_cli_list_json(monkeypatch, capsys):
    import sys

    from cuvis_ai_core.data.model_weights import download_model_cli

    monkeypatch.setattr(sys, "argv", ["download-model", "list", "--json"])

    with pytest.raises(SystemExit) as exc:
        download_model_cli()

    assert exc.value.code == 0
    rows = json.loads(capsys.readouterr().out)
    assert {row["name"] for row in rows} == set(ModelWeights._models)


def test_cli_download_invokes_downloader(monkeypatch, tmp_path):
    import sys

    from cuvis_ai_core.data.model_weights import download_model_cli

    fake = _fake_download_factory()
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download-model",
            "download",
            "--repo-id",
            "a/b",
            "--filename",
            "w.pt",
            "--cache-dir",
            str(tmp_path / "c"),
        ],
    )

    with pytest.raises(SystemExit) as exc:
        download_model_cli()

    assert exc.value.code == 0
    assert fake.calls[-1]["repo_id"] == "a/b"


def test_cli_download_error_exits_nonzero(monkeypatch):
    import sys

    from cuvis_ai_core.data.model_weights import download_model_cli

    monkeypatch.setattr(sys, "argv", ["download-model", "download", "does-not-exist"])

    with pytest.raises(SystemExit) as exc:
        download_model_cli()

    assert "error:" in str(exc.value)
