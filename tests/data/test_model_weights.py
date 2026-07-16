"""Tests for the model-weight provisioning downloader (``download-model``).

huggingface_hub is mocked (attribute patch on ``huggingface_hub.hf_hub_download``)
so no network / gated access is needed; these assert the seam behaviour: token
forwarding, revision pinning, sha validation, output contract, and error mapping.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cuvis_ai_core.data.model_weights import ModelDownloadError, ModelWeights


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

    # Main checkpoint call carries the registry repo / filename / revision / token.
    main = next(c for c in fake.calls if c["filename"] == "sam3.pt")
    assert main["repo_id"] == "facebook/sam3"
    assert main["token"] == "tok-123"
    assert main["revision"] == "3c879f39826c281e95690f02c7821c4de09afae7"
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
    import hashlib

    content = b"exact-bytes"
    expected = hashlib.sha256(content).hexdigest()
    fake = _fake_download_factory(content=content)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake)

    resolved = ModelWeights.download_model(
        repo_id="a/b", filename="w.pt", sha256=expected, cache_dir=tmp_path / "c"
    )
    assert Path(resolved).exists()


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
        ModelWeights.download_model("sam3", cache_dir=tmp_path / "c")


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


def test_list_models_prints_registry(capsys):
    ModelWeights.list_models()

    out = capsys.readouterr().out
    assert "sam3" in out
    assert "facebook/sam3" in out


def test_require_hf_hub_missing_raises(monkeypatch):
    import sys

    # A None entry in sys.modules makes ``import huggingface_hub`` raise ImportError,
    # which the seam turns into an actionable install hint.
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)

    with pytest.raises(ModelDownloadError, match="huggingface_hub is not installed"):
        ModelWeights._require_hf_hub()


def test_cli_list_prints_registry(monkeypatch, capsys):
    import sys

    from cuvis_ai_core.data.model_weights import download_model_cli

    monkeypatch.setattr(sys, "argv", ["download-model", "list"])

    with pytest.raises(SystemExit) as exc:
        download_model_cli()

    assert exc.value.code == 0
    assert "sam3" in capsys.readouterr().out


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

    monkeypatch.setattr(
        sys, "argv", ["download-model", "download", "does-not-exist"]
    )

    with pytest.raises(SystemExit) as exc:
        download_model_cli()

    assert "error:" in str(exc.value)
