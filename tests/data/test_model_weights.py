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
        target = Path(cache_dir) / filename
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
    monkeypatch.setenv("HF_TOKEN", "tok-123")

    resolved = ModelWeights.download_model("sam3", cache_dir=tmp_path / "cache")

    call = fake.calls[-1]
    assert call["repo_id"] == "facebook/sam3"
    assert call["filename"] == "sam3.pt"
    assert call["token"] == "tok-123"
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
