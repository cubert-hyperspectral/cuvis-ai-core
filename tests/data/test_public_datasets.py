"""Tests for public dataset registry and downloader helpers."""

from __future__ import annotations

import builtins
import sys
from pathlib import Path
from types import ModuleType

import pytest

from cuvis_ai_core.data.public_datasets import PublicDatasets


def _install_fake_hf(
    monkeypatch: pytest.MonkeyPatch,
    *,
    download_fn,
) -> None:
    module = ModuleType("huggingface_hub")
    module.snapshot_download = download_fn
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)


def test_download_dataset_rejects_unknown_name(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert PublicDatasets.download_dataset("missing") is False

    out = capsys.readouterr().out
    assert "Dataset 'missing' not found." in out
    assert "Lentils_Anomaly" in out
    assert "Blood_Perfusion" in out


def test_download_dataset_skips_existing_directory(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    target = tmp_path / PublicDatasets.get_target_dir("lentils")
    target.mkdir(parents=True)

    assert (
        PublicDatasets.download_dataset("lentils", download_path=str(tmp_path)) is True
    )

    out = capsys.readouterr().out
    assert "already exists" in out
    assert "force=True" in out


def test_download_dataset_handles_missing_huggingface_hub(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "huggingface_hub":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    assert (
        PublicDatasets.download_dataset(
            "Lentils_Anomaly",
            download_path=str(tmp_path / "downloads"),
            force=True,
        )
        is False
    )

    out = capsys.readouterr().out
    assert "huggingface_hub is not installed." in out


def test_download_dataset_success_and_failure_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[dict[str, str]] = []

    def _fake_snapshot_download(
        *, repo_id: str, repo_type: str, local_dir: str
    ) -> None:
        calls.append(
            {"repo_id": repo_id, "repo_type": repo_type, "local_dir": local_dir}
        )

    _install_fake_hf(monkeypatch, download_fn=_fake_snapshot_download)
    download_root = tmp_path / "downloads"

    assert (
        PublicDatasets.download_dataset(
            "Blood_Perfusion",
            download_path=str(download_root),
            force=True,
        )
        is True
    )

    assert calls == [
        {
            "repo_id": "cubert-gmbh/XMR_Blood_Perfusion",
            "repo_type": "dataset",
            "local_dir": str(download_root / "XMR_Blood_Perfusion"),
        }
    ]
    out = capsys.readouterr().out
    assert "Downloaded 'Blood_Perfusion' successfully." in out

    def _failing_snapshot_download(**kwargs) -> None:
        raise RuntimeError("boom")

    _install_fake_hf(monkeypatch, download_fn=_failing_snapshot_download)

    assert (
        PublicDatasets.download_dataset(
            "Blood_Perfusion",
            download_path=str(download_root),
            force=True,
        )
        is False
    )
    out = capsys.readouterr().out
    assert "Download failed: boom" in out
    assert "Manual download:" in out


def test_list_datasets_verbose_and_canonical_names(
    capsys: pytest.CaptureFixture[str],
) -> None:
    canonical = PublicDatasets._canonical_names()

    PublicDatasets.list_datasets(verbose=True)

    out = capsys.readouterr().out
    assert canonical == ["Lentils_Anomaly", "Blood_Perfusion"]
    assert "(alias: lentils)" in out
    assert "(alias: blood_perfusion)" in out
    assert "repo:" in out
    assert "dir:" in out
