"""Tests for public dataset registry and downloader helpers."""

from __future__ import annotations

import builtins
import sys
from pathlib import Path
from types import ModuleType

import pytest

from cuvis_ai_core.data.public_datasets import PublicDatasets, download_data_cli


def _install_fake_hf(
    monkeypatch: pytest.MonkeyPatch,
    *,
    download_fn,
) -> None:
    module = ModuleType("huggingface_hub")
    module.snapshot_download = download_fn
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)


def _assert_cli_success(exc_info: pytest.ExceptionInfo[SystemExit]) -> None:
    assert exc_info.value.code in (0, None)


def test_download_dataset_rejects_unknown_name(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert PublicDatasets.download_dataset("missing") is False

    out = capsys.readouterr().out
    assert "Dataset 'missing' not found." in out
    assert "Demo_Industrial_FOD_Lentils" in out
    assert "Blood_Perfusion" in out


def test_download_dataset_skips_existing_directory(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    target = tmp_path / PublicDatasets.get_target_dir("demo_industrial_fod_lentils")
    target.mkdir(parents=True)

    assert (
        PublicDatasets.download_dataset(
            "demo_industrial_fod_lentils", download_path=str(tmp_path)
        )
        is True
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
            "Demo_Industrial_FOD_Lentils",
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
            "repo_id": "cubert-gmbh/XMR_Demo_Blood_Perfusion",
            "repo_type": "dataset",
            "local_dir": str(download_root / "XMR_Demo_Blood_Perfusion"),
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


def test_lookup_accepts_hyphen_form(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    assert (
        PublicDatasets.get_target_dir("Demo-Object-Tracking")
        == "XMR_Demo_Object_Tracking"
    )
    assert (
        PublicDatasets.get_target_dir("blood-perfusion") == "XMR_Demo_Blood_Perfusion"
    )

    calls: list[dict[str, str]] = []

    def _fake_snapshot_download(
        *, repo_id: str, repo_type: str, local_dir: str
    ) -> None:
        calls.append({"repo_id": repo_id, "local_dir": local_dir})

    _install_fake_hf(monkeypatch, download_fn=_fake_snapshot_download)

    assert (
        PublicDatasets.download_dataset(
            "Demo-Object-Tracking",
            download_path=str(tmp_path / "downloads"),
            force=True,
        )
        is True
    )
    assert calls and calls[0]["repo_id"] == "cubert-gmbh/XMR_Demo_Object_Tracking"


def test_list_datasets_verbose_and_canonical_names(
    capsys: pytest.CaptureFixture[str],
) -> None:
    canonical = PublicDatasets._canonical_names()

    PublicDatasets.list_datasets(verbose=True)

    out = capsys.readouterr().out
    assert canonical == [
        "Lentils",
        "Demo_Industrial_FOD_Lentils",
        "Industrial_FOD_Lentils",
        "Blood_Perfusion",
        "Demo_Object_Tracking",
    ]
    assert "(alias: lentils)" in out
    assert "(alias: demo_industrial_fod_lentils)" in out
    assert "(alias: industrial_fod_lentils)" in out
    assert "(alias: blood_perfusion)" in out
    assert "alias: demo_object_tracking" in out
    assert "repo:" in out
    assert "dir:" in out


def test_download_data_cli_lists_datasets(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["dataset", "list", "--verbose"])

    with pytest.raises(SystemExit) as exc_info:
        download_data_cli()

    _assert_cli_success(exc_info)
    out = capsys.readouterr().out
    assert "Demo_Industrial_FOD_Lentils" in out
    assert "Blood_Perfusion" in out
    assert "repo:" in out


def test_download_data_cli_exits_nonzero_on_failed_download(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        PublicDatasets,
        "download_dataset",
        staticmethod(lambda *args, **kwargs: False),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["dataset", "download", "blood_perfusion", "--data-dir", str(tmp_path)],
    )

    with pytest.raises(SystemExit) as exc_info:
        download_data_cli()

    assert exc_info.value.code == 1


def test_download_data_cli_returns_when_target_dir_lookup_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        PublicDatasets, "download_dataset", staticmethod(lambda *args, **kwargs: True)
    )
    monkeypatch.setattr(
        PublicDatasets,
        "get_target_dir",
        staticmethod(lambda _name: (_ for _ in ()).throw(KeyError("missing"))),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dataset",
            "download",
            "demo_industrial_fod_lentils",
            "--data-dir",
            str(tmp_path),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        download_data_cli()

    _assert_cli_success(exc_info)
