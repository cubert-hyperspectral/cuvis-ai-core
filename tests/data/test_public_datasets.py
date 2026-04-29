"""Tests for public dataset registry and downloader helpers."""

from __future__ import annotations

import builtins
import shutil
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
    assert canonical == ["Lentils_Anomaly", "Blood_Perfusion", "Demo_Object_Tracking"]
    assert "(alias: lentils)" in out
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
    assert "Lentils_Anomaly" in out
    assert "Blood_Perfusion" in out
    assert "repo:" in out


def test_download_data_cli_validates_and_creates_lentils_symlink(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    created_links: list[tuple[Path, str, bool]] = []
    path_type = type(tmp_path)
    original_exists = path_type.exists

    def _fake_download_dataset(
        name: str, *, download_path: str = ".", force: bool = False
    ) -> bool:
        del force
        target = Path(download_path) / PublicDatasets.get_target_dir(name)
        target.mkdir(parents=True, exist_ok=True)
        (target / "sample.cu3s").write_text("", encoding="utf-8")
        return True

    def _fake_symlink_to(self: Path, target: str, target_is_directory: bool = False):
        created_links.append((self, target, target_is_directory))

    def _fake_exists(self: Path) -> bool:
        if self.parent == tmp_path and self.name == "lentils":
            return False
        return original_exists(self)

    monkeypatch.setattr(
        PublicDatasets, "download_dataset", staticmethod(_fake_download_dataset)
    )
    monkeypatch.setattr(Path, "symlink_to", _fake_symlink_to)
    monkeypatch.setattr(path_type, "exists", _fake_exists)
    monkeypatch.setattr(
        sys,
        "argv",
        ["dataset", "download", "lentils", "--data-dir", str(tmp_path), "--force"],
    )

    with pytest.raises(SystemExit) as exc_info:
        download_data_cli()

    _assert_cli_success(exc_info)
    assert created_links == [(tmp_path / "lentils", "Lentils", True)]


def test_download_data_cli_copies_when_symlink_creation_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    copied: list[tuple[Path, Path]] = []
    path_type = type(tmp_path)
    original_exists = path_type.exists

    def _fake_download_dataset(
        name: str, *, download_path: str = ".", force: bool = False
    ) -> bool:
        del force
        target = Path(download_path) / PublicDatasets.get_target_dir(name)
        target.mkdir(parents=True, exist_ok=True)
        return True

    def _failing_symlink(self: Path, target: str, target_is_directory: bool = False):
        del self, target, target_is_directory
        raise OSError("no symlink privileges")

    def _fake_copytree(src: Path | str, dst: Path | str) -> None:
        src_path = Path(src)
        dst_path = Path(dst)
        copied.append((src_path, dst_path))
        dst_path.mkdir(parents=True, exist_ok=True)

    def _fake_exists(self: Path) -> bool:
        if self.parent == tmp_path and self.name == "lentils":
            return False
        return original_exists(self)

    monkeypatch.setattr(
        PublicDatasets, "download_dataset", staticmethod(_fake_download_dataset)
    )
    monkeypatch.setattr(Path, "symlink_to", _failing_symlink)
    monkeypatch.setattr(path_type, "exists", _fake_exists)
    monkeypatch.setattr(shutil, "copytree", _fake_copytree)
    monkeypatch.setattr(
        sys,
        "argv",
        ["dataset", "download", "lentils", "--data-dir", str(tmp_path)],
    )

    with pytest.raises(SystemExit) as exc_info:
        download_data_cli()

    _assert_cli_success(exc_info)
    assert copied == [(tmp_path / "Lentils", tmp_path / "lentils")]


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
        ["dataset", "download", "lentils", "--data-dir", str(tmp_path)],
    )

    with pytest.raises(SystemExit) as exc_info:
        download_data_cli()

    _assert_cli_success(exc_info)
