"""Tests for the public dataset registry, the marker contract, downloads and the CLI.

``huggingface_hub.snapshot_download`` and ``HfApi`` are replaced by fakes that
write a configurable file list into ``local_dir``; every marker and status path
then runs against real files under ``tmp_path`` without network.
"""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner
from filelock import FileLock

from cuvis_ai_core.data import _provisioning as prov
from cuvis_ai_core.data._datasets_cli import build_cli
from cuvis_ai_core.data.public_datasets import (
    CAMERAS,
    DATASET_TAGS,
    DATASETS,
    HF_ORG,
    MARKER_NAME,
    DatasetError,
    PublicDatasets,
)

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
LENTILS = PublicDatasets.get_spec("Lentils")
FILES = ["README.md", "Lentils_000.cu3s", "Lentils_000.info", "annotations/labels.json"]


class Recorder(prov.ProgressEmitter):
    def __init__(self) -> None:
        super().__init__(enabled=True)
        self.events: list[dict] = []

    def _write(self, event: dict) -> None:
        self.events.append(event)


@pytest.fixture
def fake_hub(monkeypatch):
    """Fake snapshot_download + HfApi.list_repo_files driven by a file list."""

    def factory(
        files: list[str] | None = None,
        *,
        fail_after: int | None = None,
        write: int | None = None,
    ):
        table = list(FILES if files is None else files)
        calls: list[dict] = []

        def _snapshot(
            repo_id,
            *,
            repo_type,
            revision,
            local_dir,
            token,
            force_download=False,
            tqdm_class=None,
        ):
            calls.append(
                {
                    "repo_id": repo_id,
                    "repo_type": repo_type,
                    "revision": revision,
                    "local_dir": local_dir,
                    "token": token,
                    "force_download": force_download,
                    "tqdm_class": tqdm_class,
                }
            )
            bar = tqdm_class(total=len(table), desc="files") if tqdm_class else None
            limit = len(table) if write is None else write
            for i, rel in enumerate(table[:limit]):
                if fail_after is not None and i >= fail_after:
                    raise OSError("connection reset")
                target = Path(local_dir) / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(rel.encode() * 3)
                if bar is not None:
                    bar.update(1)
            (Path(local_dir) / ".cache" / "huggingface").mkdir(
                parents=True, exist_ok=True
            )
            return local_dir

        class _Api:
            def list_repo_files(self, repo_id, *, repo_type, revision):
                return list(table)

        monkeypatch.setattr("huggingface_hub.snapshot_download", _snapshot)
        monkeypatch.setattr("huggingface_hub.HfApi", _Api)
        return calls

    return factory


def _marker(path: Path, payload: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / MARKER_NAME).write_text(json.dumps(payload), encoding="utf-8")


def _complete_marker(spec, files: list[str], revision: str | None = None) -> dict:
    return {
        "state": "complete",
        "repo_id": spec.repo_id,
        "revision": revision or spec.revision,
        "files": [{"path": f, "size_bytes": len(f.encode()) * 3} for f in files],
    }


def _write_files(path: Path, files: list[str]) -> None:
    for rel in files:
        (path / rel).parent.mkdir(parents=True, exist_ok=True)
        (path / rel).write_bytes(rel.encode() * 3)


# ----------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------


def test_registry_specs_pin_cubert_datasets():
    assert len(DATASETS) == 6
    names, aliases, dirs = set(), set(), set()
    for spec in DATASETS:
        assert spec.repo_id.startswith(f"{HF_ORG}/")
        assert _HEX40.match(spec.revision), spec.name
        assert spec.size_bytes > 0 and spec.file_count > 0
        assert spec.tags and set(spec.tags) <= set(DATASET_TAGS), spec.name
        assert spec.camera in CAMERAS
        assert len(spec.summary) <= 60 and spec.license == "Apache-2.0"
        for key in (spec.name, *spec.aliases):
            assert key not in names | aliases, key
        names.add(spec.name)
        aliases.update(spec.aliases)
        assert spec.target_dir not in dirs
        dirs.add(spec.target_dir)
    total_gib = sum(s.size_bytes for s in DATASETS) / 2**30
    assert 250 < total_gib < 300  # the six cards add up to about 259 GiB


def test_get_spec_accepts_aliases_hyphens_and_case():
    assert PublicDatasets.get_spec("lentils") is LENTILS
    assert (
        PublicDatasets.get_spec("Demo-Industrial-FOD-Lentils").name
        == "Demo_Industrial_FOD_Lentils"
    )
    assert PublicDatasets.get_spec("INDUSTRIAL_FOD_BEDDING").camera == "X4 SWIR"
    with pytest.raises(
        DatasetError, match="Dataset 'nope' not found. Available: Lentils"
    ):
        PublicDatasets.get_spec("nope")
    assert (
        PublicDatasets.get_target_dir("blood-perfusion") == "XMR_Demo_Blood_Perfusion"
    )
    with pytest.raises(KeyError):
        PublicDatasets.get_target_dir("nope")


def test_list_payload_validates_against_the_schema():
    jsonschema = pytest.importorskip("jsonschema")
    payload = PublicDatasets.list_payload()
    jsonschema.Draft202012Validator(prov.load_schema("dataset_list")).validate(payload)
    assert payload["tags"] == list(DATASET_TAGS) and payload["cameras"] == list(CAMERAS)
    assert [d["name"] for d in payload["datasets"]] == [s.name for s in DATASETS]


# ----------------------------------------------------------------------------
# Status: the marker contract
# ----------------------------------------------------------------------------


def test_status_maps_every_marker_state(tmp_path):
    target = tmp_path / LENTILS.target_dir
    assert PublicDatasets.status("Lentils", tmp_path).state == "absent"
    target.mkdir()
    assert PublicDatasets.status("Lentils", tmp_path).state == "absent"  # empty folder
    (target / ".cache" / "huggingface").mkdir(parents=True)
    (target / ".cache" / "huggingface" / "x.metadata").write_bytes(b"m")
    assert (
        PublicDatasets.status("Lentils", tmp_path).state == "absent"
    )  # hub bookkeeping only
    (target / "my_recording.cu3s").write_bytes(b"mine")
    st = PublicDatasets.status("Lentils", tmp_path)
    assert (
        st.state == "foreign"
        and st.present is False
        and st.bytes_on_disk
        and st.marker is None
    )
    _marker(
        target,
        {
            "state": "downloading",
            "repo_id": LENTILS.repo_id,
            "revision": LENTILS.revision,
        },
    )
    assert PublicDatasets.status("Lentils", tmp_path).state == "incomplete"
    (target / MARKER_NAME).write_text("{not json", encoding="utf-8")
    assert PublicDatasets.status("Lentils", tmp_path).state == "incomplete"
    _write_files(target, FILES)
    _marker(target, _complete_marker(LENTILS, FILES, revision="a" * 40))
    st = PublicDatasets.status("Lentils", tmp_path)
    assert st.state == "outdated" and st.files_ok is True
    _marker(target, _complete_marker(LENTILS, FILES))
    st = PublicDatasets.status("Lentils", tmp_path)
    assert st.state == "present" and st.present and st.files_ok is True
    assert st.bytes_on_disk == sum(len(f.encode()) * 3 for f in FILES)
    (target / FILES[1]).write_bytes(b"short")
    st = PublicDatasets.status("Lentils", tmp_path)
    assert st.state == "damaged" and st.files_ok is False
    (target / FILES[1]).unlink()
    assert PublicDatasets.status("Lentils", tmp_path).state == "damaged"
    assert PublicDatasets.status_all(tmp_path)[0].spec is LENTILS
    assert [
        s.spec.name for s in PublicDatasets.status_all(tmp_path, ["blood_perfusion"])
    ] == ["Blood_Perfusion"]


# ----------------------------------------------------------------------------
# Download
# ----------------------------------------------------------------------------


def test_download_writes_markers_manifest_and_is_idempotent(tmp_path, fake_hub):
    calls = fake_hub()
    result = PublicDatasets.download("lentils", tmp_path)
    target = tmp_path / LENTILS.target_dir
    assert result.path == target and result.stale_files == ()
    call = calls[0]
    assert call["repo_id"] == LENTILS.repo_id and call["repo_type"] == "dataset"
    assert (
        call["revision"] == LENTILS.revision
        and call["token"] is False
        and call["force_download"] is False
    )
    marker = json.loads((target / MARKER_NAME).read_text(encoding="utf-8"))
    assert marker["state"] == "complete" and marker["revision"] == LENTILS.revision
    assert marker["file_count"] == len(FILES) and marker["size_bytes"] == sum(
        len(f.encode()) * 3 for f in FILES
    )
    assert {f["path"] for f in marker["files"]} == set(
        FILES
    ) and "finished_at" in marker
    assert not list(target.glob("*.tmp"))
    assert PublicDatasets.status("Lentils", tmp_path).present
    PublicDatasets.download("Lentils", tmp_path)
    assert len(calls) == 1  # present: no second snapshot
    PublicDatasets.download("Lentils", tmp_path, force=True)
    assert len(calls) == 2 and calls[1]["force_download"] is True


def test_downloading_marker_is_written_before_snapshot_and_survives_a_failure(
    tmp_path, fake_hub
):
    fake_hub(fail_after=1)
    with pytest.raises(
        DatasetError, match="Download of 'Lentils'.*connection reset.*Manual download"
    ):
        PublicDatasets.download("Lentils", tmp_path)
    target = tmp_path / LENTILS.target_dir
    marker = json.loads((target / MARKER_NAME).read_text(encoding="utf-8"))
    assert (
        marker["state"] == "downloading"
        and marker["repo_id"] == LENTILS.repo_id
        and "started_at" in marker
    )
    assert PublicDatasets.status("Lentils", tmp_path).state == "incomplete"
    # The next call resumes into the same folder without --adopt.
    fake_hub()
    PublicDatasets.download("Lentils", tmp_path)
    assert PublicDatasets.status("Lentils", tmp_path).present


def test_incomplete_snapshot_never_becomes_complete(tmp_path, fake_hub):
    fake_hub(write=2)  # the hub returns but two files are missing
    with pytest.raises(DatasetError, match="2 file\\(s\\) of revision .* are missing"):
        PublicDatasets.download("Lentils", tmp_path)
    assert PublicDatasets.status("Lentils", tmp_path).state == "incomplete"


def test_foreign_folder_is_refused_unless_adopted(tmp_path, fake_hub):
    calls = fake_hub()
    target = tmp_path / LENTILS.target_dir
    target.mkdir()
    (target / "mine.cu3s").write_bytes(b"mine")
    with pytest.raises(DatasetError, match="is not a Cubert download; pass --adopt"):
        PublicDatasets.download("Lentils", tmp_path)
    assert calls == [] and (target / "mine.cu3s").exists()
    PublicDatasets.download("Lentils", tmp_path, adopt=True)
    assert (
        PublicDatasets.status("Lentils", tmp_path).present
        and (target / "mine.cu3s").exists()
    )


def test_outdated_refresh_reports_and_prunes_stale_files(tmp_path, fake_hub):
    target = tmp_path / LENTILS.target_dir
    old_files = FILES[:2] + ["old_split.json"]
    _write_files(target, old_files)
    _marker(target, _complete_marker(LENTILS, old_files, revision="b" * 40))
    assert PublicDatasets.status("Lentils", tmp_path).state == "outdated"
    fake_hub()
    result = PublicDatasets.download("Lentils", tmp_path)
    assert (
        result.stale_files == ("old_split.json",)
        and (target / "old_split.json").exists()
    )
    assert PublicDatasets.status("Lentils", tmp_path).present
    _marker(target, _complete_marker(LENTILS, old_files, revision="b" * 40))
    result = PublicDatasets.download("Lentils", tmp_path, prune_stale=True)
    assert result.stale_files == () and not (target / "old_split.json").exists()


def test_remove_requires_a_marker_naming_the_repo(tmp_path):
    target = tmp_path / LENTILS.target_dir
    assert PublicDatasets.remove("Lentils", tmp_path) == 0  # absent
    target.mkdir()
    (target / "mine.cu3s").write_bytes(b"mine")
    with pytest.raises(DatasetError, match="not a downloaded Cubert dataset"):
        PublicDatasets.remove("Lentils", tmp_path)
    _marker(
        target,
        {"state": "complete", "repo_id": "someone/else", "revision": LENTILS.revision},
    )
    with pytest.raises(DatasetError, match="refusing to delete"):
        PublicDatasets.remove("Lentils", tmp_path)
    _marker(
        target,
        {
            "state": "downloading",
            "repo_id": LENTILS.repo_id,
            "revision": LENTILS.revision,
        },
    )
    freed = PublicDatasets.remove("Lentils", tmp_path)
    assert freed >= 4 and not target.exists()


def test_progress_events_count_files_and_end_with_done(tmp_path, fake_hub):
    fake_hub()
    rec = Recorder()
    PublicDatasets.download("Lentils", tmp_path, progress=rec)
    kinds = [e["event"] for e in rec.events]
    assert kinds == ["progress"] * len(FILES) + ["done"]
    assert [e["files_done"] for e in rec.events[:-1]] == list(range(1, len(FILES) + 1))
    assert all(
        e["files_total"] == len(FILES) and e["bytes_done"] is None
        for e in rec.events[:-1]
    )
    assert rec.events[-2]["pct"] == 100.0
    jsonschema = pytest.importorskip("jsonschema")
    validator = jsonschema.Draft202012Validator(prov.load_schema("progress_event"))
    for event in rec.events:
        validator.validate(event)


def test_download_waits_for_the_folder_lock(tmp_path, fake_hub):
    fake_hub()
    held = FileLock(str(tmp_path / prov.LOCK_FILENAME))
    held.acquire(timeout=0)
    rec = Recorder()
    done = threading.Event()

    def worker():
        PublicDatasets.download("Lentils", tmp_path, progress=rec)
        done.set()

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    assert not done.wait(0.4)
    assert {"event": "waiting"} in rec.events
    held.release()
    assert done.wait(5)
    t.join()
    assert PublicDatasets.status("Lentils", tmp_path).present


def test_download_dataset_compat_wrapper_returns_bool(tmp_path, fake_hub, capsys):
    fake_hub()
    assert (
        PublicDatasets.download_dataset("Lentils", download_path=str(tmp_path)) is True
    )
    assert PublicDatasets.download_dataset("nope", download_path=str(tmp_path)) is False
    captured = capsys.readouterr()
    assert captured.out == "" and "not found" in captured.err


def test_library_methods_write_nothing_to_stdout(tmp_path, fake_hub, capsys):
    fake_hub()
    PublicDatasets.download("Lentils", tmp_path)
    PublicDatasets.status_all(tmp_path)
    PublicDatasets.remove("Lentils", tmp_path)
    assert capsys.readouterr().out == ""


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def _validate(name: str, payload) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.Draft202012Validator(prov.load_schema(name)).validate(payload)


@pytest.fixture
def cli_env(tmp_path):
    return SimpleNamespace(runner=CliRunner(), cli=build_cli(), data=tmp_path / "data")


def test_cli_list_json_and_table(cli_env):
    result = cli_env.runner.invoke(cli_env.cli, ["list", "--json"])
    assert result.exit_code == 0, result.output
    _validate("dataset_list", json.loads(result.stdout))
    table = cli_env.runner.invoke(cli_env.cli, ["list", "-v"])
    assert (
        table.exit_code == 0
        and "Industrial_FOD_Bedding" in table.stdout
        and "repo:" in table.stdout
    )


def test_cli_status_download_remove_json_shapes(cli_env, fake_hub):
    fake_hub()
    absent = cli_env.runner.invoke(
        cli_env.cli, ["status", "--data-dir", str(cli_env.data), "--json"]
    )
    assert absent.exit_code == 0, absent.output
    payload = json.loads(absent.stdout)
    _validate("dataset_status", payload)
    assert {d["state"] for d in payload["datasets"]} == {"absent"} and len(
        payload["datasets"]
    ) == 6
    downloaded = cli_env.runner.invoke(
        cli_env.cli, ["download", "lentils", "--data-dir", str(cli_env.data), "--json"]
    )
    assert downloaded.exit_code == 0, downloaded.output
    payload = json.loads(downloaded.stdout)
    _validate("dataset_status", payload)
    (row,) = payload["datasets"]
    assert (
        row["name"] == "Lentils"
        and row["state"] == "present"
        and row["downloaded"] is True
    )
    assert row["stale_files"] == []
    plain = cli_env.runner.invoke(
        cli_env.cli, ["download", "Lentils", "--data-dir", str(cli_env.data)]
    )
    assert plain.exit_code == 0 and plain.stdout.strip().endswith(LENTILS.target_dir)
    table = cli_env.runner.invoke(
        cli_env.cli, ["status", "Lentils", "--data-dir", str(cli_env.data)]
    )
    assert table.exit_code == 0 and "present" in table.stdout
    removed = cli_env.runner.invoke(
        cli_env.cli, ["remove", "Lentils", "--data-dir", str(cli_env.data), "--json"]
    )
    assert removed.exit_code == 0, removed.output
    assert json.loads(removed.stdout)["removed"][0]["name"] == "Lentils"
    assert not (cli_env.data / LENTILS.target_dir).exists()


def test_cli_progress_json_and_errors(cli_env, fake_hub):
    fake_hub()
    result = cli_env.runner.invoke(
        cli_env.cli,
        ["download", "Lentils", "--data-dir", str(cli_env.data), "--progress-json"],
    )
    assert result.exit_code == 0, result.output
    events = [json.loads(line) for line in result.stdout.splitlines()]
    assert events[-1]["event"] == "done" and events[0]["event"] == "progress"
    both = cli_env.runner.invoke(
        cli_env.cli,
        [
            "download",
            "Lentils",
            "--data-dir",
            str(cli_env.data),
            "--json",
            "--progress-json",
        ],
    )
    assert both.exit_code == 2
    unknown = cli_env.runner.invoke(
        cli_env.cli, ["download", "nope", "--data-dir", str(cli_env.data)]
    )
    assert unknown.exit_code == 1 and unknown.stderr.startswith(
        "error: Dataset 'nope' not found"
    )
    fake_hub(fail_after=0)
    failed = cli_env.runner.invoke(
        cli_env.cli,
        [
            "download",
            "Blood_Perfusion",
            "--data-dir",
            str(cli_env.data),
            "--progress-json",
        ],
    )
    assert failed.exit_code == 1
    last = json.loads(failed.stdout.splitlines()[-1])
    assert last["event"] == "error" and "connection reset" in last["message"]
    # The failed download left a folder with a `downloading` marker: ours, so removable.
    cleaned = cli_env.runner.invoke(
        cli_env.cli, ["remove", "Blood_Perfusion", "--data-dir", str(cli_env.data)]
    )
    assert cleaned.exit_code == 0 and "freed" in cleaned.stdout
    assert not (cli_env.data / "XMR_Demo_Blood_Perfusion").exists()
    # A same-named folder without a marker is somebody's data: refused.
    foreign = cli_env.data / "XMR_Demo_Object_Tracking"
    foreign.mkdir(parents=True)
    (foreign / "session.cu3s").write_bytes(b"mine")
    refused = cli_env.runner.invoke(
        cli_env.cli, ["remove", "Demo_Object_Tracking", "--data-dir", str(cli_env.data)]
    )
    assert (
        refused.exit_code == 1 and "not a downloaded Cubert dataset" in refused.stderr
    )
    assert (foreign / "session.cu3s").exists()


def test_cli_schema_prints_shipped_files(cli_env):
    for name in ("dataset_list", "dataset_status", "progress_event"):
        result = cli_env.runner.invoke(cli_env.cli, ["schema", name])
        assert result.exit_code == 0 and json.loads(result.stdout) == prov.load_schema(
            name
        )


def test_marker_write_is_atomic(tmp_path, monkeypatch):
    target = tmp_path / "d"
    target.mkdir()
    PublicDatasets._write_marker(target, {"state": "complete"})

    def _boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", _boom)
    with pytest.raises(OSError):
        PublicDatasets._write_marker(target, {"state": "downloading"})
    assert json.loads((target / MARKER_NAME).read_text())["state"] == "complete"
