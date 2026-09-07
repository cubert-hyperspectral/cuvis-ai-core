"""Tests for the shared provisioning plumbing: lock, progress protocol, schemas, helpers."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest
from filelock import FileLock

from cuvis_ai_core.data import _provisioning as prov


class Recorder(prov.ProgressEmitter):
    """A ProgressEmitter that keeps the events instead of writing stdout."""

    def __init__(self) -> None:
        super().__init__(enabled=True)
        self.events: list[dict] = []

    def _write(self, event: dict) -> None:
        self.events.append(event)


def test_format_bytes_iec_units():
    assert prov.format_bytes(None) == "-"
    assert prov.format_bytes(0) == "0 B"
    assert prov.format_bytes(1023) == "1023 B"
    assert prov.format_bytes(1024) == "1.0 KiB"
    assert prov.format_bytes(3_450_062_241) == "3.2 GiB"
    assert prov.format_bytes(5 * 1024**4) == "5.0 TiB"


def test_sha256_of_matches_hashlib(tmp_path):
    import hashlib

    p = tmp_path / "f.bin"
    p.write_bytes(b"x" * 3000)
    assert prov.sha256_of(p) == hashlib.sha256(b"x" * 3000).hexdigest()


def test_root_lock_creates_the_lock_file_and_releases(tmp_path):
    root = tmp_path / "cache"
    with prov.root_lock(root):
        assert (root / prov.LOCK_FILENAME).exists()
    # Released: a second, non-blocking acquire succeeds immediately.
    lock = FileLock(str(root / prov.LOCK_FILENAME))
    lock.acquire(timeout=0)
    lock.release()


def test_root_lock_waits_reports_and_times_out(tmp_path):
    root = tmp_path / "cache"
    root.mkdir()
    outer = FileLock(str(root / prov.LOCK_FILENAME))
    outer.acquire(timeout=0)
    waited: list[bool] = []
    try:
        with pytest.raises(
            prov.CacheBusyError, match="another operation is using the cache"
        ):
            with prov.root_lock(root, timeout=0.2, on_wait=lambda: waited.append(True)):
                pass
    finally:
        outer.release()
    assert waited == [True]
    # Once released the same call goes through and on_wait is not called.
    waited.clear()
    with prov.root_lock(root, timeout=0.2, on_wait=lambda: waited.append(True)):
        pass
    assert waited == []


def test_progress_poller_emits_monotonic_lines_from_a_growing_file(tmp_path):
    target = tmp_path / "blob.incomplete"
    rec = Recorder()
    poller = prov.ProgressPoller(
        rec, "sam3", 40, lambda: target if target.exists() else None, interval=0.02
    )
    poller.start()
    for i in range(1, 5):
        with target.open("ab") as fh:
            fh.write(b"0123456789")
        time.sleep(0.06)
    poller.stop()
    rec.done("sam3", target)
    progress = [e for e in rec.events if e["event"] == "progress"]
    assert progress, rec.events
    sizes = [e["bytes_done"] for e in progress]
    assert sizes == sorted(sizes) and sizes[-1] == 40
    assert all(e["bytes_total"] == 40 for e in progress)
    assert progress[-1]["pct"] == 100.0
    assert rec.events[-1]["event"] == "done"  # nothing follows the terminal event


def test_progress_poller_preallocated_file_emits_null_pct(tmp_path):
    target = tmp_path / "blob.incomplete"
    target.write_bytes(b"\0" * 100)  # reserved up front, nothing landed yet
    rec = Recorder()
    with prov.ProgressPoller(rec, "sam3", 100, lambda: target, interval=0.02):
        time.sleep(0.15)
    progress = [e for e in rec.events if e["event"] == "progress"]
    assert len(progress) == 1
    assert progress[0]["pct"] is None and progress[0]["bytes_done"] is None
    assert progress[0]["bytes_total"] == 100


def test_progress_poller_without_a_file_emits_nothing(tmp_path):
    rec = Recorder()
    with prov.ProgressPoller(rec, "sam3", 10, lambda: None, interval=0.01):
        time.sleep(0.05)
    assert rec.events == []


def test_progress_emitter_writes_one_json_object_per_line(capsys):
    emitter = prov.ProgressEmitter()
    emitter.waiting()
    emitter.progress("a", 1, 2, 50.0)
    emitter.files_progress("d", 3, 6)
    emitter.verifying("a")
    emitter.done("a", Path("/x/a.pt"))
    emitter.error("b", "boom")
    lines = capsys.readouterr().out.splitlines()
    events = [json.loads(line) for line in lines]
    assert [e["event"] for e in events] == [
        "waiting",
        "progress",
        "progress",
        "verifying",
        "done",
        "error",
    ]
    assert events[2] == {
        "event": "progress",
        "name": "d",
        "bytes_done": None,
        "bytes_total": None,
        "files_done": 3,
        "files_total": 6,
        "pct": 50.0,
    }
    disabled = prov.ProgressEmitter(enabled=False)
    disabled.done("a", "p")
    assert capsys.readouterr().out == ""


def test_progress_events_validate_against_the_shipped_schema(capsys):
    jsonschema = pytest.importorskip("jsonschema")
    validator = jsonschema.Draft202012Validator(prov.load_schema("progress_event"))
    emitter = prov.ProgressEmitter()
    emitter.waiting()
    emitter.progress("a", None, 10, None)
    emitter.files_progress("d", 1, None)
    emitter.verifying("a")
    emitter.done("a", "p")
    emitter.error("a", "m")
    for line in capsys.readouterr().out.splitlines():
        validator.validate(json.loads(line))


def test_newest_incomplete_blob_respects_the_time_bound(tmp_path):
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    stale = blobs / "old.incomplete"
    stale.write_bytes(b"1")
    import os

    os.utime(stale, (time.time() - 100, time.time() - 100))
    assert prov.newest_incomplete_blob(blobs, time.time() - 10) is None
    fresh = blobs / "new.incomplete"
    fresh.write_bytes(b"12")
    assert prov.newest_incomplete_blob(blobs, time.time() - 10) == fresh
    assert prov.newest_incomplete_blob(tmp_path / "missing", 0.0) is None


def test_schema_names_and_load_schema():
    names = prov.schema_names()
    assert {"model_list", "status", "export", "remove", "progress_event"} <= set(names)
    assert prov.load_schema("model_list")["properties"]["schema_version"] == {
        "const": 1
    }
    with pytest.raises(KeyError, match="unknown schema"):
        prov.load_schema("nope")


def test_status_schema_spec_matches_model_list_spec():
    """The SPEC is declared twice (two files); they must not drift apart."""
    spec = prov.load_schema("model_list")["$defs"]["spec"]
    status_row = prov.load_schema("status")["$defs"]["status_row"]
    spec_props = prov.load_schema("status")["$defs"]["spec_properties"]
    assert spec["properties"] == spec_props
    assert set(spec["required"]) <= set(status_row["required"])


def test_emit_json_is_pretty_and_thread_safe(capsys):
    def worker(i: int) -> None:
        prov.emit_json({"i": i, "payload": list(range(50))})

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    out = capsys.readouterr().out
    # Eight complete documents, none interleaved.
    docs = [d for d in out.split("}\n{") if d]
    assert len(docs) == 8
