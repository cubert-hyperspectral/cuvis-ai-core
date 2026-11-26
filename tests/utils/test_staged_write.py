"""write_staged: a save replaces its target in one move and keeps who may read it."""

from __future__ import annotations

import os
import stat

import pytest

from cuvis_ai_core.utils.staged_write import write_staged

posix_only = pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")


def test_the_target_holds_what_was_written(tmp_path):
    target = tmp_path / "pipeline.yaml"
    target.write_text("old", encoding="utf-8")

    write_staged(target, lambda path: path.write_text("new", encoding="utf-8"))

    assert target.read_text(encoding="utf-8") == "new"
    assert [p.name for p in tmp_path.iterdir()] == ["pipeline.yaml"]


def test_a_failed_write_keeps_the_previous_file_and_no_staging_file(tmp_path):
    target = tmp_path / "pipeline.yaml"
    target.write_text("old", encoding="utf-8")

    def fail(path):
        path.write_text("half", encoding="utf-8")
        raise RuntimeError("disk full")

    with pytest.raises(RuntimeError):
        write_staged(target, fail)

    assert target.read_text(encoding="utf-8") == "old"
    assert [p.name for p in tmp_path.iterdir()] == ["pipeline.yaml"]


@posix_only
def test_a_replaced_file_keeps_its_mode(tmp_path):
    target = tmp_path / "pipeline.yaml"
    target.write_text("old", encoding="utf-8")
    target.chmod(0o640)

    write_staged(target, lambda path: path.write_text("new", encoding="utf-8"))

    assert stat.S_IMODE(target.stat().st_mode) == 0o640


@posix_only
def test_a_new_file_gets_the_mode_a_plain_open_would_give(tmp_path):
    umask = os.umask(0)
    os.umask(umask)
    target = tmp_path / "pipeline.yaml"

    write_staged(target, lambda path: path.write_text("new", encoding="utf-8"))

    assert stat.S_IMODE(target.stat().st_mode) == 0o666 & ~umask
