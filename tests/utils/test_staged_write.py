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
def test_a_replaced_private_file_is_not_readable_by_others_while_written(tmp_path):
    # Another user who opens the staging file during the write keeps that handle.
    target = tmp_path / "pipeline.yaml"
    target.write_text("old", encoding="utf-8")
    target.chmod(0o600)
    modes: list[int] = []

    def write(path):
        modes.append(stat.S_IMODE(path.stat().st_mode))
        path.write_text("secret", encoding="utf-8")

    write_staged(target, write)

    assert modes and modes[0] & 0o077 == 0
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


@posix_only
def test_a_new_file_is_readable_by_its_owner_only(tmp_path):
    # Nobody but the account that saved it reads a new save, during the write or after.
    umask = os.umask(0)
    os.umask(umask)
    target = tmp_path / "pipeline.yaml"
    modes: list[int] = []

    def write(path):
        modes.append(stat.S_IMODE(path.stat().st_mode))
        path.write_text("new", encoding="utf-8")

    write_staged(target, write)

    assert modes and modes[0] & 0o077 == 0
    assert stat.S_IMODE(target.stat().st_mode) == 0o600 & ~umask


@pytest.mark.parametrize(
    "name", [f"{'p' * 245}.yaml", f"p.{'x' * 248}"], ids=["long stem", "long suffix"]
)
def test_a_long_file_name_still_saves(tmp_path, name):
    # A name may take 255 characters; the staging name next to it may not need more.
    target = tmp_path / name
    staging_names: list[str] = []

    def write(path):
        staging_names.append(path.name)
        path.write_text("new", encoding="utf-8")

    write_staged(target, write)

    assert target.read_text(encoding="utf-8") == "new"
    assert [p.name for p in tmp_path.iterdir()] == [name]
    assert len(staging_names[0]) < 100
