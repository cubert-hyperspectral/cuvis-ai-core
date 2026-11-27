"""write_staged: a save replaces its target in one move and keeps who may read it."""

from __future__ import annotations

import errno
import os
import stat
import struct

import pytest

from cuvis_ai_core.utils.staged_write import write_staged

posix_only = pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")
linux_acl = pytest.mark.skipif(
    not hasattr(os, "setxattr"), reason="POSIX ACLs as Linux extended attributes"
)
root_only = pytest.mark.skipif(
    os.name != "posix" or os.geteuid() != 0,
    reason="only root can give a file to another owner",
)

_ACCESS_ACL = "system.posix_acl_access"
_DEFAULT_ACL = "system.posix_acl_default"
_NAMED_USER = 4242


def _acl(named_user_perm: int) -> bytes:
    """An ACL in the kernel's xattr form: owner rw, user 4242, group r, mask r, others none."""
    undefined = 0xFFFFFFFF
    entries = [
        (0x01, 6, undefined),  # owner
        (0x02, named_user_perm, _NAMED_USER),  # one named user
        (0x04, 4, undefined),  # owning group
        (0x10, 4, undefined),  # mask
        (0x20, 0, undefined),  # others
    ]
    return struct.pack("<I", 2) + b"".join(struct.pack("<HHI", *e) for e in entries)


def _set_xattr_or_skip(path, name: str, value: bytes) -> None:
    try:
        os.setxattr(path, name, value)
    except OSError as exc:
        pytest.skip(f"this file system keeps no POSIX ACLs: {exc}")


def _write_new(staging) -> None:
    staging.write(b"new")


def test_the_target_holds_what_was_written(tmp_path):
    target = tmp_path / "pipeline.yaml"
    target.write_text("old", encoding="utf-8")

    write_staged(target, _write_new)

    assert target.read_text(encoding="utf-8") == "new"
    assert [p.name for p in tmp_path.iterdir()] == ["pipeline.yaml"]


def test_a_failed_write_keeps_the_previous_file_and_no_staging_file(tmp_path):
    target = tmp_path / "pipeline.yaml"
    target.write_text("old", encoding="utf-8")

    def fail(staging):
        staging.write(b"half")
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

    write_staged(target, _write_new)

    assert stat.S_IMODE(target.stat().st_mode) == 0o640


@posix_only
def test_a_replaced_private_file_is_not_readable_by_others_while_written(tmp_path):
    # Another user who opens the staging file during the write keeps that handle.
    target = tmp_path / "pipeline.yaml"
    target.write_text("old", encoding="utf-8")
    target.chmod(0o600)
    modes: list[int] = []

    def write(staging):
        modes.append(stat.S_IMODE(os.fstat(staging.fileno()).st_mode))
        staging.write(b"secret")

    write_staged(target, write)

    assert modes and modes[0] & 0o077 == 0
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


@posix_only
@pytest.mark.parametrize("swap", [os.symlink, os.link], ids=["symlink", "hard link"])
def test_a_staging_file_swapped_during_the_save_leaves_the_other_file_alone(
    tmp_path, swap
):
    # Another account that can write the folder swaps the staging file for a link to a
    # file of its choice while the save runs. The save writes into, and sets the mode and
    # owner of, the file it created, then refuses the move.
    victim = tmp_path / "victim.txt"
    victim.write_text("secret", encoding="utf-8")
    victim.chmod(0o600)
    folder = tmp_path / "shared"
    folder.mkdir()
    target = folder / "pipeline.yaml"
    target.write_text("old", encoding="utf-8")
    target.chmod(0o666)
    if os.geteuid() == 0:  # a root server saves over another account's file
        os.chown(target, _NAMED_USER, -1)
    before = victim.stat()

    def write(staging):
        (name,) = [p for p in folder.iterdir() if p != target]
        name.unlink()
        swap(victim, name)
        staging.write(b"new")

    with pytest.raises(OSError, match="replaced"):
        write_staged(target, write)

    after = victim.stat()
    assert victim.read_text(encoding="utf-8") == "secret"
    assert (stat.S_IMODE(after.st_mode), after.st_uid) == (0o600, before.st_uid)
    assert target.read_text(encoding="utf-8") == "old"
    assert [p.name for p in folder.iterdir()] == ["pipeline.yaml"]


@posix_only
def test_a_new_file_is_readable_by_its_owner_only(tmp_path):
    # Nobody but the account that saved it reads a new save, during the write or after.
    umask = os.umask(0)
    os.umask(umask)
    target = tmp_path / "pipeline.yaml"
    modes: list[int] = []

    def write(staging):
        modes.append(stat.S_IMODE(os.fstat(staging.fileno()).st_mode))
        staging.write(b"new")

    write_staged(target, write)

    assert modes and modes[0] & 0o077 == 0
    assert stat.S_IMODE(target.stat().st_mode) == 0o600 & ~umask


@posix_only
def test_a_replaced_file_keeps_its_group(tmp_path):
    # In a setgid folder a new file takes the folder's group; the replaced file's stays.
    folder = tmp_path / "shared"
    folder.mkdir()
    target = folder / "pipeline.yaml"
    target.write_text("old", encoding="utf-8")
    target.chmod(0o640)
    own_group = target.stat().st_gid
    others = [group for group in os.getgroups() if group != own_group]
    if not others:
        pytest.skip("needs an account in a second group")
    os.chown(folder, -1, others[0])
    folder.chmod(0o2770)

    write_staged(target, _write_new)

    assert target.stat().st_gid == own_group
    assert stat.S_IMODE(target.stat().st_mode) == 0o640


@root_only
def test_a_file_saved_by_root_keeps_its_owner(tmp_path):
    # A server running as root saves over a user's file; the user still owns it.
    target = tmp_path / "pipeline.yaml"
    target.write_text("old", encoding="utf-8")
    target.chmod(0o640)
    os.chown(target, _NAMED_USER, -1)

    write_staged(target, _write_new)

    assert target.stat().st_uid == _NAMED_USER
    assert stat.S_IMODE(target.stat().st_mode) == 0o640


@root_only
def test_a_file_whose_owner_cannot_be_carried_over_becomes_owner_only(
    tmp_path, monkeypatch
):
    # A file system that refuses the change of owner (root squash on NFS) leaves root as owner.
    target = tmp_path / "pipeline.yaml"
    target.write_text("old", encoding="utf-8")
    target.chmod(0o640)
    os.chown(target, _NAMED_USER, -1)

    def refuse(*args, **kwargs):
        raise PermissionError(errno.EPERM, "refused")

    monkeypatch.setattr(os, "fchown", refuse)

    write_staged(target, _write_new)

    assert target.read_text(encoding="utf-8") == "new"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


@linux_acl
def test_a_replaced_file_keeps_its_acl(tmp_path):
    target = tmp_path / "pipeline.yaml"
    target.write_text("old", encoding="utf-8")
    _set_xattr_or_skip(target, _ACCESS_ACL, _acl(4))
    before = os.getxattr(target, _ACCESS_ACL)

    write_staged(target, _write_new)

    assert os.getxattr(target, _ACCESS_ACL) == before


@linux_acl
def test_a_replaced_file_without_an_acl_takes_none_from_its_folder(tmp_path):
    # The folder's default ACL names user 4242; the file it replaces granted nobody else.
    target = tmp_path / "pipeline.yaml"
    target.write_text("old", encoding="utf-8")
    target.chmod(0o640)
    _set_xattr_or_skip(tmp_path, _DEFAULT_ACL, _acl(4))

    write_staged(target, _write_new)

    with pytest.raises(OSError) as missing:
        os.getxattr(target, _ACCESS_ACL)
    assert missing.value.errno == errno.ENODATA
    assert stat.S_IMODE(target.stat().st_mode) == 0o640


@linux_acl
def test_a_file_whose_acl_cannot_be_carried_over_becomes_owner_only(
    tmp_path, monkeypatch
):
    # A save never lets more accounts read a file than before.
    target = tmp_path / "pipeline.yaml"
    target.write_text("old", encoding="utf-8")
    target.chmod(0o640)

    def refuse(*args, **kwargs):
        raise PermissionError(errno.EPERM, "refused")

    monkeypatch.setattr(os, "setxattr", refuse)
    monkeypatch.setattr(os, "removexattr", refuse)

    write_staged(target, _write_new)

    assert target.read_text(encoding="utf-8") == "new"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


@pytest.mark.parametrize(
    "name",
    [f"{'p' * 245}.yaml", f"p.{'x' * 248}", "p." + "\U0001f600" * 60],
    ids=["long stem", "long suffix", "four-byte characters"],
)
def test_a_long_file_name_still_saves(tmp_path, name):
    # A name may take 255 bytes; the staging name next to it may not need more.
    target = tmp_path / name
    staging_names: list[str] = []

    def write(staging):
        staging_names.extend(os.listdir(tmp_path))
        staging.write(b"new")

    write_staged(target, write)

    assert target.read_text(encoding="utf-8") == "new"
    assert [p.name for p in tmp_path.iterdir()] == [name]
    assert len(staging_names[0]) < 100
    assert len(os.fsencode(staging_names[0])) <= 255
