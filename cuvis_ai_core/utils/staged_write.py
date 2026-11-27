"""Write a file through a staging sibling that is moved into place."""

from __future__ import annotations

import errno
import os
import secrets
import stat
import sys
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

# Linux keeps a file's access ACL in this extended attribute.
_ACCESS_ACL = "system.posix_acl_access"
# The errors that say a file has no access ACL: none is set, or the file system keeps none.
_NO_ACL = frozenset(
    code
    for code in (
        getattr(errno, name, None)
        for name in ("ENODATA", "ENOATTR", "ENOTSUP", "EOPNOTSUPP")
    )
    if code is not None
)


def write_staged(target: Path, write: Callable[[BinaryIO], None]) -> None:
    """Write ``target`` through a unique sibling that is moved into place.

    ``write`` gets the staging file open for binary writing and leaves it open.
    A failed write never leaves a truncated or half-written file at ``target``
    (the previous file stays), and two saves to one path cannot clobber each
    other's staging file. The move replaces a link at ``target`` instead of
    writing through it, so a link put there after the path was checked does
    not redirect the write. The save writes into the file it created and never
    opens the staging name again, so a link another account swaps in for the
    staging file receives nothing; on POSIX the move is then refused. On POSIX
    only the owner can read the file while it is written. The saved file then
    gets the group, the access ACL (on Linux) and the mode of the regular file
    it replaces, and its owner when root saves it; any other account that
    saves becomes the owner. When the owner, the group or the ACL cannot be
    carried over, only its owner can read it, so a save never lets more
    accounts read a file than before. A new file (a link at the target counts
    as no file) stays readable by its owner only. Windows ignores all of this:
    there the saved file gets the permissions its folder passes on.
    """
    replaced = _regular_file(target)
    staging_path, fd = _create_staging(target)
    try:
        with os.fdopen(fd, "wb") as staging:
            write(staging)
            staging.flush()
            if replaced is not None:
                _carry_access(replaced, target, staging.fileno())
            _require_staging_name(staging_path, staging.fileno(), target)
        os.replace(staging_path, target)
    except BaseException:
        try:
            staging_path.unlink(missing_ok=True)
        except OSError:  # cleanup must not mask the failure
            pass
        raise


def _create_staging(target: Path) -> tuple[Path, int]:
    """A new, empty, owner-only sibling of ``target`` and a descriptor open for writing it.

    The name holds at most 24 characters of the target's name and of its suffix. A character
    takes four bytes at most, so the name stays within the 255 bytes a file system allows for
    a name the target may use in full. ``O_EXCL`` refuses an existing name, a link included.
    """
    staging = target.with_name(
        f"{target.name[:24]}.{secrets.token_hex(8)}.tmp{target.suffix[:24]}"
    )
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
    return staging, os.open(staging, flags, 0o600)


def _require_staging_name(staging: Path, fd: int, target: Path) -> None:
    """Refuse the move when ``staging`` no longer names the file open at ``fd`` (POSIX).

    Another account that can write the folder may have put a link or another file in its
    place. Windows refuses to delete or rename a file this process holds open, and its file
    ids from a name and from a descriptor need not match, so it is not checked there.
    """
    if sys.platform == "win32":
        return
    if not os.path.samestat(os.lstat(staging), os.fstat(fd)):
        raise OSError(
            f"{staging} was replaced during the save; {target} is left as it was"
        )


def _regular_file(target: Path) -> os.stat_result | None:
    """The status of the regular file at ``target``; ``None`` for no file or a link."""
    try:
        found = os.lstat(target)
    except FileNotFoundError:
        return None
    return found if stat.S_ISREG(found.st_mode) else None


def _carry_access(replaced: os.stat_result, target: Path, staging: int) -> None:
    """Give the file open at ``staging`` the owner, group, access ACL and mode of the file
    it replaces (POSIX).

    Only root can give a file to another account, so another saver becomes the owner. When
    the owner, the group or the ACL cannot be carried over, the file keeps the owner's
    permission bits only.
    """
    if sys.platform == "win32":  # no POSIX owner, group, ACL or mode bits to carry
        return
    mode = stat.S_IMODE(replaced.st_mode)
    try:
        current = os.fstat(staging)
        uid = replaced.st_uid
        if os.geteuid() != 0 or current.st_uid == uid:
            uid = -1
        gid = replaced.st_gid if current.st_gid != replaced.st_gid else -1
        if uid != -1 or gid != -1:
            os.fchown(staging, uid, gid)
        _copy_access_acl(target, staging)
    except OSError:
        mode &= stat.S_IRWXU
    os.fchmod(staging, mode)


def _copy_access_acl(source: Path, staging: int) -> None:
    """Give the file open at ``staging`` the access ACL of ``source``, or none when
    ``source`` has none.

    The file may carry one from its folder's default ACL, which the replaced file did not
    grant. Only Linux exposes ACLs as extended attributes; elsewhere nothing is copied.
    """
    if sys.platform != "linux":
        return
    try:
        acl: bytes | None = os.getxattr(source, _ACCESS_ACL, follow_symlinks=False)
    except OSError as exc:
        if exc.errno not in _NO_ACL:
            raise
        acl = None
    if acl is not None:
        os.setxattr(staging, _ACCESS_ACL, acl)
        return
    try:
        os.removexattr(staging, _ACCESS_ACL)
    except OSError as exc:
        if exc.errno not in _NO_ACL:
            raise
