"""Write a file through a staging sibling that is moved into place."""

from __future__ import annotations

import os
import secrets
import stat
from collections.abc import Callable
from pathlib import Path


def write_staged(target: Path, write: Callable[[Path], None]) -> None:
    """Write ``target`` through a unique sibling that is moved into place.

    A failed write never leaves a truncated or half-written file at ``target``
    (the previous file stays), and two saves to one path cannot clobber each
    other's staging file. The move replaces a link at ``target`` instead of
    writing through it, so a link put there after the path was checked does
    not redirect the write. The staging name keeps the target's suffix, so a
    writer that picks its format by extension sees the same one. On POSIX only
    the owner can read the file while it is written; the saved file then gets
    the mode the regular file it replaces had, and a new one (a link at the
    target counts as no file) stays readable by its owner only. Windows ignores
    these mode bits: there the saved file gets the permissions its folder
    passes on.
    """
    replaced_mode = _regular_file_mode(target)
    staging_path = _create_staging(target)
    try:
        write(staging_path)
        if replaced_mode is not None and os.name == "posix":
            os.chmod(staging_path, replaced_mode)
        os.replace(staging_path, target)
    except BaseException:
        try:
            staging_path.unlink(missing_ok=True)
        except OSError:  # cleanup must not mask the failure
            pass
        raise


def _create_staging(target: Path) -> Path:
    """A new, empty, owner-only sibling of ``target``.

    The name holds at most 32 characters of the target's name and of its suffix, so it stays
    short of the 255 a file system allows for a name the target may use in full. ``O_EXCL``
    refuses an existing name, a link included.
    """
    staging = target.with_name(
        f"{target.name[:32]}.{secrets.token_hex(8)}.tmp{target.suffix[:32]}"
    )
    os.close(os.open(staging, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
    return staging


def _regular_file_mode(target: Path) -> int | None:
    """The permission bits of the regular file at ``target``; ``None`` for no file or a link."""
    try:
        found = os.lstat(target)
    except FileNotFoundError:
        return None
    return stat.S_IMODE(found.st_mode) if stat.S_ISREG(found.st_mode) else None
