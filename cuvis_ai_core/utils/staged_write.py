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
    writer that picks its format by extension sees the same one. The saved file
    gets the mode the file it replaces had, or the mode a plain open would give.
    """
    staging_path = _create_staging(target)
    try:
        write(staging_path)
        _keep_mode(target, staging_path)
        os.replace(staging_path, target)
    except BaseException:
        try:
            staging_path.unlink(missing_ok=True)
        except OSError:  # cleanup must not mask the failure
            pass
        raise


def _create_staging(target: Path) -> Path:
    """A new, empty sibling of ``target``, created like a plain open would (0666 less umask).

    ``O_EXCL`` refuses an existing name, a link included.
    """
    staging = target.with_name(
        f"{target.name}.{secrets.token_hex(8)}.tmp{target.suffix}"
    )
    os.close(os.open(staging, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666))
    return staging


def _keep_mode(target: Path, staging: Path) -> None:
    """Give ``staging`` the mode of the regular file it replaces (POSIX only)."""
    if os.name != "posix":
        return
    try:
        replaced = os.lstat(target)
    except FileNotFoundError:
        return
    if stat.S_ISREG(replaced.st_mode):
        os.chmod(staging, stat.S_IMODE(replaced.st_mode))
