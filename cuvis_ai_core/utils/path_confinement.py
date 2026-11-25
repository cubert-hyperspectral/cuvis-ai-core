"""Keep the paths a client sends inside the folders the launcher allows.

``CUVIS_ALLOWED_ROOTS`` lists the folders, separated by ``os.pathsep``. Unset or empty, nothing
is checked and every path behaves as it always did. Set, a client path is refused when it is a
remote form (a UNC or device path), contains a ``..`` segment, or resolves outside every root.
Remote forms are refused on the text alone: resolving or even testing such a path makes
Windows open an SMB session to the named host, which hands it the user's NTLM credentials.
"""

from __future__ import annotations

import os
import re
from pathlib import Path, PureWindowsPath

ALLOWED_ROOTS_ENV = "CUVIS_ALLOWED_ROOTS"

# \\host\share, //host/share, \\?\..., \\.\device and their mixed-slash spellings.
_REMOTE_PREFIX = re.compile(r"^[\\/]{2}")


class PathNotAllowedError(PermissionError):
    """A client path lies outside ``CUVIS_ALLOWED_ROOTS`` or has a form that is never allowed."""


def allowed_roots() -> list[Path] | None:
    """The resolved roots from ``CUVIS_ALLOWED_ROOTS``, or ``None`` when it is unset or empty."""
    raw = os.environ.get(ALLOWED_ROOTS_ENV, "")
    entries = [entry.strip() for entry in raw.split(os.pathsep) if entry.strip()]
    if not entries:
        return None
    return [Path(entry).resolve() for entry in entries]


def _key(path: Path) -> str:
    return os.path.normcase(str(path))


def _within(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath([_key(root), _key(path)]) == _key(root)
    except ValueError:  # different drives
        return False


def _check_remote(raw: str, what: str) -> None:
    """Refuse a network or device path on its text, before anything touches it."""
    if "\x00" in raw or _REMOTE_PREFIX.match(raw):
        raise PathNotAllowedError(
            f"{what}: network and device paths are not allowed: {raw!r}"
        )


def _check_text(raw: str, what: str) -> None:
    """Refuse the forms a client path may never take, without touching the filesystem."""
    _check_remote(raw, what)
    if ".." in re.split(r"[\\/]", raw):
        raise PathNotAllowedError(f"{what}: '..' is not allowed in a path: {raw!r}")
    if os.name == "nt":
        windows = PureWindowsPath(raw)
        if windows.drive and not windows.root:
            raise PathNotAllowedError(
                f"{what}: a drive-relative path is not allowed: {raw!r}"
            )


def _contain(path: Path, roots: list[Path], what: str) -> Path:
    resolved = path.resolve()
    if not any(_within(resolved, root) for root in roots):
        raise PathNotAllowedError(
            f"{what}: {resolved} is outside the allowed folders ({ALLOWED_ROOTS_ENV})"
        )
    return resolved


def confine_path(raw: str, *, field: str) -> str:
    """Screen a path a client sent in request field ``field``.

    Without roots ``raw`` comes back unchanged. With roots an absolute path comes back
    resolved (callers open that, not the text they got), and a relative one comes back as is:
    it becomes a location only where a resolver joins it to a folder, and that resolver checks
    the result with :func:`confine_resolved`.
    """
    roots = allowed_roots()
    if roots is None or not raw:
        return raw
    _check_text(raw, field)
    path = Path(raw)
    if not path.is_absolute():
        return raw
    return str(_contain(path, roots, field))


def confine_resolved(path: Path, *, what: str) -> Path:
    """Check a path a resolver built from client input; resolved, or ``path`` without roots."""
    roots = allowed_roots()
    if roots is None:
        return path
    _check_remote(str(path), what)
    return _contain(path, roots, what)


def check_not_remote(raw: str, *, what: str) -> None:
    """Refuse a network or device path before a resolver touches ``raw`` (only with roots set).

    For references read from a file, such as a trainrun's ``pipeline:``, which may climb out of
    their folder with ``..`` and still land inside the roots; the resolver's result then goes
    through :func:`confine_resolved`.
    """
    if allowed_roots() is not None and raw:
        _check_remote(raw, what)
