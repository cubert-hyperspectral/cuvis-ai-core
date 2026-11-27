"""Keep the paths a client sends inside the folders the launcher allows.

``CUVIS_ALLOWED_ROOTS`` lists the folders, separated by ``os.pathsep``. Unset or empty, nothing
is checked and every path behaves as it always did. Set, a client path is refused when it is a
remote form (a UNC or device path, a URL or an fsspec chain such as ``simplecache::``), contains
a ``..`` segment, or resolves outside every root.
Remote forms are refused on the text alone: resolving or even testing such a path makes
Windows open an SMB session to the named host, which hands it the user's NTLM credentials.
A read may also land in the server's own pipeline library (``CUVIS_CONFIGS_DIR``); a write
may not.
"""

from __future__ import annotations

import os
import re
from pathlib import Path, PureWindowsPath

ALLOWED_ROOTS_ENV = "CUVIS_ALLOWED_ROOTS"
CONFIGS_DIR_ENV = "CUVIS_CONFIGS_DIR"

# \\host\share, //host/share, \\?\..., \\.\device, the NT object prefix \??\ (which
# Windows passes through, so \??\UNC\host\share is a share) and their mixed-slash spellings.
_REMOTE_PREFIX = re.compile(r"^(?:[\\/]{2}|[\\/]\?\?[\\/])")
# A URL (http://, s3://, file://) anywhere in the text: fsspec, which Lightning writes through,
# opens it remotely. Two scheme characters at least, so a drive such as C:// stays a path. The
# lookbehind keeps the search linear; a pattern that starts with the scheme rescans it from
# every letter, which takes minutes on a path as long as a gRPC message.
_URL = re.compile(r"(?<=[A-Za-z0-9+.-]{2})://")

# DOS device names, which Windows opens as the device in any folder and with any extension.
_DEVICE_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
    | {f"{port}{n}" for port in ("COM", "LPT") for n in [*"123456789", "¹", "²", "³"]}
)


def _names_a_device(raw: str) -> bool:
    """True when a component of ``raw`` is a device name (Windows drops trailing spaces).

    A colon separates components here too, so ``C:NUL`` and ``CON:`` count.
    """
    return any(
        part.split(".", 1)[0].rstrip(" ").upper() in _DEVICE_NAMES
        for part in re.split(r"[\\/:]", raw)
    )


_DRIVE = re.compile(r"[A-Za-z]:")


def _names_a_stream(raw: str) -> bool:
    """True when ``raw`` has a colon after its drive.

    Windows reads ``pipeline.yaml:hidden`` as a second data stream of ``pipeline.yaml``, which
    no folder listing shows.
    """
    return ":" in (raw[2:] if _DRIVE.match(raw) else raw)


class PathNotAllowedError(PermissionError):
    """A client path lies outside ``CUVIS_ALLOWED_ROOTS`` or has a form that is never allowed."""


def allowed_roots() -> list[Path] | None:
    """The resolved roots from ``CUVIS_ALLOWED_ROOTS``, or ``None`` when it is unset or empty."""
    raw = os.environ.get(ALLOWED_ROOTS_ENV, "")
    entries = [entry.strip() for entry in raw.split(os.pathsep) if entry.strip()]
    if not entries:
        return None
    return [Path(entry).resolve() for entry in entries]


def library_root() -> Path:
    """The server's own pipeline library: ``CUVIS_CONFIGS_DIR``, else ``./configs``."""
    env_dir = os.environ.get(CONFIGS_DIR_ENV)
    return Path(env_dir) if env_dir else Path.cwd() / "configs"


def _roots(read: bool) -> list[Path] | None:
    """The allowed roots and, for a read, the pipeline library; ``None`` without roots."""
    roots = allowed_roots()
    if roots is None or not read:
        return roots
    return [*roots, library_root().resolve()]


def _key(path: Path) -> str:
    return os.path.normcase(str(path))


def _within(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath([_key(root), _key(path)]) == _key(root)
    except ValueError:  # different drives
        return False


def _check_remote(raw: str, what: str) -> None:
    """Refuse a network or device path on its text, before anything touches it."""
    if (
        "\x00" in raw
        or "::" in raw  # an fsspec chain such as simplecache::file:///elsewhere
        or _REMOTE_PREFIX.match(raw)
        or _URL.search(raw)
        or (os.name == "nt" and (_names_a_device(raw) or _names_a_stream(raw)))
    ):
        raise PathNotAllowedError(
            f"{what}: network, device and stream paths are not allowed: {raw!r}"
        )


def _check_text(raw: str, what: str) -> None:
    """Refuse the forms a client path may never take, without touching the filesystem."""
    _check_remote(raw, what)
    if ".." in re.split(r"[\\/]", raw):
        raise PathNotAllowedError(f"{what}: '..' is not allowed in a path: {raw!r}")
    if raw.startswith("~"):
        # expanduser (Lightning applies it to its folders) would turn it into a home directory.
        raise PathNotAllowedError(
            f"{what}: a path starting with '~' is not allowed: {raw!r}"
        )
    if os.name == "nt":
        windows = PureWindowsPath(raw)
        if windows.drive and not windows.root:
            raise PathNotAllowedError(
                f"{what}: a drive-relative path is not allowed: {raw!r}"
            )
        if windows.root and not windows.drive:
            # Not absolute to Python, yet a join keeps only the drive of the folder.
            raise PathNotAllowedError(
                f"{what}: a path from the drive root has to name its drive: {raw!r}"
            )


def _contain(path: Path, roots: list[Path], what: str) -> Path:
    resolved = path.resolve()
    if not any(_within(resolved, root) for root in roots):
        raise PathNotAllowedError(
            f"{what}: {resolved} is outside the allowed folders ({ALLOWED_ROOTS_ENV})"
        )
    return resolved


def confine_path(raw: str, *, field: str, read: bool = False) -> str:
    """Screen a path a client sent in request field ``field``.

    Without roots ``raw`` comes back unchanged. With roots an absolute path comes back
    resolved (callers open that, not the text they got), and a relative one comes back as is:
    it becomes a location only where a resolver joins it to a folder, and that resolver checks
    the result with :func:`confine_resolved`. ``read`` also allows the pipeline library.
    """
    roots = _roots(read)
    if roots is None or not raw:
        return raw
    _check_text(raw, field)
    path = Path(raw)
    if not path.is_absolute():
        return raw
    return str(_contain(path, roots, field))


def confine_resolved(path: Path, *, what: str, read: bool = False) -> Path:
    """Check a path a resolver built from client input; resolved, or ``path`` without roots.

    ``read`` also allows the pipeline library.
    """
    roots = _roots(read)
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
