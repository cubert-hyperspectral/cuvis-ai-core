"""Cross-platform venv path helpers.

Windows venvs use ``Scripts/python.exe``; POSIX venvs use ``bin/python``.
A single helper avoids hard-coding either form at the call site.
"""

from __future__ import annotations

import sys
from pathlib import Path


def venv_bin_dir(venv_path: Path) -> Path:
    """Return the venv's script directory (``Scripts`` on Windows, ``bin`` elsewhere)."""
    return venv_path / ("Scripts" if sys.platform == "win32" else "bin")


def venv_python(venv_path: Path) -> Path:
    """Return the venv's python interpreter path."""
    if sys.platform == "win32":
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"
