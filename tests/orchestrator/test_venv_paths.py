"""Cross-platform venv python path helper tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from cuvis_ai_core.orchestrator.venv_paths import venv_bin_dir, venv_python


def test_venv_python_on_windows():
    with patch("cuvis_ai_core.orchestrator.venv_paths.sys.platform", "win32"):
        assert venv_python(Path("C:/runs/key/.venv")) == Path(
            "C:/runs/key/.venv/Scripts/python.exe"
        )


def test_venv_python_on_linux():
    with patch("cuvis_ai_core.orchestrator.venv_paths.sys.platform", "linux"):
        assert venv_python(Path("/runs/key/.venv")) == Path(
            "/runs/key/.venv/bin/python"
        )


def test_venv_python_on_macos():
    with patch("cuvis_ai_core.orchestrator.venv_paths.sys.platform", "darwin"):
        assert venv_python(Path("/runs/key/.venv")) == Path(
            "/runs/key/.venv/bin/python"
        )


def test_venv_bin_dir_on_windows():
    with patch("cuvis_ai_core.orchestrator.venv_paths.sys.platform", "win32"):
        assert venv_bin_dir(Path("C:/v")) == Path("C:/v/Scripts")


def test_venv_bin_dir_on_posix():
    with patch("cuvis_ai_core.orchestrator.venv_paths.sys.platform", "linux"):
        assert venv_bin_dir(Path("/v")) == Path("/v/bin")
