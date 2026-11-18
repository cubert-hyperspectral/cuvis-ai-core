"""The development lock covers CPython 3.11.5 and newer and forks no package by Python version.

soupsieve 3 and later need CPython 3.11.5. With ``requires-python = ">=3.11"`` alone uv locks an
older soupsieve for 3.11.0 to 3.11.4 next to the current one, and the audit job, which installs
the lock on one interpreter, never checks that side. ``[tool.uv].environments`` leaves those
interpreters out of the lock; the package itself still installs from 3.11.0.
"""

from __future__ import annotations

import functools
import tomllib
from pathlib import Path

import pytest
from packaging.markers import Marker

ROOT = Path(__file__).resolve().parents[1]
PLATFORMS = {
    "jetson": {"sys_platform": "linux", "platform_machine": "aarch64"},
    "linux-x86_64": {"sys_platform": "linux", "platform_machine": "x86_64"},
    "windows": {"sys_platform": "win32", "platform_machine": "AMD64"},
}
PYTHON_MINORS = ("3.11", "3.12", "3.13")
# The oldest CPython the lock covers and the newest one it leaves out.
LOCK_FLOOR = "3.11.5"
BELOW_LOCK_FLOOR = "3.11.4"
# Every CPython release a marker boundary can fall on.
PYTHON_RELEASES = tuple(
    f"{minor}.{patch}" for minor in PYTHON_MINORS for patch in range(40)
)
# Packages that may resolve to different versions by Python version, each with its reason.
PYTHON_FORK_ALLOWLIST: dict[str, str] = {}


def _environment(platform: str, release: str) -> dict[str, str]:
    """A marker environment for one platform and one CPython release."""
    return {
        **PLATFORMS[platform],
        "python_version": release.rsplit(".", 1)[0],
        "python_full_version": release,
    }


@functools.cache
def _marker(text: str) -> Marker:
    """Parse a marker once: the lock repeats the same few on every forked entry."""
    return Marker(text)


def _applies(pkg: dict, environment: dict[str, str]) -> bool:
    """Whether a lock entry is selected in ``environment`` (no markers = every environment)."""
    markers = pkg.get("resolution-markers")
    return markers is None or any(_marker(m).evaluate(environment) for m in markers)


@pytest.fixture(scope="module")
def pyproject() -> dict:
    """The parsed ``pyproject.toml``."""
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def lock() -> dict:
    """The parsed ``uv.lock``."""
    return tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))


def test_dev_lock_leaves_out_pythons_below_its_floor(pyproject: dict) -> None:
    """``[tool.uv].environments`` keeps the lock to CPython 3.11.5 and newer, on every platform."""
    environments = [Marker(m) for m in pyproject["tool"]["uv"].get("environments", [])]
    assert environments, "[tool.uv].environments must restrict the dev lock"
    for platform in PLATFORMS:
        too_old = _environment(platform, BELOW_LOCK_FLOOR)
        assert not any(m.evaluate(too_old) for m in environments), (
            f"{platform}: {too_old}"
        )
        supported = [_environment(platform, LOCK_FLOOR)]
        supported += [_environment(platform, f"{minor}.9") for minor in PYTHON_MINORS]
        for environment in supported:
            assert any(m.evaluate(environment) for m in environments), (
                f"{platform}: {environment}"
            )


def test_lock_forks_no_package_by_python_version(lock: dict) -> None:
    """One platform gets one version of a package, whatever the CPython release.

    A platform fork is fine. A Python fork is not: pip-audit runs on one interpreter and never
    sees the other side. One that is wanted goes into ``PYTHON_FORK_ALLOWLIST`` with its reason.
    """
    entries: dict[str, list[dict]] = {}
    for pkg in lock["package"]:
        entries.setdefault(pkg["name"], []).append(pkg)
    forked = set()
    for name, pkgs in entries.items():
        if len(pkgs) < 2:
            continue
        for platform in PLATFORMS:
            versions = {
                pkg["version"]
                for release in PYTHON_RELEASES
                for pkg in pkgs
                if _applies(pkg, _environment(platform, release))
            }
            if len(versions) > 1:
                forked.add(name)
    assert forked == set(PYTHON_FORK_ALLOWLIST), (
        f"forked by Python version: {sorted(forked)}"
    )
