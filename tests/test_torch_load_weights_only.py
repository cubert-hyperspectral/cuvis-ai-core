"""Every ``torch.load`` in the shipped code passes ``weights_only=True``.

``weights_only=False`` unpickles arbitrary objects, so a checkpoint from a client, a download
or a shared folder could run code. This test reads the source instead of running it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCANNED_DIRS = ("cuvis_ai_core", "scripts")


def _is_torch_load(call: ast.Call) -> bool:
    func = call.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "load"
        and isinstance(func.value, ast.Name)
        and func.value.id == "torch"
    )


def _passes_weights_only_true(call: ast.Call) -> bool:
    return any(
        keyword.arg == "weights_only"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is True
        for keyword in call.keywords
    )


def unsafe_torch_loads(source: str, filename: str) -> list[str]:
    """``file:line`` of each ``torch.load`` call without a literal ``weights_only=True``."""
    tree = ast.parse(source, filename=filename)
    return [
        f"{filename}:{node.lineno}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _is_torch_load(node)
        and not _passes_weights_only_true(node)
    ]


@pytest.mark.parametrize(
    ("source", "flagged"),
    [
        ("torch.load(p, weights_only=True)", False),
        ("torch.load(p, map_location='cpu', weights_only=True)", False),
        ("torch.load(p)", True),
        ("torch.load(p, weights_only=False)", True),
        ("torch.load(p, weights_only=flag)", True),
        ("other.load(p)", False),
    ],
)
def test_the_check_flags_what_it_should(source: str, flagged: bool) -> None:
    assert bool(unsafe_torch_loads(source, "probe.py")) is flagged


def test_shipped_code_loads_checkpoints_with_weights_only() -> None:
    files = [
        path
        for directory in SCANNED_DIRS
        for path in sorted((REPO_ROOT / directory).rglob("*.py"))
    ]
    assert files, "no source files found; the scan paths are wrong"

    offenders = [
        hit
        for path in files
        for hit in unsafe_torch_loads(
            path.read_text(encoding="utf-8"), str(path.relative_to(REPO_ROOT))
        )
    ]

    assert offenders == []
