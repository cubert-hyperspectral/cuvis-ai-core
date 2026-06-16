"""Load, shape-validate, and fingerprint a ``splits.json`` (a serialized DataSplitConfig).

``splits.json`` is the committable form of a split assignment. Loading validates shape
only (parseable, schema-valid, legal selector kinds); it does not touch the sample
universe. ``verify_universe`` is the one universe-aware check: a committed split that still
carries ``dir_indices`` (positional) must match the ``universe_hash`` it was written
against, so adding a file to a folder cannot silently shift it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

from cuvis_ai_schemas.training.data import DataSplitConfig, SelectorKind

if TYPE_CHECKING:  # pragma: no cover - typing only
    from cuvis_ai_schemas.training.data import SampleRef, Selector


def load_splits(path: str | Path) -> DataSplitConfig:
    """Load + shape-validate a ``splits.json`` into a ``DataSplitConfig``."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"splits file not found: {file_path}")
    data = json.loads(file_path.read_text(encoding="utf-8"))
    return DataSplitConfig.model_validate(data)


def save_splits(splits: DataSplitConfig, path: str | Path) -> None:
    """Write a ``DataSplitConfig`` to ``splits.json`` (pretty, stable key order)."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(splits.to_dict(), indent=2), encoding="utf-8")


def universe_hash(refs: list[SampleRef]) -> str:
    """Stable sha256 fingerprint of the ordered universe (by ``uid``)."""
    digest = hashlib.sha256()
    for ref in refs:
        digest.update(ref.uid.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def has_dir_indices(splits: DataSplitConfig) -> bool:
    """True if any stage uses a positional ``dir_indices`` selector (incl. nested)."""

    def visit(sel: Selector) -> bool:
        if sel.kind == SelectorKind.DIR_INDICES:
            return True
        return any(visit(child) for child in sel.of)

    for stage in (splits.train, splits.val, splits.test, splits.predict):
        if any(visit(sel) for sel in stage):
            return True
    return False


def verify_universe(splits: DataSplitConfig, refs: list[SampleRef]) -> None:
    """Fail loud if a committed ``dir_indices`` split no longer matches its universe.

    Only enforced when ``splits.universe_hash`` is set AND a ``dir_indices`` selector
    remains (committed splits should have had their ``dir_indices`` lowered to concrete
    selectors, leaving this a guard for hand-authored ones).
    """
    if not splits.universe_hash or not has_dir_indices(splits):
        return
    actual = universe_hash(refs)
    if actual != splits.universe_hash:
        raise ValueError(
            "universe changed since splits.json was committed "
            f"(expected {splits.universe_hash[:12]}..., got {actual[:12]}...). "
            "Re-run resolve-splits, or remove dir_indices in favor of files/file_indices."
        )


__all__ = [
    "has_dir_indices",
    "load_splits",
    "save_splits",
    "universe_hash",
    "verify_universe",
]
