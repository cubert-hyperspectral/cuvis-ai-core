"""Resolve composable selectors over an attributed sample universe + leakage check.

This is the one resolver used by ``BaseCuvisAIDataModule.setup`` (and, later, by the
split tooling and the view API). It turns a stage's ``list[Selector]`` into a list of
``SampleRef``, unioned in selector order with first-occurrence dedup, keying identity on
``SampleRef.uid``. Membership and name validation happen here (against the live universe),
not at schema-parse time.
"""

from __future__ import annotations

import fnmatch
import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from cuvis_ai_core.utils.general import expand_range_selectors

if TYPE_CHECKING:  # pragma: no cover - typing only
    from cuvis_ai_schemas.training.data import (
        Constraint,
        DataSplitConfig,
        SampleRef,
        Selector,
    )

logger = logging.getLogger(__name__)


class SplitConstraintError(RuntimeError):
    """Raised when an ``error``-severity split constraint is violated or unverifiable."""


class SplitLeakageError(SplitConstraintError):
    """Back-compat: raised by ``validate_leakage`` on train/val/test uid overlap."""


@dataclass(frozen=True)
class ConstraintResult:
    """The outcome of evaluating one constraint (pure; ``enforce_constraints`` acts on these).

    ``status`` is ``satisfied`` (green), ``violated`` (red), or ``unavailable`` (the data
    module cannot supply the labels the constraint needs — never silently ``satisfied``).
    ``offending`` is a sorted, deduped list of uids (``no_split_overlap``) or canonical
    sources (``no_source_overlap`` / ``no_train_anomalous``); ``count`` is its length.
    """

    kind: str
    severity: str
    status: str
    count: int = 0
    offending: tuple[str, ...] = ()
    reason: str = field(default="")


def required_attrs(splits: DataSplitConfig) -> frozenset[str]:
    """Which ``SampleRef`` attributes the selectors need populated.

    Only ``tag`` / ``categories`` selectors read attributes; everything else resolves
    structurally. The returned set is passed to a module's ``enumerate`` so it can skip
    parsing COCO / PNG masks when no stage needs them.
    """
    from cuvis_ai_schemas.training.data import SelectorKind

    needed: set[str] = set()

    def visit(sel: Selector) -> None:
        if sel.kind == SelectorKind.TAG:
            needed.add("tags")
        elif sel.kind == SelectorKind.CATEGORIES:
            needed.add("category_ids")
        for child in sel.of:
            visit(child)

    for stage in (splits.train, splits.val, splits.test, splits.predict):
        for sel in stage:
            visit(sel)
    return frozenset(needed)


def resolve_selectors(
    selectors: list[Selector],
    refs: list[SampleRef],
    *,
    name_to_id: dict[str, int] | None = None,
) -> list[SampleRef]:
    """Resolve a stage's selector list to ``SampleRef``s (unioned, order-preserving).

    ``refs`` must be in the module's canonical order. The result preserves selector
    order with first-occurrence dedup on ``uid``. Raises ``ValueError`` on an unknown
    category/tag name, a negative / out-of-range index, or a selector that matches no
    sample.
    """
    out: list[SampleRef] = []
    seen: set[str] = set()
    for sel in selectors:
        matched = _resolve_one(sel, refs, name_to_id=name_to_id)
        if not matched:
            raise ValueError(f"selector '{sel.kind.value}' matched 0 samples")
        for ref in matched:
            if ref.uid not in seen:
                seen.add(ref.uid)
                out.append(ref)
    return out


def _uid_overlaps(
    train: list[SampleRef], val: list[SampleRef], test: list[SampleRef]
) -> list[tuple[str, str, set[str]]]:
    """Pairwise train/val/test uid intersections (``predict`` excluded); empty pairs dropped."""
    train_u = {r.uid for r in train}
    val_u = {r.uid for r in val}
    test_u = {r.uid for r in test}
    pairs = [
        ("train", "val", train_u & val_u),
        ("train", "test", train_u & test_u),
        ("val", "test", val_u & test_u),
    ]
    return [(a, b, shared) for a, b, shared in pairs if shared]


def validate_leakage(
    train: list[SampleRef],
    val: list[SampleRef],
    test: list[SampleRef],
    *,
    mode: str = "error",
) -> None:
    """Assert train/val/test are pairwise disjoint by ``uid`` (per ``mode``).

    ``error`` raises ``SplitLeakageError``; ``warn`` logs and continues; ``off`` skips.
    ``predict`` is intentionally excluded (it may overlap ``test``). Retained for direct
    callers; the ``no_split_overlap`` constraint shares the same ``_uid_overlaps`` core.
    """
    if mode == "off":
        return
    leaks = _uid_overlaps(train, val, test)
    if not leaks:
        return
    msg = "; ".join(
        f"{len(shared)} samples shared between {a} and {b}" for a, b, shared in leaks
    )
    if mode == "warn":
        logger.warning("split leakage: %s", msg)
        return
    raise SplitLeakageError(f"split leakage: {msg}")


def constraint_required_attrs(constraints: list[Constraint]) -> frozenset[str]:
    """Which ``SampleRef`` attributes the constraints *would like* populated.

    Only ``no_train_anomalous`` reads metadata (``category_ids``). This is
    **opportunistic**: the datamodule intersects it with what the module can supply
    (``supported_attrs``), so an unsupported attribute yields an ``unavailable`` result
    rather than forcing ``enumerate`` to raise.
    """
    from cuvis_ai_schemas.training.data import ConstraintKind

    for c in constraints:
        if c.kind == ConstraintKind.NO_TRAIN_ANOMALOUS:
            return frozenset({"category_ids"})
    return frozenset()


def _is_anomalous(ref: SampleRef) -> bool:
    """A sample is anomalous iff it carries an annotation with ``category_id != 0`` (0 = background)."""
    return any(c != 0 for c in ref.category_ids)


def _sources_spanning_stages(
    train: list[SampleRef], val: list[SampleRef], test: list[SampleRef]
) -> list[str]:
    """Canonical sources whose frames land in more than one of train/val/test (sorted)."""
    stages = (("train", train), ("val", val), ("test", test))
    spans: dict[str, set[str]] = {}
    repr_src: dict[str, str] = {}
    for stage_name, refs in stages:
        for r in refs:
            key = _norm_source(r.source)
            spans.setdefault(key, set()).add(stage_name)
            repr_src.setdefault(key, r.source)
    return sorted(repr_src[key] for key, seen in spans.items() if len(seen) > 1)


def evaluate_constraints(
    train: list[SampleRef],
    val: list[SampleRef],
    test: list[SampleRef],
    *,
    constraints: list[Constraint],
    available_attrs: frozenset[str] = frozenset(),
) -> list[ConstraintResult]:
    """Evaluate every constraint against the resolved stages (pure; never raises).

    Returns one :class:`ConstraintResult` per constraint. ``no_train_anomalous`` is
    ``unavailable`` when ``category_ids`` was not populated (a module that cannot supply
    labels), so a skipped check never reads as ``satisfied``. Call
    :func:`enforce_constraints` to act on the results.
    """
    from cuvis_ai_schemas.training.data import ConstraintKind

    results: list[ConstraintResult] = []
    for c in constraints:
        kind = c.kind
        kv = kind.value
        sev = c.severity.value
        if kind == ConstraintKind.NO_SPLIT_OVERLAP:
            leaks = _uid_overlaps(train, val, test)
            offending = tuple(sorted({uid for _, _, shared in leaks for uid in shared}))
            if offending:
                reason = "; ".join(
                    f"{len(shared)} shared between {a} and {b}"
                    for a, b, shared in leaks
                )
                results.append(
                    ConstraintResult(
                        kv, sev, "violated", len(offending), offending, reason
                    )
                )
            else:
                results.append(ConstraintResult(kv, sev, "satisfied"))
        elif kind == ConstraintKind.NO_SOURCE_OVERLAP:
            offending = tuple(_sources_spanning_stages(train, val, test))
            if offending:
                results.append(
                    ConstraintResult(
                        kv,
                        sev,
                        "violated",
                        len(offending),
                        offending,
                        f"{len(offending)} source(s) span >1 of train/val/test",
                    )
                )
            else:
                results.append(ConstraintResult(kv, sev, "satisfied"))
        elif kind == ConstraintKind.NO_TRAIN_ANOMALOUS:
            if "category_ids" not in available_attrs:
                results.append(
                    ConstraintResult(
                        kv,
                        sev,
                        "unavailable",
                        reason="data module cannot supply category labels",
                    )
                )
            else:
                offending = tuple(sorted({r.source for r in train if _is_anomalous(r)}))
                if offending:
                    results.append(
                        ConstraintResult(
                            kv,
                            sev,
                            "violated",
                            len(offending),
                            offending,
                            f"{len(offending)} source(s) with an anomalous frame in train",
                        )
                    )
                else:
                    results.append(ConstraintResult(kv, sev, "satisfied"))
        else:  # pragma: no cover - schema validates the kind set
            results.append(
                ConstraintResult(kv, sev, "unavailable", reason=f"unknown kind {kv}")
            )
    return results


def enforce_constraints(results: list[ConstraintResult]) -> None:
    """Raise on any ``error``-severity ``violated``/``unavailable`` result; ``warn`` logs.

    ``unavailable`` is treated like a failure at ``error`` severity (Codex): a required
    invariant that cannot be verified must fail loud, never silently permit the run.
    """
    hard = [
        r
        for r in results
        if r.severity == "error" and r.status in ("violated", "unavailable")
    ]
    for r in results:
        if r.severity == "warn" and r.status in ("violated", "unavailable"):
            logger.warning(
                "split constraint %s %s: %s",
                r.kind,
                r.status,
                r.reason or f"{r.count} offending",
            )
    if hard:
        msg = "; ".join(
            f"{r.kind} {r.status}" + (f" ({r.reason})" if r.reason else "")
            for r in hard
        )
        raise SplitConstraintError(f"split constraint(s) failed: {msg}")


# -- internals ----------------------------------------------------------------


def _norm_source(path: str) -> str:
    """Normalize a source path for comparison (never for identity).

    ``files`` / ``file_indices`` selectors written by an external author (e.g.
    the CuvisNEXT split designer) may differ from the module's enumerated
    ``SampleRef.source`` in separator style or drive-letter case even when they
    name the same file; on Windows that made an exact string match silently
    select 0 samples. Comparison-only defense: ``SampleRef.uid`` derivation is
    untouched, so frozen splits and hashes stay stable.
    """
    return os.path.normcase(os.path.normpath(path))


def _resolve_one(
    sel: Selector, refs: list[SampleRef], *, name_to_id: dict[str, int] | None
) -> list[SampleRef]:
    from cuvis_ai_schemas.training.data import SelectorKind

    kind = sel.kind
    if kind == SelectorKind.ALL:
        return list(refs)
    if kind == SelectorKind.FILES:
        wanted = {_norm_source(p) for p in sel.paths}
        return [r for r in refs if _norm_source(r.source) in wanted]
    if kind == SelectorKind.FILE_INDICES:
        wanted_ids = set(_expand_int_ids(sel.ids, sel))
        wanted_source = _norm_source(sel.source)
        return [
            r
            for r in refs
            if _norm_source(r.source) == wanted_source and r.index in wanted_ids
        ]
    if kind == SelectorKind.DIR_INDICES:
        positions = _expand_int_ids(sel.ids, sel)
        size = len(refs)
        out_of_range = [p for p in positions if p >= size]
        if out_of_range:
            raise ValueError(
                f"dir_indices positions {out_of_range} out of range (universe size {size})"
            )
        # preserve requested order, dedup positions
        seen: set[int] = set()
        return [refs[p] for p in positions if not (p in seen or seen.add(p))]
    if kind == SelectorKind.STEMS:
        wanted = set(sel.stems)
        return [r for r in refs if r.stem in wanted]
    if kind == SelectorKind.GLOB:
        pat = sel.pattern or ""
        return [
            r
            for r in refs
            if fnmatch.fnmatch(r.stem, pat) or fnmatch.fnmatch(r.source, pat)
        ]
    if kind == SelectorKind.TAG:
        wanted = set(sel.any_of)
        return [r for r in refs if any(t in wanted for t in r.tags)]
    if kind == SelectorKind.CATEGORIES:
        wanted_ids = set(_names_to_ids(sel.any_of, name_to_id))
        return [r for r in refs if any(c in wanted_ids for c in r.category_ids)]
    if kind in (SelectorKind.UNION, SelectorKind.EXCEPT, SelectorKind.INTERSECT):
        return _resolve_setop(sel, refs, name_to_id=name_to_id)
    raise ValueError(f"unknown selector kind {kind!r}")  # pragma: no cover


def _resolve_setop(
    sel: Selector, refs: list[SampleRef], *, name_to_id: dict[str, int] | None
) -> list[SampleRef]:
    from cuvis_ai_schemas.training.data import SelectorKind

    parts = [_resolve_one(child, refs, name_to_id=name_to_id) for child in sel.of]
    if sel.kind == SelectorKind.UNION:
        out: list[SampleRef] = []
        seen: set[str] = set()
        for part in parts:
            for ref in part:
                if ref.uid not in seen:
                    seen.add(ref.uid)
                    out.append(ref)
        return out
    if sel.kind == SelectorKind.EXCEPT:
        first = parts[0]
        remove: set[str] = set()
        for part in parts[1:]:
            remove |= {r.uid for r in part}
        return [r for r in first if r.uid not in remove]
    # INTERSECT: keep refs (ordered by the left operand) present in every operand.
    first = parts[0]
    common = {r.uid for r in first}
    for part in parts[1:]:
        common &= {r.uid for r in part}
    return [r for r in first if r.uid in common]


def _expand_int_ids(ids: list[int | str], sel: Selector) -> list[int]:
    expanded = expand_range_selectors(ids)
    out: list[int] = []
    for value in expanded:
        if not isinstance(value, int):
            raise ValueError(
                f"selector '{sel.kind.value}' ids must be ints or 'a-b' ranges, got {value!r}"
            )
        if value < 0:
            raise ValueError(f"selector '{sel.kind.value}' index {value} must be >= 0")
        out.append(value)
    return out


def _names_to_ids(names: list[str], name_to_id: dict[str, int] | None) -> list[int]:
    if name_to_id is None:
        raise ValueError(
            "a 'categories' selector needs a category name->id map, but this dataset has none"
        )
    ids: list[int] = []
    for name in names:
        if name not in name_to_id:
            raise ValueError(
                f"category '{name}' not in dataset categories {sorted(name_to_id)}"
            )
        ids.append(name_to_id[name])
    return ids


__all__ = [
    "SplitConstraintError",
    "SplitLeakageError",
    "ConstraintResult",
    "required_attrs",
    "constraint_required_attrs",
    "resolve_selectors",
    "validate_leakage",
    "evaluate_constraints",
    "enforce_constraints",
]
