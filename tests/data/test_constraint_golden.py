"""Cross-language golden fixture for split-constraint evaluation.

The same ``constraint_cases.json`` is vendored byte-identical into the cuvis-next C++ suite
(``libs/pilot_utility/test``) and the cuvis-next-mock vitest suite. Each repo asserts the same
expected results and pins the LF-normalized SHA-256, so an accidental local edit to any copy
turns that repo's suite red. (A truly shared cross-repo guard would need one CI job over all
three; this catches local drift, which is the common failure mode.)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cuvis_ai_core.data.selectors import evaluate_constraints
from cuvis_ai_schemas.training.data import (
    Constraint,
    ConstraintKind,
    ConstraintSeverity,
    SampleRef,
)

FIXTURE = Path(__file__).parent / "fixtures" / "constraint_cases.json"

#: SHA-256 of the fixture with CRLF normalized to LF. Update in EVERY vendored copy's pin
#: whenever the fixture changes (the update-all-copies ritual).
GOLDEN_SHA256 = "99941a3d8083af482726d088573bb6f538560fe91c79e69eece11e5d072db3fc"


def _cases() -> list[dict]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]


def test_golden_fixture_sha_pinned() -> None:
    normalized = FIXTURE.read_bytes().replace(b"\r\n", b"\n")
    digest = hashlib.sha256(normalized).hexdigest()
    assert digest == GOLDEN_SHA256, (
        "constraint_cases.json changed; update every vendored copy (cuvis-next, mock) and "
        "each repo's GOLDEN_SHA256 pin together."
    )


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c["name"])
def test_constraint_case(case: dict) -> None:
    by_key: dict[tuple[str, int], SampleRef] = {}
    for row in case["universe"]:
        ref = SampleRef(
            source=row["source"], index=row["index"], category_ids=row.get("category_ids", [])
        )
        by_key[(row["source"], row["index"])] = ref

    def stage(name: str) -> list[SampleRef]:
        return [by_key[(src, idx)] for src, idx in case.get(name, [])]

    constraints = [
        Constraint(kind=ConstraintKind(c["kind"]), severity=ConstraintSeverity(c["severity"]))
        for c in case["constraints"]
    ]
    results = evaluate_constraints(
        stage("train"),
        stage("val"),
        stage("test"),
        constraints=constraints,
        available_attrs=frozenset(case.get("available_attrs", [])),
    )
    got = [
        {"kind": r.kind, "status": r.status, "count": r.count, "offending": list(r.offending)}
        for r in results
    ]
    assert got == case["expected"]
