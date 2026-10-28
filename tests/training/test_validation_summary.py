"""``write_validation_summary``: the file a calibration pass leaves beside the saved pipeline.

The writer is pure: it takes the saved pipeline yaml's path, a :class:`CalibrationOutcome`
and the effective data config the pass ran on, and writes ``validation_summary.json`` beside
the yaml from whatever decider reports carry a ``"summary"`` dict. No model, no cuvis-ai
import, no I/O beyond reading the sibling ``.pt`` and the yaml it was asked about. The content
is opaque per decider: the writer forwards ``report["summary"]`` unchanged under ``**`` next
to the node's name; the math and the schema of that dict belong to cuvis-ai.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from cuvis_ai_core.training.calibration import (
    CalibrationOutcome,
    DeciderCalibration,
    DeciderCalibrationStatus,
)
from cuvis_ai_core.training.validation_summary import write_validation_summary
from cuvis_ai_schemas.training import DataConfig

pytestmark = pytest.mark.unit


def _outcome(*records: DeciderCalibration, frames: int = 48, anomalous: int = 12):
    return CalibrationOutcome(
        split="val",
        applicable=True,
        deciders=tuple(records),
        frames=frames,
        anomalous_frames=anomalous,
    )


def _calibrated(name: str, summary: dict | None, **extra) -> DeciderCalibration:
    report = {"class": "TwoStageBinaryDecider", **extra}
    if summary is not None:
        report["summary"] = summary
    return DeciderCalibration.from_report(name, report)


def _write_yaml(path: Path, *, crlf: bool = False) -> bytes:
    content = "metadata:\n  name: test\nnodes: []\nconnections: []\n"
    data = content.encode("utf-8")
    if crlf:
        data = data.replace(b"\n", b"\r\n")
    path.write_bytes(data)
    return data


def _write_weights(path: Path, content: bytes = b"fake weights") -> bytes:
    path.write_bytes(content)
    return content


# ---------------------------------------------------------------------------
# Happy path: content, both hashes, CRLF handling
# ---------------------------------------------------------------------------


def test_writes_the_summary_with_both_hashes_and_the_decider_payload(tmp_path: Path):
    yaml_path = tmp_path / "trainrun_pipeline.yaml"
    yaml_bytes = _write_yaml(yaml_path)
    weights_bytes = _write_weights(tmp_path / "trainrun_pipeline.pt")

    summary_payload = {
        "version": 1,
        "kind": "two_stage",
        "calibrated": {"thresholds": {"image_threshold": 0.41}},
    }
    outcome = _outcome(_calibrated("gate", summary_payload))
    data_config = DataConfig(data_module="cu3s", params={"cu3s_file_path": "/x.cu3s"})

    result = write_validation_summary(yaml_path, outcome, data_config)

    expected_path = tmp_path / "validation_summary.json"
    assert result == expected_path
    content = json.loads(expected_path.read_text(encoding="utf-8"))

    assert content["version"] == 1
    assert (
        content["provenance"]["weights_sha256"]
        == hashlib.sha256(weights_bytes).hexdigest()
    )
    assert (
        content["provenance"]["pipeline_yaml_sha256"]
        == hashlib.sha256(yaml_bytes).hexdigest()
    )
    assert content["provenance"]["data"] == data_config.model_dump(mode="json")
    assert content["provenance"]["split"] == "val"
    assert content["provenance"]["frames"] == 48
    assert content["provenance"]["anomalous_frames"] == 12

    assert content["deciders"] == [{"node": "gate", **summary_payload}]


def test_yaml_hash_strips_cr_bytes_so_a_crlf_file_matches_the_lf_hash(tmp_path: Path):
    """The CuvisNEXT client computes the yaml sha the same way: strip every CR byte first.
    A file saved with CRLF line endings must hash identically to its LF form."""
    yaml_lf = tmp_path / "lf" / "pipeline.yaml"
    yaml_lf.parent.mkdir()
    lf_bytes = _write_yaml(yaml_lf, crlf=False)
    _write_weights(yaml_lf.with_suffix(".pt"))

    yaml_crlf = tmp_path / "crlf" / "pipeline.yaml"
    yaml_crlf.parent.mkdir()
    _write_yaml(yaml_crlf, crlf=True)
    _write_weights(yaml_crlf.with_suffix(".pt"))

    summary = {"version": 1}
    outcome = _outcome(_calibrated("dec", summary))

    write_validation_summary(yaml_lf, outcome, None)
    write_validation_summary(yaml_crlf, outcome, None)

    lf_content = json.loads(
        (yaml_lf.parent / "validation_summary.json").read_text(encoding="utf-8")
    )
    crlf_content = json.loads(
        (yaml_crlf.parent / "validation_summary.json").read_text(encoding="utf-8")
    )
    expected = hashlib.sha256(lf_bytes.replace(b"\r", b"")).hexdigest()
    assert lf_content["provenance"]["pipeline_yaml_sha256"] == expected
    assert crlf_content["provenance"]["pipeline_yaml_sha256"] == expected


def test_data_config_none_is_recorded_as_null(tmp_path: Path):
    yaml_path = tmp_path / "pipeline.yaml"
    _write_yaml(yaml_path)
    _write_weights(yaml_path.with_suffix(".pt"))
    outcome = _outcome(_calibrated("dec", {"version": 1}))

    write_validation_summary(yaml_path, outcome, None)

    content = json.loads(
        (tmp_path / "validation_summary.json").read_text(encoding="utf-8")
    )
    assert content["provenance"]["data"] is None


def test_data_config_as_a_plain_mapping_is_recorded_as_is(tmp_path: Path):
    """``data_config`` accepts a mapping, not only a ``DataConfig`` instance."""
    yaml_path = tmp_path / "pipeline.yaml"
    _write_yaml(yaml_path)
    _write_weights(yaml_path.with_suffix(".pt"))
    outcome = _outcome(_calibrated("dec", {"version": 1}))
    plain = {"data_module": "cu3s", "params": {"cu3s_file_path": "/x.cu3s"}}

    write_validation_summary(yaml_path, outcome, plain)

    content = json.loads(
        (tmp_path / "validation_summary.json").read_text(encoding="utf-8")
    )
    assert content["provenance"]["data"] == plain


# ---------------------------------------------------------------------------
# None without a .pt, none without summaries
# ---------------------------------------------------------------------------


def test_returns_none_and_logs_when_no_pt_beside_the_yaml(tmp_path: Path, monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(
        "cuvis_ai_core.training.validation_summary.logger.warning",
        lambda msg: warnings.append(msg),
    )
    yaml_path = tmp_path / "pipeline.yaml"
    _write_yaml(yaml_path)  # no sibling .pt written
    outcome = _outcome(_calibrated("dec", {"version": 1}))

    result = write_validation_summary(yaml_path, outcome, None)

    assert result is None
    assert not (tmp_path / "validation_summary.json").exists()
    assert len(warnings) == 1
    assert ".pt" in warnings[0]


def test_returns_none_and_logs_when_no_decider_carries_a_summary(
    tmp_path: Path, monkeypatch
):
    warnings: list[str] = []
    monkeypatch.setattr(
        "cuvis_ai_core.training.validation_summary.logger.warning",
        lambda msg: warnings.append(msg),
    )
    yaml_path = tmp_path / "pipeline.yaml"
    _write_yaml(yaml_path)
    _write_weights(yaml_path.with_suffix(".pt"))
    # Calibrated, but its report carries no "summary" key.
    outcome = _outcome(_calibrated("dec", summary=None, f1=0.8))

    result = write_validation_summary(yaml_path, outcome, None)

    assert result is None
    assert not (tmp_path / "validation_summary.json").exists()
    assert len(warnings) == 1
    assert "summary" in warnings[0].lower()


def test_returns_none_when_outcome_has_no_decider_records_at_all(tmp_path: Path):
    yaml_path = tmp_path / "pipeline.yaml"
    _write_yaml(yaml_path)
    _write_weights(yaml_path.with_suffix(".pt"))
    outcome = CalibrationOutcome(split="val", applicable=False)

    assert write_validation_summary(yaml_path, outcome, None) is None


def test_skipped_decider_without_a_report_is_not_a_summary_source(tmp_path: Path):
    """A decider the phase skipped (no report at all) never contributes to the summary,
    even when another decider did."""
    yaml_path = tmp_path / "pipeline.yaml"
    _write_yaml(yaml_path)
    _write_weights(yaml_path.with_suffix(".pt"))
    good = _calibrated("good", {"version": 1})
    skipped = DeciderCalibration(
        name="bad",
        class_name="SomeDecider",
        status=DeciderCalibrationStatus.SKIPPED,
        skip=None,
        report=None,
    )
    outcome = CalibrationOutcome(
        split="val",
        applicable=True,
        deciders=(good, skipped),
        frames=48,
        anomalous_frames=12,
    )

    result = write_validation_summary(yaml_path, outcome, None)
    assert result is not None
    content = json.loads(result.read_text(encoding="utf-8"))
    assert [d["node"] for d in content["deciders"]] == ["good"]


def test_only_deciders_with_a_summary_are_included_mixed_calibration(tmp_path: Path):
    """Some deciders calibrated with a summary, one without: only the former are listed."""
    yaml_path = tmp_path / "pipeline.yaml"
    _write_yaml(yaml_path)
    _write_weights(yaml_path.with_suffix(".pt"))
    with_summary = _calibrated("gate", {"version": 1, "kind": "two_stage"})
    without_summary = _calibrated("plain", summary=None, f1=0.5)
    outcome = _outcome(with_summary, without_summary)

    result = write_validation_summary(yaml_path, outcome, None)
    assert result is not None
    content = json.loads(result.read_text(encoding="utf-8"))
    assert [d["node"] for d in content["deciders"]] == ["gate"]


# ---------------------------------------------------------------------------
# No NaN, existing file left alone, atomic write
# ---------------------------------------------------------------------------


def test_rejects_non_finite_numbers_in_a_summary(tmp_path: Path):
    """``json.dumps(..., allow_nan=False)`` must reject NaN/Infinity rather than emit
    the non-standard ``NaN`` / ``Infinity`` tokens into the file."""
    yaml_path = tmp_path / "pipeline.yaml"
    _write_yaml(yaml_path)
    _write_weights(yaml_path.with_suffix(".pt"))
    outcome = _outcome(_calibrated("dec", {"version": 1, "score": math.nan}))

    with pytest.raises(ValueError):
        write_validation_summary(yaml_path, outcome, None)
    assert not (tmp_path / "validation_summary.json").exists()


def test_existing_file_is_left_alone_when_nothing_is_written(tmp_path: Path):
    """A ``None``-returning call (missing .pt, no summaries) never deletes a file that
    was already there from an earlier, successful pass."""
    yaml_path = tmp_path / "pipeline.yaml"
    _write_yaml(yaml_path)
    existing = tmp_path / "validation_summary.json"
    existing.write_text('{"version": 1, "deciders": []}', encoding="utf-8")

    outcome = _outcome(_calibrated("dec", summary=None, f1=0.8))
    result = write_validation_summary(yaml_path, outcome, None)

    assert result is None
    assert existing.read_text(encoding="utf-8") == '{"version": 1, "deciders": []}'


def test_write_is_atomic_no_leftover_temp_file(tmp_path: Path):
    yaml_path = tmp_path / "pipeline.yaml"
    _write_yaml(yaml_path)
    _write_weights(yaml_path.with_suffix(".pt"))
    outcome = _outcome(_calibrated("dec", {"version": 1}))

    write_validation_summary(yaml_path, outcome, None)

    siblings = list(tmp_path.iterdir())
    names = {p.name for p in siblings}
    assert "validation_summary.json" in names
    assert not any(
        name.endswith(".tmp") or ".tmp" in name
        for name in names
        if name != "pipeline.yaml"
    )


def test_write_overwrites_a_previous_summary_from_an_earlier_pass(tmp_path: Path):
    """A later, successful calibration pass's summary must replace the older file's
    content (the 'never deletes' rule only protects a call that writes nothing)."""
    yaml_path = tmp_path / "pipeline.yaml"
    _write_yaml(yaml_path)
    _write_weights(yaml_path.with_suffix(".pt"))

    first = write_validation_summary(
        yaml_path, _outcome(_calibrated("dec", {"version": 1, "pass": "first"})), None
    )
    second = write_validation_summary(
        yaml_path, _outcome(_calibrated("dec", {"version": 1, "pass": "second"})), None
    )

    assert first == second
    content = json.loads(first.read_text(encoding="utf-8"))
    assert content["deciders"][0]["pass"] == "second"
