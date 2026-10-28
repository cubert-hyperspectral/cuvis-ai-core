"""Write the validation summary a calibration pass leaves beside a saved pipeline.

Every calibration pass that moved at least one decider's thresholds can hand back, in its
report, a small JSON-serializable ``"summary"`` describing what the decider found on the
validation split (the operating-point tables and the calibrated point): see cuvis-ai's
``calibrate()`` implementations. This module treats that dict as opaque data and does one
thing: when the saved pipeline has weights and at least one decider's report carries a
summary, it writes them, plus the provenance that binds the file to those exact weights
and that exact yaml, to ``validation_summary.json`` beside the yaml. No cuvis-ai import,
no model knowledge: a decider without a ``"summary"`` key is simply left out.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from loguru import logger

from cuvis_ai_core.training.calibration import CalibrationOutcome


def _sha256_without_cr(data: bytes) -> str:
    """Sha256 of ``data`` with every CR byte stripped first.

    The CuvisNEXT client hashes a pipeline yaml the same way (``results.json``'s binding),
    so a file saved with CRLF line endings hashes identically to its LF form.
    """
    return hashlib.sha256(data.replace(b"\r", b"")).hexdigest()


def _file_sha256(path: Path) -> str:
    """Sha256 of the file at ``path``, read in chunks so a large checkpoint never sits in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_json_mapping(data_config: Any) -> dict[str, Any] | None:
    """``data_config`` as a plain JSON-safe mapping, or ``None``.

    Accepts a pydantic ``DataConfig`` (via ``model_dump(mode="json")``), a plain mapping
    (recorded as is), or ``None``.
    """
    if data_config is None:
        return None
    model_dump = getattr(data_config, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    if isinstance(data_config, Mapping):
        return dict(data_config)
    raise TypeError(
        f"data_config must be a DataConfig, a mapping, or None; got {type(data_config)!r}"
    )


def write_validation_summary(
    saved_pipeline_yaml: str | Path,
    outcome: CalibrationOutcome,
    data_config: Any,
) -> Path | None:
    """Write ``validation_summary.json`` beside ``saved_pipeline_yaml``; return its path.

    Collects every decider record in ``outcome.deciders`` whose report carries a
    ``"summary"`` dict (deciders skipped, or calibrated without one, are left out) and
    writes them under ``deciders``, each as ``{"node": <name>, **report["summary"]}``,
    alongside the provenance that binds the file: the sha256 of the weights file (the
    ``.pt`` sibling of the yaml, hashed from the bytes already on disk, so it matches
    what was actually saved) and of the yaml itself (CR bytes stripped, matching the
    CuvisNEXT client's own hash), the effective data config the pass ran on, and the
    split's frame counts.

    Returns ``None``, with one warning line, and writes nothing when no ``.pt`` sits
    beside the yaml or when no decider's report carries a summary; an existing file from
    an earlier pass is left untouched in either case. The write is atomic (a temporary
    file in the same directory, moved into place) and rejects non-finite numbers
    (``NaN`` / ``Infinity``) rather than emit them as the non-standard JSON extension
    ``json.dumps`` would otherwise produce.
    """
    yaml_path = Path(saved_pipeline_yaml)
    weights_path = yaml_path.with_suffix(".pt")
    if not weights_path.is_file():
        logger.warning(
            f"Validation summary: no weights file ({weights_path}) beside "
            f"{yaml_path}; not writing a summary."
        )
        return None

    deciders: list[dict[str, Any]] = []
    for record in outcome.deciders:
        report = record.report
        if report is None:
            continue
        summary = report.get("summary")
        if not isinstance(summary, dict):
            continue
        deciders.append({"node": record.name, **summary})

    if not deciders:
        logger.warning(
            f"Validation summary: no calibrated decider reported a summary for "
            f"{yaml_path}; not writing a summary."
        )
        return None

    content = {
        "version": 1,
        "provenance": {
            "weights_sha256": _file_sha256(weights_path),
            "pipeline_yaml_sha256": _sha256_without_cr(yaml_path.read_bytes()),
            "data": _as_json_mapping(data_config),
            "split": outcome.split,
            "frames": outcome.frames,
            "anomalous_frames": outcome.anomalous_frames,
        },
        "deciders": deciders,
    }

    target = yaml_path.with_name("validation_summary.json")
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=target.parent,
        prefix=target.name + ".",
        suffix=".tmp",
        delete=False,
    ) as staging:
        staging_path = Path(staging.name)
        try:
            json.dump(content, staging, allow_nan=False)
        except BaseException:
            staging.close()
            staging_path.unlink(missing_ok=True)
            raise
    os.replace(staging_path, target)
    return target


__all__ = ["write_validation_summary"]
