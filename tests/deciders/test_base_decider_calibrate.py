"""The calibration contract of the base decider.

A decider declares its rule (``CALIBRATION_RULE``), the hparams the rule moves
(``CALIBRATION_GATE``, ``CALIBRATION_CUTOFF``) and the levels it serves
(``CALIBRATION_LEVELS``), and reaches into the search through two hooks
(``calibration_scores``, ``calibration_image_scores``); the base ``calibrate`` does the rest.
These tests pin that contract on minimal subclasses with pure tensors, the report shape
``DeciderCalibration.from_report`` reads, and the detection rule of the calibration phase.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from cuvis_ai_schemas.training import CalibrationConfig
from torch import Tensor

from cuvis_ai_core.deciders import threshold_search
from cuvis_ai_core.deciders.base_decider import BinaryDecider
from cuvis_ai_core.training.calibration import DeciderCalibration, calibratable_deciders


class Plain(BinaryDecider):
    """No rule: the base ``calibrate`` is a no-op and the phase skips the node."""

    def forward(self, logits: Tensor, context=None, **_: Any) -> dict[str, Tensor]:
        return {"decisions": logits >= self.threshold}


class Custom(BinaryDecider):
    """No rule but its own ``calibrate``: a plugin decider from before the contract."""

    def forward(self, logits: Tensor, context=None, **_: Any) -> dict[str, Tensor]:
        return {"decisions": logits >= self.threshold}

    def calibrate(self, scores, targets, objective=None):
        return {"class": "Custom", "threshold": {"old": self.threshold, "new": 0.25}}


class Gated(BinaryDecider):
    """The gate-and-cutoff rule: an image gate on the top-k mean and a pixel cutoff."""

    CALIBRATION_RULE = "gate_cutoff"
    CALIBRATION_GATE = "image_threshold"
    CALIBRATION_CUTOFF = "pixel_threshold"

    def __init__(
        self, image_threshold: float = 0.5, pixel_threshold: float = 0.5, **kwargs
    ):
        self.image_threshold = image_threshold
        self.pixel_threshold = pixel_threshold
        super().__init__(
            image_threshold=image_threshold, pixel_threshold=pixel_threshold, **kwargs
        )

    def calibration_image_scores(self, pixel_scores: np.ndarray) -> np.ndarray:
        return threshold_search.topk_mean_scores(pixel_scores, 0.01)

    def forward(self, logits: Tensor, context=None, **_: Any) -> dict[str, Tensor]:
        return {"decisions": logits >= self.pixel_threshold}


class Sigmoid(BinaryDecider):
    """The cutoff rule on sigmoid probabilities, the space its ``forward`` compares in."""

    CALIBRATION_RULE = "cutoff"
    CALIBRATION_CUTOFF = "threshold"

    def calibration_scores(self, scores: Tensor) -> Tensor:
        return torch.sigmoid(scores.detach().to("cpu", torch.float32))

    def forward(self, logits: Tensor, context=None, **_: Any) -> dict[str, Tensor]:
        return {"decisions": torch.sigmoid(logits) >= self.threshold}


class Quant(BinaryDecider):
    """The quantile rule: a per-frame cutoff, pixels only."""

    CALIBRATION_RULE = "quantile"
    CALIBRATION_CUTOFF = "quantile"
    CALIBRATION_LEVELS = ("pixels",)

    def __init__(self, quantile: float = 0.995, **kwargs):
        self.quantile = quantile
        super().__init__(quantile=quantile, **kwargs)

    def forward(self, logits: Tensor, context=None, **_: Any) -> dict[str, Tensor]:
        cut = torch.quantile(logits.flatten(1), self.quantile, dim=1)
        return {"decisions": logits >= cut.view(-1, 1, 1, 1)}


def _logits(seed: int = 0, n: int = 6, size: int = 12, positive_rate: float = 0.1):
    """Distinct float32 logits where anomalies score higher on average, as tensors."""
    rng = np.random.default_rng(seed)
    scores = torch.from_numpy(rng.normal(size=(n, size, size, 1)).astype(np.float32))
    targets = torch.from_numpy(rng.random((n, size, size, 1)) < positive_rate)
    scores[targets] += 1.5
    targets[0, 0, 0, 0] = True
    targets[1] = False
    return scores, targets


def _pipeline(*nodes):
    """What ``calibratable_deciders`` reads of a pipeline: its nodes."""
    return SimpleNamespace(nodes=lambda: list(nodes))


def _block(objective: str = "f1", target: float | None = None, level: str = "pixels"):
    return CalibrationConfig(objective=objective, target=target, level=level)


# --- detection -------------------------------------------------------------------------------


def test_a_decider_without_a_rule_is_not_calibratable():
    plain = Plain()
    assert calibratable_deciders(_pipeline(plain)) == []
    assert plain.calibrate(*_logits()) is None
    assert plain.threshold == 0.5


def test_a_rule_or_an_own_calibrate_makes_a_decider_calibratable():
    gated, custom, plain = Gated(), Custom(), Plain()
    assert calibratable_deciders(_pipeline(plain, gated, custom)) == [gated, custom]


def test_a_gate_rule_without_a_gate_name_is_refused_at_class_creation():
    with pytest.raises(TypeError, match="CALIBRATION_GATE"):

        class Half(BinaryDecider):
            CALIBRATION_RULE = "gate_cutoff"

            def forward(self, logits, context=None, **_):
                return {"decisions": logits >= 0}


def test_an_unknown_rule_is_refused_at_class_creation():
    with pytest.raises(TypeError, match="CALIBRATION_RULE"):

        class Odd(BinaryDecider):
            CALIBRATION_RULE = "median"

            def forward(self, logits, context=None, **_):
                return {"decisions": logits >= 0}


# --- the gate-and-cutoff rule ----------------------------------------------------------------


def test_gate_cutoff_sets_both_hparams_and_reports_every_key_of_the_old_report():
    scores, targets = _logits()
    decider = Gated()
    report = decider.calibrate(scores, targets)
    assert report["class"] == "Gated"
    new_gate, new_cutoff = (
        report["image_threshold"]["new"],
        report["pixel_threshold"]["new"],
    )
    assert decider.image_threshold == decider.hparams["image_threshold"] == new_gate
    assert decider.pixel_threshold == decider.hparams["pixel_threshold"] == new_cutoff
    assert report["image_threshold"]["old"] == report["pixel_threshold"]["old"] == 0.5
    assert set(report["on_point"]) == {"image_threshold", "pixel_threshold"}
    assert set(report["joint"]) == {"image_threshold", "pixel_threshold", "f1"}
    universal = {
        "objective",
        "level",
        "pixel_f1",
        "frame_recall",
        "frames",
        "anomalous_frames",
    }
    assert universal | {"image_f1", "f1", "precision", "recall"} <= set(report)
    assert (report["objective"], report["level"]) == ("f1", "pixels")
    assert "target" not in report and "reached" not in report


def test_gate_cutoff_matches_the_entry_point_on_the_hooks():
    scores, targets = _logits()
    _, pixel, gt, labels = threshold_search.reduce_scores_targets(scores, targets)
    goal = threshold_search.CalibrationObjective("recall", 0.9, "frames")
    image_scores = threshold_search.topk_mean_scores(pixel, 0.01)
    found = threshold_search.calibrate_gate_cutoff(
        pixel, gt, image_scores, labels, goal
    )
    decider = Gated()
    report = decider.calibrate(
        scores, targets, objective=_block("recall", 0.9, "frames")
    )
    assert (decider.image_threshold, decider.pixel_threshold) == (
        found.values["gate"],
        found.values["cutoff"],
    )
    assert (report["target"], report["reached"]) == (0.9, 1.0)
    assert report["frame_recall"] == found.counts["frame_recall"]
    assert "joint" not in report


# --- the cutoff rule -------------------------------------------------------------------------


def test_cutoff_runs_on_the_scores_hook_and_reports_reached():
    scores, targets = _logits(seed=2)
    _, pixel, gt, labels = threshold_search.reduce_scores_targets(
        torch.sigmoid(scores), targets
    )
    goal = threshold_search.CalibrationObjective("precision", 0.8, "pixels")
    found = threshold_search.calibrate_cutoff(pixel, gt, labels, goal)
    decider = Sigmoid()
    report = decider.calibrate(scores, targets, objective=_block("precision", 0.8))
    assert decider.threshold == decider.hparams["threshold"] == found.values["cutoff"]
    assert 0.0 < decider.threshold < 1.0
    assert report["on_point"] == {"threshold": found.on_point["cutoff"]}
    assert report["reached"] == (1.0 if found.reached else 0.0)
    assert report["precision"] == found.counts["precision"]


# --- the quantile rule -----------------------------------------------------------------------


def test_quantile_calibrates_pixels_and_refuses_frames():
    scores, targets = _logits(seed=5)
    decider = Quant()
    with pytest.raises(ValueError, match="frames"):
        decider.calibrate(scores, targets, objective=_block("recall", 0.9, "frames"))
    assert decider.quantile == 0.995
    report = decider.calibrate(scores, targets, objective=_block("recall", 0.9))
    assert decider.quantile == decider.hparams["quantile"] == report["quantile"]["new"]
    assert report["quantile"]["old"] == 0.995
    assert report["on_point"] == {"quantile": decider.quantile}
    assert report["reached"] == 1.0
    assert report["recall"] >= 0.9


# --- the phase reads the report --------------------------------------------------------------


@pytest.mark.parametrize("cls", [Gated, Sigmoid, Quant])
def test_from_report_reads_the_thresholds_and_the_metrics(cls):
    scores, targets = _logits(seed=1)
    decider = cls()
    report = decider.calibrate(scores, targets, objective=_block("recall", 0.9))
    record = DeciderCalibration.from_report("decider", report)
    names = {cls.CALIBRATION_CUTOFF} | (
        {cls.CALIBRATION_GATE} if cls.CALIBRATION_GATE else set()
    )
    assert set(record.thresholds) == names
    assert all(
        change.after == decider.hparams[name]
        for name, change in record.thresholds.items()
    )
    assert record.metrics["target"] == 0.9
    assert record.metrics["reached"] in (0.0, 1.0)
    assert {"pixel_f1", "frame_f1", "f1", "precision", "recall"} <= set(record.metrics)
