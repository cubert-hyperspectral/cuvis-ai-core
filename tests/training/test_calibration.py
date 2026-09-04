"""Post-training decider calibration phase (``calibrate_pipeline_deciders``).

The phase runs the trained pipeline over the val split, collects each decider's input scores
and the ground-truth mask, and calls ``decider.calibrate(...)``. These tests use mock nodes
(the real F1-max sweep lives on the cuvis-ai decider subclasses) to check the orchestration:
the right decider is found and handed the stacked val scores + targets; single-class and
no-decider cases are skipped; and the calibrated hparams reach ``pipeline.serialize()`` - the
config the gRPC production save re-derives from the live pipeline.
"""

import pytest
import pytorch_lightning as pl
import torch

from cuvis_ai_core.deciders.base_decider import BinaryDecider
from cuvis_ai_core.node import Node
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.training import calibrate_pipeline_deciders
from cuvis_ai_schemas.pipeline import PortSpec

pytestmark = pytest.mark.unit


class _ScoreMaskSource(Node):
    """Source node: passes the batch's scores + mask through as outputs."""

    INPUT_SPECS = {
        "scores": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, -1)),
        "mask": PortSpec(dtype=torch.bool, shape=(-1, -1, -1, -1)),
    }
    OUTPUT_SPECS = {
        "scores": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, -1)),
        "mask": PortSpec(dtype=torch.bool, shape=(-1, -1, -1, -1)),
    }

    def forward(self, scores, mask, **_):
        """Echo the score and mask tensors unchanged."""
        return {"scores": scores, "mask": mask}


class _MockCalibratableDecider(BinaryDecider):
    """Decider that overrides ``calibrate`` to record the call and set a known threshold."""

    OUTPUT_SPECS = {"decisions": PortSpec(dtype=torch.bool, shape=(-1, -1, -1, 1))}

    def __init__(self, threshold: float = 0.5, **kwargs):
        super().__init__(threshold=threshold, **kwargs)
        self.calibrated_with: tuple | None = None

    def forward(self, logits, **_):
        """Threshold the scores at the current (possibly calibrated) value."""
        return {"decisions": (logits >= self.threshold)}

    def calibrate(self, scores, targets):
        """Record the stacked shapes and move the threshold to a fixed known value."""
        self.calibrated_with = (tuple(scores.shape), tuple(targets.shape))
        self.threshold = 0.42
        self.hparams["threshold"] = 0.42
        return {"class": type(self).__name__, "threshold": {"old": 0.5, "new": 0.42}}


class _PlainDecider(BinaryDecider):
    """Decider that does NOT override ``calibrate`` - must be treated as non-calibratable."""

    OUTPUT_SPECS = {"decisions": PortSpec(dtype=torch.bool, shape=(-1, -1, -1, 1))}

    def forward(self, logits, **_):
        """Plain threshold, no calibration support."""
        return {"decisions": (logits >= self.threshold)}


class _ListDataModule(pl.LightningDataModule):
    """Minimal datamodule serving a fixed list of batches as its val split."""

    def __init__(self, batches):
        super().__init__()
        self._batches = batches
        self.val_ds = list(batches)  # non-None => the split exists

    def val_dataloader(self):
        return self._batches


def _batch(score: float, anomalous: bool):
    scores = torch.full((1, 4, 4, 1), score)
    mask = torch.zeros((1, 4, 4, 1), dtype=torch.bool)
    if anomalous:
        mask[0, 0, 0, 0] = True
    return {"scores": scores, "mask": mask}


_MIXED = [
    _batch(1.0, True),
    _batch(1.0, True),
    _batch(0.0, False),
]  # 2 anomalous, 1 normal


def _pipeline(decider):
    src = _ScoreMaskSource(name="src")
    pipe = CuvisPipeline("cal_test")
    pipe.connect(src.outputs.scores, decider.inputs.logits)
    return pipe


def test_calibrates_the_decider_over_val():
    dec = _MockCalibratableDecider(name="dec")
    reports = calibrate_pipeline_deciders(
        _pipeline(dec), _ListDataModule(_MIXED), split="val"
    )

    assert len(reports) == 1
    # scores + targets were stacked over the 3 val frames and handed to the decider.
    assert dec.calibrated_with == ((3, 4, 4, 1), (3, 4, 4, 1))
    assert dec.threshold == 0.42
    assert dec.hparams["threshold"] == 0.42


def test_skips_single_class_val():
    dec = _MockCalibratableDecider(name="dec")
    normals = [_batch(0.0, False), _batch(0.0, False)]
    reports = calibrate_pipeline_deciders(
        _pipeline(dec), _ListDataModule(normals), split="val"
    )

    assert reports == []
    assert dec.calibrated_with is None  # never called
    assert dec.threshold == 0.5  # shipped value left untouched


def test_skips_when_no_calibratable_decider():
    dec = _PlainDecider(name="dec")  # inherits the base no-op calibrate
    reports = calibrate_pipeline_deciders(
        _pipeline(dec), _ListDataModule(_MIXED), split="val"
    )
    assert reports == []


def test_skips_when_no_val_split():
    dec = _MockCalibratableDecider(name="dec")
    dm = _ListDataModule(_MIXED)
    dm.val_ds = None
    assert calibrate_pipeline_deciders(_pipeline(dec), dm, split="val") == []
    assert dec.calibrated_with is None


def test_calibrated_hparams_reach_serialize():
    """The gRPC save re-derives config from the live pipeline; serialize() must carry it."""
    dec = _MockCalibratableDecider(name="dec")
    pipe = _pipeline(dec)
    calibrate_pipeline_deciders(pipe, _ListDataModule(_MIXED), split="val")

    config = pipe.serialize()
    dec_cfg = next(node for node in config.nodes if node.name == "dec")
    assert dec_cfg.hparams["threshold"] == 0.42
