"""Post-training decider calibration phase (``calibrate_pipeline_deciders``).

The phase runs the trained pipeline over the val split, collects each decider's input scores
and the ground-truth mask, and calls ``decider.calibrate(...)``. These tests use mock nodes
(the real F1-max sweep lives on the cuvis-ai decider subclasses) to check the orchestration:
the right decider is found and handed the stacked val scores + targets; the outcome says what
happened; single-class, no-decider, unwired, failing and ambiguous-mask cases are skipped
instead of raised; a mask that is merely re-emitted does not hide its originator; and the
calibrated hparams reach ``pipeline.serialize()`` - the config the gRPC production save
re-derives from the live pipeline.
"""

import pytest
import pytorch_lightning as pl
import torch

from cuvis_ai_core.deciders.base_decider import BinaryDecider
from cuvis_ai_core.node import Node
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.training import CalibrationOutcome, calibrate_pipeline_deciders
from cuvis_ai_core.training.calibration import (
    CalibrationPhaseSkip,
    DeciderCalibrationStatus,
    DeciderSkip,
    ThresholdChange,
    uncalibrated_required_consumers,
)
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_core.utils.restore import _log_calibration_outcome
from cuvis_ai_schemas.pipeline import PortSpec
from cuvis_ai_schemas.training import CalibrationConfig

pytestmark = pytest.mark.unit

_SCORES = PortSpec(dtype=torch.float32, shape=(-1, -1, -1, -1))
_MASK = PortSpec(dtype=torch.bool, shape=(-1, -1, -1, -1))
_DECISIONS = PortSpec(dtype=torch.bool, shape=(-1, -1, -1, 1))


class _ScoreMaskSource(Node):
    """Source node: passes the batch's scores + mask through as outputs."""

    INPUT_SPECS = {"scores": _SCORES, "mask": _MASK}
    OUTPUT_SPECS = {"scores": _SCORES, "mask": _MASK}

    def forward(self, scores, mask, **_):
        """Echo the score and mask tensors unchanged."""
        return {"scores": scores, "mask": mask}


class _ScoresOnlySource(Node):
    """Source node without a ground-truth mask output."""

    INPUT_SPECS = {"scores": _SCORES}
    OUTPUT_SPECS = {"scores": _SCORES}

    def forward(self, scores, **_):
        """Echo the scores only."""
        return {"scores": scores}


class _BufferedSource(_ScoreMaskSource):
    """Source carrying a buffer, so the pipeline has a device to read."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.register_buffer("offset", torch.zeros(1))


class _ListScoresSource(_ScoreMaskSource):
    """Source whose scores come out as a Python list, never as a tensor."""

    def forward(self, scores, mask, **_):
        """Echo the mask, but hand the scores over as a nested list."""
        return {"scores": scores.tolist(), "mask": mask}


class _FailingSource(_ScoreMaskSource):
    """Source that blows up at inference."""

    def forward(self, scores, mask, **_):
        """Raise on every call."""
        raise RuntimeError("boom at inference")


class _MaskEcho(Node):
    """Consumes and re-emits ``mask`` (an augmentation wrapper at inference)."""

    INPUT_SPECS = {"mask": _MASK}
    OUTPUT_SPECS = {"mask": _MASK}

    def forward(self, mask, **_):
        """Echo the mask unchanged."""
        return {"mask": mask}


class _MockCalibratableDecider(BinaryDecider):
    """Decider that overrides ``calibrate`` to record the call and set a known threshold."""

    OUTPUT_SPECS = {"decisions": _DECISIONS}

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
        self.calibrated_report = {
            "class": type(self).__name__,
            "threshold": {"old": 0.5, "new": 0.42},
        }
        return self.calibrated_report


class _RaisingDecider(_MockCalibratableDecider):
    """Decider whose ``calibrate`` raises, like a cuvis-ai guard would."""

    def calibrate(self, scores, targets):
        """Refuse the data."""
        raise ValueError("scores must be finite")


class _DecliningDecider(_MockCalibratableDecider):
    """Decider that overrides ``calibrate`` but returns no report."""

    def calibrate(self, scores, targets):
        """Record the call and decline."""
        self.calibrated_with = (tuple(scores.shape), tuple(targets.shape))
        return None


class _TolerantDecider(_MockCalibratableDecider):
    """Decider whose ``forward`` accepts anything, so odd sources still run end to end."""

    def forward(self, logits, **_):
        """Emit a constant decision map."""
        return {"decisions": torch.zeros(1, 4, 4, 1, dtype=torch.bool)}


class _PlainDecider(BinaryDecider):
    """Decider that does NOT override ``calibrate`` - must be treated as non-calibratable."""

    OUTPUT_SPECS = {"decisions": _DECISIONS}

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


def _batch(score: float, anomalous: bool, with_mask: bool = True):
    scores = torch.full((1, 4, 4, 1), score)
    if not with_mask:
        return {"scores": scores}
    mask = torch.zeros((1, 4, 4, 1), dtype=torch.bool)
    if anomalous:
        mask[0, 0, 0, 0] = True
    return {"scores": scores, "mask": mask}


_MIXED = [
    _batch(1.0, True),
    _batch(1.0, True),
    _batch(0.0, False),
]  # 2 anomalous, 1 normal


def _pipeline(decider, source=None, strict=True):
    src = source if source is not None else _ScoreMaskSource(name="src")
    pipe = CuvisPipeline("cal_test", strict_runtime_io_validation=strict)
    pipe.connect(src.outputs.scores, decider.inputs.logits)
    return pipe


def test_calibrates_the_decider_over_val():
    dec = _MockCalibratableDecider(name="dec")
    outcome = calibrate_pipeline_deciders(
        _pipeline(dec), _ListDataModule(_MIXED), split="val"
    )

    assert isinstance(outcome, CalibrationOutcome)
    assert outcome.applicable and outcome.calibrated
    assert set(outcome.reports) == {"dec"}
    assert outcome.skipped == {} and outcome.reason is None
    assert outcome.summary() == "thresholds calibrated on val: dec (threshold 0.42)"
    # scores + targets were stacked over the 3 val frames and handed to the decider.
    assert dec.calibrated_with == ((3, 4, 4, 1), (3, 4, 4, 1))
    assert dec.threshold == 0.42
    assert dec.hparams["threshold"] == 0.42


def test_skips_single_class_val():
    dec = _MockCalibratableDecider(name="dec")
    normals = [_batch(0.0, False), _batch(0.0, False)]
    outcome = calibrate_pipeline_deciders(
        _pipeline(dec), _ListDataModule(normals), split="val"
    )

    assert outcome.applicable and not outcome.calibrated
    assert outcome.reason == "val split is single-class (0/2 frames anomalous)"
    assert outcome.summary() == (
        "thresholds not calibrated: val split is single-class (0/2 frames anomalous)"
    )
    assert dec.calibrated_with is None  # never called
    assert dec.threshold == 0.5  # shipped value left untouched


def test_skips_when_no_calibratable_decider():
    dec = _PlainDecider(name="dec")  # inherits the base no-op calibrate
    outcome = calibrate_pipeline_deciders(
        _pipeline(dec), _ListDataModule(_MIXED), split="val"
    )
    assert outcome.applicable is False
    assert not outcome.calibrated
    assert outcome.summary() == "thresholds not calibrated: no calibratable decider"


def test_base_calibrate_is_a_no_op():
    dec = _PlainDecider(name="dec")
    scores = torch.zeros(1, 4, 4, 1)
    targets = torch.zeros(1, 4, 4, 1, dtype=torch.bool)
    assert dec.calibrate(scores, targets) is None


def test_skips_when_no_val_split():
    dec = _MockCalibratableDecider(name="dec")
    dm = _ListDataModule(_MIXED)
    dm.val_ds = None
    outcome = calibrate_pipeline_deciders(_pipeline(dec), dm, split="val")
    assert outcome.reason == "no val split"
    assert not outcome.calibrated
    assert dec.calibrated_with is None


def test_failing_calibrate_is_skipped_and_the_rest_still_calibrates():
    good = _MockCalibratableDecider(name="good")
    bad = _RaisingDecider(name="bad")
    src = _ScoreMaskSource(name="src")
    pipe = CuvisPipeline("cal_test")
    pipe.connect(src.outputs.scores, good.inputs.logits)
    pipe.connect(src.outputs.scores, bad.inputs.logits)

    outcome = calibrate_pipeline_deciders(pipe, _ListDataModule(_MIXED), split="val")

    assert good.threshold == 0.42
    assert bad.threshold == 0.5  # shipped value kept
    assert outcome.calibrated
    assert outcome.skipped == {"bad": "ValueError: scores must be finite"}
    assert outcome.summary() == (
        "thresholds calibrated on val: good (threshold 0.42); "
        "skipped bad: ValueError: scores must be finite"
    )


def test_all_deciders_failing_reports_not_calibrated():
    bad = _RaisingDecider(name="bad")
    outcome = calibrate_pipeline_deciders(
        _pipeline(bad), _ListDataModule(_MIXED), split="val"
    )
    assert outcome.applicable and not outcome.calibrated
    assert outcome.reason is None
    assert outcome.summary() == (
        "thresholds not calibrated: bad: ValueError: scores must be finite"
    )


def test_declining_decider_is_skipped():
    dec = _DecliningDecider(name="dec")
    outcome = calibrate_pipeline_deciders(
        _pipeline(dec), _ListDataModule(_MIXED), split="val"
    )
    assert dec.calibrated_with is not None  # it was asked
    assert outcome.skipped == {"dec": "calibrate returned no report"}
    assert not outcome.calibrated


def test_unwired_decider_is_skipped_while_the_wired_one_calibrates():
    wired = _MockCalibratableDecider(name="wired")
    loose = _MockCalibratableDecider(name="loose")
    pipe = _pipeline(wired)
    # A decider fed straight from the batch has no edge into ``logits``: the phase cannot
    # tell which score space it thresholds and leaves it alone.
    pipe._assign_counter_and_add_node(loose)
    batches = [dict(batch, logits=batch["scores"]) for batch in _MIXED]

    outcome = calibrate_pipeline_deciders(pipe, _ListDataModule(batches), split="val")

    assert wired.threshold == 0.42
    assert loose.threshold == 0.5
    assert outcome.skipped == {"loose": "nothing is wired into inputs.logits"}
    assert outcome.calibrated


def test_every_decider_unwired_returns_early():
    loose = _MockCalibratableDecider(name="loose")
    src = _ScoreMaskSource(name="src")
    sink = _PlainDecider(name="sink")
    pipe = CuvisPipeline("cal_test")
    pipe.connect(src.outputs.scores, sink.inputs.logits)
    pipe._assign_counter_and_add_node(loose)

    outcome = calibrate_pipeline_deciders(pipe, _ListDataModule(_MIXED), split="val")

    assert outcome.applicable and not outcome.calibrated
    assert outcome.reports == {}
    assert outcome.skipped == {"loose": "nothing is wired into inputs.logits"}
    assert loose.calibrated_with is None


def test_echoed_mask_is_ignored_in_favour_of_its_originator():
    dec = _MockCalibratableDecider(name="dec")
    src = _ScoreMaskSource(name="src")
    echo = _MaskEcho(name="echo")
    pipe = CuvisPipeline("cal_test")
    pipe.connect(src.outputs.scores, dec.inputs.logits)
    pipe.connect(src.outputs.mask, echo.inputs.mask)

    outcome = calibrate_pipeline_deciders(pipe, _ListDataModule(_MIXED), split="val")

    # Two nodes emit ``mask``; the one that only re-emits it does not count.
    assert outcome.calibrated and outcome.reason is None
    assert dec.calibrated_with == ((3, 4, 4, 1), (3, 4, 4, 1))


def test_two_mask_originators_skip_with_both_names():
    dec = _MockCalibratableDecider(name="dec")
    first = _ScoreMaskSource(name="first")
    second = _ScoreMaskSource(name="second")
    echo = _MaskEcho(name="echo")
    pipe = CuvisPipeline("cal_test")
    pipe.connect(first.outputs.scores, dec.inputs.logits)
    pipe.connect(second.outputs.mask, echo.inputs.mask)

    outcome = calibrate_pipeline_deciders(pipe, _ListDataModule(_MIXED), split="val")

    assert not outcome.calibrated
    assert outcome.reason is not None
    assert outcome.reason.startswith("expected one ground-truth 'mask' output, found ")
    assert "first" in outcome.reason and "second" in outcome.reason
    assert dec.threshold == 0.5


def test_no_mask_output_skips():
    dec = _MockCalibratableDecider(name="dec")
    pipe = _pipeline(dec, source=_ScoresOnlySource(name="src"))
    batches = [_batch(1.0, True, with_mask=False), _batch(0.0, False, with_mask=False)]

    outcome = calibrate_pipeline_deciders(pipe, _ListDataModule(batches), split="val")

    assert outcome.reason == "expected one ground-truth 'mask' output, found []"
    assert dec.calibrated_with is None


def test_non_tensor_scores_skip_the_decider():
    dec = _TolerantDecider(name="dec")
    pipe = _pipeline(dec, source=_ListScoresSource(name="src"), strict=False)

    outcome = calibrate_pipeline_deciders(pipe, _ListDataModule(_MIXED), split="val")

    assert outcome.skipped == {"dec": "src.scores produced no tensor over val"}
    assert not outcome.calibrated
    assert dec.threshold == 0.5


def test_failing_inference_is_reported_not_raised():
    dec = _MockCalibratableDecider(name="dec")
    pipe = _pipeline(dec, source=_FailingSource(name="src"))

    outcome = calibrate_pipeline_deciders(pipe, _ListDataModule(_MIXED), split="val")

    assert outcome.reason is not None
    assert outcome.reason.startswith("inference over val failed: ")
    assert "boom at inference" in outcome.reason
    assert not outcome.calibrated
    assert dec.threshold == 0.5


def test_pipeline_device_follows_the_first_buffer():
    dec = _MockCalibratableDecider(name="dec")
    pipe = _pipeline(dec, source=_BufferedSource(name="src"))

    assert pipe.device == torch.device("cpu")
    outcome = calibrate_pipeline_deciders(pipe, _ListDataModule(_MIXED), split="val")
    assert outcome.calibrated


def test_summary_formats_two_stage_style_reports():
    outcome = CalibrationOutcome(
        split="val",
        applicable=True,
        reports={
            "gate": {
                "class": "TwoStageBinaryDecider",
                "image_threshold": {"old": None, "new": 2.5},
                "pixel_threshold": {"old": 0.9, "new": 2.4999998},
                "joint": {"image_threshold": 2.0, "pixel_threshold": 2.0, "f1": 0.9},
                "f1": 0.93,
            },
            "topk": {"class": "SomethingDiscrete", "top_k": {"old": 1, "new": 3}},
            "plain": {"class": "SomethingWithoutKnobs"},
        },
    )
    assert outcome.summary() == (
        "thresholds calibrated on val: "
        "gate (image_threshold 2.5, pixel_threshold 2.5), topk (top_k 3), plain"
    )


def test_calibrated_hparams_reach_serialize():
    """The gRPC save re-derives config from the live pipeline; serialize() must carry it."""
    dec = _MockCalibratableDecider(name="dec")
    pipe = _pipeline(dec)
    calibrate_pipeline_deciders(pipe, _ListDataModule(_MIXED), split="val")

    config = pipe.serialize()
    dec_cfg = next(node for node in config.nodes if node.name == "dec")
    assert dec_cfg.hparams["threshold"] == 0.42


def test_cli_log_line_matches_the_outcome(monkeypatch):
    """restore_trainrun reports a calibrated outcome at info, a skipped one at warning,
    and says nothing when the pipeline has no calibratable decider."""
    calls: list[tuple[str, str]] = []
    fake_logger = type(
        "L",
        (),
        {
            "info": staticmethod(lambda msg: calls.append(("info", msg))),
            "warning": staticmethod(lambda msg: calls.append(("warning", msg))),
        },
    )
    monkeypatch.setattr("cuvis_ai_core.utils.restore.logger", fake_logger)

    _log_calibration_outcome(CalibrationOutcome(split="val", applicable=False))
    _log_calibration_outcome(
        CalibrationOutcome(
            split="val",
            applicable=True,
            reports={"dec": {"t": {"old": 1.0, "new": 2.0}}},
        )
    )
    _log_calibration_outcome(
        CalibrationOutcome(split="val", applicable=True, reason="no val split")
    )

    assert calls == [
        ("info", "  Calibration: thresholds calibrated on val: dec (t 2)"),
        ("warning", "  Calibration: thresholds not calibrated: no val split"),
    ]


def test_outcome_carries_structured_records():
    """The phase reports data: counts, and one record per decider with the knobs it moved."""
    dec = _MockCalibratableDecider(name="dec")
    outcome = calibrate_pipeline_deciders(
        _pipeline(dec), _ListDataModule(_MIXED), split="val"
    )

    assert outcome.phase_skip is None and outcome.phase_detail == ""
    assert (outcome.frames, outcome.anomalous_frames) == (3, 2)
    (record,) = outcome.deciders
    assert record.name == "dec"
    assert record.class_name == "_MockCalibratableDecider"
    assert record.status is DeciderCalibrationStatus.CALIBRATED
    assert record.skip is None and record.skip_detail == ""
    assert record.thresholds == {"threshold": ThresholdChange(before=0.5, after=0.42)}
    assert record.metrics == {}
    assert record.thresholds_present is True
    assert outcome.reports == {"dec": dec.calibrated_report}


def test_phase_skips_carry_their_enum_and_counts():
    dec = _MockCalibratableDecider(name="dec")
    normals = [_batch(0.0, False), _batch(0.0, False)]
    single = calibrate_pipeline_deciders(
        _pipeline(dec), _ListDataModule(normals), split="val"
    )
    assert single.phase_skip is CalibrationPhaseSkip.SINGLE_CLASS
    assert (single.frames, single.anomalous_frames) == (2, 0)
    (record,) = single.deciders
    assert record.status is DeciderCalibrationStatus.SKIPPED
    assert record.skip is None  # the phase reason applies, not a per-decider one
    assert record.thresholds_present is True
    assert single.skipped == {}

    dm = _ListDataModule(_MIXED)
    dm.val_ds = None
    missing = calibrate_pipeline_deciders(_pipeline(dec), dm, split="val")
    assert missing.phase_skip is CalibrationPhaseSkip.NO_SPLIT
    assert (missing.frames, missing.anomalous_frames) == (0, 0)

    failing = calibrate_pipeline_deciders(
        _pipeline(dec, source=_FailingSource(name="src")),
        _ListDataModule(_MIXED),
        split="val",
    )
    assert failing.phase_skip is CalibrationPhaseSkip.INFERENCE_FAILED
    assert failing.phase_detail == "RuntimeError: boom at inference"
    assert (
        failing.reason == "inference over val failed: RuntimeError: boom at inference"
    )

    first = _ScoreMaskSource(name="first")
    second = _ScoreMaskSource(name="second")
    echo = _MaskEcho(name="echo")
    pipe = CuvisPipeline("cal_test")
    pipe.connect(first.outputs.scores, dec.inputs.logits)
    pipe.connect(second.outputs.mask, echo.inputs.mask)
    ambiguous = calibrate_pipeline_deciders(pipe, _ListDataModule(_MIXED), split="val")
    assert ambiguous.phase_skip is CalibrationPhaseSkip.AMBIGUOUS_MASK
    assert "first" in ambiguous.phase_detail and "second" in ambiguous.phase_detail


def test_decider_skips_carry_their_enum():
    raising = calibrate_pipeline_deciders(
        _pipeline(_RaisingDecider(name="bad")), _ListDataModule(_MIXED), split="val"
    )
    (record,) = raising.deciders
    assert record.status is DeciderCalibrationStatus.SKIPPED
    assert record.skip is DeciderSkip.CALIBRATE_FAILED
    assert record.skip_detail == "ValueError: scores must be finite"
    assert record.thresholds == {} and record.thresholds_present is True

    declining = calibrate_pipeline_deciders(
        _pipeline(_DecliningDecider(name="dec")), _ListDataModule(_MIXED), split="val"
    )
    assert declining.deciders[0].skip is DeciderSkip.NO_REPORT

    wired = _MockCalibratableDecider(name="wired")
    loose = _MockCalibratableDecider(name="loose")
    pipe = _pipeline(wired)
    pipe._assign_counter_and_add_node(loose)
    batches = [dict(batch, logits=batch["scores"]) for batch in _MIXED]
    both = calibrate_pipeline_deciders(pipe, _ListDataModule(batches), split="val")
    by_name = {record.name: record for record in both.deciders}
    assert by_name["loose"].skip is DeciderSkip.NOTHING_WIRED
    assert by_name["wired"].status is DeciderCalibrationStatus.CALIBRATED

    listy = calibrate_pipeline_deciders(
        _pipeline(
            _TolerantDecider(name="dec"),
            source=_ListScoresSource(name="src"),
            strict=False,
        ),
        _ListDataModule(_MIXED),
        split="val",
    )
    (record,) = listy.deciders
    assert record.skip is DeciderSkip.NO_SCORES
    assert record.skip_detail == "src.scores produced no tensor over val"


def test_records_extract_knobs_and_metrics_from_a_two_stage_report():
    class _TwoStageLike(_MockCalibratableDecider):
        def calibrate(self, scores, targets):
            self.calibrated_with = (tuple(scores.shape), tuple(targets.shape))
            return {
                "class": "TwoStageBinaryDecider",
                "image_threshold": {"old": None, "new": 2.5},
                "pixel_threshold": {"old": 0.9, "new": 2.4999998},
                "on_point": {"image_threshold": 2.0, "pixel_threshold": 2.1},
                "image_f1": 0.93,
                "pixel_iou": 0.4,
                "joint": {"f1": 0.9},
            }

    outcome = calibrate_pipeline_deciders(
        _pipeline(_TwoStageLike(name="gate")), _ListDataModule(_MIXED), split="val"
    )
    (record,) = outcome.deciders
    assert record.class_name == "TwoStageBinaryDecider"
    assert record.thresholds == {
        "image_threshold": ThresholdChange(before=None, after=2.5),
        "pixel_threshold": ThresholdChange(before=0.9, after=2.4999998),
    }
    assert record.metrics == {"image_f1": 0.93, "pixel_iou": 0.4}
    assert outcome.summary() == (
        "thresholds calibrated on val: gate (image_threshold 2.5, pixel_threshold 2.5)"
    )


def test_on_batch_reports_frames_per_batch():
    seen: list[tuple[int, int]] = []
    calibrate_pipeline_deciders(
        _pipeline(_MockCalibratableDecider(name="dec")),
        _ListDataModule(_MIXED),
        split="val",
        on_batch=lambda done, total: seen.append((done, total)),
    )
    assert seen == [(1, 0), (2, 0), (3, 0)]


def test_uncalibrated_required_consumers_names_the_graph_that_would_fail():
    """A node that needs decisions at a training stage from a decider without thresholds."""

    class _UncalibratedDecider(_MockCalibratableDecider):
        @property
        def is_calibrated(self) -> bool:
            return False

    class _Consumer(Node):
        INPUT_SPECS = {"decisions": _DECISIONS}
        OUTPUT_SPECS = {"count": PortSpec(dtype=torch.int64, shape=())}
        EXECUTION_STAGES = {ExecutionStage.VAL, ExecutionStage.TEST}

        def forward(self, decisions, **_):
            return {"count": decisions.sum()}

    class _OptionalConsumer(_Consumer):
        INPUT_SPECS = {
            "decisions": PortSpec(
                dtype=torch.bool, shape=(-1, -1, -1, 1), optional=True
            )
        }

        def forward(self, decisions=None, **_):
            return {"count": torch.tensor(0)}

    class _InferenceConsumer(_Consumer):
        EXECUTION_STAGES = {ExecutionStage.INFERENCE}

    def graph(decider, consumer):
        pipe = _pipeline(decider)
        pipe.connect(decider.outputs.decisions, consumer.inputs.decisions)
        return pipe

    dec = _UncalibratedDecider(name="dec")
    sink = _Consumer(name="sink")
    assert uncalibrated_required_consumers(graph(dec, sink)) == [(sink, dec)]
    assert (
        uncalibrated_required_consumers(graph(dec, _OptionalConsumer(name="sink")))
        == []
    )
    assert (
        uncalibrated_required_consumers(graph(dec, _InferenceConsumer(name="sink")))
        == []
    )
    calibrated = _MockCalibratableDecider(name="dec")
    assert (
        uncalibrated_required_consumers(graph(calibrated, _Consumer(name="sink"))) == []
    )


# --- the trainrun's calibration block and an out-of-memory pass ---------------------------


class _ObjectiveDecider(_MockCalibratableDecider):
    """Decider whose ``calibrate`` takes the trainrun's block and records it."""

    def calibrate(self, scores, targets, objective=None):
        """Record the block, then calibrate as the mock does."""
        self.objective_seen = objective
        return super().calibrate(scores, targets)


class _KeywordsDecider(_MockCalibratableDecider):
    """Decider whose ``calibrate`` takes keyword arguments it does not name."""

    def calibrate(self, scores, targets, **kwargs):
        """Record the keywords, then calibrate as the mock does."""
        self.keywords_seen = kwargs
        return super().calibrate(scores, targets)


class _OutOfMemorySource(_ScoreMaskSource):
    """Source whose pass runs the graphics card out of memory."""

    def forward(self, scores, mask, **_):
        """Fail the way torch does when an allocation does not fit."""
        raise torch.cuda.OutOfMemoryError(
            "CUDA out of memory. Tried to allocate 2.00 GiB"
        )


class _OutOfMemoryDecider(_MockCalibratableDecider):
    """Decider whose ``calibrate`` runs the graphics card out of memory."""

    def calibrate(self, scores, targets):
        """Fail the way torch does when an allocation does not fit."""
        raise torch.cuda.OutOfMemoryError(
            "CUDA out of memory. Tried to allocate 1.00 GiB"
        )


_RECALL = CalibrationConfig(objective="recall", target=0.9, level="frames")


def test_a_block_reaches_the_decider_that_takes_it():
    dec = _ObjectiveDecider(name="dec")
    outcome = calibrate_pipeline_deciders(
        _pipeline(dec), _ListDataModule(_MIXED), split="val", objective=_RECALL
    )
    assert outcome.calibrated
    assert dec.objective_seen is _RECALL


def test_a_decider_taking_keywords_gets_the_block():
    dec = _KeywordsDecider(name="dec")
    calibrate_pipeline_deciders(
        _pipeline(dec), _ListDataModule(_MIXED), split="val", objective=_RECALL
    )
    assert dec.keywords_seen == {"objective": _RECALL}


def test_without_a_block_the_call_keeps_two_arguments():
    plain, taking = _MockCalibratableDecider(name="dec"), _ObjectiveDecider(name="dec")
    assert calibrate_pipeline_deciders(
        _pipeline(plain), _ListDataModule(_MIXED)
    ).calibrated
    assert calibrate_pipeline_deciders(
        _pipeline(taking), _ListDataModule(_MIXED)
    ).calibrated
    assert taking.objective_seen is None


def test_a_block_fails_a_decider_that_takes_none_and_keeps_its_threshold():
    dec = _MockCalibratableDecider(name="dec")
    outcome = calibrate_pipeline_deciders(
        _pipeline(dec), _ListDataModule(_MIXED), split="val", objective=_RECALL
    )
    (record,) = outcome.deciders
    assert record.skip is DeciderSkip.CALIBRATE_FAILED
    assert (
        record.skip_detail
        == "_MockCalibratableDecider.calibrate takes no calibration objective"
    )
    assert dec.calibrated_with is None and dec.threshold == 0.5


def test_an_out_of_memory_pass_is_recorded_unless_asked_to_raise():
    def run(**kwargs):
        dec = _MockCalibratableDecider(name="dec")
        return calibrate_pipeline_deciders(
            _pipeline(dec, source=_OutOfMemorySource(name="src")),
            _ListDataModule(_MIXED),
            split="val",
            **kwargs,
        )

    outcome = run()
    assert outcome.phase_skip is CalibrationPhaseSkip.INFERENCE_FAILED
    assert outcome.phase_detail.startswith("OutOfMemoryError: CUDA out of memory")
    with pytest.raises(torch.cuda.OutOfMemoryError):
        run(raise_out_of_memory=True)


def test_an_out_of_memory_calibrate_raises_when_asked_and_other_errors_stay_recorded():
    with pytest.raises(torch.cuda.OutOfMemoryError):
        calibrate_pipeline_deciders(
            _pipeline(_OutOfMemoryDecider(name="dec")),
            _ListDataModule(_MIXED),
            raise_out_of_memory=True,
        )
    outcome = calibrate_pipeline_deciders(
        _pipeline(_RaisingDecider(name="dec")),
        _ListDataModule(_MIXED),
        raise_out_of_memory=True,
    )
    (record,) = outcome.deciders
    assert record.skip is DeciderSkip.CALIBRATE_FAILED


class _PositionalObjectiveDecider(_MockCalibratableDecider):
    """Decider whose ``calibrate`` names ``objective`` but takes it by position only."""

    def calibrate(self, scores, targets, objective=None, /):
        """Calibrate as the mock does; a keyword ``objective`` cannot reach it."""
        return super().calibrate(scores, targets)


def test_an_f1_block_lets_a_decider_without_the_parameter_calibrate_by_its_own_rule():
    plain, taking = _MockCalibratableDecider(name="dec"), _ObjectiveDecider(name="dec")
    f1 = CalibrationConfig()
    assert calibrate_pipeline_deciders(
        _pipeline(plain), _ListDataModule(_MIXED), split="val", objective=f1
    ).calibrated
    assert plain.calibrated_with is not None
    assert calibrate_pipeline_deciders(
        _pipeline(taking), _ListDataModule(_MIXED), split="val", objective=f1
    ).calibrated
    assert taking.objective_seen is f1


def test_a_positional_only_objective_is_no_objective_parameter():
    dec = _PositionalObjectiveDecider(name="dec")
    outcome = calibrate_pipeline_deciders(
        _pipeline(dec), _ListDataModule(_MIXED), split="val", objective=_RECALL
    )
    (record,) = outcome.deciders
    assert record.skip is DeciderSkip.CALIBRATE_FAILED
    assert (
        record.skip_detail
        == "_PositionalObjectiveDecider.calibrate takes no calibration objective"
    )
