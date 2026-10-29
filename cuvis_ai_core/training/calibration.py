"""Post-training threshold calibration for anomaly deciders.

After training a decider's shipped threshold hparams are stale: training moved the score
distribution, so a value tuned for one set of weights misfits the next. This phase runs
the trained pipeline over the labelled validation split, collects the scores feeding each
decider and the ground-truth ``mask``, and calls ``decider.calibrate(scores, targets)`` so
the saved checkpoint ships thresholds matched to its own weights. A trainrun whose
``calibration:`` block names a precision or recall target hands it on as
``decider.calibrate(scores, targets, objective=block)``.

The heavy F1-max sweep math lives on the decider subclasses (cuvis-ai); this module only
orchestrates: find the deciders, collect their input scores plus the mask over the split,
dispatch, and report. The phase is best-effort by design:

- Deciders that do not override ``calibrate``, pipelines without one, a missing split, an
  ambiguous ground-truth mask, and a single-class split are skipped, never raised.
- A ``calibrate`` that raises (cuvis-ai's ``CalibrationError`` for a shape mismatch,
  non-finite scores, single-class targets or an unsupported ``reduce_dims``, or any other
  error) is logged with its traceback and the decider keeps its shipped thresholds. The
  trained weights are never lost over a calibration guard. A caller with nothing unsaved
  (the standalone calibration phase) may ask for a GPU out-of-memory error instead, so it
  can run the pass again on the CPU.
- The result is a :class:`CalibrationOutcome`, so the trainrun can tell the user
  ``thresholds calibrated on val: ...`` or ``thresholds not calibrated: <reason>`` (the
  gRPC ``Train`` completion message, the CLI log) instead of leaving it to the log alone.

The ground-truth ``mask`` is taken from the node that originates it: a node whose ``mask``
output has no incoming ``mask`` edge. Nodes that consume and re-emit ``mask`` (augmentation
wrappers echoing it at inference) are ignored, so they do not make the phase skip.

Two caveats for consumers: the pass is a full inference forward over the split, so sink
nodes in the pipeline (video or JSON writers) fire once more; and metrics reported on the
same validation split after calibration are optimistic, because the thresholds were chosen
on it.

The outcome is data first (:class:`CalibrationOutcome` with one :class:`DeciderCalibration`
per decider, the skip reasons as enums, the frame counts): the Train stream sends it as a
typed event and the client words it; ``summary()`` is the one-line text for the logs.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import torch
from loguru import logger

from cuvis_ai_core.deciders.base_decider import BinaryDecider
from cuvis_ai_core.training.evaluation import (
    OnBatch,
    TrainingCancelled,
    forward_split,
    is_cuda_out_of_memory,
)
from cuvis_ai_schemas.enums import ExecutionStage

if TYPE_CHECKING:
    import pytorch_lightning as pl
    from cuvis_ai_schemas.training import CalibrationConfig

    from cuvis_ai_core.node.node import Node
    from cuvis_ai_core.pipeline.pipeline import CuvisPipeline


class CalibrationPhaseSkip(Enum):
    """Why the phase did nothing for any decider. The names mirror the proto enum."""

    NO_SPLIT = "no_split"
    INFERENCE_FAILED = "inference_failed"
    AMBIGUOUS_MASK = "ambiguous_mask"
    SINGLE_CLASS = "single_class"


class DeciderSkip(Enum):
    """Why one decider kept its thresholds. The names mirror the proto enum."""

    NOTHING_WIRED = "nothing_wired"
    NO_SCORES = "no_scores"
    CALIBRATE_FAILED = "calibrate_failed"
    NO_REPORT = "no_report"


class DeciderCalibrationStatus(Enum):
    """Whether a decider's thresholds were updated on this run."""

    CALIBRATED = "calibrated"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class ThresholdChange:
    """One threshold knob before and after the step; ``before`` is None for an unset one."""

    before: float | None
    after: float


@dataclass(frozen=True)
class DeciderCalibration:
    """What the phase did to one decider, as data.

    Attributes:
        name: The node name.
        class_name: The decider class, from the report when it names one.
        status: Updated on this run, or skipped.
        skip: Why a skipped decider kept its thresholds; ``None`` when the phase-level
            skip applies to it (see :attr:`CalibrationOutcome.phase_skip`).
        skip_detail: The error class and text behind ``CALIBRATE_FAILED``, the source
            behind ``NO_SCORES``.
        thresholds: The knobs the report moved, by name (``image_threshold``,
            ``pixel_threshold``, ``threshold``, ``quantile``).
        metrics: The report's top-level numbers (``f1``, ``precision``, ...), the metrics at
            the chosen thresholds; nested entries stay in ``report``.
        thresholds_present: Whether the decider holds thresholds after the step, also
            when it kept earlier ones (``BinaryDecider.is_calibrated``).
        report: The decider's full report, for the log.
    """

    name: str
    class_name: str
    status: DeciderCalibrationStatus
    skip: DeciderSkip | None = None
    skip_detail: str = ""
    thresholds: dict[str, ThresholdChange] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    thresholds_present: bool = True
    report: dict[str, Any] | None = None

    @classmethod
    def from_report(
        cls, name: str, report: dict[str, Any], *, thresholds_present: bool = True
    ) -> DeciderCalibration:
        """Build the record of a calibrated decider from the report ``calibrate`` returned."""
        thresholds: dict[str, ThresholdChange] = {}
        metrics: dict[str, float] = {}
        for key, value in report.items():
            if isinstance(value, dict) and "new" in value:
                before = value.get("old")
                thresholds[key] = ThresholdChange(
                    before=None if before is None else float(before),
                    after=float(value["new"]),
                )
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics[key] = float(value)
        return cls(
            name=name,
            class_name=str(report.get("class", "")),
            status=DeciderCalibrationStatus.CALIBRATED,
            thresholds=thresholds,
            metrics=metrics,
            thresholds_present=thresholds_present,
            report=report,
        )

    @classmethod
    def skipped_decider(
        cls, decider: BinaryDecider, skip: DeciderSkip | None, detail: str = ""
    ) -> DeciderCalibration:
        """Build the record of a decider that kept its thresholds."""
        return cls(
            name=decider.name,
            class_name=type(decider).__name__,
            status=DeciderCalibrationStatus.SKIPPED,
            skip=skip,
            skip_detail=detail,
            thresholds_present=decider.is_calibrated,
        )

    @property
    def skip_text(self) -> str:
        """The skip as the one-line reason the summary and the logs use."""
        if self.skip is DeciderSkip.NOTHING_WIRED:
            return "nothing is wired into inputs.logits"
        if self.skip is DeciderSkip.NO_REPORT:
            return "calibrate returned no report"
        return self.skip_detail


@dataclass(frozen=True)
class CalibrationOutcome:
    """What the calibration phase did, for the Train stream, the logs and the trainrun result.

    The structured fields are the truth: ``deciders`` holds one record per calibratable
    decider, ``phase_skip`` says why nothing ran at all. ``reports``, ``skipped`` and
    ``reason`` are the same facts as text, kept for the summary line and for callers that
    read them; when an outcome is built from records they are derived, when it is built
    from the text fields (older callers, tests) the records are derived from ``reports``.

    Attributes:
        split: Name of the split the phase ran on.
        applicable: Whether the pipeline holds at least one decider that implements
            ``calibrate``. ``False`` means the phase had nothing to do; callers should
            not mention calibration to the user in that case.
        reports: Calibration report per decider (decider name -> report) for the
            deciders whose thresholds were updated.
        skipped: Skip reason per calibratable decider (decider name -> reason) that was
            not updated: nothing wired into ``logits``, no scores produced, a
            ``calibrate`` that raised, or one that returned no report.
        reason: Phase-level skip reason when no decider could be handled at all (missing
            split, ambiguous ground-truth mask, single-class split); ``None`` otherwise.
        deciders: One :class:`DeciderCalibration` per calibratable decider.
        phase_skip: Why the phase ran for no decider; ``None`` when it ran.
        phase_detail: The error class and text, or the mask candidates, behind the skip.
        frames: Frames seen over the split (0 when the pass did not run).
        anomalous_frames: Of which anomalous, by the ground-truth mask.
    """

    split: str
    applicable: bool
    reports: dict[str, dict[str, Any]] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)
    reason: str | None = None
    deciders: tuple[DeciderCalibration, ...] = ()
    phase_skip: CalibrationPhaseSkip | None = None
    phase_detail: str = ""
    frames: int = 0
    anomalous_frames: int = 0

    def __post_init__(self) -> None:
        """Keep the text fields and the records in step, whichever side was given."""
        if self.deciders and not self.reports and not self.skipped:
            object.__setattr__(
                self,
                "reports",
                {
                    record.name: record.report
                    for record in self.deciders
                    if record.status is DeciderCalibrationStatus.CALIBRATED
                    and record.report is not None
                },
            )
            object.__setattr__(
                self,
                "skipped",
                {
                    record.name: record.skip_text
                    for record in self.deciders
                    if record.status is DeciderCalibrationStatus.SKIPPED
                    and record.skip is not None
                },
            )
        elif not self.deciders and self.reports:
            object.__setattr__(
                self,
                "deciders",
                tuple(
                    DeciderCalibration.from_report(name, report)
                    for name, report in self.reports.items()
                ),
            )
        if self.reason is None and self.phase_skip is not None:
            object.__setattr__(self, "reason", self._phase_reason())

    def _phase_reason(self) -> str:
        """The phase skip as the one-line reason the summary and the logs use."""
        if self.phase_skip is CalibrationPhaseSkip.NO_SPLIT:
            return f"no {self.split} split"
        if self.phase_skip is CalibrationPhaseSkip.INFERENCE_FAILED:
            return f"inference over {self.split} failed: {self.phase_detail}"
        if self.phase_skip is CalibrationPhaseSkip.AMBIGUOUS_MASK:
            return f"expected one ground-truth 'mask' output, found {self.phase_detail}"
        return (
            f"{self.split} split is single-class "
            f"({self.anomalous_frames}/{self.frames} frames anomalous)"
        )

    @property
    def calibrated(self) -> bool:
        """Whether at least one decider's thresholds were updated."""
        return bool(self.reports)

    def summary(self) -> str:
        """One line for the user: what was calibrated, or why nothing was."""
        skipped = "; ".join(f"{name}: {why}" for name, why in self.skipped.items())
        if self.calibrated:
            described = ", ".join(
                _describe_report(name, report) for name, report in self.reports.items()
            )
            text = f"thresholds calibrated on {self.split}: {described}"
            return f"{text}; skipped {skipped}" if skipped else text
        if self.reason is not None:
            why = self.reason
        elif skipped:
            why = skipped
        else:
            why = "no calibratable decider"
        return f"thresholds not calibrated: {why}"


def _describe_report(name: str, report: dict[str, Any]) -> str:
    """``dec (image_threshold 2.5, pixel_threshold 2.5)`` from a decider's report.

    Deciders report each updated knob as ``{"old": ..., "new": ...}``; other entries
    (class name, metrics, alternatives) are left out of the one-liner.
    """
    changes = [
        f"{key} {_format_value(value['new'])}"
        for key, value in report.items()
        if isinstance(value, dict) and "new" in value
    ]
    return f"{name} ({', '.join(changes)})" if changes else name


def _format_value(value: Any) -> str:
    """Compact rendering of a threshold value (floats to four significant digits)."""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def calibratable_deciders(pipeline: CuvisPipeline) -> list[BinaryDecider]:
    """Decider nodes that override the base ``calibrate`` (i.e. actually support it)."""
    return [
        node
        for node in pipeline.nodes()
        if isinstance(node, BinaryDecider)
        and type(node).calibrate is not BinaryDecider.calibrate
    ]


def calibration_pending(pipeline: CuvisPipeline) -> bool:
    """Whether a calibratable decider of ``pipeline`` still has no thresholds."""
    return any(not decider.is_calibrated for decider in calibratable_deciders(pipeline))


_TRAINING_STAGES = frozenset(
    {
        ExecutionStage.TRAIN,
        ExecutionStage.VAL,
        ExecutionStage.TEST,
        ExecutionStage.ALWAYS,
    }
)


def uncalibrated_required_consumers(
    pipeline: CuvisPipeline,
) -> list[tuple[Node, BinaryDecider]]:
    """Nodes that need ``decisions`` at a training stage from a decider without thresholds.

    An uncalibrated calibratable decider emits no decisions outside inference, so a node
    wired to its ``decisions`` output through a required input, and running at the train,
    validation or test stage, would fail its first forward. Callers refuse such a graph
    before the fit with the node and decider names. An optional input or an
    inference-only consumer is fine.
    """
    found: list[tuple[Node, BinaryDecider]] = []
    for decider in calibratable_deciders(pipeline):
        if decider.is_calibrated:
            continue
        for src, dst, edge in pipeline._graph.edges(data=True):
            if src is not decider or edge.get("from_port") != "decisions":
                continue
            spec = dst.INPUT_SPECS.get(edge.get("to_port"))
            if spec is None or spec.optional:
                continue
            if dst.execution_stages & _TRAINING_STAGES:
                found.append((dst, decider))
    return found


def _logits_source(
    pipeline: CuvisPipeline, decider: BinaryDecider
) -> tuple[str, str] | None:
    """The ``(node, port)`` wired into ``decider.inputs.logits`` - the space it thresholds."""
    for src, dst, edge in pipeline._graph.edges(data=True):
        if dst is decider and edge.get("to_port") == "logits":
            return src.name, edge.get("from_port")
    return None


def _mask_originators(
    pipeline: CuvisPipeline, mask_keys: list[tuple[str, str]]
) -> list[tuple[str, str]]:
    """The ``mask`` outputs whose node does not itself consume a ``mask`` over an edge.

    A node fed ``mask`` by another node only re-emits it (an augmentation wrapper at
    inference), so the ground truth is the output of the node that originates the mask.
    Batch keys reach nodes without graph edges, so a data node that reads ``mask`` from
    the batch still counts as an originator.
    """
    re_emitters = {
        dst.name
        for _src, dst, edge in pipeline._graph.edges(data=True)
        if edge.get("to_port") == "mask"
    }
    return [key for key in mask_keys if key[0] not in re_emitters]


def _takes_objective(decider: BinaryDecider) -> bool:
    """Whether ``decider.calibrate`` accepts the trainrun's block as ``objective=``.

    A parameter of that name counts unless it is positional-only; ``**kwargs`` counts too.
    """
    return any(
        (
            parameter.name == "objective"
            and parameter.kind is not inspect.Parameter.POSITIONAL_ONLY
        )
        or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in inspect.signature(decider.calibrate).parameters.values()
    )


def calibrate_pipeline_deciders(
    pipeline: CuvisPipeline,
    datamodule: pl.LightningDataModule,
    *,
    split: str = "val",
    on_batch: OnBatch | None = None,
    objective: CalibrationConfig | None = None,
    raise_out_of_memory: bool = False,
) -> CalibrationOutcome:
    """Re-fit calibratable deciders' thresholds on the labelled ``split`` of ``datamodule``.

    Runs the trained ``pipeline`` over the split at the inference stage (so metric / loss
    nodes stay quiet), collects each decider's input scores and the originating
    ground-truth ``mask`` output, and calls ``decider.calibrate(scores, targets)``. The
    deciders are mutated in place, so a subsequent ``pipeline.save_to_file`` (or the gRPC
    save, once the session's cached config is dropped) persists the calibrated thresholds
    into the yaml. Never raises for the documented skip cases or a failing ``calibrate``;
    the returned :class:`CalibrationOutcome` says what happened. ``on_batch`` receives the
    frames seen so far and the split's frame total after each batch of the pass. A
    cooperative stop (:class:`~cuvis_ai_core.training.evaluation.TrainingCancelled`
    from the pass) propagates.

    ``objective`` is the trainrun's calibration block. With one, each decider gets it as
    ``calibrate(scores, targets, objective=objective)``. A decider whose ``calibrate`` takes
    no such parameter keeps the two-argument call under an f1 block, which asks for its own
    rule, and is recorded as ``CALIBRATE_FAILED`` under a precision or recall block, keeping
    its thresholds. Without a block the call keeps its two arguments.
    ``raise_out_of_memory`` lets a GPU out-of-memory error from the pass or from a
    ``calibrate`` propagate instead of being recorded; every other error is still recorded.
    """
    deciders = calibratable_deciders(pipeline)
    if not deciders:
        logger.info("Threshold calibration: no calibratable decider; skipping.")
        return CalibrationOutcome(split=split, applicable=False)

    records: dict[str, DeciderCalibration] = {}
    sources: dict[BinaryDecider, tuple[str, str]] = {}
    for decider in deciders:
        source = _logits_source(pipeline, decider)
        if source is None:
            records[decider.name] = DeciderCalibration.skipped_decider(
                decider, DeciderSkip.NOTHING_WIRED
            )
            logger.warning(
                f"Threshold calibration: nothing feeds {decider.name}.inputs.logits; "
                "skipping this decider."
            )
            continue
        sources[decider] = source
    if not sources:
        return CalibrationOutcome(
            split=split, applicable=True, deciders=tuple(records.values())
        )

    def skip_all(
        kind: CalibrationPhaseSkip,
        detail: str = "",
        frames: int = 0,
        anomalous: int = 0,
    ) -> CalibrationOutcome:
        phase_records = dict(records)
        for decider in sources:
            phase_records[decider.name] = DeciderCalibration.skipped_decider(
                decider, None
            )
        outcome = CalibrationOutcome(
            split=split,
            applicable=True,
            deciders=tuple(phase_records.values()),
            phase_skip=kind,
            phase_detail=detail,
            frames=frames,
            anomalous_frames=anomalous,
        )
        logger.warning(
            f"Threshold calibration: {outcome.reason}; leaving shipped thresholds unchanged."
        )
        return outcome

    if getattr(datamodule, f"{split}_ds", None) is None:
        return skip_all(CalibrationPhaseSkip.NO_SPLIT)
    loader = getattr(datamodule, f"{split}_dataloader")()

    wanted = {port for (_, port) in sources.values()} | {"mask"}
    collected: dict[tuple[str, str], list[torch.Tensor]] = {}
    try:
        for item in forward_split(
            pipeline, loader, stage=ExecutionStage.INFERENCE, on_batch=on_batch
        ):
            for (node_name, port), value in item.outputs.items():
                if port in wanted and isinstance(value, torch.Tensor):
                    collected.setdefault((node_name, port), []).append(
                        value.detach().cpu()
                    )
    except TrainingCancelled:
        raise
    except Exception as exc:  # noqa: BLE001 - best effort: never lose the weights
        if raise_out_of_memory and is_cuda_out_of_memory(exc):
            raise
        logger.opt(exception=True).warning(
            f"Threshold calibration: inference over {split} failed."
        )
        return skip_all(
            CalibrationPhaseSkip.INFERENCE_FAILED, f"{type(exc).__name__}: {exc}"
        )

    mask_keys = [key for key in collected if key[1] == "mask"]
    candidates = _mask_originators(pipeline, mask_keys) or mask_keys
    if len(candidates) != 1:
        found = [key[0] for key in candidates]
        return skip_all(CalibrationPhaseSkip.AMBIGUOUS_MASK, str(found))
    targets = torch.cat(collected[candidates[0]], dim=0)
    frame_labels = targets.flatten(1).any(dim=1)
    n_anom, n_total = int(frame_labels.sum()), int(frame_labels.numel())
    if n_anom == 0 or n_anom == n_total:
        return skip_all(
            CalibrationPhaseSkip.SINGLE_CLASS, frames=n_total, anomalous=n_anom
        )

    for decider, key in sources.items():
        if key not in collected:
            records[decider.name] = DeciderCalibration.skipped_decider(
                decider,
                DeciderSkip.NO_SCORES,
                f"{key[0]}.{key[1]} produced no tensor over {split}",
            )
            logger.warning(
                f"Threshold calibration: {key[0]}.{key[1]} produced nothing over "
                f"{split}; skipping {decider.name}."
            )
            continue
        hands_objective = objective is not None and _takes_objective(decider)
        if (
            objective is not None
            and objective.objective != "f1"
            and not hands_objective
        ):
            records[decider.name] = DeciderCalibration.skipped_decider(
                decider,
                DeciderSkip.CALIBRATE_FAILED,
                f"{type(decider).__name__}.calibrate takes no calibration objective",
            )
            logger.warning(
                f"Threshold calibration: {decider.name}.calibrate takes no calibration "
                "objective; keeping its shipped thresholds."
            )
            continue
        scores = torch.cat(collected[key], dim=0)
        try:
            report = (
                decider.calibrate(scores, targets, objective=objective)
                if hands_objective
                else decider.calibrate(scores, targets)
            )
        except Exception as exc:  # noqa: BLE001 - best effort: never lose the weights
            if raise_out_of_memory and is_cuda_out_of_memory(exc):
                raise
            records[decider.name] = DeciderCalibration.skipped_decider(
                decider, DeciderSkip.CALIBRATE_FAILED, f"{type(exc).__name__}: {exc}"
            )
            logger.opt(exception=True).warning(
                f"Threshold calibration: {decider.name}.calibrate failed; keeping its "
                "shipped thresholds."
            )
            continue
        if report is None:
            records[decider.name] = DeciderCalibration.skipped_decider(
                decider, DeciderSkip.NO_REPORT
            )
            continue
        logger.info(f"Threshold calibration: {decider.name} -> {report}")
        records[decider.name] = DeciderCalibration.from_report(
            decider.name, report, thresholds_present=decider.is_calibrated
        )

    outcome = CalibrationOutcome(
        split=split,
        applicable=True,
        deciders=tuple(records.values()),
        frames=n_total,
        anomalous_frames=n_anom,
    )
    logger.info(f"Threshold calibration: {outcome.summary()}")
    return outcome
