"""Post-training threshold calibration for anomaly deciders.

After training a decider's shipped threshold hparams are stale: training moved the score
distribution, so a value tuned for one set of weights misfits the next. This phase runs
the trained pipeline over the labelled validation split, collects the scores feeding each
decider and the ground-truth ``mask``, and calls ``decider.calibrate(scores, targets)`` so
the saved checkpoint ships thresholds matched to its own weights.

The heavy F1-max sweep math lives on the decider subclasses (cuvis-ai); this module only
orchestrates: find the deciders, collect their input scores plus the mask over the split,
dispatch, and report. The phase is best-effort by design:

- Deciders that do not override ``calibrate``, pipelines without one, a missing split, an
  ambiguous ground-truth mask, and a single-class split are skipped, never raised.
- A ``calibrate`` that raises (cuvis-ai's ``CalibrationError`` for a shape mismatch,
  non-finite scores, single-class targets or an unsupported ``reduce_dims``, or any other
  error) is logged with its traceback and the decider keeps its shipped thresholds. The
  trained weights are never lost over a calibration guard.
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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from loguru import logger

from cuvis_ai_core.deciders.base_decider import BinaryDecider
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.execution import Context

if TYPE_CHECKING:
    import pytorch_lightning as pl

    from cuvis_ai_core.pipeline.pipeline import CuvisPipeline


@dataclass(frozen=True)
class CalibrationOutcome:
    """What the calibration phase did, for logs and the trainrun result.

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
    """

    split: str
    applicable: bool
    reports: dict[str, dict[str, Any]] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)
    reason: str | None = None

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


def _calibratable_deciders(pipeline: CuvisPipeline) -> list[BinaryDecider]:
    """Decider nodes that override the base ``calibrate`` (i.e. actually support it)."""
    return [
        node
        for node in pipeline.nodes()
        if isinstance(node, BinaryDecider)
        and type(node).calibrate is not BinaryDecider.calibrate
    ]


def _logits_source(
    pipeline: CuvisPipeline, decider: BinaryDecider
) -> tuple[str, str] | None:
    """The ``(node, port)`` wired into ``decider.inputs.logits`` - the space it thresholds."""
    for src, dst, edge in pipeline._graph.edges(data=True):
        if dst is decider and edge.get("to_port") == "logits":
            return src.name, edge.get("from_port")
    return None


def _pipeline_device(pipeline: CuvisPipeline) -> torch.device:
    """The device the pipeline's parameters/buffers live on (CPU when it has neither)."""
    for layer in pipeline.torch_layers:
        for tensor in (*layer.parameters(), *layer.buffers()):
            return tensor.device
    return torch.device("cpu")


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


def calibrate_pipeline_deciders(
    pipeline: CuvisPipeline,
    datamodule: pl.LightningDataModule,
    *,
    split: str = "val",
) -> CalibrationOutcome:
    """Re-fit calibratable deciders' thresholds on the labelled ``split`` of ``datamodule``.

    Runs the trained ``pipeline`` over the split at the inference stage (so metric / loss
    nodes stay quiet), collects each decider's input scores and the originating
    ground-truth ``mask`` output, and calls ``decider.calibrate(scores, targets)``. The
    deciders are mutated in place, so a subsequent ``pipeline.save_to_file`` (or the gRPC
    save, once the session's cached config is dropped) persists the calibrated thresholds
    into the yaml. Never raises for the documented skip cases or a failing ``calibrate``;
    the returned :class:`CalibrationOutcome` says what happened.
    """
    deciders = _calibratable_deciders(pipeline)
    if not deciders:
        logger.info("Threshold calibration: no calibratable decider; skipping.")
        return CalibrationOutcome(split=split, applicable=False)

    skipped: dict[str, str] = {}
    sources: dict[BinaryDecider, tuple[str, str]] = {}
    for decider in deciders:
        source = _logits_source(pipeline, decider)
        if source is None:
            skipped[decider.name] = "nothing is wired into inputs.logits"
            logger.warning(
                f"Threshold calibration: nothing feeds {decider.name}.inputs.logits; "
                "skipping this decider."
            )
            continue
        sources[decider] = source
    if not sources:
        return CalibrationOutcome(split=split, applicable=True, skipped=skipped)

    def skip_all(reason: str) -> CalibrationOutcome:
        logger.warning(
            f"Threshold calibration: {reason}; leaving shipped thresholds unchanged."
        )
        return CalibrationOutcome(
            split=split, applicable=True, skipped=skipped, reason=reason
        )

    if getattr(datamodule, f"{split}_ds", None) is None:
        return skip_all(f"no {split} split")
    loader = getattr(datamodule, f"{split}_dataloader")()

    device = _pipeline_device(pipeline)
    for module in pipeline.torch_layers:
        module.eval()
    wanted = {port for (_, port) in sources.values()} | {"mask"}
    collected: dict[tuple[str, str], list[torch.Tensor]] = {}
    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                moved = {
                    key: (
                        value.to(device) if isinstance(value, torch.Tensor) else value
                    )
                    for key, value in batch.items()
                }
                outputs = pipeline.forward(
                    batch=moved,
                    context=Context(
                        stage=ExecutionStage.INFERENCE, batch_idx=batch_idx
                    ),
                )
                for (node_name, port), value in outputs.items():
                    if port in wanted and isinstance(value, torch.Tensor):
                        collected.setdefault((node_name, port), []).append(
                            value.detach().cpu()
                        )
    except Exception as exc:  # noqa: BLE001 - best effort: never lose the weights
        logger.opt(exception=True).warning(
            f"Threshold calibration: inference over {split} failed."
        )
        return skip_all(f"inference over {split} failed: {type(exc).__name__}: {exc}")

    mask_keys = [key for key in collected if key[1] == "mask"]
    candidates = _mask_originators(pipeline, mask_keys) or mask_keys
    if len(candidates) != 1:
        found = [key[0] for key in candidates]
        return skip_all(f"expected one ground-truth 'mask' output, found {found}")
    targets = torch.cat(collected[candidates[0]], dim=0)
    frame_labels = targets.flatten(1).any(dim=1)
    n_anom, n_total = int(frame_labels.sum()), int(frame_labels.numel())
    if n_anom == 0 or n_anom == n_total:
        return skip_all(
            f"{split} split is single-class ({n_anom}/{n_total} frames anomalous)"
        )

    reports: dict[str, dict[str, Any]] = {}
    for decider, key in sources.items():
        if key not in collected:
            skipped[decider.name] = f"{key[0]}.{key[1]} produced no tensor over {split}"
            logger.warning(
                f"Threshold calibration: {key[0]}.{key[1]} produced nothing over "
                f"{split}; skipping {decider.name}."
            )
            continue
        scores = torch.cat(collected[key], dim=0)
        try:
            report = decider.calibrate(scores, targets)
        except Exception as exc:  # noqa: BLE001 - best effort: never lose the weights
            skipped[decider.name] = f"{type(exc).__name__}: {exc}"
            logger.opt(exception=True).warning(
                f"Threshold calibration: {decider.name}.calibrate failed; keeping its "
                "shipped thresholds."
            )
            continue
        if report is None:
            skipped[decider.name] = "calibrate returned no report"
            continue
        logger.info(f"Threshold calibration: {decider.name} -> {report}")
        reports[decider.name] = report

    outcome = CalibrationOutcome(
        split=split, applicable=True, reports=reports, skipped=skipped
    )
    logger.info(f"Threshold calibration: {outcome.summary()}")
    return outcome
