"""Post-training threshold calibration for anomaly deciders.

After gradient training a decider's shipped threshold hparams are stale: training moved the
score distribution, so a value tuned for one set of weights misfits the next. This phase runs
the trained pipeline over the labelled validation split, collects the scores feeding each
decider and the ground-truth ``mask``, and calls ``decider.calibrate(scores, targets)`` so the
saved checkpoint ships thresholds matched to its own weights. Deciders that do not implement
calibration, pipelines without one, and single-class val splits are skipped with a log.

The heavy F1-max sweep math lives on the decider subclasses (cuvis-ai); this module only
orchestrates - find the deciders, collect their input scores + the mask over val, dispatch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from loguru import logger

from cuvis_ai_core.deciders.base_decider import BinaryDecider
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.execution import Context

if TYPE_CHECKING:
    import pytorch_lightning as pl

    from cuvis_ai_core.pipeline.pipeline import CuvisPipeline


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


def calibrate_pipeline_deciders(
    pipeline: CuvisPipeline,
    datamodule: pl.LightningDataModule,
    *,
    split: str = "val",
) -> list[dict[str, Any]]:
    """Re-fit calibratable deciders' thresholds on the labelled ``split`` of ``datamodule``.

    Runs the trained ``pipeline`` over the split at the inference stage (so metric / loss
    nodes stay quiet), collects each decider's input scores and the single ground-truth
    ``mask`` output, and calls ``decider.calibrate(scores, targets)``. The deciders are
    mutated in place, so a subsequent ``pipeline.save_to_file`` persists the calibrated
    thresholds into the yaml. No-ops (with a log) when there is no calibratable decider, the
    split is missing, or it is single-class. Returns the per-decider reports.
    """
    deciders = _calibratable_deciders(pipeline)
    if not deciders:
        logger.info("Threshold calibration: no calibratable decider; skipping.")
        return []

    sources: dict[BinaryDecider, tuple[str, str]] = {}
    for decider in deciders:
        source = _logits_source(pipeline, decider)
        if source is None:
            logger.warning(
                f"Threshold calibration: nothing feeds {decider.name}.inputs.logits; "
                "skipping this decider."
            )
            continue
        sources[decider] = source
    if not sources:
        return []

    split_ds = getattr(datamodule, f"{split}_ds", None)
    if split_ds is None:
        logger.info(f"Threshold calibration: no {split} split; skipping.")
        return []
    loader = getattr(datamodule, f"{split}_dataloader")()

    device = _pipeline_device(pipeline)
    for module in pipeline.torch_layers:
        module.eval()
    wanted = {port for (_, port) in sources.values()} | {"mask"}
    collected: dict[tuple[str, str], list[torch.Tensor]] = {}
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            moved = {
                key: (value.to(device) if isinstance(value, torch.Tensor) else value)
                for key, value in batch.items()
            }
            outputs = pipeline.forward(
                batch=moved,
                context=Context(stage=ExecutionStage.INFERENCE, batch_idx=batch_idx),
            )
            for (node_name, port), value in outputs.items():
                if port in wanted and isinstance(value, torch.Tensor):
                    collected.setdefault((node_name, port), []).append(
                        value.detach().cpu()
                    )

    mask_keys = [key for key in collected if key[1] == "mask"]
    if len(mask_keys) != 1:
        logger.warning(
            f"Threshold calibration: expected one 'mask' output, found "
            f"{[key[0] for key in mask_keys]}; skipping."
        )
        return []
    targets = torch.cat(collected[mask_keys[0]], dim=0)
    frame_labels = targets.flatten(1).any(dim=1)
    n_anom, n_total = int(frame_labels.sum()), int(frame_labels.numel())
    if n_anom == 0 or n_anom == n_total:
        logger.warning(
            f"Threshold calibration: {split} split is single-class "
            f"({n_anom}/{n_total} anomalous); leaving shipped thresholds unchanged."
        )
        return []

    reports: list[dict[str, Any]] = []
    for decider, key in sources.items():
        if key not in collected:
            logger.warning(
                f"Threshold calibration: {key[0]}.{key[1]} produced nothing over {split}; "
                f"skipping {decider.name}."
            )
            continue
        scores = torch.cat(collected[key], dim=0)
        report = decider.calibrate(scores, targets)
        if report is not None:
            logger.info(f"Threshold calibration: {decider.name} -> {report}")
            reports.append(report)
    return reports
