from abc import abstractmethod
from typing import Any, ClassVar

import numpy as np
import torch
from cuvis_ai_schemas.execution.context import Context
from cuvis_ai_schemas.pipeline import PortSpec
from cuvis_ai_schemas.training import CalibrationConfig
from torch import Tensor

from cuvis_ai_core.deciders.threshold_search import (
    CALIBRATION_LEVELS,
    SearchResult,
    calibrate_cutoff,
    calibrate_gate_cutoff,
    calibrate_quantile,
    calibration_objective_of,
    calibration_report,
    reduce_scores_targets,
)
from cuvis_ai_core.node.node import Node

# The search rules a decider may declare in ``CALIBRATION_RULE``.
CALIBRATION_RULES = ("gate_cutoff", "cutoff", "quantile")


class BinaryDecider(Node):
    """
    Abstract class for Decision Making Nodes.

    The decider nodes transform a prediction state into a final prediction
    based on the task that needs to be accomplished.

    A decider takes part in the post-fit calibration by declaring its rule: ``CALIBRATION_RULE``
    names the search (``"gate_cutoff"``, an image gate and a pixel cutoff; ``"cutoff"``, one
    absolute threshold; ``"quantile"``, one per-frame quantile), ``CALIBRATION_GATE`` and
    ``CALIBRATION_CUTOFF`` the hparams the rule moves, ``CALIBRATION_LEVELS`` the levels a
    precision or recall target may be measured at. The search reads the decider through two
    hooks, :meth:`calibration_scores` and :meth:`calibration_image_scores`; the default
    :meth:`calibrate` does the rest. A decider without a rule is left out of the phase unless
    it overrides :meth:`calibrate` itself.
    """

    INPUT_SPECS = {
        "logits": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Input logits to threshold (BHWC format)",
        )
    }

    OUTPUT_SPECS = {
        "decisions": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, 1),
            description="Binary decision mask (BHWC format)",
        )
    }

    # The calibration contract; see the class docstring.
    CALIBRATION_RULE: ClassVar[str | None] = None
    CALIBRATION_GATE: ClassVar[str | None] = None
    CALIBRATION_CUTOFF: ClassVar[str] = "threshold"
    CALIBRATION_LEVELS: ClassVar[tuple[str, ...]] = CALIBRATION_LEVELS

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if (
            cls.CALIBRATION_RULE is not None
            and cls.CALIBRATION_RULE not in CALIBRATION_RULES
        ):
            raise TypeError(
                f"{cls.__name__}.CALIBRATION_RULE {cls.CALIBRATION_RULE!r} is not one of "
                f"{', '.join(map(repr, CALIBRATION_RULES))}"
            )
        if cls.CALIBRATION_RULE == "gate_cutoff" and not cls.CALIBRATION_GATE:
            raise TypeError(
                f"{cls.__name__} declares the gate_cutoff rule without CALIBRATION_GATE, "
                "the hparam that holds its image gate"
            )

    def __init__(self, threshold: float = 0.5, **kwargs) -> None:
        self.threshold = threshold
        # Accept arbitrary args/kwargs so subclasses can forward their init
        # parameters to the Serializable base for automatic hparam capture.
        super().__init__(threshold=threshold, **kwargs)

    @property
    def is_calibrated(self) -> bool:
        """Whether the decider holds the thresholds its decisions need.

        A decider constructed with its thresholds is calibrated from the start. A
        subclass whose thresholds may be unset until the post-fit calibration (a null
        value in the pipeline yaml) overrides this to report the live state; the post-fit
        lifecycle reads it to announce a pending calibration and to say, after the step,
        whether thresholds are present.
        """
        return True

    @abstractmethod
    def forward(  # type: ignore[override]
        self,
        logits: Tensor,
        context: Context,
        **_: Any,
    ) -> dict[str, Tensor]:
        """Apply decisioning on channels-last data.

        Args:
            logits: Tensor shaped (B, H, W, C) containing logits.

        Returns:
            Dictionary with "decisions" key containing (B, H, W, 1) decision mask.
        """

    def calibration_scores(self, scores: Tensor) -> Tensor:
        """The tensor the search runs on: what ``forward`` compares, the raw scores by default.

        A decider that compares a transform of its input (a decider thresholding
        ``sigmoid(logits)`` in float32) returns that transform, so the fitted value lives in
        the space ``forward`` decides in. A decider whose rule the search cannot reproduce for
        a given input raises ``CalibrationError`` here.
        """
        return scores

    def calibration_image_scores(self, pixel_scores: np.ndarray) -> np.ndarray | None:
        """The per-frame image score the gate rule gates on, one value per frame.

        ``pixel_scores`` is the ``[N, H, W]`` float32 per-pixel max the search works on. The
        gate rule requires the override (a two-stage decider returns the mean of its top-k
        pixel scores, the statistic its ``forward`` computes); the other rules ignore it.
        """
        return None

    def calibrate(
        self,
        scores: Tensor,
        targets: Tensor,
        objective: CalibrationConfig | None = None,
        *,
        num_candidates: int = 256,
    ) -> dict[str, Any] | None:
        """Re-fit this decider's threshold(s) on a labelled split by its declared rule.

        The in-training calibration phase collects the scores feeding this decider and the
        ground-truth mask over the validation split, then calls this method. A decider that
        declares ``CALIBRATION_RULE`` gets the search for free: the scores pass through
        :meth:`calibration_scores`, are reduced to float32 per-pixel scores, masks and frame
        labels, and the rule's entry point in :mod:`cuvis_ai_core.deciders.threshold_search`
        sweeps the decision rule to the F1 optimum, or to the precision or recall target the
        trainrun's calibration block names. The new values go to the live attributes and to
        ``hparams``; ``pipeline.save_to_file`` then carries them in the pipeline yaml and the
        ``.pt`` weights are unchanged. A decider without a rule is skipped (``None``) unless
        it overrides this method, which a decider from before the contract may still do.

        The phase passes ``objective`` only when the trainrun carries a calibration block, so
        an override without the parameter keeps calibrating to F1 on every other trainrun.

        The report carries ``class``, ``{"old", "new"}`` per moved hparam, ``on_point`` (the
        values before the plateau margin, by hparam name), the objective and level, the
        precision, recall, F1 and IoU at the pixel and at the frame level with the counts
        behind them, the ``frames`` and ``anomalous_frames`` of the split, ``target`` and
        ``reached`` (1 or 0) for a target objective, the pixel-level ``f1``, ``precision`` and
        ``recall`` once more at the top level, and for the gate rule on the F1 objective
        ``image_f1`` and the ``joint`` optimum.

        Args:
            scores: Decider-input scores stacked over the split, shaped (N, H, W, C).
            targets: Ground-truth anomaly mask, (N, H, W, C) or (N, H, W).
            objective: The trainrun's calibration block; ``None`` means the F1 optimum.
            num_candidates: Size of the sampled cutoff grid (the grid is exact below it).

        Returns:
            The calibration report, or ``None`` when the decider declares no rule.

        Raises:
            ValueError: The block names a level the decider does not serve.
            CalibrationError: Shape mismatch, non-finite scores, or a single-class split.
        """
        rule = type(self).CALIBRATION_RULE
        if rule is None:
            return None
        goal = calibration_objective_of(objective, levels=self.CALIBRATION_LEVELS)
        full, pixel, gt, frame_labels = reduce_scores_targets(
            self.calibration_scores(scores), targets
        )
        names: dict[str, str] = {}
        if rule == "gate_cutoff":
            names["gate"] = str(self.CALIBRATION_GATE)
        names["cutoff"] = self.CALIBRATION_CUTOFF
        found: SearchResult
        if rule == "gate_cutoff":
            image_scores = self.calibration_image_scores(pixel)
            if image_scores is None:
                raise TypeError(
                    f"{type(self).__name__} declares the gate_cutoff rule, so "
                    "calibration_image_scores must return one image score per frame"
                )
            found = calibrate_gate_cutoff(
                pixel, gt, image_scores, frame_labels, goal, num_candidates
            )
        elif rule == "cutoff":
            found = calibrate_cutoff(pixel, gt, frame_labels, goal, num_candidates)
        else:
            preset = float(getattr(self, names["cutoff"]))
            found = calibrate_quantile(
                full, pixel, gt, frame_labels, preset, goal, num_candidates
            )
        old = {role: getattr(self, name) for role, name in names.items()}
        for role, name in names.items():
            setattr(self, name, found.values[role])
            self.hparams[name] = found.values[role]
        report: dict[str, Any] = {"class": type(self).__name__}
        if goal.targeted:
            report["target"] = goal.target
            report["reached"] = 1.0 if found.reached else 0.0
        report.update(calibration_report(goal, found.counts, frame_labels))
        for role, name in names.items():
            report[name] = {"old": old[role], "new": found.values[role]}
        report["on_point"] = {
            name: found.on_point[role] for role, name in names.items()
        }
        report.update({key: found.counts[key] for key in ("f1", "precision", "recall")})
        for key, value in found.extra.items():
            if isinstance(value, dict):
                value = {names.get(role, role): item for role, item in value.items()}
            report[key] = value
        return report
