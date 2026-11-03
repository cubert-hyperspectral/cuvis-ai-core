"""Pure-numpy tests for the threshold search the deciders calibrate with.

These pin the numerics cuvis-ai's deciders and the ``calibrate-thresholds`` CLI both rely on:
the one-pass ``searchsorted`` confusion counts match the plain per-candidate loop bit for bit,
candidates and margins are exact in float32 (the dtype the deciders compare pixel scores in),
the stage-1 image score is the statistic ``TwoStageBinaryDecider.forward`` computes, unusable
input is refused with a ``CalibrationError`` that names the problem, the selection rule for a
precision or recall target on hand-built rows, how the trainrun's block maps to the objective,
and the three entry points the base decider's default ``calibrate`` dispatches to.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cuvis_ai_schemas.training import CalibrationConfig

from cuvis_ai_core.deciders import threshold_search as calib


def _random_split(
    seed: int = 0, n: int = 4, size: int = 16, positive_rate: float = 0.05
):
    """Random float32 scores with ties plus a sparse random mask, as numpy arrays."""
    rng = np.random.default_rng(seed)
    # Integer-valued scores divided by 10 give plenty of exact ties.
    pixel = (rng.integers(0, 40, size=(n, size, size)) / 10).astype(np.float32)
    gt = rng.random((n, size, size)) < positive_rate
    gt[0, 0, 0] = True  # guarantee at least one positive
    gt[1, 0, 0] = False  # and one negative
    return pixel, gt


def _reference_confusions(pixel_scores, gt_masks, candidates):
    """The pre-searchsorted implementation: one full pass per candidate."""
    n = pixel_scores.shape[0]
    flat_scores = pixel_scores.reshape(n, -1)
    flat_gt = gt_masks.reshape(n, -1)
    tp = np.empty((n, candidates.size), dtype=np.float64)
    fp = np.empty_like(tp)
    for j, thr in enumerate(candidates):
        flagged = flat_scores >= thr
        tp[:, j] = (flagged & flat_gt).sum(axis=1)
        fp[:, j] = (flagged & ~flat_gt).sum(axis=1)
    return tp, fp


def _reference_image_score(pixel_scores: torch.Tensor, top_k_fraction: float) -> float:
    """The stage-1 statistic as it was written inline in ``TwoStageBinaryDecider.forward``."""
    flat = pixel_scores.reshape(-1)
    k = max(
        1,
        int(
            torch.ceil(
                torch.tensor(flat.numel() * top_k_fraction, dtype=torch.float32)
            ).item()
        ),
    )
    return torch.topk(flat, k).values.mean().item()


# --- margin_below ---------------------------------------------------------------------


def test_margin_below_is_the_plateau_midpoint():
    samples = np.array([0.0, 0.0, 5.0, 5.0], dtype=np.float32)
    assert calib.margin_below(5.0, samples, float32=True) == 2.5
    assert calib.margin_below(5.0, samples, float32=False) == 2.5


def test_margin_below_without_a_lower_sample_returns_the_threshold():
    samples = np.array([5.0, 5.0], dtype=np.float32)
    assert calib.margin_below(5.0, samples, float32=True) == 5.0


def test_margin_below_float32_falls_back_when_the_midpoint_collapses():
    upper = np.float32(1.0)
    lower = np.nextafter(upper, np.float32(0.0))  # adjacent float32 values
    samples = np.array([lower, upper], dtype=np.float32)
    margin = calib.margin_below(float(upper), samples, float32=True)
    assert margin == float(upper)  # no float32 value fits strictly between the two
    assert np.float32(margin) == margin


# --- candidates and confusions ----------------------------------------------------------


def test_pixel_candidates_are_float32_exact_sorted_and_include_the_max():
    pixel, _ = _random_split()
    candidates = calib.pixel_candidates(pixel, 64)
    assert candidates.dtype == np.float32
    assert np.all(np.diff(candidates) > 0)
    assert np.float32(pixel.max()) in candidates
    assert all(np.float32(float(c)) == c for c in candidates)


def test_frame_confusions_match_the_per_candidate_loop():
    for seed in range(3):
        pixel, gt = _random_split(seed=seed)
        candidates = calib.pixel_candidates(pixel, 64)
        tp, fp = calib.frame_confusions(pixel, gt, candidates)
        ref_tp, ref_fp = _reference_confusions(pixel, gt, candidates)
        np.testing.assert_array_equal(tp, ref_tp)
        np.testing.assert_array_equal(fp, ref_fp)


def test_sweep_absolute_margin_lies_strictly_between_adjacent_samples():
    pixel, gt = _random_split(seed=1)
    best = calib.sweep_absolute(pixel, gt, 64)
    threshold, margin = best["threshold"], best["margin_threshold"]
    assert np.float32(threshold) == threshold
    assert np.float32(margin) == margin
    below = pixel[pixel < threshold]
    if below.size:
        assert float(below.max()) < margin <= threshold
    else:
        assert margin == threshold
    # Same flagged set, so the same F1 as the on-point optimum.
    flagged = pixel >= np.float32(margin)
    tp = float((flagged & gt).sum())
    assert (
        calib.prf(tp, float((flagged & ~gt).sum()), float(gt.sum()) - tp)["f1"]
        == best["f1"]
    )


def test_sweep_two_stage_carries_on_point_and_margin_keys():
    pixel, gt = _random_split(seed=2)
    frame_labels = gt.any(axis=(1, 2))
    image_scores = calib.topk_mean_scores(pixel, 0.01)
    best_image, joint, conditional = calib.sweep_two_stage(
        pixel, gt, image_scores, frame_labels, 64
    )
    assert {"threshold", "margin_threshold", "f1"} <= best_image.keys()
    for row in (joint, conditional):
        assert {
            "pixel_threshold",
            "margin_pixel_threshold",
            "margin_image_threshold",
        } <= row.keys()
        assert (
            np.float32(row["margin_pixel_threshold"]) == row["margin_pixel_threshold"]
        )
    assert conditional["image_threshold"] == best_image["threshold"]
    assert joint["f1"] >= conditional["f1"]


def test_sweep_ties_resolve_to_the_highest_candidate():
    # Every threshold in (0, 5] separates perfectly: the highest candidate (5.0) must win.
    pixel = np.zeros((2, 4, 4), dtype=np.float32)
    gt = np.zeros((2, 4, 4), dtype=bool)
    pixel[1, :2, :2] = 5.0
    gt[1, :2, :2] = True
    best = calib.sweep_absolute(pixel, gt, 32)
    assert best["threshold"] == 5.0
    assert best["margin_threshold"] == 2.5


# --- stage-1 image score ----------------------------------------------------------------


@pytest.mark.parametrize("top_k_fraction", [0.001, 0.25, 0.5, 1.0])
def test_frame_image_score_matches_the_inline_forward_statistic(top_k_fraction):
    frames = torch.rand(5, 23, 31, generator=torch.Generator().manual_seed(0))
    for frame in frames:
        assert calib.frame_image_score(frame, top_k_fraction) == _reference_image_score(
            frame, top_k_fraction
        )
    stacked = calib.topk_mean_scores(frames.numpy(), top_k_fraction)
    assert stacked.dtype == np.float64
    assert stacked.tolist() == [
        _reference_image_score(f, top_k_fraction) for f in frames
    ]


def test_topk_count_rounds_the_product_in_float32():
    # 1000 * 0.001 is 1.0000000000000002 in float64 but exactly 1 in float32: k stays 1.
    assert calib.topk_count(1000, 0.001) == 1
    assert calib.topk_count(1, 1e-9) == 1  # never below one pixel


# --- sigmoid space ------------------------------------------------------------------------


def test_sigmoid_float32_matches_torch_and_saturates():
    values = np.array([-40.0, 0.0, 5.0, 20.0, 40.0], dtype=np.float32)
    probs = calib.sigmoid_float32(values)
    assert probs.dtype == np.float32
    np.testing.assert_array_equal(
        probs, torch.sigmoid(torch.from_numpy(values)).numpy()
    )
    assert probs[3] == 1.0 and probs[4] == 1.0  # float32 saturation above logits of ~17


# --- guards --------------------------------------------------------------------------------


def _scores_and_targets(n=2, size=4):
    scores = torch.zeros(n, size, size, 1)
    targets = torch.zeros(n, size, size, 1, dtype=torch.bool)
    scores[1, 0, 0, 0] = 5.0
    targets[1, 0, 0, 0] = True
    return scores, targets


def test_reduce_scores_targets_accepts_3d_and_4d_layouts():
    scores, targets = _scores_and_targets()
    full, pixel, gt, frame_labels = calib.reduce_scores_targets(scores, targets[..., 0])
    assert full.shape == (2, 4, 4, 1) and pixel.shape == (2, 4, 4)
    assert pixel.dtype == np.float32 and gt.dtype == bool
    assert frame_labels.tolist() == [False, True]


def test_reduce_scores_targets_refuses_shape_mismatch():
    scores, targets = _scores_and_targets()
    with pytest.raises(calib.CalibrationError, match=r"disagree on \[N, H, W\]"):
        calib.reduce_scores_targets(scores, targets[:1])
    with pytest.raises(calib.CalibrationError, match="scores must be"):
        calib.reduce_scores_targets(
            scores[0, 0], targets[0, 0]
        )  # rank 2: not a frame stack


def test_reduce_scores_targets_refuses_non_finite_scores():
    scores, targets = _scores_and_targets()
    scores[0, 1, 1, 0] = float("nan")
    with pytest.raises(calib.CalibrationError, match="nan or inf"):
        calib.reduce_scores_targets(scores, targets)


def test_reduce_scores_targets_refuses_single_class_targets():
    scores, targets = _scores_and_targets()
    with pytest.raises(calib.CalibrationError, match="no anomalous pixel"):
        calib.reduce_scores_targets(scores, torch.zeros_like(targets))
    with pytest.raises(calib.CalibrationError, match="every pixel anomalous"):
        calib.reduce_scores_targets(scores, torch.ones_like(targets))


# --- auroc ---------------------------------------------------------------------------------


def test_binary_auroc_separable_ties_and_single_class():
    labels = np.array([False, False, True, True])
    assert calib.binary_auroc(np.array([0.1, 0.2, 0.8, 0.9]), labels) == 1.0
    assert calib.binary_auroc(np.array([0.5, 0.5, 0.5, 0.5]), labels) == 0.5  # midranks
    assert np.isnan(calib.binary_auroc(np.array([0.1, 0.9]), np.array([True, True])))


def _row(**values):
    """A sweep row with every metric the selection reads, zero unless given."""
    row = {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "frame_precision": 0.0,
        "frame_recall": 0.0,
        "frame_f1": 0.0,
        "image_threshold": 0.0,
        "pixel_threshold": 0.0,
    }
    row.update(values)
    return row


def _block(objective: str = "f1", target: float | None = None, level: str = "pixels"):
    """The trainrun's calibration block, as core hands it to ``calibrate``."""
    return CalibrationConfig(objective=objective, target=target, level=level)


# --- the selection rule on hand-built rows -------------------------------------------------


def test_recall_target_takes_the_most_precise_row_at_or_above_the_target():
    rows = [
        _row(recall=0.95, precision=0.5, pixel_threshold=1.0),
        _row(recall=0.92, precision=0.7, pixel_threshold=2.0),
        _row(recall=0.85, precision=0.9, pixel_threshold=3.0),
    ]
    row, reached = calib.select_calibration_row(
        rows, calib.CalibrationObjective("recall", 0.9, "pixels")
    )
    assert reached
    assert row["pixel_threshold"] == 2.0


def test_precision_target_takes_the_highest_recall_row_at_or_above_the_target():
    rows = [
        _row(recall=0.95, precision=0.5, pixel_threshold=1.0),
        _row(recall=0.92, precision=0.7, pixel_threshold=2.0),
        _row(recall=0.85, precision=0.9, pixel_threshold=3.0),
    ]
    row, reached = calib.select_calibration_row(
        rows, calib.CalibrationObjective("precision", 0.6, "pixels")
    )
    assert reached
    assert row["pixel_threshold"] == 2.0


def test_ties_resolve_by_the_other_level_then_the_higher_thresholds():
    rows = [
        _row(
            recall=0.9,
            precision=0.8,
            frame_f1=0.5,
            image_threshold=1.0,
            pixel_threshold=2.0,
        ),
        _row(
            recall=0.9,
            precision=0.8,
            frame_f1=0.9,
            image_threshold=1.0,
            pixel_threshold=1.0,
        ),
        _row(
            recall=0.9,
            precision=0.8,
            frame_f1=0.9,
            image_threshold=2.0,
            pixel_threshold=1.0,
        ),
    ]
    row, _ = calib.select_calibration_row(
        rows, calib.CalibrationObjective("recall", 0.9, "pixels")
    )
    assert row["frame_f1"] == 0.9
    assert (row["image_threshold"], row["pixel_threshold"]) == (2.0, 1.0)


def test_unreachable_recall_lands_on_the_highest_recall_row_and_says_so():
    rows = [
        _row(recall=0.95, precision=0.5, pixel_threshold=1.0),
        _row(recall=0.95, precision=0.6, pixel_threshold=2.0),
        _row(recall=0.85, precision=0.9, pixel_threshold=3.0),
    ]
    row, reached = calib.select_calibration_row(
        rows, calib.CalibrationObjective("recall", 0.99, "pixels")
    )
    assert not reached
    assert row["pixel_threshold"] == 2.0  # the most precise of the highest-recall rows


def test_unreachable_precision_lands_on_the_most_precise_row():
    rows = [
        _row(recall=0.95, precision=0.5, pixel_threshold=1.0),
        _row(recall=0.85, precision=0.9, pixel_threshold=3.0),
        _row(recall=0.80, precision=0.9, pixel_threshold=4.0),
    ]
    row, reached = calib.select_calibration_row(
        rows, calib.CalibrationObjective("precision", 0.99, "pixels")
    )
    assert not reached
    assert (
        row["pixel_threshold"] == 3.0
    )  # the highest recall among the most precise rows


def test_frame_level_reads_the_frame_columns():
    rows = [
        _row(
            frame_recall=0.9,
            frame_precision=0.5,
            recall=0.1,
            precision=0.1,
            pixel_threshold=1.0,
        ),
        _row(
            frame_recall=0.9,
            frame_precision=0.8,
            recall=0.1,
            precision=0.1,
            pixel_threshold=2.0,
        ),
    ]
    row, reached = calib.select_calibration_row(
        rows, calib.CalibrationObjective("recall", 0.9, "frames")
    )
    assert reached
    assert row["pixel_threshold"] == 2.0


# --- the objective argument ----------------------------------------------------------------


def test_no_block_is_the_f1_optimum():
    assert calib.calibration_objective_of(None) == calib.CalibrationObjective(
        "f1", None, "pixels"
    )


def test_the_block_maps_to_the_objective():
    objective = calib.calibration_objective_of(_block("recall", 0.9, "frames"))
    assert objective == calib.CalibrationObjective("recall", 0.9, "frames")


# --- the entry points of the base decider's default calibrate -------------------------------


def _split_arrays(seed: int = 3):
    """``reduce_scores_targets`` on distinct logits: full, pixel, gt masks, frame labels."""
    rng = np.random.default_rng(seed)
    scores = torch.from_numpy(rng.normal(size=(6, 12, 12, 1)).astype(np.float32))
    targets = torch.from_numpy(rng.random((6, 12, 12, 1)) < 0.1)
    scores[targets] += 1.5
    targets[0, 0, 0, 0] = True
    targets[1] = False
    return calib.reduce_scores_targets(scores, targets)


def test_calibrate_cutoff_f1_applies_the_absolute_sweep_margin():
    _, pixel, gt, labels = _split_arrays()
    found = calib.calibrate_cutoff(pixel, gt, labels, calib.CalibrationObjective(), 64)
    best = calib.sweep_absolute(pixel, gt, 64)
    assert found.values == {"cutoff": best["margin_threshold"]}
    assert found.on_point == {"cutoff": best["threshold"]}
    assert found.reached is None
    assert found.extra == {}
    admitted = np.ones(pixel.shape[0], dtype=bool)
    cutoffs = np.full(pixel.shape[0], np.float32(best["margin_threshold"]))
    assert found.counts == calib.counts_at_point(pixel, gt, labels, admitted, cutoffs)


def test_calibrate_cutoff_recall_target_reaches_it_and_says_so():
    _, pixel, gt, labels = _split_arrays()
    goal = calib.CalibrationObjective("recall", 0.9, "pixels")
    found = calib.calibrate_cutoff(pixel, gt, labels, goal, 64)
    assert found.reached is True
    assert found.counts["recall"] >= 0.9


def test_calibrate_gate_cutoff_f1_carries_the_image_f1_and_the_joint_optimum():
    _, pixel, gt, labels = _split_arrays()
    image_scores = calib.topk_mean_scores(pixel, 0.01)
    found = calib.calibrate_gate_cutoff(
        pixel, gt, image_scores, labels, calib.CalibrationObjective(), 64
    )
    best_image, joint, conditional = calib.sweep_two_stage(
        pixel, gt, image_scores, labels, 64
    )
    assert found.values == {
        "gate": best_image["margin_threshold"],
        "cutoff": conditional["margin_pixel_threshold"],
    }
    assert found.on_point == {
        "gate": best_image["threshold"],
        "cutoff": conditional["pixel_threshold"],
    }
    assert found.extra["image_f1"] == best_image["f1"]
    assert found.extra["joint"] == {
        "gate": joint["margin_image_threshold"],
        "cutoff": joint["margin_pixel_threshold"],
        "f1": joint["f1"],
    }
    assert found.reached is None


def test_calibrate_gate_cutoff_frame_recall_target_counts_frames():
    _, pixel, gt, labels = _split_arrays()
    image_scores = calib.topk_mean_scores(pixel, 0.01)
    goal = calib.CalibrationObjective("recall", 0.9, "frames")
    found = calib.calibrate_gate_cutoff(pixel, gt, image_scores, labels, goal, 64)
    assert found.reached is True
    assert found.counts["frame_recall"] >= 0.9
    assert found.extra == {}


def test_calibrate_quantile_f1_picks_the_quantile_sweep_optimum_on_the_same_grid():
    full, pixel, gt, labels = _split_arrays()
    found = calib.calibrate_quantile(
        full, pixel, gt, labels, 0.995, calib.CalibrationObjective(), 64
    )
    flat = full.reshape(full.shape[0], -1)
    grid = calib.quantile_grid_builder(0.995, 64)(flat.shape[1])
    per_frame = np.quantile(flat, grid, axis=1)
    best = calib.sweep_quantile(
        pixel, gt, {float(q): per_frame[i] for i, q in enumerate(grid)}
    )
    assert found.values == {"cutoff": best["quantile"]}
    assert found.on_point == {"cutoff": best["quantile"]}
    assert found.reached is None
    assert set(found.counts) >= {"f1", "frame_f1", "tp", "frame_tp"}


def test_calibrate_quantile_recall_target_reaches_it():
    full, pixel, gt, labels = _split_arrays()
    goal = calib.CalibrationObjective("recall", 0.9, "pixels")
    found = calib.calibrate_quantile(full, pixel, gt, labels, 0.995, goal, 64)
    assert found.reached is True
    assert found.counts["recall"] >= 0.9
