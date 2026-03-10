"""Unit tests for PipelineProfiler and _ScalarAccumulator math.

Covers Welford mean/std accuracy, P² median approximation, warm-up skip,
reset, empty snapshot, and thread-safety stress.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from cuvis_ai_core.pipeline.profiling import (
    PipelineProfiler,
    _ScalarAccumulator,
    format_profiling_table,
)
from cuvis_ai_schemas.pipeline.profiling import NodeProfilingStats


class TestScalarAccumulator:
    """Tests for _ScalarAccumulator Welford + P² logic."""

    def test_welford_mean_std_match_numpy(self) -> None:
        """Welford mean/std should match NumPy on deterministic samples."""
        rng = np.random.default_rng(42)
        samples = rng.normal(loc=50.0, scale=10.0, size=200).tolist()

        acc = _ScalarAccumulator()
        for s in samples:
            acc.record(s)

        snap = acc.snapshot()
        assert snap["count"] == 200
        np.testing.assert_allclose(snap["mean_ms"], np.mean(samples), rtol=1e-10)
        # Population std (ddof=0)
        np.testing.assert_allclose(snap["std_ms"], np.std(samples, ddof=0), rtol=1e-10)

    def test_min_max_total_last(self) -> None:
        """Min, max, total, and last should be exact."""
        acc = _ScalarAccumulator()
        values = [3.0, 1.0, 4.0, 1.5, 9.0, 2.6]
        for v in values:
            acc.record(v)

        snap = acc.snapshot()
        assert snap["min_ms"] == 1.0
        assert snap["max_ms"] == 9.0
        assert snap["last_ms"] == 2.6
        np.testing.assert_allclose(snap["total_ms"], sum(values))

    def test_p2_median_within_tolerance(self) -> None:
        """P² median should be within 10% of true median on large data."""
        rng = np.random.default_rng(123)
        samples = rng.exponential(scale=5.0, size=1000).tolist()

        acc = _ScalarAccumulator()
        for s in samples:
            acc.record(s)

        snap = acc.snapshot()
        true_median = float(np.median(samples))
        # Allow 10% relative tolerance for P² approximation
        assert abs(snap["median_ms"] - true_median) / true_median < 0.10

    def test_median_exact_for_small_samples(self) -> None:
        """Median should be exact when <=5 samples (warm-up buffer)."""
        acc = _ScalarAccumulator()
        acc.record(10.0)
        acc.record(20.0)
        acc.record(30.0)
        assert acc.snapshot()["median_ms"] == 20.0

        acc2 = _ScalarAccumulator()
        acc2.record(1.0)
        acc2.record(2.0)
        assert acc2.snapshot()["median_ms"] == 1.5

    def test_empty_snapshot(self) -> None:
        """Empty accumulator should return zeroed snapshot."""
        acc = _ScalarAccumulator()
        snap = acc.snapshot()
        assert snap["count"] == 0
        assert snap["mean_ms"] == 0.0
        assert snap["median_ms"] == 0.0
        assert snap["std_ms"] == 0.0

    def test_single_sample_std_is_zero(self) -> None:
        """A single sample should yield std=0."""
        acc = _ScalarAccumulator()
        acc.record(42.0)
        snap = acc.snapshot()
        assert snap["count"] == 1
        assert snap["mean_ms"] == 42.0
        assert snap["std_ms"] == 0.0


class TestSkipFirstN:
    """Tests for the warm-up skip mechanism."""

    def test_skip_discards_initial_samples(self) -> None:
        """First skip_first_n samples should be discarded."""
        acc = _ScalarAccumulator(skip_target=3)
        for v in [100.0, 200.0, 300.0, 1.0, 2.0, 3.0]:
            acc.record(v)

        snap = acc.snapshot()
        assert snap["count"] == 3  # Only last 3 recorded
        np.testing.assert_allclose(snap["mean_ms"], 2.0)

    def test_skip_zero_records_all(self) -> None:
        """skip_target=0 should record all samples."""
        acc = _ScalarAccumulator(skip_target=0)
        acc.record(5.0)
        acc.record(10.0)
        assert acc.snapshot()["count"] == 2


class TestPipelineProfiler:
    """Tests for PipelineProfiler."""

    def test_negative_skip_raises(self) -> None:
        """Negative skip_first_n should raise ValueError."""
        with pytest.raises(ValueError, match="skip_first_n must be >= 0"):
            PipelineProfiler(skip_first_n=-1)

    def test_record_and_snapshot(self) -> None:
        """Recording and snapshotting should work end-to-end."""
        profiler = PipelineProfiler()
        profiler.record("inference", "NodeA", 10.0)
        profiler.record("inference", "NodeA", 20.0)
        profiler.record("inference", "NodeB", 5.0)

        stats = profiler.snapshot()
        assert len(stats) == 2

        a_stats = [s for s in stats if s.node_name == "NodeA"][0]
        assert a_stats.count == 2
        assert a_stats.stage == "inference"
        np.testing.assert_allclose(a_stats.mean_ms, 15.0)

    def test_snapshot_stage_filter(self) -> None:
        """Snapshot with stage filter should only return matching entries."""
        profiler = PipelineProfiler()
        profiler.record("inference", "NodeA", 10.0)
        profiler.record("train", "NodeA", 20.0)

        inference_stats = profiler.snapshot(stage="inference")
        assert len(inference_stats) == 1
        assert inference_stats[0].stage == "inference"

        all_stats = profiler.snapshot()
        assert len(all_stats) == 2

    def test_reset_clears_all(self) -> None:
        """Reset should clear all accumulated stats."""
        profiler = PipelineProfiler()
        profiler.record("inference", "NodeA", 10.0)
        assert len(profiler.snapshot()) == 1

        profiler.reset()
        assert len(profiler.snapshot()) == 0

    def test_empty_snapshot_returns_empty(self) -> None:
        """Fresh profiler snapshot should return empty list."""
        profiler = PipelineProfiler()
        assert profiler.snapshot() == []

    def test_skip_first_n_applied(self) -> None:
        """Skip should apply per (stage, node_name) key."""
        profiler = PipelineProfiler(skip_first_n=2)
        for v in [100.0, 200.0, 1.0, 2.0, 3.0]:
            profiler.record("inference", "NodeA", v)

        stats = profiler.snapshot()
        assert len(stats) == 1
        assert stats[0].count == 3  # Only 1.0, 2.0, 3.0 recorded

    def test_frozen_dataclass(self) -> None:
        """Returned stats should be frozen (immutable)."""
        profiler = PipelineProfiler()
        profiler.record("inference", "NodeA", 10.0)
        stats = profiler.snapshot()
        with pytest.raises(AttributeError):
            stats[0].count = 999  # type: ignore[misc]

    def test_thread_safety_stress(self) -> None:
        """Concurrent record/reset/snapshot should not raise."""
        profiler = PipelineProfiler()
        errors: list[Exception] = []
        barrier = threading.Barrier(6)

        def recorder(stage: str, node: str) -> None:
            try:
                barrier.wait(timeout=5)
                for i in range(500):
                    profiler.record(stage, node, float(i))
            except Exception as e:
                errors.append(e)

        def resetter() -> None:
            try:
                barrier.wait(timeout=5)
                for _ in range(50):
                    profiler.reset()
            except Exception as e:
                errors.append(e)

        def snapshotter() -> None:
            try:
                barrier.wait(timeout=5)
                for _ in range(100):
                    profiler.snapshot()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=recorder, args=("inference", "NodeA")),
            threading.Thread(target=recorder, args=("inference", "NodeB")),
            threading.Thread(target=recorder, args=("train", "NodeA")),
            threading.Thread(target=recorder, args=("train", "NodeB")),
            threading.Thread(target=resetter),
            threading.Thread(target=snapshotter),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert errors == [], f"Thread errors: {errors}"


class TestFormatProfilingTable:
    """Tests for format_profiling_table()."""

    def _make_stats(self) -> list[NodeProfilingStats]:
        return [
            NodeProfilingStats(
                node_name="node_a",
                stage="inference",
                count=100,
                mean_ms=10.0,
                median_ms=9.5,
                std_ms=2.0,
                min_ms=5.0,
                max_ms=20.0,
                total_ms=1000.0,
                last_ms=10.0,
            ),
            NodeProfilingStats(
                node_name="node_b",
                stage="inference",
                count=100,
                mean_ms=5.0,
                median_ms=4.8,
                std_ms=1.0,
                min_ms=3.0,
                max_ms=8.0,
                total_ms=500.0,
                last_ms=5.0,
            ),
        ]

    def test_empty_returns_message(self) -> None:
        assert format_profiling_table([]) == "No profiling data collected."

    def test_contains_header_and_nodes(self) -> None:
        table = format_profiling_table(
            self._make_stats(), total_frames=103, skip_first_n=3
        )
        assert "Profiling Summary (103 frames, skip_first_n=3)" in table
        assert "node_a" in table
        assert "node_b" in table
        assert "TOTAL" in table

    def test_sorted_by_total_descending(self) -> None:
        table = format_profiling_table(self._make_stats())
        lines = table.split("\n")
        # node_a (1000ms) should appear before node_b (500ms) in data rows
        node_a_idx = next(i for i, line in enumerate(lines) if "node_a" in line)
        node_b_idx = next(i for i, line in enumerate(lines) if "node_b" in line)
        assert node_a_idx < node_b_idx

    def test_total_line(self) -> None:
        table = format_profiling_table(self._make_stats())
        assert "1.500" in table  # 1500ms = 1.500s total

    def test_fps_line(self) -> None:
        table = format_profiling_table(self._make_stats())
        # 1500ms total / 100 frames = 15ms per frame → ~66.7 FPS
        assert "15.00 ms" in table
        assert "66.7 FPS" in table

    def test_skip_first_n_only(self) -> None:
        table = format_profiling_table(self._make_stats(), skip_first_n=5)
        assert "skip_first_n=5" in table

    def test_no_metadata(self) -> None:
        table = format_profiling_table(self._make_stats())
        assert table.startswith("Profiling Summary\n")
