"""Pipeline node runtime profiling with online statistics.

This module implements a lightweight scalar Welford accumulator with P² approximate
median, purpose-built for profiling ``node.forward()`` durations.

**Why not reuse** ``cuvis_ai.utils.welford.WelfordAccumulator`` **from cuvis-ai-tracking?**

1. That class is an ``nn.Module`` with float64 torch buffers, designed for
   multi-feature (N, C) tensor statistics during statistical node initialization.
2. Profiling needs a pure-Python *scalar* accumulator (one float per sample) with
   P² approximate median, ``threading.Lock`` thread safety, min/max/total/last
   tracking, and skip-first-N warm-up — none of which exist in the tensor class.
3. Importing from cuvis-ai-tracking into cuvis-ai-core would invert the
   one-directional dependency (core ← tracking), which is architecturally wrong.
"""

from __future__ import annotations

import math
import threading

from cuvis_ai_schemas.pipeline.profiling import NodeProfilingStats


# ---------------------------------------------------------------------------
# P² approximate quantile estimator (constant memory)
# ---------------------------------------------------------------------------


class _P2MedianEstimator:
    """Piecewise-parabolic (P²) quantile estimator for the median (q = 0.5).

    After the 5-sample warm-up buffer, this estimates the running median in
    constant memory with no sample history.

    Reference: Jain & Chlamtac, "The P² Algorithm for Dynamic Calculation of
    Quantiles and Histograms Without Storing Observations", 1985.
    """

    __slots__ = ("_warmup", "_q", "_n", "_ns", "_dn", "_heights")

    def __init__(self) -> None:
        self._warmup: list[float] = []
        # Once we transition out of warm-up, these hold P² state.
        self._q: list[float] = []  # marker heights
        self._n: list[int] = []  # marker positions
        self._ns: list[float] = []  # desired marker positions
        self._dn: list[float] = []  # desired position increments
        self._heights: bool = False  # True once P² is active

    def add(self, x: float) -> None:
        if not self._heights:
            self._warmup.append(x)
            if len(self._warmup) == 5:
                self._init_p2()
            return
        self._update_p2(x)

    @property
    def median(self) -> float:
        if not self._heights:
            if not self._warmup:
                return 0.0
            s = sorted(self._warmup)
            m = len(s)
            mid = m // 2
            if m % 2 == 1:
                return s[mid]
            return (s[mid - 1] + s[mid]) / 2.0
        return self._q[2]

    # -- internal P² helpers ------------------------------------------------

    def _init_p2(self) -> None:
        self._warmup.sort()
        self._q = list(self._warmup)
        self._n = [1, 2, 3, 4, 5]
        self._ns = [1.0, 2.0, 3.0, 4.0, 5.0]
        self._dn = [0.0, 0.25, 0.5, 0.75, 1.0]
        self._heights = True

    def _update_p2(self, x: float) -> None:
        q, n, ns, dn = self._q, self._n, self._ns, self._dn

        # Find cell k
        if x < q[0]:
            q[0] = x
            k = 0
        elif x < q[1]:
            k = 0
        elif x < q[2]:
            k = 1
        elif x < q[3]:
            k = 2
        elif x <= q[4]:
            k = 3
        else:
            q[4] = x
            k = 3

        for i in range(k + 1, 5):
            n[i] += 1
        for i in range(5):
            ns[i] += dn[i]

        for i in (1, 2, 3):
            d = ns[i] - n[i]
            if (d >= 1.0 and n[i + 1] - n[i] > 1) or (
                d <= -1.0 and n[i - 1] - n[i] < -1
            ):
                d_sign = 1 if d > 0 else -1
                qn = self._parabolic(i, d_sign)
                if q[i - 1] < qn < q[i + 1]:
                    q[i] = qn
                else:
                    q[i] = q[i] + d_sign * (q[i + d_sign] - q[i]) / (
                        n[i + d_sign] - n[i]
                    )
                n[i] += d_sign

    def _parabolic(self, i: int, d: int) -> float:
        q, n = self._q, self._n
        ni = n[i]
        qi = q[i]
        nim1 = n[i - 1]
        nip1 = n[i + 1]
        return qi + (d / (nip1 - nim1)) * (
            (ni - nim1 + d) * (q[i + 1] - qi) / (nip1 - ni)
            + (nip1 - ni - d) * (qi - q[i - 1]) / (ni - nim1)
        )


# ---------------------------------------------------------------------------
# Scalar accumulator with Welford mean/std + P² median
# ---------------------------------------------------------------------------


class _ScalarAccumulator:
    """Online scalar statistics accumulator.

    Tracks count, mean, variance (Welford), min, max, total, last, and an
    approximate median (P²).  Supports warm-up skip: the first *skip_target*
    samples are silently discarded.
    """

    __slots__ = (
        "count",
        "mean",
        "m2",
        "min_val",
        "max_val",
        "total",
        "last",
        "skipped",
        "skip_target",
        "_median",
    )

    def __init__(self, skip_target: int = 0) -> None:
        self.count: int = 0
        self.mean: float = 0.0
        self.m2: float = 0.0
        self.min_val: float = float("inf")
        self.max_val: float = float("-inf")
        self.total: float = 0.0
        self.last: float = 0.0
        self.skipped: int = 0
        self.skip_target: int = skip_target
        self._median = _P2MedianEstimator()

    def record(self, value: float) -> None:
        """Record a single scalar sample (after warm-up skip)."""
        if self.skipped < self.skip_target:
            self.skipped += 1
            return

        self.last = value
        self.count += 1
        self.total += value

        # Welford online update
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

        if value < self.min_val:
            self.min_val = value
        if value > self.max_val:
            self.max_val = value

        self._median.add(value)

    def snapshot(self) -> dict:
        """Return a snapshot dict of all accumulated stats."""
        if self.count == 0:
            return {
                "count": 0,
                "mean_ms": 0.0,
                "median_ms": 0.0,
                "std_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
                "total_ms": 0.0,
                "last_ms": 0.0,
            }
        std = math.sqrt(self.m2 / self.count) if self.count > 0 else 0.0
        return {
            "count": self.count,
            "mean_ms": self.mean,
            "median_ms": self._median.median,
            "std_ms": std,
            "min_ms": self.min_val,
            "max_ms": self.max_val,
            "total_ms": self.total,
            "last_ms": self.last,
        }


# ---------------------------------------------------------------------------
# Pipeline profiler
# ---------------------------------------------------------------------------


class PipelineProfiler:
    """Thread-safe per-node runtime profiler for pipeline execution.

    Accumulates timing samples keyed by ``(stage_value, node_name)`` and
    exposes frozen :class:`NodeProfilingStats` snapshots.

    Parameters
    ----------
    skip_first_n : int
        Number of initial samples to discard per accumulator key (warm-up skip).
        Must be >= 0.
    """

    def __init__(self, skip_first_n: int = 0) -> None:
        if skip_first_n < 0:
            raise ValueError(f"skip_first_n must be >= 0, got {skip_first_n}")
        self._skip_first_n = skip_first_n
        self._accumulators: dict[tuple[str, str], _ScalarAccumulator] = {}
        self._lock = threading.Lock()

    def record(self, stage_value: str, node_name: str, elapsed_ms: float) -> None:
        """Record a timing sample for the given stage and node."""
        key = (stage_value, node_name)
        with self._lock:
            acc = self._accumulators.get(key)
            if acc is None:
                acc = _ScalarAccumulator(skip_target=self._skip_first_n)
                self._accumulators[key] = acc
            acc.record(elapsed_ms)

    def reset(self) -> None:
        """Clear all accumulated statistics."""
        with self._lock:
            self._accumulators.clear()

    def snapshot(self, stage: str | None = None) -> list[NodeProfilingStats]:
        """Return a list of frozen stats for all (or filtered) accumulators.

        Parameters
        ----------
        stage : str or None
            If provided, only return stats for this stage value
            (e.g. ``"inference"``).  ``None`` returns all stages.
        """
        with self._lock:
            results: list[NodeProfilingStats] = []
            for (stage_val, node_name), acc in self._accumulators.items():
                if stage is not None and stage_val != stage:
                    continue
                snap = acc.snapshot()
                results.append(
                    NodeProfilingStats(
                        node_name=node_name,
                        stage=stage_val,
                        **snap,
                    )
                )
        return results


__all__ = ["PipelineProfiler"]
