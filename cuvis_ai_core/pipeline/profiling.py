"""Pipeline node runtime profiling with online statistics.

This module implements a lightweight scalar Welford accumulator with P² approximate
median, purpose-built for profiling ``node.forward()`` durations.

**Why not reuse** ``cuvis_ai.utils.welford.WelfordAccumulator`` **from cuvis-ai?**

1. That class is an ``nn.Module`` with float64 torch buffers, designed for
   multi-feature (N, C) tensor statistics during statistical node initialization.
2. Profiling needs a pure-Python *scalar* accumulator (one float per sample) with
   P² approximate median, ``threading.Lock`` thread safety, min/max/total/last
   tracking, and skip-first-N warm-up — none of which exist in the tensor class.
3. Importing from cuvis-ai into cuvis-ai-core would invert the one-directional
   dependency (cuvis-ai builds on core), which is architecturally wrong.
"""

from __future__ import annotations

import math
import statistics
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

    __slots__ = ("_warmup", "_q", "_n", "_ns", "_dn", "_active")

    def __init__(self) -> None:
        self._warmup: list[float] = []
        # Once we transition out of warm-up, these hold P² state.
        self._q: list[float] = []  # marker heights
        self._n: list[int] = []  # marker positions
        self._ns: list[float] = []  # desired marker positions
        self._dn: list[float] = []  # desired position increments
        self._active: bool = False  # True once P² is active

    def add(self, x: float) -> None:
        if not self._active:
            self._warmup.append(x)
            if len(self._warmup) == 5:
                self._init_p2()
            return
        self._update_p2(x)

    @property
    def median(self) -> float:
        """Exact median during the warm-up buffer, P² estimate afterwards."""
        if not self._active:
            return statistics.median(self._warmup) if self._warmup else 0.0
        return self._q[2]

    # -- internal P² helpers ------------------------------------------------

    def _init_p2(self) -> None:
        self._warmup.sort()
        self._q = list(self._warmup)
        self._n = [1, 2, 3, 4, 5]
        self._ns = [1.0, 2.0, 3.0, 4.0, 5.0]
        self._dn = [0.0, 0.25, 0.5, 0.75, 1.0]
        self._active = True

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
        return {
            "count": self.count,
            "mean_ms": self.mean,
            "median_ms": self._median.median,
            "std_ms": math.sqrt(self.m2 / self.count),
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

    @property
    def skip_first_n(self) -> int:
        """Number of initial samples discarded per accumulator key."""
        return self._skip_first_n

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


# ---------------------------------------------------------------------------
# Formatted table output
# ---------------------------------------------------------------------------


#: Step names a batch loop records through ``CuvisPipeline.iter_profiled_batches``.
DATA_LOAD = "data_load"
TO_DEVICE = "to_device"
BATCH_LOOP = "batch_loop"
DATA_STEPS = (DATA_LOAD, TO_DEVICE, BATCH_LOOP)


def format_profiling_table(
    stats: list[NodeProfilingStats],
    *,
    total_frames: int | None = None,
    skip_first_n: int = 0,
    data_stats: list[NodeProfilingStats] | None = None,
    first_batch_ms: dict[str, list[float]] | None = None,
    cuda_synchronized: bool = False,
) -> str:
    """Format profiling stats as a pretty-printed text table.

    The node table is followed by a "Data loading" block when ``data_stats`` or
    ``first_batch_ms`` carry samples: one row per step (``data_load``, ``to_device``,
    ``batch_loop``) and stage, the first-batch line per stage, and a per-batch line
    from the ``batch_loop`` mean. Rows with a zero count (every sample skipped) are
    left out. Nothing is summed across stages or between the two blocks.

    Parameters
    ----------
    stats : list[NodeProfilingStats]
        Profiling stats as returned by ``PipelineProfiler.snapshot()`` or
        ``CuvisPipeline.get_profiling_summary()``.
    total_frames : int or None
        Total number of frames/batches processed (shown in the header).
    skip_first_n : int
        Number of warm-up samples that were skipped (shown in the header).
    data_stats : list[NodeProfilingStats] or None
        Data-loading stats as returned by ``CuvisPipeline.get_data_profiling_summary()``.
    first_batch_ms : dict[str, list[float]] or None
        Per stage value, the first fetch of every profiled pass (milliseconds); it is
        excluded from the rows because it carries one-time setup.
    cuda_synchronized : bool
        Whether the loop samples were taken after a CUDA synchronize; the per-batch
        line names host wall time otherwise.

    Returns
    -------
    str
        Multi-line formatted table ready for logging or printing.
    """
    data_rows = [s for s in (data_stats or []) if s.count > 0]
    first_batches = {k: v for k, v in (first_batch_ms or {}).items() if v}
    if not stats and not data_rows and not first_batches:
        return "No profiling data collected."

    # Header
    parts: list[str] = []
    header_meta = "Profiling Summary"
    if total_frames is not None:
        header_meta += f" ({total_frames} frames"
        if skip_first_n > 0:
            header_meta += f", skip_first_n={skip_first_n}"
        header_meta += ")"
    elif skip_first_n > 0:
        header_meta += f" (skip_first_n={skip_first_n})"
    parts.append(header_meta)
    if stats:
        parts.extend(_node_block(stats))
    if data_rows or first_batches:
        parts.extend(_data_block(data_rows, first_batches, cuda_synchronized))
    return "\n".join(parts)


def _column_header(label: str) -> str:
    """The column header line, with ``label`` over the name column."""
    return (
        f"{label:<40} {'Stage':<12} {'Count':>5} {'Mean(ms)':>10} "
        f"{'Std(ms)':>10} {'Min(ms)':>10} {'Max(ms)':>10} "
        f"{'Median(ms)':>10} {'Total(s)':>10}"
    )


def _format_row(s: NodeProfilingStats) -> str:
    """One table row for a node or a data-loading step."""
    return (
        f"{s.node_name:<40} {s.stage:<12} {s.count:>5} {s.mean_ms:>10.2f} "
        f"{s.std_ms:>10.2f} {s.min_ms:>10.2f} {s.max_ms:>10.2f} "
        f"{s.median_ms:>10.2f} {s.total_ms / 1000:>10.3f}"
    )


def _node_block(stats: list[NodeProfilingStats]) -> list[str]:
    """The node rows sorted by total time, the TOTAL footer and the per-frame line."""
    sorted_stats = sorted(stats, key=lambda s: s.total_ms, reverse=True)
    col_header = _column_header("Node")
    separator = "-" * len(col_header)
    parts = [col_header, separator]

    total_pipeline_ms = 0.0
    for s in sorted_stats:
        total_pipeline_ms += s.total_ms
        parts.append(_format_row(s))

    parts.append(separator)
    parts.append(
        f"{'TOTAL':<40} {'':12} {'':>5} {'':>10} {'':>10} {'':>10} "
        f"{'':>10} {'':>10} {total_pipeline_ms / 1000:>10.3f}"
    )

    # FPS line
    frame_count = sorted_stats[0].count
    if frame_count > 0:
        avg_frame_ms = total_pipeline_ms / frame_count
        fps = 1000.0 / avg_frame_ms if avg_frame_ms > 0 else 0.0
        parts.append(
            f"Average per-frame pipeline time: {avg_frame_ms:.2f} ms ({fps:.1f} FPS)"
        )
    return parts


def _data_block(
    rows: list[NodeProfilingStats],
    first_batches: dict[str, list[float]],
    cuda_synchronized: bool,
) -> list[str]:
    """The data-loading rows per stage, then the first-batch and per-batch lines."""
    col_header = _column_header("Step")
    separator = "-" * len(col_header)
    parts = ["", "Data loading (outside the nodes)", col_header, separator]
    order = {name: i for i, name in enumerate(DATA_STEPS)}
    stages = sorted({s.stage for s in rows} | set(first_batches))
    for stage in stages:
        stage_rows = sorted(
            (s for s in rows if s.stage == stage),
            key=lambda s: order.get(s.node_name, len(order)),
        )
        parts.extend(_format_row(s) for s in stage_rows)
    parts.append(separator)

    timing = "CUDA-synchronized" if cuda_synchronized else "host wall time"
    for stage in stages:
        values = first_batches.get(stage)
        if values:
            if len(values) == 1:
                text = f"{values[0] / 1000:.2f} s"
            else:
                text = (
                    f"mean {sum(values) / len(values) / 1000:.2f} s over {len(values)} "
                    f"passes (min {min(values) / 1000:.2f} s, max {max(values) / 1000:.2f} s)"
                )
            parts.append(
                f"First batch data load ({stage}, excluded from the rows): {text}"
            )
        loop = next(
            (s for s in rows if s.stage == stage and s.node_name == BATCH_LOOP), None
        )
        if loop is not None and loop.mean_ms > 0:
            parts.append(
                f"Time per batch ({stage}, {BATCH_LOOP}, {timing}): "
                f"{loop.mean_ms:.2f} ms ({1000.0 / loop.mean_ms:.1f} batches/s)"
            )
    return parts


__all__ = [
    "BATCH_LOOP",
    "DATA_LOAD",
    "DATA_STEPS",
    "PipelineProfiler",
    "TO_DEVICE",
    "format_profiling_table",
]
