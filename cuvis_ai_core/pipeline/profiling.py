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

import functools
import math
import statistics
import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import torch
from loguru import logger
from torch._ops import HigherOrderOperator
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten, tree_map

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
# FLOP counting
# ---------------------------------------------------------------------------

_aten = torch.ops.aten
_SDPA_OPS = (
    "scaled_dot_product_attention",
    "_scaled_dot_product_flash_attention_for_cpu",
    "_scaled_dot_product_flash_attention",
    "_scaled_dot_product_efficient_attention",
    "_scaled_dot_product_cudnn_attention",
)
_DECOMPOSITION_CACHE_SIZE = 4096

# FLOPs of one composite call, keyed by op and argument shapes. Decomposition runs
# on meta copies, so the count never depends on the real device or values.
_decomposition_cache: OrderedDict[tuple, int] = OrderedDict()
_decomposition_cache_lock = threading.Lock()
_logged_decomposition_failures: set[str] = set()


def _sdpa_flop(query, key, value, *args, out_shape=None, **kwargs) -> int:
    """FLOPs of ``softmax(q k^T) v`` for any batch layout, grouped-query included.

    With ``q (..., L, E)``, ``k (..., S, E)`` and ``v (..., S, Ev)``, every query
    batch entry does one ``q k^T`` and one ``attn v`` product. torch's own
    ``sdpa_flop_count`` requires 4-D inputs with equal head counts.
    """
    *batch, length, embed = query
    source, value_embed = key[-2], value[-1]
    return 2 * math.prod(batch) * length * source * (embed + value_embed)


@functools.cache
def _flop_formulas() -> dict[Any, Callable[..., int]] | None:
    """torch's FLOP formulas plus the attention formula; ``None`` when unavailable.

    Imported lazily: importing ``torch.utils.flop_counter`` logs a "triton not
    found" line on machines without triton, which only a counted forward pays.
    """
    try:
        from torch.utils.flop_counter import flop_registry, shape_wrapper
    except ImportError as exc:
        logger.warning("FLOP counting is off: torch.utils.flop_counter lacks {}", exc)
        return None
    formulas = dict(flop_registry)
    sdpa = shape_wrapper(_sdpa_flop)
    for name in _SDPA_OPS:
        packet = getattr(_aten, name, None)
        if packet is not None:
            formulas[packet] = sdpa
    return formulas


def flop_counting_available() -> bool:
    """Whether this torch build has the FLOP formulas the counter needs."""
    return _flop_formulas() is not None


def _to_meta(value: Any) -> Any:
    """A meta copy of a tensor; anything else as is."""
    return value.to("meta") if isinstance(value, torch.Tensor) else value


def _leaf_key(leaf: Any) -> Any:
    """One argument as a cache key part: shape and dtype for a tensor."""
    if isinstance(leaf, torch.Tensor):
        return (tuple(leaf.shape), leaf.dtype)
    if isinstance(leaf, (bool, int, float, str, type(None), torch.dtype)):
        return leaf
    return repr(leaf)


def _call_key(func: Any, args: tuple, kwargs: dict) -> tuple:
    """The decomposition cache key of one call: op, argument layout and leaves."""
    leaves, spec = tree_flatten((args, kwargs))
    return (func, str(spec), tuple(_leaf_key(leaf) for leaf in leaves))


def _cached_flops(key: tuple) -> int | None:
    """The cached count of a composite call, or ``None`` on a miss."""
    with _decomposition_cache_lock:
        flops = _decomposition_cache.get(key)
        if flops is not None:
            _decomposition_cache.move_to_end(key)
        return flops


def _cache_flops(key: tuple, flops: int) -> None:
    """Cache the count of a composite call, evicting the least recently used."""
    with _decomposition_cache_lock:
        _decomposition_cache[key] = flops
        if len(_decomposition_cache) > _DECOMPOSITION_CACHE_SIZE:
            _decomposition_cache.popitem(last=False)


class NodeFlopCounter(TorchDispatchMode):
    """Count the FLOPs of the ops one node forward runs, without changing them.

    Every op runs exactly as it would without the mode; the mode adds torch's
    formula count for matrix multiplication, convolution and attention ops.
    Composite ops that reach the mode undecomposed (``aten.linear``,
    ``aten.matmul``, ``aten.conv2d`` or ``aten.einsum`` under
    ``torch.inference_mode``) are counted by decomposing them on meta copies of
    their arguments. Higher-order operators such as ``torch.cond`` run
    unchanged and uncounted. Work in other threads, compiled kernels and engines
    outside torch (TensorRT, ONNX) is not seen.
    """

    supports_higher_order_operators = True

    def __init__(self) -> None:
        super().__init__()
        self._formulas = _flop_formulas() or {}
        self._flops = 0
        self._on_meta = False
        #: The first op whose formula failed; the count is void once set.
        self.failed_op: str | None = None

    @classmethod
    def ignore_compile_internals(cls) -> bool:
        """Let Dynamo compile under the mode.

        Dynamo skips a frame while a dispatch mode is on the stack, and a
        compiled node whose first call happened under such a mode never compiles
        afterwards.
        """
        return True

    @property
    def total(self) -> int | None:
        """The FLOP count, or ``None`` when a formula failed."""
        return None if self.failed_op is not None else self._flops

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        """Run the op as is, then add its FLOPs when it has a formula or decomposes."""
        kwargs = kwargs or {}
        if isinstance(func, HigherOrderOperator):
            return func(*args, **kwargs)
        formula = self._formulas.get(func._overloadpacket)
        if self._on_meta:
            return self._run_meta(func, formula, args, kwargs)
        out = func(*args, **kwargs)
        if formula is not None:
            self._add(formula, func, args, kwargs, out)
        elif func.has_kernel_for_dispatch_key(
            torch._C.DispatchKey.CompositeImplicitAutograd
        ):
            self._flops += self._composite_flops(func, args, kwargs)
        return out

    def _add(self, formula, func, args, kwargs, out) -> None:
        """Add one formula count; a raising or invalid formula voids the total."""
        try:
            flops = formula(*args, **kwargs, out_val=out)
        except Exception:
            flops = None
        if isinstance(flops, int) and not isinstance(flops, bool) and flops >= 0:
            self._flops += flops
        elif self.failed_op is None:
            self.failed_op = str(func)

    def _run_meta(self, func, formula, args, kwargs):
        """Inside a meta decomposition: decompose composites, count the rest."""
        if formula is None:
            # The mode is off inside its own handler; re-enter it so the ops of a
            # nested composite (``aten.matmul`` inside ``aten.linear``) are seen.
            with self:
                out = func.decompose(*args, **kwargs)
            if out is not NotImplemented:
                return out
        out = func(*args, **kwargs)
        if formula is not None:
            self._add(formula, func, args, kwargs, out)
        return out

    def _composite_flops(self, func, args, kwargs) -> int:
        """FLOPs of one composite call, from its decomposition on meta copies."""
        key = _call_key(func, args, kwargs)
        cached = _cached_flops(key)
        if cached is not None:
            return cached
        before, failed_before = self._flops, self.failed_op
        self._on_meta = True
        try:
            meta_args, meta_kwargs = tree_map(_to_meta, (args, kwargs))
            # A composite that draws random numbers without naming a device must
            # not advance the real generator, so the meta pass gets its own. (A
            # ``torch.device("meta")`` context would re-dispatch the composite
            # instead of decomposing it.)
            with torch.random.fork_rng(devices=[]), torch.no_grad(), self:
                func.decompose(*meta_args, **meta_kwargs)
            flops = self._flops - before
        except Exception as exc:
            name = str(func)
            if name not in _logged_decomposition_failures:
                _logged_decomposition_failures.add(name)
                logger.debug("FLOP count of {} skipped: {}", name, exc)
            flops = 0
        finally:
            self._on_meta = False
            self._flops = before
        if self.failed_op is failed_before:
            _cache_flops(key, flops)
        return flops


class FlopClaim:
    """One (stage, node) FLOP count in flight: its counter and where it goes."""

    __slots__ = ("counter", "_profiler", "_token", "_key")

    def __init__(
        self, profiler: PipelineProfiler, token: int, key: tuple[str, str]
    ) -> None:
        self.counter = NodeFlopCounter()
        self._profiler = profiler
        self._token = token
        self._key = key

    def record(self) -> None:
        """Store the count in the profiler it was claimed from (dropped after a reset)."""
        flops = self.counter.total
        if flops is None:
            logger.warning(
                "FLOP count of node '{}' ({}) failed at {}; its GFLOP stays empty",
                self._key[1],
                self._key[0],
                self.counter.failed_op,
            )
        self._profiler.record_flops(self._token, self._key, flops)

    def release(self) -> None:
        """Hand the claim back after the counted forward raised; the next one counts."""
        self._profiler.release_flop_claim(self._token, self._key)


# ---------------------------------------------------------------------------
# Pipeline profiler
# ---------------------------------------------------------------------------


class PipelineProfiler:
    """Thread-safe per-node runtime profiler for pipeline execution.

    Accumulates timing samples keyed by ``(stage_value, node_name)`` and
    exposes frozen :class:`NodeProfilingStats` snapshots. Each key also gets one
    FLOP count per reset, from the forward that claims it first.

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
        self._flops: dict[tuple[str, str], int | None] = {}
        self._flop_claims: set[tuple[str, str]] = set()
        # Bumped by reset(): a claim taken before it records nothing.
        self._generation = 0
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

    def claim_flop_count(self, stage_value: str, node_name: str) -> FlopClaim | None:
        """Claim the FLOP count of this key; ``None`` when it is already claimed.

        Only the first forward per key after a reset counts, so two concurrent
        forwards never both run under a counter.
        """
        key = (stage_value, node_name)
        if key in self._flop_claims:  # warm keys skip the lock
            return None
        with self._lock:
            if key in self._flop_claims:
                return None
            self._flop_claims.add(key)
            token = self._generation
        return FlopClaim(self, token, key)

    def record_flops(self, token: int, key: tuple[str, str], flops: int | None) -> None:
        """Store a claimed count, unless a reset happened since the claim."""
        with self._lock:
            if token == self._generation:
                self._flops[key] = flops

    def release_flop_claim(self, token: int, key: tuple[str, str]) -> None:
        """Free a claim whose forward raised, unless a reset already freed it."""
        with self._lock:
            if token == self._generation:
                self._flop_claims.discard(key)

    def reset(self) -> None:
        """Clear all accumulated statistics and FLOP counts."""
        with self._lock:
            self._accumulators.clear()
            self._flops.clear()
            self._flop_claims.clear()
            self._generation += 1

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
                        flops_per_call=self._flops.get((stage_val, node_name)),
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
        parts.extend(_node_block(stats, cuda_synchronized))
    if data_rows or first_batches:
        parts.extend(_data_block(data_rows, first_batches, cuda_synchronized))
    return "\n".join(parts)


def _column_header(label: str, flops: bool = False) -> str:
    """The column header line, with ``label`` over the name column."""
    header = (
        f"{label:<40} {'Stage':<12} {'Count':>5} {'Mean(ms)':>10} "
        f"{'Std(ms)':>10} {'Min(ms)':>10} {'Max(ms)':>10} "
        f"{'Median(ms)':>10} {'Total(s)':>10}"
    )
    if flops:
        header += f" {'GFLOP':>10} {'GFLOP/s':>10}"
    return header


def _format_row(s: NodeProfilingStats, flops: bool = False) -> str:
    """One table row for a node or a data-loading step."""
    row = (
        f"{s.node_name:<40} {s.stage:<12} {s.count:>5} {s.mean_ms:>10.2f} "
        f"{s.std_ms:>10.2f} {s.min_ms:>10.2f} {s.max_ms:>10.2f} "
        f"{s.median_ms:>10.2f} {s.total_ms / 1000:>10.3f}"
    )
    if flops:
        gflop = "" if s.flops_per_call is None else f"{s.flops_per_call / 1e9:.3f}"
        rate = "" if s.gflop_per_s is None else f"{s.gflop_per_s:.2f}"
        row = f"{row} {gflop:>10} {rate:>10}".rstrip()
    return row


def _per_forward_flops(stats: list[NodeProfilingStats], frame_ms: float) -> str:
    """The FLOP part of the per-frame line: the counts summed over the nodes."""
    counts = [s.flops_per_call for s in stats if s.flops_per_call is not None]
    total = sum(counts)
    text = f", {total / 1e9:.3f} GFLOP per forward"
    if frame_ms > 0:
        text += f" ({total / (frame_ms * 1e6):.2f} GFLOP/s)"
    missing = len(stats) - len(counts)
    if missing:
        text += f", {missing} node{'s' if missing > 1 else ''} not counted"
    return text


def _node_block(
    stats: list[NodeProfilingStats], cuda_synchronized: bool = False
) -> list[str]:
    """The node rows sorted by total time, the TOTAL footer and the per-frame line.

    The GFLOP columns appear when any row has a FLOP count. The per-frame line
    sums the counts only when every row belongs to one stage.
    """
    sorted_stats = sorted(stats, key=lambda s: s.total_ms, reverse=True)
    flops = any(s.flops_per_call is not None for s in stats)
    col_header = _column_header("Node", flops)
    separator = "-" * len(col_header)
    parts = [col_header, separator]

    total_pipeline_ms = 0.0
    for s in sorted_stats:
        total_pipeline_ms += s.total_ms
        parts.append(_format_row(s, flops))

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
        line = f"Average per-frame pipeline time: {avg_frame_ms:.2f} ms ({fps:.1f} FPS)"
        if flops and len({s.stage for s in stats}) == 1:
            line += _per_forward_flops(stats, avg_frame_ms)
        parts.append(line)
    if flops and not cuda_synchronized:
        parts.append(
            "Node times are host wall time (no CUDA synchronize): "
            "GFLOP/s of GPU nodes reads high."
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
    order = {DATA_LOAD: 0, TO_DEVICE: 1, BATCH_LOOP: 2}
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
    "FlopClaim",
    "NodeFlopCounter",
    "PipelineProfiler",
    "TO_DEVICE",
    "flop_counting_available",
    "format_profiling_table",
]
