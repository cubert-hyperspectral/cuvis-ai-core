"""Tests for NodeFlopCounter: the FLOPs of one forward, with every op run unchanged."""

from __future__ import annotations

import math
import sys
import threading

import pytest
import torch
import torch.nn.functional as F
from loguru import logger
from torch.utils._python_dispatch import _get_current_dispatch_mode_stack

from cuvis_ai_core.pipeline import profiling
from cuvis_ai_core.pipeline.profiling import NodeFlopCounter

aten = torch.ops.aten

GRAD_MODES = {
    "grad": torch.enable_grad,
    "no_grad": torch.no_grad,
    "inference_mode": torch.inference_mode,
}

# A composite custom op that draws from the global RNG without naming a device.
_TEST_LIB = torch.library.Library("cuvis_flop_test", "DEF")
_TEST_LIB.define("noisy(Tensor x) -> Tensor")
_TEST_LIB.impl(
    "noisy", lambda x: x + torch.rand(x.shape[-1]), "CompositeImplicitAutograd"
)


@pytest.fixture(autouse=True)
def _fresh_decomposition_cache():
    profiling._decomposition_cache.clear()
    yield
    profiling._decomposition_cache.clear()


@pytest.fixture
def warnings_sink():
    messages: list[str] = []
    sink_id = logger.add(
        lambda m: messages.append(m.record["message"]), level="WARNING"
    )
    yield messages
    logger.remove(sink_id)


def _counted(fn, grad_mode: str = "no_grad"):
    """Run ``fn`` under the counter; return the counter and the output."""
    with GRAD_MODES[grad_mode]():
        counter = NodeFlopCounter()
        with counter:
            out = fn()
    return counter, out


def _plain(fn, grad_mode: str = "no_grad"):
    with GRAD_MODES[grad_mode]():
        return fn()


torch.manual_seed(0)
_X = torch.randn(2, 3, 16, 16)
_CONV_W = torch.randn(4, 3, 3, 3)
_CONV_B = torch.randn(4)
_A = torch.randn(8, 16)
_W = torch.randn(32, 16)
_B = torch.randn(32)
_BA = torch.randn(2, 8, 16)
_BB = torch.randn(2, 16, 4)
_Q = torch.randn(2, 4, 8, 16)

CASES = {
    "conv2d": (
        lambda: F.conv2d(_X, _CONV_W, _CONV_B, padding=1),
        2 * 2 * 4 * 16 * 16 * 3 * 3 * 3,
    ),
    "linear": (lambda: F.linear(_A, _W, _B), 2 * 8 * 16 * 32),
    "matmul": (lambda: _A @ _W.T, 2 * 8 * 16 * 32),
    "einsum": (lambda: torch.einsum("bij,bjk->bik", _BA, _BB), 2 * 2 * 8 * 16 * 4),
    "sdpa": (
        lambda: F.scaled_dot_product_attention(_Q, _Q, _Q),
        2 * (2 * 8 * 8 * 8 * 16),
    ),
    "elementwise": (lambda: (_X - _X.mean()) / _X.std(), 0),
    "bilinear": (
        lambda: F.interpolate(
            _X, scale_factor=0.5, mode="bilinear", align_corners=False
        ),
        0,
    ),
}


@pytest.mark.parametrize("grad_mode", list(GRAD_MODES))
@pytest.mark.parametrize("case", list(CASES))
def test_counts_match_the_formulas(case: str, grad_mode: str) -> None:
    """matmul, conv and attention are counted exactly; elementwise ops count nothing."""
    fn, expected = CASES[case]
    counter, _ = _counted(fn, grad_mode)
    assert counter.total == expected


@pytest.mark.parametrize("grad_mode", list(GRAD_MODES))
@pytest.mark.parametrize("case", list(CASES))
def test_outputs_equal_an_uncounted_run(case: str, grad_mode: str) -> None:
    """Every op runs as it would without the counter, bit for bit."""
    fn, _ = CASES[case]
    _, out = _counted(fn, grad_mode)
    assert torch.equal(out, _plain(fn, grad_mode))


@pytest.mark.parametrize("grad_mode", ["no_grad", "inference_mode"])
def test_state_is_mutated_exactly_once(grad_mode: str) -> None:
    with GRAD_MODES[grad_mode]():
        state = torch.zeros(3)
        weight = torch.ones(3, 3)
        with NodeFlopCounter():
            state.add_(weight @ torch.ones(3))
    assert torch.equal(state, torch.full((3,), 3.0))


def test_cond_with_a_false_predicate_returns_the_false_branch() -> None:
    x = torch.randn(4, 4)
    _, out = _counted(
        lambda: torch.cond(torch.tensor(False), lambda t: t + 1, lambda t: t - 1, (x,))
    )
    assert torch.equal(out, x - 1)


def test_work_in_a_worker_thread_runs_and_is_not_counted() -> None:
    a = torch.randn(8, 8)
    results: dict[str, torch.Tensor] = {}

    def work() -> None:
        results["y"] = a @ a

    def run() -> None:
        worker = threading.Thread(target=work)
        worker.start()
        worker.join()

    counter, _ = _counted(run)
    assert torch.equal(results["y"], _plain(lambda: a @ a))
    assert counter.total == 0


def test_the_node_exception_propagates_and_no_mode_stays_on_the_stack() -> None:
    class NodeFailure(RuntimeError):
        pass

    with pytest.raises(NodeFailure, match="node failed"):
        with torch.no_grad(), NodeFlopCounter():
            _ = _A @ _W.T
            raise NodeFailure("node failed")
    assert _get_current_dispatch_mode_stack() == []


def _spy_decompose(monkeypatch: pytest.MonkeyPatch, target, *, fail: bool = False):
    """Record calls of ``target.decompose``; raise from it when ``fail``."""
    calls: list[object] = []
    original = torch._ops.OpOverload.decompose

    def spy(self, *args, **kwargs):
        if self is target:
            calls.append(self)
            if fail:
                raise RuntimeError("no meta kernel")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(torch._ops.OpOverload, "decompose", spy)
    return calls


def test_identical_composite_calls_decompose_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _spy_decompose(monkeypatch, aten.linear.default)
    counter, _ = _counted(
        lambda: (F.linear(_A, _W), F.linear(_A, _W)), "inference_mode"
    )
    assert len(calls) == 1
    assert counter.total == 2 * (2 * 8 * 16 * 32)


def test_a_raising_decomposition_counts_zero_and_is_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _spy_decompose(monkeypatch, aten.linear.default, fail=True)
    counter, out = _counted(
        lambda: (F.linear(_A, _W), F.linear(_A, _W)), "inference_mode"
    )
    assert counter.total == 0
    assert len(calls) == 1
    assert torch.equal(out[0], _plain(lambda: F.linear(_A, _W), "inference_mode"))


@pytest.mark.parametrize("result", [RuntimeError("bad shape"), -1, 1.5])
def test_a_failing_formula_keeps_the_output_and_voids_the_count(
    monkeypatch: pytest.MonkeyPatch, result: object
) -> None:
    def formula(*args, **kwargs):
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(profiling, "_flop_formulas", lambda: {aten.mm: formula})
    counter, out = _counted(lambda: _A @ _W.T)
    assert torch.equal(out, _plain(lambda: _A @ _W.T))
    assert counter.total is None
    assert counter.failed_op is not None and "mm" in counter.failed_op


def test_compiled_module_first_called_under_the_counter_compiles_once() -> None:
    torch._dynamo.reset()
    compiles: list[object] = []

    def backend(gm, example_inputs):
        compiles.append(gm)
        return gm.forward

    module = torch.nn.Linear(16, 16)
    compiled = torch.compile(module, backend=backend)
    x = torch.randn(4, 16)
    try:
        with torch.no_grad():
            expected = module(x)
            with NodeFlopCounter():
                out = compiled(x)
            compiled(x)
            compiled(x)
    finally:
        torch._dynamo.reset()
    assert len(compiles) == 1
    assert torch.equal(out, expected)


def test_meta_decomposition_leaves_the_global_rng_alone() -> None:
    x = torch.randn(3, 4)
    op = torch.ops.cuvis_flop_test.noisy

    torch.manual_seed(7)
    expected = _plain(lambda: op(x), "inference_mode")
    expected_state = torch.get_rng_state()

    torch.manual_seed(7)
    _, out = _counted(lambda: op(x), "inference_mode")

    assert torch.equal(out, expected)
    assert torch.equal(torch.get_rng_state(), expected_state)


@pytest.mark.parametrize("grad_mode", ["no_grad", "inference_mode"])
@pytest.mark.parametrize(
    ("q_shape", "k_shape", "v_shape", "kwargs"),
    [
        ((2, 8, 16), (2, 6, 16), (2, 6, 12), {}),
        ((2, 4, 8, 16), (2, 4, 6, 16), (2, 4, 6, 12), {}),
        ((2, 3, 4, 8, 16), (2, 3, 4, 6, 16), (2, 3, 4, 6, 12), {}),
        ((2, 8, 8, 16), (2, 2, 6, 16), (2, 2, 6, 16), {"enable_gqa": True}),
    ],
    ids=["3d", "4d", "5d", "grouped-query"],
)
def test_attention_counts_any_layout(
    q_shape, k_shape, v_shape, kwargs, grad_mode
) -> None:
    q, k, v = torch.randn(q_shape), torch.randn(k_shape), torch.randn(v_shape)
    batch, length, embed = math.prod(q_shape[:-2]), q_shape[-2], q_shape[-1]
    source, value_embed = k_shape[-2], v_shape[-1]
    expected = 2 * batch * length * source * (embed + value_embed)

    counter, _ = _counted(
        lambda: F.scaled_dot_product_attention(q, k, v, **kwargs), grad_mode
    )
    assert counter.total == expected


def test_formula_table_holds_matmul_conv_and_attention() -> None:
    formulas = profiling._flop_formulas()
    assert formulas is not None
    assert aten.mm in formulas
    assert aten.convolution in formulas
    assert aten.scaled_dot_product_attention in formulas
    assert aten._scaled_dot_product_flash_attention_for_cpu in formulas


def test_missing_formula_table_turns_counting_off_with_one_warning(
    monkeypatch: pytest.MonkeyPatch, warnings_sink: list[str]
) -> None:
    profiling._flop_formulas.cache_clear()
    monkeypatch.setitem(sys.modules, "torch.utils.flop_counter", None)
    try:
        assert profiling._flop_formulas() is None
        assert profiling._flop_formulas() is None
        assert not profiling.flop_counting_available()
    finally:
        profiling._flop_formulas.cache_clear()
    assert len(warnings_sink) == 1
    assert "FLOP" in warnings_sink[0]
