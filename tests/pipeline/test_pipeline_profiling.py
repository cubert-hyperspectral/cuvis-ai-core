"""Integration tests for CuvisPipeline profiling API.

Tests profiling through the pipeline's public API using lightweight inline
Node subclasses (same pattern as test_graph_routing.py).
"""

from __future__ import annotations

import time

import pytest
import torch

from cuvis_ai_core.node import Node
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.pipeline import PortSpec


# ---------------------------------------------------------------------------
# Lightweight test nodes
# ---------------------------------------------------------------------------


class _SlowNode(Node):
    """Node that sleeps briefly to produce measurable timing."""

    INPUT_SPECS = {"x": PortSpec(dtype=torch.float32, shape=(-1,))}
    OUTPUT_SPECS = {"y": PortSpec(dtype=torch.float32, shape=(-1,))}

    def __init__(self, delay_ms: float = 1.0, **kwargs) -> None:
        self._delay = delay_ms / 1000.0
        super().__init__(**kwargs)

    def forward(self, x, **kwargs):
        time.sleep(self._delay)
        return {"y": x * 2}


class _IdentityNode(Node):
    INPUT_SPECS = {"x": PortSpec(dtype=torch.float32, shape=(-1,))}
    OUTPUT_SPECS = {"y": PortSpec(dtype=torch.float32, shape=(-1,))}

    def forward(self, x, **kwargs):
        return {"y": x}


class _FailingNode(Node):
    INPUT_SPECS = {"x": PortSpec(dtype=torch.float32, shape=(-1,))}
    OUTPUT_SPECS = {"y": PortSpec(dtype=torch.float32, shape=(-1,))}

    def forward(self, x, **kwargs):
        raise RuntimeError("intentional failure")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_single_node_pipeline(node_cls, **node_kwargs):
    """Create a pipeline with a single node (source → sink pattern)."""
    pipeline = CuvisPipeline("test")
    n1 = node_cls(**node_kwargs)
    n2 = _IdentityNode()
    pipeline.connect(n1.outputs.y, n2.x)
    return pipeline, n1, n2


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestProfilingDisabledByDefault:
    def test_profiling_off_by_default(self) -> None:
        pipeline = CuvisPipeline("test")
        assert pipeline.profiling_enabled is False
        assert pipeline.get_profiling_summary() == []


class TestProfilingBasic:
    def test_enable_and_accumulate(self) -> None:
        """Enabling profiling should accumulate count across forwards."""
        pipeline = CuvisPipeline("test")
        n1 = _IdentityNode()
        n2 = _IdentityNode()
        pipeline.connect(n1.outputs.y, n2.x)

        pipeline.set_profiling(enabled=True)
        assert pipeline.profiling_enabled is True

        batch = {"x": torch.tensor([1.0, 2.0])}
        pipeline.forward(batch=batch, stage=ExecutionStage.INFERENCE)
        pipeline.forward(batch=batch, stage=ExecutionStage.INFERENCE)

        stats = pipeline.get_profiling_summary()
        assert len(stats) == 2  # Two nodes

        for s in stats:
            assert s.count == 2
            assert s.mean_ms >= 0
            assert s.std_ms >= 0
            assert s.min_ms >= 0
            assert s.max_ms >= s.min_ms
            assert s.total_ms >= 0
            assert s.last_ms >= 0
            assert s.stage == "inference"

    def test_stats_timing_is_positive(self) -> None:
        """Slow node should produce measurable timing."""
        pipeline, n1, n2 = _make_single_node_pipeline(_SlowNode, delay_ms=5.0)

        pipeline.set_profiling(enabled=True)
        pipeline.forward(batch={"x": torch.tensor([1.0])})

        stats = pipeline.get_profiling_summary()
        assert len(stats) == 2  # SlowNode + IdentityNode
        slow_stat = [s for s in stats if s.node_name == n1.name][0]
        assert slow_stat.mean_ms >= 1.0  # Should be at least 1ms


class TestSkipFirstN:
    def test_skip_discards_initial_calls(self) -> None:
        pipeline, n1, n2 = _make_single_node_pipeline(_IdentityNode)

        pipeline.set_profiling(enabled=True, skip_first_n=2)
        batch = {"x": torch.tensor([1.0])}

        for _ in range(5):
            pipeline.forward(batch=batch)

        stats = pipeline.get_profiling_summary()
        for s in stats:
            assert s.count == 3  # 5 - 2 skipped per node

    def test_negative_skip_raises(self) -> None:
        pipeline = CuvisPipeline("test")
        with pytest.raises(ValueError, match="skip_first_n"):
            pipeline.set_profiling(enabled=True, skip_first_n=-1)


class TestStageSeparation:
    def test_stages_accumulated_separately(self) -> None:
        """Stats should be keyed by (stage, node_name)."""
        pipeline = CuvisPipeline("test")
        n1 = _IdentityNode(execution_stages={ExecutionStage.ALWAYS})
        n2 = _IdentityNode(execution_stages={ExecutionStage.ALWAYS})
        pipeline.connect(n1.outputs.y, n2.x)

        pipeline.set_profiling(enabled=True)
        batch = {"x": torch.tensor([1.0])}

        pipeline.forward(batch=batch, stage=ExecutionStage.INFERENCE)
        pipeline.forward(batch=batch, stage=ExecutionStage.TRAIN)

        all_stats = pipeline.get_profiling_summary()
        assert len(all_stats) == 4  # 2 nodes x 2 stages

        inf_stats = pipeline.get_profiling_summary(stage=ExecutionStage.INFERENCE)
        assert len(inf_stats) == 2
        assert all(s.stage == "inference" for s in inf_stats)

        train_stats = pipeline.get_profiling_summary(stage=ExecutionStage.TRAIN)
        assert len(train_stats) == 2
        assert all(s.stage == "train" for s in train_stats)


class TestUptoNode:
    def test_upto_profiles_only_executed_nodes(self) -> None:
        """Only ancestors of upto_node should be profiled."""
        pipeline = CuvisPipeline("test")
        n1 = _IdentityNode()
        n2 = _IdentityNode()
        n3 = _IdentityNode()
        pipeline.connect(n1.outputs.y, n2.x)
        pipeline.connect(n2.outputs.y, n3.x)

        pipeline.set_profiling(enabled=True)
        batch = {"x": torch.tensor([1.0])}

        # Execute only ancestors of n3 (which are n1 and n2, not n3 itself)
        pipeline.forward(batch=batch, upto_node=n3)

        stats = pipeline.get_profiling_summary()
        profiled_names = {s.node_name for s in stats}
        assert n1.name in profiled_names
        assert n2.name in profiled_names
        assert n3.name not in profiled_names


class TestReset:
    def test_reset_clears_stats(self) -> None:
        pipeline, n1, n2 = _make_single_node_pipeline(_IdentityNode)

        pipeline.set_profiling(enabled=True)
        pipeline.forward(batch={"x": torch.tensor([1.0])})
        assert len(pipeline.get_profiling_summary()) > 0

        pipeline.reset_profiling()
        assert pipeline.get_profiling_summary() == []

    def test_set_profiling_with_reset(self) -> None:
        pipeline, n1, n2 = _make_single_node_pipeline(_IdentityNode)

        pipeline.set_profiling(enabled=True)
        pipeline.forward(batch={"x": torch.tensor([1.0])})

        pipeline.set_profiling(enabled=True, reset=True)
        assert pipeline.get_profiling_summary() == []


class TestCudaSync:
    def test_cuda_sync_no_error_on_cpu(self) -> None:
        """synchronize_cuda=True should not error on CPU-only nodes."""
        pipeline, n1, n2 = _make_single_node_pipeline(_IdentityNode)

        pipeline.set_profiling(enabled=True, synchronize_cuda=True)
        # Should not raise — sync is skipped when no CUDA device detected
        pipeline.forward(batch={"x": torch.tensor([1.0])})

        stats = pipeline.get_profiling_summary()
        assert len(stats) == 2


class TestNodeException:
    def test_exception_no_sample_recorded(self) -> None:
        """If node raises, no profiling sample should be recorded."""
        pipeline = CuvisPipeline("test")
        n1 = _FailingNode()
        n2 = _IdentityNode()
        pipeline.connect(n1.outputs.y, n2.x)

        pipeline.set_profiling(enabled=True)

        with pytest.raises(RuntimeError, match="intentional failure"):
            pipeline.forward(batch={"x": torch.tensor([1.0])})

        # n1 failed so its sample was not recorded; n2 never ran
        stats = pipeline.get_profiling_summary()
        assert stats == []


class TestLifecycle:
    def test_disable_reenable_preserves_stats(self) -> None:
        """Disabling then re-enabling should preserve accumulated stats."""
        pipeline, n1, n2 = _make_single_node_pipeline(_IdentityNode)

        pipeline.set_profiling(enabled=True)
        pipeline.forward(batch={"x": torch.tensor([1.0])})
        for s in pipeline.get_profiling_summary():
            assert s.count == 1

        # Disable
        pipeline.set_profiling(enabled=False)
        assert pipeline.profiling_enabled is False

        # Forward without profiling — should not accumulate
        pipeline.forward(batch={"x": torch.tensor([1.0])})

        # Re-enable — stats should still be there from before
        pipeline.set_profiling(enabled=True)
        for s in pipeline.get_profiling_summary():
            assert s.count == 1

        # One more forward — should now be 2
        pipeline.forward(batch={"x": torch.tensor([1.0])})
        for s in pipeline.get_profiling_summary():
            assert s.count == 2


class TestFormattingAndSaveGuards:
    def test_format_profiling_summary_includes_header_metadata(self) -> None:
        pipeline, n1, n2 = _make_single_node_pipeline(_IdentityNode)

        pipeline.set_profiling(enabled=True, skip_first_n=2)
        for _ in range(3):
            pipeline.forward(batch={"x": torch.tensor([1.0])})

        table = pipeline.format_profiling_summary(total_frames=3)

        assert "Profiling Summary (3 frames, skip_first_n=2)" in table
        assert n1.name in table
        assert n2.name in table
        assert "TOTAL" in table

    def test_save_to_file_requires_weights_for_optimizer_or_scheduler(self, tmp_path):
        pipeline = CuvisPipeline("test")

        with pytest.raises(
            ValueError,
            match="include_optimizer/include_scheduler require save_weights=True",
        ):
            pipeline.save_to_file(
                tmp_path / "pipeline.yaml",
                save_weights=False,
                include_optimizer=True,
            )
