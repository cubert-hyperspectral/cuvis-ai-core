"""Smoke tests for restore module imports."""


def test_restore_module_imports():
    """Verify restore module imports resolve correctly after schema migration."""
    from cuvis_ai_core.utils.restore import ExecutionStage, Context

    assert ExecutionStage is not None
    assert Context is not None
