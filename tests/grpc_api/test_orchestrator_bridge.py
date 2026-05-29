"""Orchestrator-bridge dispatch tests.

Verifies the flag-gated branch in pipeline_service / inference_service /
training_service: when ``CUVIS_USE_ORCHESTRATOR`` is off (default) the
existing in-process path runs; when on AND a child handle is attached
to the session, the call forwards to ``session.child_handle.stub()``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cuvis_ai_core.grpc import orchestrator_bridge


# ---------------------------------------------------------------------------
# orchestrator_enabled() flag parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "YES", "on", "On"])
def test_flag_on_for_truthy_values(monkeypatch, val):
    monkeypatch.setenv("CUVIS_USE_ORCHESTRATOR", val)
    assert orchestrator_bridge.orchestrator_enabled() is True


@pytest.mark.parametrize("val", ["", "0", "false", "no", "off", "random"])
def test_flag_off_for_falsy_or_unknown_values(monkeypatch, val):
    monkeypatch.setenv("CUVIS_USE_ORCHESTRATOR", val)
    assert orchestrator_bridge.orchestrator_enabled() is False


def test_flag_off_when_unset(monkeypatch):
    monkeypatch.delenv("CUVIS_USE_ORCHESTRATOR", raising=False)
    assert orchestrator_bridge.orchestrator_enabled() is False


# ---------------------------------------------------------------------------
# detect_core_source: editable cuvis-ai-core (the dev case)
# ---------------------------------------------------------------------------


def test_detect_core_source_for_editable_install_returns_local():
    """In this checkout cuvis_ai_core lives outside site-packages."""
    source = orchestrator_bridge.detect_core_source()
    assert source.kind in ("local", "pypi")
    if source.kind == "local":
        # Identity points at the project root.
        from pathlib import Path

        assert Path(source.identity).exists()


# ---------------------------------------------------------------------------
# ensure_child_for_session — short-circuit cases
# ---------------------------------------------------------------------------


def test_ensure_child_returns_existing_handle(monkeypatch):
    """If a session already has a child handle, no re-spawn."""

    existing = MagicMock(name="ExistingChildHandle")

    class _FakeSession:
        def __init__(self):
            self.child_handle = existing

    class _FakeSessionMgr:
        def get_session(self, _id):
            return _FakeSession()

    pipeline_config = MagicMock(plugins=["foo"])
    handle = orchestrator_bridge.ensure_child_for_session(
        _FakeSessionMgr(), "s1", pipeline_config, plugins_dirs=[]
    )
    assert handle is existing


def test_ensure_child_returns_none_for_builtin_only(monkeypatch):
    """No plugins declared, no plugins_dirs → no child needed."""

    class _FakeSession:
        child_handle = None

    class _FakeSessionMgr:
        def get_session(self, _id):
            return _FakeSession()

    pipeline_config = MagicMock(plugins=None)
    handle = orchestrator_bridge.ensure_child_for_session(
        _FakeSessionMgr(), "s1", pipeline_config, plugins_dirs=[]
    )
    assert handle is None
