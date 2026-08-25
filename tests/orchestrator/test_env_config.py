"""Tests for the shared warn-and-default numeric env parser."""

from __future__ import annotations

from cuvis_ai_core.orchestrator.env_config import number_from_env


def test_unset_returns_default(monkeypatch):
    monkeypatch.delenv("CUVIS_TEST_NUMBER", raising=False)
    assert number_from_env("CUVIS_TEST_NUMBER", 7, cast=int) == 7


def test_valid_int_and_float_are_used(monkeypatch):
    monkeypatch.setenv("CUVIS_TEST_NUMBER", "3")
    assert number_from_env("CUVIS_TEST_NUMBER", 7, cast=int) == 3
    monkeypatch.setenv("CUVIS_TEST_NUMBER", "2.5")
    assert number_from_env("CUVIS_TEST_NUMBER", 7.0, cast=float) == 2.5


def test_garbage_falls_back(monkeypatch):
    monkeypatch.setenv("CUVIS_TEST_NUMBER", "ten")
    assert number_from_env("CUVIS_TEST_NUMBER", 7, cast=int) == 7


def test_negative_falls_back(monkeypatch):
    monkeypatch.setenv("CUVIS_TEST_NUMBER", "-1")
    assert number_from_env("CUVIS_TEST_NUMBER", 7, cast=int) == 7


def test_zero_allowed_by_default_disallowed_for_timeouts(monkeypatch):
    monkeypatch.setenv("CUVIS_TEST_NUMBER", "0")
    # Cache knobs: 0 is the conventional "disabled" value.
    assert number_from_env("CUVIS_TEST_NUMBER", 7, cast=int) == 0
    # Timeouts: 0 makes no sense, fall back.
    assert (
        number_from_env("CUVIS_TEST_NUMBER", 7.0, cast=float, allow_zero=False) == 7.0
    )
