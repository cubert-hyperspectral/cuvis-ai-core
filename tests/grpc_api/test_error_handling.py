"""Unit tests for gRPC error handling helpers."""

from unittest.mock import Mock

import grpc

from cuvis_ai_core.grpc.error_handling import (
    get_session_and_pipeline,
    get_session_or_error,
    require_pipeline,
)
from cuvis_ai_core.grpc.session_manager import SessionManager
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline


class TestGetSessionOrError:
    """Tests for get_session_or_error helper."""

    def setup_method(self):
        self.session_manager = SessionManager()
        self.context = Mock(spec=grpc.ServicerContext)

    def teardown_method(self):
        for sid in list(self.session_manager._sessions.keys()):
            self.session_manager.close_session(sid)

    def test_returns_session_on_valid_id(self):
        session_id = self.session_manager.create_session()

        result = get_session_or_error(self.session_manager, session_id, self.context)

        assert result is not None
        assert result.session_id == session_id
        self.context.set_code.assert_not_called()
        self.context.set_details.assert_not_called()

    def test_sets_not_found_on_invalid_id(self):
        result = get_session_or_error(self.session_manager, "nonexistent", self.context)

        assert result is None
        self.context.set_code.assert_called_once_with(grpc.StatusCode.NOT_FOUND)
        self.context.set_details.assert_called_once()
        assert "nonexistent" in self.context.set_details.call_args[0][0]


class TestRequirePipeline:
    """Tests for require_pipeline helper."""

    def test_returns_true_when_pipeline_exists(self):
        session = Mock()
        session.pipeline = Mock()
        context = Mock(spec=grpc.ServicerContext)

        assert require_pipeline(session, context) is True
        context.set_code.assert_not_called()

    def test_returns_false_and_sets_precondition_when_no_pipeline(self):
        session = Mock()
        session.pipeline = None
        context = Mock(spec=grpc.ServicerContext)

        assert require_pipeline(session, context) is False
        context.set_code.assert_called_once_with(grpc.StatusCode.FAILED_PRECONDITION)
        context.set_details.assert_called_once()
        assert "No pipeline" in context.set_details.call_args[0][0]


class TestGetSessionAndPipeline:
    """get_session_and_pipeline: both checks, the same statuses as the two steps."""

    def setup_method(self):
        self.session_manager = SessionManager()
        self.context = Mock(spec=grpc.ServicerContext)

    def teardown_method(self):
        for sid in list(self.session_manager._sessions.keys()):
            self.session_manager.close_session(sid)

    def test_missing_session_is_not_found(self):
        resolved = get_session_and_pipeline(
            self.session_manager, "nonexistent", self.context
        )

        assert resolved is None
        self.context.set_code.assert_called_once_with(grpc.StatusCode.NOT_FOUND)
        assert "nonexistent" in self.context.set_details.call_args[0][0]

    def test_session_without_pipeline_is_failed_precondition(self):
        session_id = self.session_manager.create_session()

        resolved = get_session_and_pipeline(
            self.session_manager, session_id, self.context
        )

        assert resolved is None
        self.context.set_code.assert_called_once_with(
            grpc.StatusCode.FAILED_PRECONDITION
        )
        self.context.set_details.assert_called_once_with(
            "No pipeline is available for this session. Build pipeline first."
        )

    def test_returns_the_session_and_its_pipeline(self):
        session_id = self.session_manager.create_session()
        session = self.session_manager.get_session(session_id)
        session.pipeline = CuvisPipeline("guard")

        resolved = get_session_and_pipeline(
            self.session_manager, session_id, self.context
        )

        assert resolved == (session, session.pipeline)
        self.context.set_code.assert_not_called()
        self.context.set_details.assert_not_called()
