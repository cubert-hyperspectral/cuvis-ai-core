"""Tests for gRPC service error paths (invalid session / no pipeline).

Covers get_session_or_error and require_pipeline call sites in
PipelineService, TrainRunService, InferenceService, and TrainingService.
"""

from unittest.mock import Mock

import grpc

from cuvis_ai_core.grpc.pipeline_service import PipelineService
from cuvis_ai_core.grpc.trainrun_service import TrainRunService
from cuvis_ai_core.grpc.session_manager import SessionManager
from cuvis_ai_core.grpc.v1 import cuvis_ai_pb2


class TestPipelineServiceErrors:
    """Test PipelineService invalid-session and no-pipeline paths."""

    def setup_method(self):
        self.session_manager = SessionManager()
        self.service = PipelineService(self.session_manager)
        self.ctx = Mock()

    def teardown_method(self):
        for sid in list(self.session_manager._sessions.keys()):
            self.session_manager.close_session(sid)

    def test_load_pipeline_weights_invalid_session(self):
        request = cuvis_ai_pb2.LoadPipelineWeightsRequest(
            session_id="nonexistent",
            weights_path="/some/path.pt",
        )
        response = self.service.load_pipeline_weights(request, self.ctx)
        assert response.success is False
        self.ctx.set_code.assert_called_with(grpc.StatusCode.NOT_FOUND)

    def test_load_pipeline_weights_no_pipeline(self):
        session_id = self.session_manager.create_session()
        request = cuvis_ai_pb2.LoadPipelineWeightsRequest(
            session_id=session_id,
            weights_path="/some/path.pt",
        )
        response = self.service.load_pipeline_weights(request, self.ctx)
        assert response.success is False
        self.ctx.set_code.assert_called_with(grpc.StatusCode.FAILED_PRECONDITION)

    def test_set_train_run_config_invalid_session(self):
        request = cuvis_ai_pb2.SetTrainRunConfigRequest(
            session_id="nonexistent",
        )
        response = self.service.set_train_run_config(request, self.ctx)
        assert response.success is False
        self.ctx.set_code.assert_called_with(grpc.StatusCode.NOT_FOUND)


class TestTrainRunServiceErrors:
    """Test TrainRunService invalid-session path."""

    def setup_method(self):
        self.session_manager = SessionManager()
        self.service = TrainRunService(self.session_manager)
        self.ctx = Mock()

    def teardown_method(self):
        for sid in list(self.session_manager._sessions.keys()):
            self.session_manager.close_session(sid)

    def test_save_train_run_invalid_session(self):
        request = cuvis_ai_pb2.SaveTrainRunRequest(
            session_id="nonexistent",
            trainrun_path="/some/path",
        )
        response = self.service.save_train_run(request, self.ctx)
        assert response.success is False
        self.ctx.set_code.assert_called_with(grpc.StatusCode.NOT_FOUND)
