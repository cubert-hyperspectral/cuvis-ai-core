"""Tests for gRPC service error paths (invalid session / no pipeline).

Covers get_session_or_error, require_pipeline, and @grpc_handler call sites
in PipelineService, TrainRunService, InferenceService, TrainingService,
and PluginService.
"""

from unittest.mock import Mock

import grpc

from cuvis_ai_core.grpc.error_handling import grpc_handler
from cuvis_ai_core.grpc.inference_service import InferenceService
from cuvis_ai_core.grpc.pipeline_service import PipelineService
from cuvis_ai_core.grpc.plugin_service import PluginService
from cuvis_ai_core.grpc.training_service import TrainingService
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

    def test_save_pipeline_invalid_session(self):
        request = cuvis_ai_pb2.SavePipelineRequest(
            session_id="nonexistent",
            pipeline_path="/some/path.yaml",
        )
        response = self.service.save_pipeline(request, self.ctx)
        assert response.success is False
        self.ctx.set_code.assert_called_with(grpc.StatusCode.NOT_FOUND)

    def test_save_pipeline_success(self, tmp_path):
        """Exercise save_pipeline success path (covers __version__ import)."""
        session_id = self.session_manager.create_session()
        session = self.session_manager.get_session(session_id)
        session.pipeline = Mock()
        session.pipeline.save_to_file = Mock()

        save_path = str(tmp_path / "test_pipeline.yaml")
        request = cuvis_ai_pb2.SavePipelineRequest(
            session_id=session_id,
            pipeline_path=save_path,
        )
        response = self.service.save_pipeline(request, self.ctx)
        assert response.success is True
        session.pipeline.save_to_file.assert_called_once()

    def test_load_pipeline_invalid_session(self):
        request = cuvis_ai_pb2.LoadPipelineRequest(
            session_id="nonexistent",
        )
        response = self.service.load_pipeline(request, self.ctx)
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


class TestPluginServiceErrors:
    """Test PluginService error paths for all methods."""

    def setup_method(self):
        self.session_manager = SessionManager()
        self.service = PluginService(self.session_manager)
        self.ctx = Mock()

    def teardown_method(self):
        for sid in list(self.session_manager._sessions.keys()):
            self.session_manager.close_session(sid)

    def test_load_plugins_invalid_session(self):
        request = cuvis_ai_pb2.LoadPluginsRequest(
            session_id="nonexistent",
        )
        response = self.service.load_plugins(request, self.ctx)
        assert len(response.loaded_plugins) == 0
        self.ctx.set_code.assert_called_with(grpc.StatusCode.NOT_FOUND)

    def test_list_loaded_plugins_invalid_session(self):
        request = cuvis_ai_pb2.ListLoadedPluginsRequest(
            session_id="nonexistent",
        )
        response = self.service.list_loaded_plugins(request, self.ctx)
        assert len(response.plugins) == 0
        self.ctx.set_code.assert_called_with(grpc.StatusCode.NOT_FOUND)

    def test_get_plugin_info_invalid_session(self):
        request = cuvis_ai_pb2.GetPluginInfoRequest(
            session_id="nonexistent",
            plugin_name="any",
        )
        response = self.service.get_plugin_info(request, self.ctx)
        assert response.plugin.name == ""
        self.ctx.set_code.assert_called_with(grpc.StatusCode.NOT_FOUND)

    def test_list_available_nodes_invalid_session(self):
        request = cuvis_ai_pb2.ListAvailableNodesRequest(
            session_id="nonexistent",
        )
        response = self.service.list_available_nodes(request, self.ctx)
        assert len(response.nodes) == 0
        self.ctx.set_code.assert_called_with(grpc.StatusCode.NOT_FOUND)


class TestInferenceServiceErrors:
    """Test InferenceService invalid-session and no-pipeline paths."""

    def setup_method(self):
        self.session_manager = SessionManager()
        self.service = InferenceService(self.session_manager)
        self.ctx = Mock()

    def teardown_method(self):
        for sid in list(self.session_manager._sessions.keys()):
            self.session_manager.close_session(sid)

    def test_inference_invalid_session(self):
        request = cuvis_ai_pb2.InferenceRequest(
            session_id="nonexistent",
        )
        response = self.service.inference(request, self.ctx)
        assert response == cuvis_ai_pb2.InferenceResponse()
        self.ctx.set_code.assert_called_with(grpc.StatusCode.NOT_FOUND)

    def test_inference_no_pipeline(self):
        session_id = self.session_manager.create_session()
        request = cuvis_ai_pb2.InferenceRequest(
            session_id=session_id,
        )
        response = self.service.inference(request, self.ctx)
        assert response == cuvis_ai_pb2.InferenceResponse()
        self.ctx.set_code.assert_called_with(grpc.StatusCode.FAILED_PRECONDITION)


class TestTrainingServiceErrors:
    """Test TrainingService invalid-session and no-pipeline paths."""

    def setup_method(self):
        self.session_manager = SessionManager()
        self.service = TrainingService(self.session_manager)
        self.ctx = Mock()

    def teardown_method(self):
        for sid in list(self.session_manager._sessions.keys()):
            self.session_manager.close_session(sid)

    def test_train_invalid_session(self):
        request = cuvis_ai_pb2.TrainRequest(
            session_id="nonexistent",
        )
        # train() is a generator — consume it to trigger the error path
        responses = list(self.service.train(request, self.ctx))
        assert responses == []
        self.ctx.set_code.assert_called_with(grpc.StatusCode.NOT_FOUND)

    def test_train_no_pipeline(self):
        session_id = self.session_manager.create_session()
        request = cuvis_ai_pb2.TrainRequest(
            session_id=session_id,
        )
        responses = list(self.service.train(request, self.ctx))
        assert responses == []
        self.ctx.set_code.assert_called_with(grpc.StatusCode.FAILED_PRECONDITION)


class TestGrpcHandlerDecorator:
    """Test the @grpc_handler decorator in isolation."""

    def _make_ctx(self):
        return Mock(spec=["set_code", "set_details"])

    def test_success_passthrough(self):
        """Successful method calls pass through unchanged."""

        class FakeService:
            @grpc_handler()
            def my_method(self, request, context) -> cuvis_ai_pb2.CreateSessionResponse:
                return cuvis_ai_pb2.CreateSessionResponse(session_id="ok")

        ctx = self._make_ctx()
        resp = FakeService().my_method("request", ctx)
        assert resp.session_id == "ok"
        ctx.set_code.assert_not_called()

    def test_value_error_maps_to_invalid_argument(self):
        """ValueError → INVALID_ARGUMENT."""

        class FakeService:
            @grpc_handler()
            def my_method(self, request, context) -> cuvis_ai_pb2.CreateSessionResponse:
                raise ValueError("bad input")

        ctx = self._make_ctx()
        resp = FakeService().my_method("request", ctx)
        assert resp == cuvis_ai_pb2.CreateSessionResponse()
        ctx.set_code.assert_called_with(grpc.StatusCode.INVALID_ARGUMENT)
        ctx.set_details.assert_called_with("bad input")

    def test_file_not_found_maps_to_not_found(self):
        """FileNotFoundError → NOT_FOUND."""

        class FakeService:
            @grpc_handler()
            def my_method(self, request, context) -> cuvis_ai_pb2.CreateSessionResponse:
                raise FileNotFoundError("missing.yaml")

        ctx = self._make_ctx()
        resp = FakeService().my_method("request", ctx)
        assert resp == cuvis_ai_pb2.CreateSessionResponse()
        ctx.set_code.assert_called_with(grpc.StatusCode.NOT_FOUND)

    def test_generic_exception_maps_to_internal(self):
        """Exception (catch-all) → INTERNAL with error_prefix."""

        class FakeService:
            @grpc_handler("Custom prefix")
            def my_method(self, request, context) -> cuvis_ai_pb2.CreateSessionResponse:
                raise RuntimeError("unexpected")

        ctx = self._make_ctx()
        resp = FakeService().my_method("request", ctx)
        assert resp == cuvis_ai_pb2.CreateSessionResponse()
        ctx.set_code.assert_called_with(grpc.StatusCode.INTERNAL)
        ctx.set_details.assert_called_with("Custom prefix: unexpected")

    def test_generic_exception_no_prefix(self):
        """Exception (catch-all) → INTERNAL without error_prefix."""

        class FakeService:
            @grpc_handler()
            def my_method(self, request, context) -> cuvis_ai_pb2.CreateSessionResponse:
                raise RuntimeError("boom")

        ctx = self._make_ctx()
        FakeService().my_method("request", ctx)
        ctx.set_details.assert_called_with("boom")
