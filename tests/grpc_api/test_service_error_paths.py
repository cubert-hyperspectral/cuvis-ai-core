"""Tests for gRPC service error paths (invalid session / no pipeline).

Covers get_session_or_error, require_pipeline, and @grpc_handler call sites
in PipelineService, TrainRunService, InferenceService, TrainingService,
and PluginService.
"""

import json
from unittest.mock import Mock, patch

import grpc
import yaml

from cuvis_ai_core.grpc.error_handling import grpc_handler
from cuvis_ai_core.grpc.inference_service import InferenceService
from cuvis_ai_core.grpc.pipeline_service import PipelineService
from cuvis_ai_core.grpc.plugin_service import PluginService
from cuvis_ai_core.grpc.training_service import TrainingService
from cuvis_ai_core.grpc.trainrun_service import TrainRunService
from cuvis_ai_core.grpc.session_manager import SessionManager
from cuvis_ai_core.grpc.v1 import cuvis_ai_pb2
from cuvis_ai_core.training.config import DataConfig, TrainingConfig, TrainRunConfig


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

    def test_context_as_keyword_argument(self):
        """Context passed as kwarg is extracted correctly (line 131)."""

        class FakeService:
            @grpc_handler()
            def my_method(self, request, context) -> cuvis_ai_pb2.CreateSessionResponse:
                raise ValueError("kwarg test")

        ctx = self._make_ctx()
        FakeService().my_method("request", context=ctx)
        ctx.set_code.assert_called_with(grpc.StatusCode.INVALID_ARGUMENT)
        ctx.set_details.assert_called_with("kwarg test")


class TestPipelineServiceValidation:
    """Test PipelineService request validation paths."""

    def setup_method(self):
        self.session_manager = SessionManager()
        self.service = PipelineService(self.session_manager)
        self.ctx = Mock()

    def teardown_method(self):
        for sid in list(self.session_manager._sessions.keys()):
            self.session_manager.close_session(sid)

    def test_load_pipeline_weights_missing_path_and_bytes(self):
        """Neither weights_path nor weights_bytes → INVALID_ARGUMENT (lines 68-69)."""
        session_id = self.session_manager.create_session()
        session = self.session_manager.get_session(session_id)
        session.pipeline = Mock()

        request = cuvis_ai_pb2.LoadPipelineWeightsRequest(session_id=session_id)
        response = self.service.load_pipeline_weights(request, self.ctx)
        assert response.success is False
        self.ctx.set_code.assert_called_with(grpc.StatusCode.INVALID_ARGUMENT)

    def test_set_train_run_config_no_pipeline_no_config(self):
        """No pipeline in session AND no pipeline in trainrun → FAILED_PRECONDITION (lines 108-113)."""
        session_id = self.session_manager.create_session()

        # Build a minimal trainrun config without pipeline section
        from cuvis_ai_core.training.config import (
            DataConfig,
            TrainingConfig,
            TrainRunConfig,
        )

        trainrun = TrainRunConfig(
            name="test",
            pipeline=None,
            data=DataConfig(cu3s_file_path=""),
            training=TrainingConfig(),
            loss_nodes=[],
            metric_nodes=[],
            freeze_nodes=[],
            unfreeze_nodes=[],
            output_dir=".",
            tags={},
        )
        config_bytes = json.dumps(trainrun.to_dict()).encode()

        request = cuvis_ai_pb2.SetTrainRunConfigRequest(
            session_id=session_id,
            config=cuvis_ai_pb2.TrainRunConfig(config_bytes=config_bytes),
        )
        response = self.service.set_train_run_config(request, self.ctx)
        assert response.success is False
        self.ctx.set_code.assert_called_with(grpc.StatusCode.FAILED_PRECONDITION)


class TestTrainRunServiceValidation:
    """Test TrainRunService error and edge case paths."""

    def setup_method(self):
        self.session_manager = SessionManager()
        self.service = TrainRunService(self.session_manager)
        self.ctx = Mock()

    def teardown_method(self):
        for sid in list(self.session_manager._sessions.keys()):
            self.session_manager.close_session(sid)

    def test_save_train_run_no_pipeline_config(self, tmp_path):
        """No pipeline and no pipeline config → FAILED_PRECONDITION (lines 50-54)."""
        session_id = self.session_manager.create_session()

        request = cuvis_ai_pb2.SaveTrainRunRequest(
            session_id=session_id,
            trainrun_path=str(tmp_path / "trainrun.yaml"),
        )
        response = self.service.save_train_run(request, self.ctx)
        assert response.success is False
        self.ctx.set_code.assert_called_with(grpc.StatusCode.FAILED_PRECONDITION)

    def test_save_train_run_weights_failure(self, tmp_path):
        """Weights save error is non-fatal (lines 108-110)."""
        session_id = self.session_manager.create_session()
        session = self.session_manager.get_session(session_id)

        # Set up a mock pipeline with nodes
        mock_node = Mock()
        mock_node.name = "node1"
        mock_node.state_dict.return_value = {"w": "data"}
        mock_pipeline = Mock()
        mock_pipeline.nodes.return_value = [mock_node]
        mock_pipeline.name = "test"
        session.pipeline = mock_pipeline

        # Set trainrun_config directly to skip pipeline config construction
        session.trainrun_config = TrainRunConfig(
            name="test",
            pipeline=None,
            data=DataConfig(cu3s_file_path=""),
            training=TrainingConfig(),
            loss_nodes=[],
            metric_nodes=[],
            freeze_nodes=[],
            unfreeze_nodes=[],
            output_dir=".",
            tags={},
        )

        request = cuvis_ai_pb2.SaveTrainRunRequest(
            session_id=session_id,
            trainrun_path=str(tmp_path / "trainrun.yaml"),
            save_weights=True,
        )

        # torch is imported inside the function; patch it at module level
        with patch("torch.save", side_effect=RuntimeError("disk full")):
            response = self.service.save_train_run(request, self.ctx)

        # Should still succeed (weights failure is non-fatal)
        assert response.success is True
        self.ctx.set_details.assert_called()

    def test_restore_train_run_hydra_defaults(self, tmp_path):
        """Config with Hydra defaults raises ValueError (line 140)."""
        trainrun_file = tmp_path / "trainrun.yaml"
        trainrun_file.write_text(
            yaml.dump({"defaults": [{"optimizer": "adam"}], "name": "test"})
        )

        request = cuvis_ai_pb2.RestoreTrainRunRequest(
            trainrun_path=str(trainrun_file),
        )
        self.service.restore_train_run(request, self.ctx)
        self.ctx.set_code.assert_called_with(grpc.StatusCode.INVALID_ARGUMENT)

    def test_restore_train_run_invalid_pipeline_config_file(self, tmp_path):
        """Pipeline config file that isn't a mapping → ValueError (line 159)."""
        # Create a pipeline config file that is a list, not a dict
        pipeline_file = tmp_path / "pipeline.yaml"
        pipeline_file.write_text(yaml.dump(["not", "a", "dict"]))

        trainrun_file = tmp_path / "trainrun.yaml"
        trainrun_file.write_text(
            yaml.dump(
                {
                    "name": "test",
                    "pipeline": {"config_path": str(pipeline_file)},
                    "data": {"cu3s_file_path": ""},
                    "training": {},
                }
            )
        )

        with patch(
            "cuvis_ai_core.grpc.trainrun_service.helpers.resolve_pipeline_path",
            return_value=pipeline_file,
        ):
            request = cuvis_ai_pb2.RestoreTrainRunRequest(
                trainrun_path=str(trainrun_file),
            )
            self.service.restore_train_run(request, self.ctx)
        self.ctx.set_code.assert_called_with(grpc.StatusCode.INVALID_ARGUMENT)

    def test_restore_train_run_missing_weights(self, tmp_path):
        """Weights path that doesn't exist → FileNotFoundError (line 167)."""
        pipeline_file = tmp_path / "pipeline.yaml"
        pipeline_file.write_text(
            yaml.dump(
                {
                    "metadata": {"name": "test"},
                    "nodes": [],
                    "connections": [],
                }
            )
        )

        trainrun_file = tmp_path / "trainrun.yaml"
        trainrun_file.write_text(
            yaml.dump(
                {
                    "name": "test",
                    "pipeline": {
                        "config_path": str(pipeline_file),
                        "weights_path": str(tmp_path / "nonexistent.pt"),
                    },
                    "data": {"cu3s_file_path": ""},
                    "training": {},
                }
            )
        )

        with patch(
            "cuvis_ai_core.grpc.trainrun_service.helpers.resolve_pipeline_path",
            return_value=pipeline_file,
        ):
            request = cuvis_ai_pb2.RestoreTrainRunRequest(
                trainrun_path=str(trainrun_file),
            )
            self.service.restore_train_run(request, self.ctx)
        self.ctx.set_code.assert_called_with(grpc.StatusCode.NOT_FOUND)

    def test_restore_train_run_missing_pipeline_section(self, tmp_path):
        """Trainrun config without pipeline → ValueError (line 177)."""
        trainrun_file = tmp_path / "trainrun.yaml"
        trainrun_file.write_text(
            yaml.dump(
                {
                    "name": "test",
                    "data": {"cu3s_file_path": ""},
                    "training": {},
                }
            )
        )

        request = cuvis_ai_pb2.RestoreTrainRunRequest(
            trainrun_path=str(trainrun_file),
        )
        self.service.restore_train_run(request, self.ctx)
        self.ctx.set_code.assert_called_with(grpc.StatusCode.INVALID_ARGUMENT)

    def test_restore_train_run_pipeline_yaml_pattern(self, tmp_path):
        """Detect _pipeline.yaml companion file (line 186)."""

        # Create companion pipeline file
        pipeline_file = tmp_path / "test_pipeline.yaml"
        pipeline_file.write_text(
            yaml.dump(
                {
                    "metadata": {"name": "test"},
                    "nodes": [],
                    "connections": [],
                }
            )
        )

        trainrun_file = tmp_path / "test.yaml"
        trainrun_file.write_text(
            yaml.dump(
                {
                    "name": "test",
                    "pipeline": {
                        "metadata": {"name": "test"},
                        "nodes": [],
                        "connections": [],
                    },
                    "data": {"cu3s_file_path": ""},
                    "training": {},
                    "loss_nodes": [],
                    "metric_nodes": [],
                    "freeze_nodes": [],
                    "unfreeze_nodes": [],
                    "output_dir": ".",
                    "tags": {},
                }
            )
        )

        mock_pipeline = Mock()
        mock_pipeline.name = "test"
        with patch(
            "cuvis_ai_core.pipeline.pipeline.CuvisPipeline.load_pipeline",
            return_value=mock_pipeline,
        ):
            request = cuvis_ai_pb2.RestoreTrainRunRequest(
                trainrun_path=str(trainrun_file),
            )
            response = self.service.restore_train_run(request, self.ctx)

        assert response.session_id != ""


class TestTrainingServiceStatus:
    """Test TrainingService get_train_status paths."""

    def setup_method(self):
        self.session_manager = SessionManager()
        self.service = TrainingService(self.session_manager)
        self.ctx = Mock()

    def teardown_method(self):
        for sid in list(self.session_manager._sessions.keys()):
            self.session_manager.close_session(sid)

    def test_get_train_status_invalid_session(self):
        """Invalid session → empty response (line 134)."""
        request = cuvis_ai_pb2.GetTrainStatusRequest(session_id="nonexistent")
        response = self.service.get_train_status(request, self.ctx)
        assert response == cuvis_ai_pb2.GetTrainStatusResponse()
        self.ctx.set_code.assert_called_with(grpc.StatusCode.NOT_FOUND)

    def test_get_train_status_with_trainer(self):
        """Session with trainer returns COMPLETE status (line 140)."""
        session_id = self.session_manager.create_session()
        session = self.session_manager.get_session(session_id)
        session.trainer = Mock()

        request = cuvis_ai_pb2.GetTrainStatusRequest(session_id=session_id)
        response = self.service.get_train_status(request, self.ctx)
        assert response.latest_progress.status == cuvis_ai_pb2.TRAIN_STATUS_COMPLETE


class TestPluginServiceClearCache:
    """Test PluginService clear_plugin_cache path."""

    def setup_method(self):
        self.session_manager = SessionManager()
        self.service = PluginService(self.session_manager)
        self.ctx = Mock()

    def test_clear_cache_with_plugin_name(self):
        """Clear cache for specific plugin (line 355)."""
        request = cuvis_ai_pb2.ClearPluginCacheRequest(plugin_name="test_plugin")
        response = self.service.clear_plugin_cache(request, self.ctx)
        assert response.cleared_count >= 0
