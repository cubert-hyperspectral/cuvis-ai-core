"""Tests for gRPC service error paths (invalid session / no pipeline).

Covers get_session_or_error, require_pipeline, and @grpc_handler call sites
in PipelineService, TrainRunService, InferenceService, TrainingService,
and PluginService.
"""

import json
import pytest
from unittest.mock import Mock, patch

import grpc
import yaml
from loguru import logger

from cuvis_ai_core.grpc.error_handling import grpc_handler
from cuvis_ai_core.grpc.inference_service import InferenceService
from cuvis_ai_core.grpc.pipeline_service import PipelineService
from cuvis_ai_core.grpc.plugin_service import PluginService
from cuvis_ai_core.grpc.training_service import TrainingService, _run_outcome
from cuvis_ai_core.training.trainers import TrainingCancelled
from cuvis_ai_core.grpc.trainrun_service import TrainRunService
from cuvis_ai_core.grpc.session_manager import SessionManager
from cuvis_ai_core.grpc.session_service import SessionService
from cuvis_ai_core.grpc.v1 import cuvis_ai_pb2
from cuvis_ai_core.training import CalibrationOutcome
from cuvis_ai_core.training.post_fit import (
    EvaluationOutcome,
    EvaluationStatus,
    PostFitCallback,
    PostFitResult,
    TrainEvent,
    TrainHook,
)
from cuvis_ai_schemas.enums import ExecutionStage
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

    def test_save_pipeline_no_pipeline(self, tmp_path):
        """No pipeline -> FAILED_PRECONDITION before anything touches the disk. The
        child runtime is in this state after a LoadPipeline that failed inside it."""
        sid = self.session_manager.create_session()
        target = tmp_path / "out" / "p.yaml"
        resp = self.service.save_pipeline(
            cuvis_ai_pb2.SavePipelineRequest(session_id=sid, pipeline_path=str(target)),
            self.ctx,
        )
        assert resp.success is False
        self.ctx.set_code.assert_called_with(grpc.StatusCode.FAILED_PRECONDITION)
        assert "pipeline" in self.ctx.set_details.call_args.args[0].lower()
        assert not target.parent.exists()

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

    def test_load_plugin_invalid_session(self):
        request = cuvis_ai_pb2.LoadPluginRequest(
            session_id="nonexistent",
        )
        response = self.service.load_plugin(request, self.ctx)
        assert response.registered_plugin == ""
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

    def test_train_relative_splits_path_is_invalid_argument(self):
        """A relative splits_path is baseless on the wire Train path -> rejected.

        RestoreTrainRun resolves a relative splits_path against the trainrun dir;
        a raw session/request with a relative path has no base dir, so Train must
        reject it rather than resolve against the child's CWD.
        """
        from cuvis_ai_schemas.training.data import DataConfig, DataSplitConfig

        session_id = self.session_manager.create_session()
        session = self.session_manager.get_session(session_id)
        session.pipeline = Mock()  # pass require_pipeline
        session.data_config = DataConfig(
            data_module="npz_multi",
            splits=DataSplitConfig(splits_path="splits.json"),  # relative
        )
        request = cuvis_ai_pb2.TrainRequest(session_id=session_id)
        responses = list(self.service.train(request, self.ctx))
        assert responses == []
        self.ctx.set_code.assert_called_with(grpc.StatusCode.INVALID_ARGUMENT)
        assert any(
            "absolute" in str(c.args[0]).lower()
            for c in self.ctx.set_details.mock_calls
        )


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


class TestSessionServiceErrors:
    """CloseSession keeps two answers apart: an unknown session is NOT_FOUND, a teardown
    failure is INTERNAL with the "Failed to close session:" prefix. A FileNotFoundError
    from the teardown must not fall through to ``@grpc_handler``, whose NOT_FOUND mapping
    is the code CloseSession uses for an unknown session."""

    def _service(self, exc):
        manager = Mock(spec=SessionManager)
        manager.close_session.side_effect = exc
        return SessionService(manager)

    def test_unknown_session_is_not_found(self):
        ctx = Mock()
        resp = self._service(ValueError("Session s missing")).close_session(
            cuvis_ai_pb2.CloseSessionRequest(session_id="s"), ctx
        )
        assert resp.success is False
        ctx.set_code.assert_called_with(grpc.StatusCode.NOT_FOUND)
        ctx.set_details.assert_called_with("Session s missing")

    def test_teardown_failure_is_internal_with_prefix(self):
        ctx = Mock()
        exc = FileNotFoundError("[Errno 2] No such file or directory: 'run.lock'")
        resp = self._service(exc).close_session(
            cuvis_ai_pb2.CloseSessionRequest(session_id="s"), ctx
        )
        assert resp.success is False
        ctx.set_code.assert_called_with(grpc.StatusCode.INTERNAL)
        detail = ctx.set_details.call_args.args[0]
        assert detail.startswith("Failed to close session: ")
        assert "run.lock" in detail


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
            data=DataConfig(),
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
            data=DataConfig(),
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

        # the service imports torch at module level, so patch torch.save itself
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

    def test_restore_train_run_rejects_inline_pipeline(self, tmp_path):
        """An inline pipeline mapping is rejected at parse → INVALID_ARGUMENT."""
        trainrun_file = tmp_path / "trainrun.yaml"
        trainrun_file.write_text(
            yaml.dump(
                {
                    "name": "test",
                    "pipeline": {"nodes": [], "connections": []},
                    "data": {"data_module": "cu3s"},
                    "training": {},
                }
            )
        )

        request = cuvis_ai_pb2.RestoreTrainRunRequest(
            trainrun_path=str(trainrun_file),
        )
        self.service.restore_train_run(request, self.ctx)
        self.ctx.set_code.assert_called_with(grpc.StatusCode.INVALID_ARGUMENT)

    def test_restore_train_run_missing_weights(self, tmp_path):
        """A request weights path that doesn't exist → FileNotFoundError → NOT_FOUND."""
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
                    "pipeline": "pipeline.yaml",
                    "data": {"data_module": "cu3s"},
                    "training": {},
                }
            )
        )

        request = cuvis_ai_pb2.RestoreTrainRunRequest(
            trainrun_path=str(trainrun_file),
            weights_path=str(tmp_path / "nonexistent.pt"),
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
                    "data": {"data_module": "cu3s"},
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
        """Resolve the pipeline from the ``<name>_pipeline.yaml`` companion (no explicit ref)."""

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
                    "data": {"data_module": "cu3s"},
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
        """Status reflects recorded stream progress, not the trainer's presence."""
        session_id = self.session_manager.create_session()
        session = self.session_manager.get_session(session_id)
        session.trainer = Mock()

        # A trainer alone (no recorded Train-stream progress) is UNSPECIFIED.
        request = cuvis_ai_pb2.GetTrainStatusRequest(session_id=session_id)
        response = self.service.get_train_status(request, self.ctx)
        assert response.latest_progress.status == cuvis_ai_pb2.TRAIN_STATUS_UNSPECIFIED

        # Once the Train stream recorded progress, that response is served.
        session.latest_train_response = cuvis_ai_pb2.TrainResponse(
            status=cuvis_ai_pb2.TRAIN_STATUS_COMPLETE, message="done"
        )
        response = self.service.get_train_status(request, self.ctx)
        assert response.latest_progress.status == cuvis_ai_pb2.TRAIN_STATUS_COMPLETE
        assert response.latest_progress.message == "done"


def _fake_post_fit(outcome, events=()):
    """A PostFitCallback stand-in: emits ``events`` through the sink and returns ``outcome``."""

    class _FakePostFit:
        instances: list = []

        def __init__(
            self,
            *,
            emit=None,
            stop_event=None,
            release_cuda_cache=False,
            metric_nodes=None,
        ):
            self.emit = emit
            self.stop_event = stop_event
            self.release_cuda_cache = release_cuda_cache
            self.metric_nodes = metric_nodes
            self.result = None
            self.runs: list[bool] = []
            type(self).instances.append(self)

        def run(
            self,
            pipeline,
            datamodule,
            *,
            module=None,
            restore=True,
            epoch=0,
            global_step=0,
        ):
            self.runs.append(restore)
            for event in events:
                self.emit(event)
            self.result = PostFitResult(None, outcome, {})
            return self.result

    return _FakePostFit


def _calibrated_outcome(
    knob: str = "threshold", before: float = 0.5, after: float = 0.42
):
    return CalibrationOutcome(
        split="val",
        applicable=True,
        reports={"dec": {"class": "Mock", knob: {"old": before, "new": after}}},
    )


def test_train_gradient_streams_the_post_fit_events_and_reports_the_calibration(
    monkeypatch,
):
    """The gradient Train RPC seeds, builds its callbacks (progress, stop, post-fit, then
    the config-derived ones) and constructs the trainer. The mocked trainer's fit stands in
    for Lightning: it hands the post-fit callback its result, the way on_train_end would.
    The results travel as events; the COMPLETE response keeps an empty metrics map."""
    import threading

    service = TrainingService(SessionManager())
    monkeypatch.setattr(
        service, "_configure_gradient_components", lambda *a, **k: ([], [])
    )
    training_config = TrainingConfig(seed=7, max_epochs=1)
    # A bare Mock's stop_event.is_set() would be truthy; the session needs the
    # real (unset) event so the drained stream is not reported as cancelled.
    session = Mock(stop_event=threading.Event(), trainrun_config=None)
    outcome = _calibrated_outcome()

    with patch("cuvis_ai_core.grpc.training_service.GradientTrainer") as mock_gt:

        def fit() -> None:
            post_fit = next(
                c
                for c in mock_gt.call_args.kwargs["callbacks"]
                if isinstance(c, PostFitCallback)
            )
            post_fit.emit(
                TrainEvent(
                    TrainHook.CALIBRATION_END,
                    split=ExecutionStage.VAL,
                    calibration=outcome,
                )
            )
            post_fit.result = PostFitResult(None, outcome, {})

        mock_gt.return_value.fit.side_effect = fit
        data_config = DataConfig(data_module="cu3s")
        responses = list(
            service._train_gradient(session, Mock(), training_config, data_config)
        )

    mock_gt.assert_called_once()
    assert mock_gt.return_value.fit.called
    registered = [type(c).__name__ for c in mock_gt.call_args.kwargs["callbacks"]]
    assert registered[:3] == [
        "ProgressStreamCallback",
        "StopTrainingCallback",
        "PostFitCallback",
    ]
    # The memory log and the cache release are installed by the trainer itself.
    assert "CudaMemoryLogCallback" not in registered

    event_responses = [r for r in responses if r.HasField("event")]
    assert [r.event.hook for r in event_responses] == [
        cuvis_ai_pb2.TRAIN_HOOK_CALIBRATION_END
    ]
    assert event_responses[0].status == cuvis_ai_pb2.TRAIN_STATUS_RUNNING
    calibration = event_responses[0].event.calibration
    assert calibration.applicable and calibration.phase_skip == (
        cuvis_ai_pb2.CALIBRATION_PHASE_SKIP_NONE
    )
    (decider,) = calibration.deciders
    assert decider.name == "dec"
    assert decider.status == cuvis_ai_pb2.DECIDER_CALIBRATION_STATUS_CALIBRATED
    assert decider.thresholds["threshold"].before == pytest.approx(0.5)
    assert decider.thresholds["threshold"].after == pytest.approx(0.42)

    assert responses[-1].status == cuvis_ai_pb2.TRAIN_STATUS_COMPLETE
    assert not responses[-1].metrics and not responses[-1].HasField("event")
    assert responses[-1].message == (
        "Gradient training complete; thresholds calibrated on val: dec (threshold 0.42)"
    )
    assert session.pipeline_config is None
    assert session.weights_source == "fit"
    assert session.last_calibration.outcome is outcome
    assert session.last_calibration.data_config is data_config


def test_train_gradient_without_a_post_fit_result_reports_plain_completion(monkeypatch):
    """A mocked fit that never reaches on_train_end leaves no result: no calibration clause,
    the cached config stays, and the session's calibration record is left untouched."""
    import threading

    service = TrainingService(SessionManager())
    monkeypatch.setattr(
        service, "_configure_gradient_components", lambda *a, **k: ([], [])
    )
    session = Mock(stop_event=threading.Event(), trainrun_config=None)
    cached = session.pipeline_config
    previous_calibration = session.last_calibration

    with patch("cuvis_ai_core.grpc.training_service.GradientTrainer"):
        responses = list(
            service._train_gradient(
                session, Mock(), TrainingConfig(seed=7, max_epochs=1), DataConfig()
            )
        )

    assert responses[-1].message == "Gradient training complete"
    assert session.pipeline_config is cached
    assert session.last_calibration is previous_calibration


def test_train_gradient_passes_the_cache_release_to_the_post_fit_callback(monkeypatch):
    """``release_cuda_cache_on_validation`` reaches the post-fit callback with the stop event."""
    import threading

    service = TrainingService(SessionManager())
    monkeypatch.setattr(
        service, "_configure_gradient_components", lambda *a, **k: ([], [])
    )
    training_config = TrainingConfig(
        seed=7, max_epochs=1, release_cuda_cache_on_validation=True
    )
    session = Mock(stop_event=threading.Event(), trainrun_config=None)

    with patch("cuvis_ai_core.grpc.training_service.GradientTrainer") as mock_gt:
        list(service._train_gradient(session, Mock(), training_config, DataConfig()))

    post_fit = next(
        c
        for c in mock_gt.call_args.kwargs["callbacks"]
        if isinstance(c, PostFitCallback)
    )
    assert post_fit.release_cuda_cache is True
    assert post_fit.stop_event is session.stop_event


def test_train_statistical_runs_the_post_fit_steps_only_for_a_statistical_only_run():
    """A statistical-only run calibrates and evaluates after its fit, streaming the events;
    a statistical phase followed by a gradient phase ends after the fit."""
    import threading

    service = TrainingService(SessionManager())
    outcome = _calibrated_outcome("quantile", 0.99, 0.995)
    fake = _fake_post_fit(
        outcome,
        events=[
            TrainEvent(
                TrainHook.CALIBRATION_END, split=ExecutionStage.VAL, calibration=outcome
            )
        ],
    )

    session = Mock(stop_event=threading.Event(), trainrun_config=None)
    data_config = DataConfig(data_module="cu3s")
    with (
        patch("cuvis_ai_core.grpc.training_service.StatisticalTrainer") as mock_st,
        patch("cuvis_ai_core.grpc.training_service.PostFitCallback", fake),
    ):
        responses = list(
            service._train_statistical(session, Mock(), data_config, post_fit=True)
        )

    assert mock_st.return_value.fit.called
    assert fake.instances[-1].runs == [
        False
    ]  # no checkpoint restore after a statistical fit
    assert [r.event.hook for r in responses if r.HasField("event")] == [
        cuvis_ai_pb2.TRAIN_HOOK_CALIBRATION_END
    ]
    assert responses[-1].status == cuvis_ai_pb2.TRAIN_STATUS_COMPLETE
    assert responses[-1].message == (
        "Statistical training complete; thresholds calibrated on val: dec (quantile 0.995)"
    )
    assert session.pipeline_config is None
    assert session.weights_source == "fit"
    assert session.last_calibration.outcome is outcome
    assert session.last_calibration.data_config is data_config

    fake.instances.clear()
    session = Mock(stop_event=threading.Event(), trainrun_config=None)
    cached = session.pipeline_config
    previous_calibration = session.last_calibration
    with (
        patch("cuvis_ai_core.grpc.training_service.StatisticalTrainer"),
        patch("cuvis_ai_core.grpc.training_service.PostFitCallback", fake),
    ):
        responses = list(
            service._train_statistical(session, Mock(), DataConfig(), post_fit=False)
        )

    assert fake.instances == []
    assert responses[-1].message == "Statistical training complete"
    assert not any(r.HasField("event") for r in responses)
    assert session.pipeline_config is cached
    assert session.last_calibration is previous_calibration  # post_fit=False: untouched


class _FakeStatisticalTrainer:
    """Drives the sink it was built with the way a fit over one node would."""

    instances: list = []
    events: list = []
    raise_cancelled = False

    def __init__(
        self, pipeline, datamodule, stop_event=None, metric_nodes=None, emit=None
    ):
        self.emit = emit
        self.stop_event = stop_event
        type(self).instances.append(self)

    def fit(self):
        for event in type(self).events:
            self.emit(event)
        if type(self).raise_cancelled:
            self.stop_event.set()
            raise TrainingCancelled("Statistical training stopped via StopTrain")


def test_train_statistical_streams_the_node_events_while_the_fit_runs():
    """The statistical fit reports each node through the typed sink, the events reach the
    stream before the post-fit steps, and the opening sentence and COMPLETE carry none."""
    import threading

    from cuvis_ai_core.training.post_fit import FrameProgress

    service = TrainingService(SessionManager())
    outcome = _calibrated_outcome("quantile", 0.99, 0.995)
    fake_post_fit = _fake_post_fit(
        outcome,
        events=[
            TrainEvent(
                TrainHook.CALIBRATION_END, split=ExecutionStage.VAL, calibration=outcome
            )
        ],
    )
    fake_trainer = type(
        "_Trainer",
        (_FakeStatisticalTrainer,),
        {
            "instances": [],
            "events": [
                TrainEvent(
                    TrainHook.STATISTICAL_NODE_START,
                    node="norm",
                    progress=FrameProgress(0, 4),
                ),
                TrainEvent(
                    TrainHook.STATISTICAL_PROGRESS,
                    node="norm",
                    progress=FrameProgress(4, 4),
                ),
                TrainEvent(TrainHook.STATISTICAL_NODE_END, node="norm"),
            ],
        },
    )
    session = Mock(stop_event=threading.Event(), trainrun_config=None)
    with (
        patch("cuvis_ai_core.grpc.training_service.StatisticalTrainer", fake_trainer),
        patch("cuvis_ai_core.grpc.training_service.PostFitCallback", fake_post_fit),
    ):
        responses = list(
            service._train_statistical(session, Mock(), DataConfig(), post_fit=True)
        )

    assert fake_trainer.instances[-1].stop_event is session.stop_event
    assert responses[0].message == "Starting statistical training"
    assert not responses[0].HasField("event")
    events = [r for r in responses if r.HasField("event")]
    assert [r.event.hook for r in events] == [
        cuvis_ai_pb2.TRAIN_HOOK_STATISTICAL_NODE_START,
        cuvis_ai_pb2.TRAIN_HOOK_STATISTICAL_PROGRESS,
        cuvis_ai_pb2.TRAIN_HOOK_STATISTICAL_NODE_END,
        cuvis_ai_pb2.TRAIN_HOOK_CALIBRATION_END,
    ]
    assert [r.message for r in events[:3]] == [
        "Fitting norm (0/4)",
        "Fitting norm (4/4)",
        "Fitted norm",
    ]
    assert events[0].event.node == "norm"
    assert (events[1].event.progress.done, events[1].event.progress.total) == (4, 4)
    assert all(not r.event.HasField("epochs") for r in events)
    assert all(r.status == cuvis_ai_pb2.TRAIN_STATUS_RUNNING for r in events)
    assert all(
        r.context.stage == cuvis_ai_pb2.EXECUTION_STAGE_TRAIN for r in events[:3]
    )
    assert responses[-1].status == cuvis_ai_pb2.TRAIN_STATUS_COMPLETE
    assert not responses[-1].HasField("event")
    assert session.weights_source == "fit"


def test_train_statistical_cancellation_surfaces_through_the_worker():
    """A StopTrain during the statistical fit still ends the phase as a cancellation."""
    import threading

    service = TrainingService(SessionManager())
    fake_trainer = type(
        "_Trainer",
        (_FakeStatisticalTrainer,),
        {"instances": [], "events": [], "raise_cancelled": True},
    )
    session = Mock(stop_event=threading.Event(), trainrun_config=None)
    with (
        patch("cuvis_ai_core.grpc.training_service.StatisticalTrainer", fake_trainer),
        patch(
            "cuvis_ai_core.grpc.training_service.PostFitCallback",
            _fake_post_fit(_calibrated_outcome()),
        ),
        pytest.raises(TrainingCancelled) as raised,
    ):
        list(service._train_statistical(session, Mock(), DataConfig(), post_fit=True))

    # The fit's own cancellation passes through: no swallowed error, no warning.
    assert raised.value.swallowed_error is None


def test_train_calibration_runs_the_steps_without_a_fit():
    """The standalone phase streams the same events and never calls a trainer."""
    import threading

    service = TrainingService(SessionManager())
    outcome = _calibrated_outcome()
    evaluation = EvaluationOutcome(
        split=ExecutionStage.VAL,
        status=EvaluationStatus.EVALUATED,
        frames=3,
        metrics={"m/f1": 0.5},
    )
    fake = _fake_post_fit(
        outcome,
        events=[
            TrainEvent(
                TrainHook.CALIBRATION_END, split=ExecutionStage.VAL, calibration=outcome
            ),
            TrainEvent(
                TrainHook.EVALUATION_END,
                split=ExecutionStage.VAL,
                evaluation=evaluation,
            ),
            TrainEvent(TrainHook.POST_FIT_END),
        ],
    )
    session = Mock(
        stop_event=threading.Event(), trainrun_config=None, weights_source="run.pt"
    )

    data_config = DataConfig(data_module="cu3s")
    with (
        patch("cuvis_ai_core.grpc.training_service.StatisticalTrainer") as mock_st,
        patch("cuvis_ai_core.grpc.training_service.GradientTrainer") as mock_gt,
        patch("cuvis_ai_core.grpc.training_service.PostFitCallback", fake),
    ):
        responses = list(service._train_calibration(session, Mock(), data_config))

    assert not mock_st.called and not mock_gt.called
    assert fake.instances[-1].runs == [False]
    assert responses[0].status == cuvis_ai_pb2.TRAIN_STATUS_RUNNING
    assert responses[0].message == "Starting calibration"
    event_responses = [r for r in responses if r.HasField("event")]
    assert [r.event.hook for r in event_responses] == [
        cuvis_ai_pb2.TRAIN_HOOK_CALIBRATION_END,
        cuvis_ai_pb2.TRAIN_HOOK_EVALUATION_END,
        cuvis_ai_pb2.TRAIN_HOOK_POST_FIT_END,
    ]
    # No fit in this phase: no epoch pair and no node on any of its events.
    assert all(
        not r.event.HasField("epochs") and r.event.node == "" for r in event_responses
    )
    evaluated = event_responses[1].event.evaluation
    assert evaluated.status == cuvis_ai_pb2.EVALUATION_STATUS_EVALUATED
    assert evaluated.split == cuvis_ai_pb2.EXECUTION_STAGE_VAL
    assert evaluated.frames == 3 and evaluated.metrics["m/f1"] == pytest.approx(0.5)
    assert event_responses[1].context.stage == cuvis_ai_pb2.EXECUTION_STAGE_VAL
    assert responses[-1].status == cuvis_ai_pb2.TRAIN_STATUS_COMPLETE
    assert responses[-1].message == (
        "Calibration complete; thresholds calibrated on val: dec (threshold 0.42)"
    )
    assert session.pipeline_config is None
    assert session.last_calibration.outcome is outcome
    assert session.last_calibration.data_config is data_config


def test_calibration_precondition_names_the_missing_weights_and_the_unfitted_nodes():
    service = TrainingService(SessionManager())

    class _Stat:
        def __init__(self, name: str, fitted: bool) -> None:
            self.name = name
            self.requires_initial_fit = True
            self._statistically_initialized = fitted

    pipeline = Mock()
    pipeline.nodes.return_value = [_Stat("norm", False)]
    session = Mock(weights_source=None, pipeline=pipeline)
    refusal = service._calibration_precondition(session)
    assert refusal is not None and refusal.startswith(
        "The session holds no trained weights"
    )

    session.weights_source = "run.pt"
    refusal = service._calibration_precondition(session)
    assert refusal is not None and "norm" in refusal

    pipeline.nodes.return_value = [_Stat("norm", True)]
    assert service._calibration_precondition(session) is None


def test_train_refuses_a_required_decisions_consumer_of_an_uncalibrated_decider():
    """The pre-fit check names the consumer and the decider, before any trainer runs."""
    from types import SimpleNamespace

    service = TrainingService(SessionManager())
    pair = (SimpleNamespace(name="sink"), SimpleNamespace(name="dec"))
    with patch(
        "cuvis_ai_core.grpc.training_service.uncalibrated_required_consumers",
        return_value=[pair],
    ):
        with pytest.raises(ValueError, match="sink reads decisions from dec"):
            service._refuse_uncalibrated_consumers(Mock())
    with patch(
        "cuvis_ai_core.grpc.training_service.uncalibrated_required_consumers",
        return_value=[],
    ):
        service._refuse_uncalibrated_consumers(Mock())


def test_event_to_proto_maps_every_enum_by_name():
    """Every Python enum member has its proto twin; a drift fails here, not on the wire."""
    from cuvis_ai_core.grpc.training_service import _event_to_proto
    from cuvis_ai_core.training.calibration import (
        CalibrationPhaseSkip,
        DeciderCalibration,
        DeciderCalibrationStatus,
        DeciderSkip,
        ThresholdChange,
    )
    from cuvis_ai_core.training.post_fit import CheckpointRestored, FrameProgress

    for hook in TrainHook:
        assert _event_to_proto(TrainEvent(hook)).hook == getattr(
            cuvis_ai_pb2, f"TRAIN_HOOK_{hook.name}"
        )
    for status in EvaluationStatus:
        assert getattr(cuvis_ai_pb2, f"EVALUATION_STATUS_{status.name}")
    for skip in CalibrationPhaseSkip:
        assert getattr(cuvis_ai_pb2, f"CALIBRATION_PHASE_SKIP_{skip.name}")
    for skip in DeciderSkip:
        assert getattr(cuvis_ai_pb2, f"DECIDER_SKIP_{skip.name}")

    skipped = CalibrationOutcome(
        split="val",
        applicable=True,
        deciders=(
            DeciderCalibration(
                name="dec",
                class_name="TwoStageBinaryDecider",
                status=DeciderCalibrationStatus.SKIPPED,
                skip=DeciderSkip.CALIBRATE_FAILED,
                skip_detail="ValueError: boom",
                thresholds={"image_threshold": ThresholdChange(before=None, after=2.5)},
                metrics={"f1": 0.9},
                thresholds_present=False,
            ),
        ),
        phase_skip=CalibrationPhaseSkip.SINGLE_CLASS,
        frames=4,
        anomalous_frames=0,
    )
    proto = _event_to_proto(
        TrainEvent(
            TrainHook.CALIBRATION_END,
            split=ExecutionStage.VAL,
            progress=FrameProgress(4, 4),
            checkpoint=CheckpointRestored(
                True, path="b.ckpt", epoch=2, monitor="m", score=0.7
            ),
            calibration=skipped,
        )
    )
    assert proto.split == cuvis_ai_pb2.EXECUTION_STAGE_VAL
    assert (proto.progress.done, proto.progress.total) == (4, 4)
    assert proto.checkpoint.restored and proto.checkpoint.score == pytest.approx(0.7)
    assert proto.checkpoint.HasField("score")
    outcome = proto.calibration
    assert outcome.phase_skip == cuvis_ai_pb2.CALIBRATION_PHASE_SKIP_SINGLE_CLASS
    assert (outcome.frames, outcome.anomalous_frames) == (4, 0)
    (decider,) = outcome.deciders
    assert decider.skip == cuvis_ai_pb2.DECIDER_SKIP_CALIBRATE_FAILED
    assert decider.skip_detail == "ValueError: boom"
    assert not decider.thresholds["image_threshold"].HasField("before")
    assert decider.thresholds["image_threshold"].after == pytest.approx(2.5)
    assert decider.metrics["f1"] == pytest.approx(0.9)
    assert decider.thresholds_present is False
    kept = _event_to_proto(
        TrainEvent(
            TrainHook.CHECKPOINT_RESTORED,
            checkpoint=CheckpointRestored(False, reason="no checkpoint was saved"),
        )
    )
    assert not kept.checkpoint.HasField("score") and kept.checkpoint.reason


def test_event_message_words_the_fit_progress_hooks():
    """The status line of a fit event names the pass and the frame counts; a total of 0
    shows the count alone; a node's end is one line."""
    from cuvis_ai_core.grpc.training_service import _event_message
    from cuvis_ai_core.training.post_fit import EpochProgress, FrameProgress

    epochs = EpochProgress(1, 2)
    cases = [
        (
            TrainEvent(
                TrainHook.EPOCH_START, epochs=epochs, progress=FrameProgress(0, 124)
            ),
            "Training (0/124)",
        ),
        (
            TrainEvent(
                TrainHook.TRAIN_PROGRESS, epochs=epochs, progress=FrameProgress(12, 124)
            ),
            "Training (12/124)",
        ),
        (
            TrainEvent(
                TrainHook.VALIDATION_START,
                epochs=epochs,
                split=ExecutionStage.VAL,
                progress=FrameProgress(0, 48),
            ),
            "Validating (0/48)",
        ),
        (
            TrainEvent(
                TrainHook.VALIDATION_PROGRESS,
                split=ExecutionStage.VAL,
                progress=FrameProgress(3, 48),
            ),
            "Validating (3/48)",
        ),
        (
            TrainEvent(
                TrainHook.STATISTICAL_NODE_START,
                node="norm",
                progress=FrameProgress(0, 124),
            ),
            "Fitting norm (0/124)",
        ),
        (
            TrainEvent(
                TrainHook.STATISTICAL_PROGRESS,
                node="norm",
                progress=FrameProgress(8, 0),
            ),
            "Fitting norm (8)",
        ),
        (TrainEvent(TrainHook.STATISTICAL_NODE_END, node="norm"), "Fitted norm"),
        (
            TrainEvent(
                TrainHook.CALIBRATION_PROGRESS,
                split=ExecutionStage.VAL,
                progress=FrameProgress(2, 48),
            ),
            "Calibrating decider thresholds (2/48)",
        ),
    ]
    for event, expected in cases:
        assert _event_message(event) == expected, event.hook


def test_event_to_proto_carries_the_epoch_pair_and_the_node():
    from cuvis_ai_core.grpc.training_service import _event_to_proto
    from cuvis_ai_core.training.post_fit import EpochProgress, FrameProgress

    proto = _event_to_proto(
        TrainEvent(
            TrainHook.TRAIN_PROGRESS,
            epochs=EpochProgress(3, 20),
            progress=FrameProgress(4, 14),
        )
    )
    assert proto.HasField("epochs")
    assert (proto.epochs.current, proto.epochs.total) == (3, 20)
    assert proto.node == ""

    fitting = _event_to_proto(
        TrainEvent(
            TrainHook.STATISTICAL_NODE_START, node="norm", progress=FrameProgress(0, 4)
        )
    )
    assert fitting.node == "norm" and not fitting.HasField("epochs")

    unlimited = _event_to_proto(
        TrainEvent(TrainHook.FIT_END, epochs=EpochProgress(3, 0))
    )
    assert unlimited.HasField("epochs") and unlimited.epochs.total == 0


class TestRunOutcome:
    """``_run_outcome``: a stop must never hide a failure that came first."""

    def test_clean_run_raises_nothing(self):
        assert _run_outcome(None, False, False) is None

    def test_a_cancellation_the_run_raised_itself_passes_through(self):
        """A statistical fit raises TrainingCancelled at a batch boundary: that is the
        cancellation, not an error a stop happened to swallow."""
        cancelled = TrainingCancelled("Statistical training stopped via StopTrain")
        assert _run_outcome(cancelled, True, True) is cancelled

    def test_plain_stop_is_a_cancellation(self):
        outcome = _run_outcome(None, False, True)
        assert isinstance(outcome, TrainingCancelled)
        assert outcome.swallowed_error is None

    def test_error_without_a_stop_wins(self):
        error = RuntimeError("CUDA out of memory")
        assert _run_outcome(error, False, False) is error

    def test_error_that_arrived_before_the_stop_still_wins(self):
        # The run died, the client (or the stream teardown) stopped it after:
        # reporting that as a clean cancellation would hide the crash.
        error = RuntimeError("CUDA out of memory")
        assert _run_outcome(error, False, True) is error

    def test_stop_that_preceded_the_error_swallows_it_but_keeps_it(self):
        error = RuntimeError("dataloader shutdown noise")
        outcome = _run_outcome(error, True, True)
        assert isinstance(outcome, TrainingCancelled)
        assert outcome.swallowed_error is error

    def test_swallowed_error_is_logged_at_warning(self):
        records: list[tuple[str, str]] = []
        sink_id = logger.add(
            lambda message: records.append(
                (message.record["level"].name, message.record["message"])
            ),
            level="WARNING",
        )
        try:
            _run_outcome(RuntimeError("late boom"), True, True)
        finally:
            logger.remove(sink_id)

        assert [level for level, _ in records] == ["WARNING"]
        assert "late boom" in records[0][1]

    def test_a_stop_without_an_error_logs_nothing(self):
        records: list[str] = []
        sink_id = logger.add(
            lambda message: records.append(message.record["message"]), level="WARNING"
        )
        try:
            _run_outcome(None, False, True)
        finally:
            logger.remove(sink_id)

        assert records == []


def test_save_train_run_failed_yaml_write_leaves_the_previous_file(tmp_path):
    """A save into an existing run folder stages its yaml: a failing dump leaves the old
    trainrun yaml intact and no staging file behind."""
    from cuvis_ai_core.grpc.trainrun_service import TrainRunService

    manager = SessionManager()
    service = TrainRunService(manager)
    session_id = manager.create_session()
    session = manager.get_session(session_id)
    mock_pipeline = Mock()
    mock_pipeline.nodes.return_value = []
    mock_pipeline.name = "test"
    session.pipeline = mock_pipeline
    session.trainrun_config = TrainRunConfig(
        name="first",
        pipeline=None,
        data=DataConfig(),
        training=TrainingConfig(),
        loss_nodes=[],
        metric_nodes=[],
        freeze_nodes=[],
        unfreeze_nodes=[],
        output_dir=".",
        tags={},
    )
    request = cuvis_ai_pb2.SaveTrainRunRequest(
        session_id=session_id, trainrun_path=str(tmp_path / "trainrun.yaml")
    )

    first = service.save_train_run(request, Mock())
    assert first.success is True
    saved = (tmp_path / "trainrun.yaml").read_text(encoding="utf-8")
    assert "name: first" in saved

    session.trainrun_config = session.trainrun_config.model_copy(
        update={"name": "second"}
    )
    ctx = Mock()
    with patch(
        "cuvis_ai_core.grpc.trainrun_service.yaml.dump",
        side_effect=OSError("disk full"),
    ):
        second = service.save_train_run(request, ctx)

    assert second.success is False
    ctx.set_code.assert_called_with(grpc.StatusCode.INTERNAL)
    assert (tmp_path / "trainrun.yaml").read_text(encoding="utf-8") == saved
    # No staging file survives; the pipeline sibling is the only other file.
    assert {p.name for p in tmp_path.iterdir()} == {
        "trainrun.yaml",
        "trainrun_pipeline.yaml",
    }
