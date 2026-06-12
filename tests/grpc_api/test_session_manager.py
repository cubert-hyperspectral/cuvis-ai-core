import time
from unittest.mock import MagicMock

import pytest

from cuvis_ai_core.grpc.helpers import resolve_pipeline_path
from cuvis_ai_core.grpc.session_manager import SessionManager, SessionState
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline


class TestSessionManager:
    def test_create_session_returns_unique_id(self):
        manager = SessionManager()

        # Load pipeline from YAML
        pipeline_path = resolve_pipeline_path("gradient_based")
        pipeline1 = CuvisPipeline.load_pipeline(str(pipeline_path))
        pipeline2 = CuvisPipeline.load_pipeline(str(pipeline_path))

        session_id1 = manager.create_session(pipeline=pipeline1)
        session_id2 = manager.create_session(pipeline=pipeline2)

        assert session_id1 != session_id2
        assert isinstance(session_id1, str)
        assert session_id1 in manager.list_sessions()
        assert session_id2 in manager.list_sessions()

    def test_get_session_returns_state(self):
        manager = SessionManager()

        # Load pipeline from YAML
        pipeline_path = resolve_pipeline_path("gradient_based")
        pipeline = CuvisPipeline.load_pipeline(str(pipeline_path))

        session_id = manager.create_session(pipeline=pipeline)
        state = manager.get_session(session_id)

        assert isinstance(state, SessionState)
        assert isinstance(state.pipeline, CuvisPipeline)
        assert isinstance(state.created_at, float)
        assert isinstance(state.last_accessed, float)
        assert state.created_at > 0
        assert state.last_accessed > 0

    def test_pipeline_config_property_derives_from_pipeline(self):
        manager = SessionManager()

        pipeline_path = resolve_pipeline_path("gradient_based")
        pipeline = CuvisPipeline.load_pipeline(str(pipeline_path))

        session_id = manager.create_session(pipeline=pipeline)
        state = manager.get_session(session_id)

        pipeline_config = state.pipeline_config
        assert pipeline_config.metadata is not None
        assert pipeline_config.connections is not None

    def test_get_session_nonexistent_raises_error(self):
        manager = SessionManager()
        with pytest.raises(ValueError, match="Session .* not found"):
            manager.get_session("missing")

    def test_close_session_removes_state(self):
        manager = SessionManager()

        # Load pipeline from YAML
        pipeline_path = resolve_pipeline_path("gradient_based")
        pipeline = CuvisPipeline.load_pipeline(str(pipeline_path))
        session_id = manager.create_session(pipeline=pipeline)

        manager.close_session(session_id)
        with pytest.raises(ValueError):
            manager.get_session(session_id)

    def test_close_nonexistent_session_raises_error(self):
        manager = SessionManager()
        with pytest.raises(ValueError):
            manager.close_session("unknown")

    def test_close_session_cleans_up_pipeline(self, monkeypatch: pytest.MonkeyPatch):
        manager = SessionManager()
        pipeline = MagicMock()
        trainer = MagicMock()
        session_id = manager.create_session(pipeline=pipeline)
        manager.get_session(session_id).trainer = trainer

        gc_collect = MagicMock(return_value=0)
        empty_cache = MagicMock()
        monkeypatch.setattr("cuvis_ai_core.grpc.session_manager.gc.collect", gc_collect)
        monkeypatch.setattr(
            "cuvis_ai_core.grpc.session_manager.torch.cuda.is_available",
            lambda: True,
        )
        monkeypatch.setattr(
            "cuvis_ai_core.grpc.session_manager.torch.cuda.empty_cache",
            empty_cache,
        )

        manager.close_session(session_id)

        pipeline.cleanup.assert_called_once_with()
        trainer.cleanup.assert_called_once_with()
        gc_collect.assert_called_once_with()
        empty_cache.assert_called_once_with()

    def test_cleanup_pipeline_tolerates_cleanup_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = SessionManager()
        pipeline = MagicMock()
        pipeline.cleanup.side_effect = RuntimeError("boom")
        session_id = manager.create_session(pipeline=pipeline)

        monkeypatch.setattr(
            "cuvis_ai_core.grpc.session_manager.gc.collect", MagicMock()
        )
        monkeypatch.setattr(
            "cuvis_ai_core.grpc.session_manager.torch.cuda.is_available",
            lambda: False,
        )

        manager.close_session(session_id)

        pipeline.cleanup.assert_called_once_with()

    def test_set_pipeline_cleans_up_previous_pipeline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = SessionManager()
        old_pipeline = MagicMock()
        new_pipeline = MagicMock()
        session_id = manager.create_session(pipeline=old_pipeline)

        gc_collect = MagicMock(return_value=0)
        monkeypatch.setattr("cuvis_ai_core.grpc.session_manager.gc.collect", gc_collect)
        monkeypatch.setattr(
            "cuvis_ai_core.grpc.session_manager.torch.cuda.is_available",
            lambda: False,
        )

        manager.set_pipeline(session_id, new_pipeline, pipeline_config=None)

        old_pipeline.cleanup.assert_called_once_with()
        gc_collect.assert_called_once_with()
        assert manager.get_session(session_id).pipeline is new_pipeline

    def test_get_session_updates_last_accessed(self):
        manager = SessionManager()

        # Load pipeline from YAML
        pipeline_path = resolve_pipeline_path("gradient_based")
        pipeline = CuvisPipeline.load_pipeline(str(pipeline_path))
        session_id = manager.create_session(pipeline=pipeline)

        first_timestamp = manager.get_session(session_id).last_accessed
        time.sleep(0.05)
        state2 = manager.get_session(session_id)

        assert state2.last_accessed > first_timestamp

    def test_cleanup_old_sessions(self):
        manager = SessionManager()

        # Load pipeline from YAML
        pipeline_path = resolve_pipeline_path("gradient_based")
        pipeline = CuvisPipeline.load_pipeline(str(pipeline_path))
        session_id = manager.create_session(pipeline=pipeline)

        # backdate the session
        manager._sessions[session_id].last_accessed = 0.0
        cleaned = manager.cleanup_old_sessions(max_age_hours=1)

        assert cleaned == 1
        assert session_id not in manager.list_sessions()

    def test_create_session_without_data_config(self):
        """Test creating an inference-only session (no trainrun_config)."""
        manager = SessionManager()

        # Load pipeline from YAML
        pipeline_path = resolve_pipeline_path("gradient_based")
        pipeline = CuvisPipeline.load_pipeline(str(pipeline_path))
        session_id = manager.create_session(pipeline=pipeline)
        state = manager.get_session(session_id)

        assert isinstance(state, SessionState)
        assert isinstance(state.pipeline, CuvisPipeline)
        assert state.trainrun_config is None
        assert session_id in manager.list_sessions()

    def test_session_state_with_optional_data_config(self):
        """Test that session state properly handles optional trainrun_config."""
        manager = SessionManager()
        from cuvis_ai_core.training.config import (
            DataConfig,
            TrainingConfig,
            TrainRunConfig,
        )
        from cuvis_ai_schemas.training import DataSplitConfig

        # Load pipeline from YAML
        pipeline_path = resolve_pipeline_path("gradient_based")
        pipeline = CuvisPipeline.load_pipeline(str(pipeline_path))

        # Create trainrun config
        trainrun_config = TrainRunConfig(
            name="test_trainrun",
            pipeline=pipeline.serialize(),
            data=DataConfig(
                splits=DataSplitConfig(
                    train_ids=[1, 2],
                    val_ids=[3],
                    test_ids=[4],
                ),
                batch_size=4,
                params={
                    "cu3s_file_path": "/tmp/data.cu3s",
                    "annotation_json_path": "/tmp/annotations.json",
                },
            ),
            training=TrainingConfig(),
        )

        # Create session with trainrun_config
        session_id_with_config = manager.create_session(
            pipeline=pipeline, trainrun_config=trainrun_config
        )
        state_with_config = manager.get_session(session_id_with_config)
        assert state_with_config.trainrun_config is not None
        assert state_with_config.trainrun_config.name == "test_trainrun"

        # Create session without trainrun_config
        pipeline2 = CuvisPipeline.load_pipeline(str(pipeline_path))
        session_id_without_config = manager.create_session(pipeline=pipeline2)
        state_without_config = manager.get_session(session_id_without_config)
        assert state_without_config.trainrun_config is None


# ---------------------------------------------------------------------------
# pipeline_config property branches
# ---------------------------------------------------------------------------


def test_pipeline_config_property_returns_cached_value():
    manager = SessionManager()
    marker = object()
    sid = manager.create_session(pipeline_config=marker)
    assert manager.get_session(sid).pipeline_config is marker


def test_pipeline_config_property_raises_without_pipeline():
    manager = SessionManager()
    sid = manager.create_session()
    with pytest.raises(ValueError, match="not initialized"):
        _ = manager.get_session(sid).pipeline_config


# ---------------------------------------------------------------------------
# create_session_with_id
# ---------------------------------------------------------------------------


def test_create_session_with_id_rejects_empty():
    manager = SessionManager()
    with pytest.raises(ValueError, match="non-empty"):
        manager.create_session_with_id("")


def test_create_session_with_id_is_idempotent():
    manager = SessionManager()
    manager.create_session_with_id("shared-id")
    state = manager.get_session("shared-id")
    manager.create_session_with_id("shared-id")
    # Reuse keeps the same state object, not a fresh one.
    assert manager.get_session("shared-id") is state


# ---------------------------------------------------------------------------
# set_search_paths / _validate_search_path
# ---------------------------------------------------------------------------


def test_set_search_paths_rejects_invalid_paths(tmp_path):
    manager = SessionManager()
    sid = manager.create_session()
    valid_dir = tmp_path / "configs"
    valid_dir.mkdir()
    accepted, rejected = manager.set_search_paths(
        sid, [str(valid_dir), str(tmp_path / "does_not_exist")], append=True
    )
    assert str(valid_dir.resolve()) in accepted
    assert str(tmp_path / "does_not_exist") in rejected


def test_validate_search_path_swallows_resolution_errors():
    manager = SessionManager()
    # An embedded NUL makes Path.resolve raise; the helper must return None.
    assert manager._validate_search_path("bad\x00path") is None


# ---------------------------------------------------------------------------
# close_session: trainer cleanup is best-effort
# ---------------------------------------------------------------------------


def test_close_session_isolates_trainer_cleanup_failure():
    manager = SessionManager()
    sid = manager.create_session()
    trainer = MagicMock()
    trainer.cleanup.side_effect = RuntimeError("trainer cleanup blew up")
    manager.get_session(sid).trainer = trainer

    # Must not raise even though trainer.cleanup() failed.
    manager.close_session(sid)
    assert sid not in manager.list_sessions()
    trainer.cleanup.assert_called_once()
