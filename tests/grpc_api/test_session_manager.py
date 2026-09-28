import time
from pathlib import Path
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
        from cuvis_ai_schemas.training import DataSplitConfig, Selector, SelectorKind

        # Load pipeline from YAML
        pipeline_path = resolve_pipeline_path("gradient_based")
        pipeline = CuvisPipeline.load_pipeline(str(pipeline_path))

        # Create trainrun config
        trainrun_config = TrainRunConfig(
            name="test_trainrun",
            pipeline="gradient_based",
            data=DataConfig(
                splits=DataSplitConfig(
                    train=[
                        Selector(
                            kind=SelectorKind.FILE_INDICES,
                            source="/tmp/data.cu3s",
                            ids=[1, 2],
                        )
                    ],
                    val=[
                        Selector(
                            kind=SelectorKind.FILE_INDICES,
                            source="/tmp/data.cu3s",
                            ids=[3],
                        )
                    ],
                    test=[
                        Selector(
                            kind=SelectorKind.FILE_INDICES,
                            source="/tmp/data.cu3s",
                            ids=[4],
                        )
                    ],
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


# ---------------------------------------------------------------------------
# close_session: crash-log preservation for a child that died on its own
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _forget_preserved_sessions():
    """Preservation is idempotent per session id via module state; reset it."""
    from cuvis_ai_core.orchestrator.crash_logs import reset_for_tests

    reset_for_tests()
    yield
    reset_for_tests()


def _session_with_child(tmp_path, *, returncode, terminate_result):
    """Session with a fake child handle and a real runtime tree on disk."""
    manager = SessionManager()
    sid = manager.create_session()
    state = manager.get_session(sid)

    runtime = tmp_path / "session_root" / "scratch" / "runtime"
    runtime.mkdir(parents=True)
    (runtime / "child.stdout.log").write_text("nodes registered", encoding="utf-8")
    (runtime / "child.stderr.log").write_text(
        "Fatal Python error: Aborted", encoding="utf-8"
    )

    child = MagicMock()
    child.returncode = returncode
    child.terminate.return_value = terminate_result
    child.endpoint = "127.0.0.1:9"
    child.stdout_log = runtime / "child.stdout.log"
    child.stderr_log = runtime / "child.stderr.log"

    state.child_handle = child
    state.runtime_base_dir = tmp_path / "session_root"
    return manager, sid


def test_close_session_preserves_logs_when_child_died_on_its_own(monkeypatch, tmp_path):
    crash_root = tmp_path / "crashes"
    monkeypatch.setenv("CUVIS_RUNTIME_CRASH_DIR", str(crash_root))
    manager, sid = _session_with_child(tmp_path, returncode=3, terminate_result=3)

    manager.close_session(sid)

    # The copies happened BEFORE the rmtree: crash dir populated, tree gone.
    crash_dirs = list(crash_root.iterdir())
    assert len(crash_dirs) == 1
    assert (crash_dirs[0] / "child.stderr.log").exists()
    assert (crash_dirs[0] / "child.stdout.log").exists()
    assert "exit_code: 3" in (crash_dirs[0] / "crash_info.txt").read_text(
        encoding="utf-8"
    )
    assert not (tmp_path / "session_root").exists()


def test_close_session_normal_exit_leaves_no_crash_dir(monkeypatch, tmp_path):
    crash_root = tmp_path / "crashes"
    monkeypatch.setenv("CUVIS_RUNTIME_CRASH_DIR", str(crash_root))
    manager, sid = _session_with_child(tmp_path, returncode=None, terminate_result=0)

    manager.close_session(sid)

    assert not crash_root.exists()
    assert not (tmp_path / "session_root").exists()


def test_close_session_no_crash_dir_for_parent_terminated_child(monkeypatch, tmp_path):
    # A child alive at close (returncode None) that terminate() kills exits
    # nonzero (TerminateProcess reports 1) — that is NOT a crash.
    crash_root = tmp_path / "crashes"
    monkeypatch.setenv("CUVIS_RUNTIME_CRASH_DIR", str(crash_root))
    manager, sid = _session_with_child(tmp_path, returncode=None, terminate_result=1)

    manager.close_session(sid)

    assert not crash_root.exists()
    assert not (tmp_path / "session_root").exists()


def test_close_session_without_child_handle_reaps_tree(monkeypatch, tmp_path):
    crash_root = tmp_path / "crashes"
    monkeypatch.setenv("CUVIS_RUNTIME_CRASH_DIR", str(crash_root))
    manager = SessionManager()
    sid = manager.create_session()
    base = tmp_path / "session_root"
    base.mkdir()
    (base / "leftover.txt").write_text("x", encoding="utf-8")
    manager.get_session(sid).runtime_base_dir = base

    manager.close_session(sid)

    assert not base.exists()
    assert not crash_root.exists()


def test_close_session_reuses_a_directory_the_failing_rpc_already_made(
    monkeypatch, tmp_path
):
    """The crash was already preserved when the RPC failed: no second copy."""
    from cuvis_ai_core.orchestrator.crash_logs import preserve_child_logs

    crash_root = tmp_path / "crashes"
    monkeypatch.setenv("CUVIS_RUNTIME_CRASH_DIR", str(crash_root))
    manager, sid = _session_with_child(tmp_path, returncode=3, terminate_result=3)
    child = manager.get_session(sid).child_handle
    first = preserve_child_logs(
        (child.stdout_log, child.stderr_log), session_id=sid, exit_code=3
    )

    manager.close_session(sid)

    assert first is not None
    assert [p.name for p in crash_root.iterdir()] == [first.name]


def test_close_session_records_the_crash_dir_on_the_session(monkeypatch, tmp_path):
    crash_root = tmp_path / "crashes"
    monkeypatch.setenv("CUVIS_RUNTIME_CRASH_DIR", str(crash_root))
    manager, sid = _session_with_child(tmp_path, returncode=3, terminate_result=3)
    state = manager.get_session(sid)
    assert state.crash_log_dir is None

    manager.close_session(sid)

    assert state.crash_log_dir is not None
    assert state.crash_log_dir.name.endswith(sid)


# ---------------------------------------------------------------------------
# default search paths
# ---------------------------------------------------------------------------


def _write(path, text="nodes: []\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path.resolve()


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def isolated_default(monkeypatch, tmp_path):
    """No ``CUVIS_CONFIGS_DIR`` and a fresh cwd, both set before any session exists:
    a session captures its default search path when it is created."""
    monkeypatch.delenv("CUVIS_CONFIGS_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _fresh_paths():
    manager = SessionManager()
    return manager, manager.get_session(manager.create_session()).search_paths


def test_default_search_paths_agree_at_every_entry_point(isolated_default):
    from cuvis_ai_core.grpc.session_manager import default_search_paths
    from cuvis_ai_core.utils.node_registry import NodeRegistry

    expected = [str((isolated_default / "configs").resolve())]
    assert default_search_paths() == expected
    state = SessionState(session_id="s", node_registry=NodeRegistry())
    assert state.search_paths == expected

    manager = SessionManager()
    sid = manager.create_session()
    assert manager.get_session(sid).search_paths == expected

    missing = str(isolated_default / "missing")
    paths, rejected = manager.set_search_paths(sid, [missing], append=False)
    assert rejected == [missing]
    assert paths == expected


def test_default_search_path_follows_cuvis_configs_dir(isolated_default, monkeypatch):
    """The directory discovery lists from and relative saves write to is the one a new
    session searches, so a pipeline ListAvailablePipelines names resolves by prefixed
    name in a fresh session. A relative value is resolved once, at session creation."""
    from cuvis_ai_core.grpc.session_manager import default_search_paths

    monkeypatch.setenv("CUVIS_CONFIGS_DIR", "custom")
    expected = [str((isolated_default / "custom").resolve())]
    assert default_search_paths() == expected
    _manager, paths = _fresh_paths()
    assert paths == expected


def test_new_session_resolves_prefixed_pipeline_names_only(isolated_default):
    from cuvis_ai_core.utils.config_helpers import _find_config_file

    demo = _write(isolated_default / "configs" / "pipeline" / "demo.yaml")
    _manager, paths = _fresh_paths()

    assert _find_config_file("pipeline/demo", paths) == demo
    with pytest.raises(FileNotFoundError):
        _find_config_file("demo", paths)


def test_new_session_composes_bundled_configs_by_prefixed_name(monkeypatch):
    from cuvis_ai_core.utils.config_helpers import resolve_config_with_hydra

    monkeypatch.delenv("CUVIS_CONFIGS_DIR", raising=False)
    monkeypatch.chdir(REPO_ROOT)
    _manager, paths = _fresh_paths()

    pipeline = resolve_config_with_hydra("pipeline", "pipeline/gradient_based", paths)
    assert pipeline["metadata"]["name"] == "gradient_based"
    with pytest.raises(FileNotFoundError):
        resolve_config_with_hydra("pipeline", "gradient_based", paths)


def test_appended_directory_is_not_shadowed_by_the_bundled_pipelines(isolated_default):
    """A bare name found in a directory the client appended must win over a bundled
    pipeline of the same name; the bundled one is still reachable by its prefix."""
    from cuvis_ai_core.grpc.helpers import find_weights_file
    from cuvis_ai_core.utils.config_helpers import _find_config_file

    bundled = _write(isolated_default / "configs" / "pipeline" / "demo.yaml")
    custom = _write(isolated_default / "custom" / "demo.yaml")
    bundled_weights = _write(isolated_default / "configs" / "pipeline" / "w.pt", "x")
    custom_weights = _write(isolated_default / "custom" / "w.pt", "y")
    manager = SessionManager()
    sid = manager.create_session()
    manager.set_search_paths(sid, [str(isolated_default / "custom")], append=True)
    paths = manager.get_session(sid).search_paths

    assert _find_config_file("demo", paths) == custom
    assert _find_config_file("pipeline/demo", paths) == bundled
    assert find_weights_file("w", paths) == custom_weights
    assert find_weights_file("pipeline/w", paths) == bundled_weights


def test_appended_trainrun_directory_resolves_the_trainrun_not_the_pipeline(
    monkeypatch,
):
    """The bundled pipeline and trainrun share the stem ``gradient_based``. With the
    trainrun directory appended (the client's usual pattern) the bare name composes
    the trainrun, whose Hydra defaults resolve from the configs root."""
    from cuvis_ai_core.utils.config_helpers import resolve_config_with_hydra

    monkeypatch.delenv("CUVIS_CONFIGS_DIR", raising=False)
    monkeypatch.chdir(REPO_ROOT)
    manager = SessionManager()
    sid = manager.create_session()
    manager.set_search_paths(
        sid, [str(REPO_ROOT / "configs" / "trainrun")], append=True
    )
    paths = manager.get_session(sid).search_paths

    trainrun = resolve_config_with_hydra("trainrun", "gradient_based", paths)
    assert trainrun["name"] == "gradient_based"
    assert trainrun["pipeline"]
