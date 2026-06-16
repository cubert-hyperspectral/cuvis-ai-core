"""Experiment/trainrun management service component."""

from __future__ import annotations

from pathlib import Path

import grpc
import torch
import yaml

from cuvis_ai_core.training.config import DataConfig, TrainingConfig, TrainRunConfig
from cuvis_ai_core.utils.restore import _resolve_pipeline_reference

from .error_handling import get_session_or_error, grpc_handler
from .session_manager import SessionManager
from .v1 import cuvis_ai_pb2


class TrainRunService:
    """Save and restore train run configuration."""

    def __init__(self, session_manager: SessionManager) -> None:
        self.session_manager = session_manager

    @grpc_handler("Failed to save train run")
    def save_train_run(
        self,
        request: cuvis_ai_pb2.SaveTrainRunRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.SaveTrainRunResponse:
        """Persist the current session TrainRunConfig to disk, optionally with weights."""
        session = get_session_or_error(
            self.session_manager, request.session_id, context
        )
        if session is None:
            return cuvis_ai_pb2.SaveTrainRunResponse(success=False)

        if not request.trainrun_path:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("trainrun_path is required")
            return cuvis_ai_pb2.SaveTrainRunResponse(success=False)

        trainrun_path = Path(request.trainrun_path)
        trainrun_path.parent.mkdir(parents=True, exist_ok=True)

        # The trainrun's pipeline is persisted as a reference: the live pipeline
        # is written to a sibling YAML and the trainrun points at it by name.
        sibling_pipeline_path = trainrun_path.with_name(
            f"{trainrun_path.stem}_pipeline.yaml"
        )

        written_pipeline: dict[str, str] = {}

        def _write_pipeline_sibling() -> str:
            """Write the session's live pipeline beside the trainrun, return its ref name."""
            session.pipeline_config.save_to_file(sibling_pipeline_path)
            written_pipeline["path"] = str(sibling_pipeline_path)
            return sibling_pipeline_path.name

        has_live_pipeline = (
            session.pipeline is not None or session._pipeline_config is not None
        )

        if session.trainrun_config is not None:
            trainrun_config = session.trainrun_config
            # Make the saved artifact self-contained: whenever the session holds a
            # live pipeline, write it beside the trainrun and (re-)point the
            # reference at that sibling. This covers both the trained-then-saved
            # case (the in-memory config carries no reference) and the
            # restored-then-saved-elsewhere case (its reference is relative to the
            # *source* directory and would not resolve from the new location).
            if has_live_pipeline:
                trainrun_config = trainrun_config.model_copy(
                    update={"pipeline": _write_pipeline_sibling()}
                )
            elif not trainrun_config.pipeline:
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                context.set_details(
                    "Pipeline configuration is required before saving a train run."
                )
                return cuvis_ai_pb2.SaveTrainRunResponse(success=False)
        else:
            if not has_live_pipeline:
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                context.set_details(
                    "Pipeline configuration is required before saving a train run."
                )
                return cuvis_ai_pb2.SaveTrainRunResponse(success=False)

            trainrun_config = TrainRunConfig(
                name=getattr(session.pipeline, "name", "trainrun"),
                pipeline=_write_pipeline_sibling(),
                data=session.data_config or DataConfig(),
                training=session.training_config or TrainingConfig(),
                loss_nodes=[],
                metric_nodes=[],
                freeze_nodes=[],
                unfreeze_nodes=[],
                output_dir="./outputs",
                tags={},
            )

        with trainrun_path.open("w", encoding="utf-8") as f:
            yaml.dump(
                trainrun_config.to_dict(),
                f,
                default_flow_style=False,
                sort_keys=False,
            )

        # Simplified: Only save weights if explicitly requested
        weights_path = None
        if request.save_weights and session.pipeline is not None:
            try:
                # Save weights with same name as trainrun but .pt extension
                weights_path_obj = trainrun_path.with_suffix(".pt")
                weights_path = str(weights_path_obj)

                # Save only the state_dict (weights) without pipeline config
                state_dict = {}
                for node in session.pipeline.nodes():
                    if hasattr(node, "state_dict"):
                        state_dict[node.name] = node.state_dict()

                pipeline_meta = getattr(session.pipeline_config, "metadata", None)
                checkpoint = {
                    "state_dict": state_dict,
                    "metadata": {
                        "name": trainrun_config.name,
                        "created": pipeline_meta.created if pipeline_meta else "",
                    },
                }

                import torch

                torch.save(checkpoint, weights_path_obj)

            except Exception as exc:
                # Log weights save failure but don't fail the entire operation
                context.set_details(
                    f"Train run config saved successfully, but weights save failed: {exc}"
                )

        return cuvis_ai_pb2.SaveTrainRunResponse(
            success=True,
            trainrun_path=str(trainrun_path),
            pipeline_path=written_pipeline.get("path", ""),
            weights_path=weights_path,
        )

    @staticmethod
    def parse_trainrun_yaml(trainrun_path: Path) -> tuple[TrainRunConfig, Path]:
        """Parse a trainrun yaml, returning the typed config + resolved pipeline YAML path.

        The trainrun's ``pipeline:`` is a path reference (resolved relative to
        the trainrun file's directory); failing that, the ``<name>_pipeline.yaml``
        sibling convention is used. The orchestrator reads the resolved pipeline
        to learn its plugin set before forwarding the restore request; the child
        rebuilds it against its own ``NodeRegistry`` via :func:`restore_train_run`.
        """
        if not trainrun_path.exists():
            raise FileNotFoundError(f"Train run file not found: {trainrun_path}")

        with trainrun_path.open("r", encoding="utf-8") as f:
            trainrun_dict = yaml.safe_load(f.read())
        if isinstance(trainrun_dict, dict) and "defaults" in trainrun_dict:
            raise ValueError(
                "Train run config contains Hydra defaults; please compose/resolve it first "
                "and pass the resolved YAML/JSON via config_bytes."
            )

        trainrun_config = TrainRunConfig.from_dict(trainrun_dict)

        if trainrun_config.pipeline is not None:
            # An explicit reference must resolve; a failure surfaces loudly with
            # the tried-paths diagnostic (mirrors the CLI restore path) rather
            # than being masked by the sibling fallback.
            return trainrun_config, _resolve_pipeline_reference(
                trainrun_config.pipeline, trainrun_path.parent
            )

        # No explicit reference: fall back to the ``<name>_pipeline.yaml`` sibling.
        sibling = trainrun_path.with_stem(f"{trainrun_path.stem}_pipeline").with_suffix(
            ".yaml"
        )
        if sibling.exists():
            return trainrun_config, sibling
        raise ValueError(
            "Train run config missing pipeline: set 'pipeline:' to a pipeline "
            f"YAML path or place a '{trainrun_path.stem}_pipeline.yaml' sibling."
        )

    @grpc_handler("Failed to restore train run")
    def restore_train_run(
        self,
        request: cuvis_ai_pb2.RestoreTrainRunRequest,
        context: grpc.ServicerContext,
        *,
        target_session_id: str | None = None,
    ) -> cuvis_ai_pb2.RestoreTrainRunResponse:
        """Restore a TrainRunConfig from disk, build its pipeline, attach to a session.

        When ``target_session_id`` is None this creates a new session.
        When provided (as the child runtime does — its session was
        created at ``InitializeSession`` time), the existing session
        is reused and ``set_pipeline`` attaches the freshly-built
        pipeline to it.
        """
        trainrun_path = Path(request.trainrun_path)
        trainrun_config, pipeline_config_path = self.parse_trainrun_yaml(trainrun_path)

        from cuvis_ai_schemas.pipeline import PipelineConfig

        from cuvis_ai_core.pipeline.pipeline import CuvisPipeline

        # Resolve weights override from the request, if any.
        weights_path_from_request: Path | None = None
        if request.HasField("weights_path") and request.weights_path:
            weights_path_from_request = Path(request.weights_path)
            if not weights_path_from_request.exists():
                raise FileNotFoundError(
                    f"Weights file not found: {weights_path_from_request}"
                )

        # Load the referenced pipeline. Use the target session's NodeRegistry if
        # we have one (so plugin-provided node classes registered via
        # InitializeSession resolve correctly).
        node_registry = None
        if target_session_id is not None:
            node_registry = self.session_manager.get_session(
                target_session_id
            ).node_registry

        pipeline = CuvisPipeline.load_pipeline(
            config_path=pipeline_config_path,
            weights_path=str(weights_path_from_request)
            if weights_path_from_request
            else None,
            strict_weight_loading=request.strict
            if request.HasField("strict")
            else True,
            device="cuda" if torch.cuda.is_available() else None,
            node_registry=node_registry,
        )
        pipeline_config = PipelineConfig.load_from_file(pipeline_config_path)

        if torch.cuda.is_available():
            pipeline = pipeline.to("cuda")

        if target_session_id is None:
            session_id = self.session_manager.create_session(
                pipeline=pipeline,
                data_config=trainrun_config.data,
                training_config=trainrun_config.training,
                trainrun_config=trainrun_config,
            )
        else:
            self.session_manager.set_pipeline(
                target_session_id,
                pipeline,
                pipeline_config=pipeline_config,
            )
            session = self.session_manager.get_session(target_session_id)
            session.data_config = trainrun_config.data
            session.training_config = trainrun_config.training
            session.trainrun_config = trainrun_config
            session_id = target_session_id

        return cuvis_ai_pb2.RestoreTrainRunResponse(
            session_id=session_id,
            trainrun=trainrun_config.to_proto(),
        )


__all__ = ["TrainRunService"]
