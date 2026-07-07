"""Inference service component."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import ExitStack
from typing import Any

import grpc
import numpy as np
import torch
from loguru import logger

from cuvis_ai_schemas.enums import ExecutionStage

from . import helpers
from .error_handling import get_session_or_error, grpc_handler, require_pipeline
from .session_manager import SessionManager
from .v1 import cuvis_ai_pb2


class InferenceService:
    """Inference operations for Cuvis AI."""

    def __init__(self, session_manager: SessionManager) -> None:
        self.session_manager = session_manager

    @grpc_handler("Inference failed")
    def inference(
        self,
        request: cuvis_ai_pb2.InferenceRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.InferenceResponse:
        """Run a forward pass for the requested session."""
        session = get_session_or_error(
            self.session_manager, request.session_id, context
        )
        if session is None:
            return cuvis_ai_pb2.InferenceResponse()

        if not require_pipeline(session, context):
            return cuvis_ai_pb2.InferenceResponse()

        with ExitStack() as stack:
            # copy_tensors=False yields zero-copy views into the input buffers; the
            # cube may be a read-only shared-memory mapping held open by `stack`.
            # Inputs are read-only: nodes must not mutate them in place, and no view
            # may outlive this block, where the SHM mappings are released.
            batch = self._parse_input_batch(
                request.inputs,
                copy_tensors=False,
                stack=stack,
            )
            # Move to the pipeline device, then register clear() on the post-move
            # dict so the batch's tensor refs drop before the SHM mappings close.
            # This must run after the reassignment: on a CPU pipeline `.to('cpu')`
            # returns the same objects, so clearing the original dict would leave
            # them referenced and `owner.close()` would hit BufferError and leak.
            batch = self._move_batch_to_pipeline_device(batch, session.pipeline)
            stack.callback(batch.clear)

            outputs = session.pipeline.forward(
                batch=batch, stage=ExecutionStage.INFERENCE
            )

            output_specs = set(request.output_specs)
            available = [self._format_output_key(k) for k in outputs]
            logger.info(
                f"Inference produced {len(available)} pipeline outputs "
                f"{sorted(available)}; requested output_specs="
                f"{sorted(output_specs) or '[] (all)'}"
            )

            tensor_outputs, metrics, dropped = self._build_outputs(
                outputs, output_specs
            )

            # A specifically requested, array-like output that failed to serialize
            # is a real error. Fail loudly instead of silently substituting unrelated
            # outputs under an OK status.
            if dropped:
                msg = (
                    f"Requested output(s) {sorted(dropped)} could not be serialized "
                    f"(unsupported dtype or value). Available: {sorted(available)}."
                )
                logger.error(msg)
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(msg)
                return cuvis_ai_pb2.InferenceResponse()

            if available and not tensor_outputs and not metrics:
                logger.warning(
                    f"No serializable outputs for session {request.session_id}; "
                    f"returning an empty response. Available: {sorted(available)}"
                )

            return cuvis_ai_pb2.InferenceResponse(
                outputs=tensor_outputs, metrics=metrics
            )

    def _build_outputs(
        self,
        outputs: dict[Any, Any],
        specs: set[str],
    ) -> tuple[dict[str, cuvis_ai_pb2.Tensor], dict[str, float], list[str]]:
        """Filter and serialize pipeline outputs into the response maps.

        Applies ``output_specs`` filtering, routes scalars to ``metrics``, and
        serializes the rest to tensors. Outputs that cannot be serialized
        (non-tensor metadata, artifacts, unsupported dtypes) are skipped with a
        warning rather than failing the whole response. An empty ``specs`` set
        means "return everything serializable".

        Returns a third value, ``dropped``: the names of *explicitly requested*
        outputs that were array-like (tensor/ndarray) but failed to serialize.
        Those are real errors the caller must surface, not silently mask — unlike
        non-array metadata (dicts, custom objects), which is expected to be
        unserializable and is simply skipped.
        """
        tensor_outputs: dict[str, cuvis_ai_pb2.Tensor] = {}
        metrics: dict[str, float] = {}
        dropped: list[str] = []

        for raw_key, value in outputs.items():
            output_name = self._format_output_key(raw_key)
            if not self._should_return(output_name, specs):
                logger.debug(f"Skipping output {output_name!r}: not in output_specs")
                continue

            # Metrics: plain scalars
            if isinstance(value, (int, float, np.number)) and not isinstance(
                value, bool
            ):
                metrics[output_name] = float(value)
                continue

            try:
                tensor = self._to_tensor(value)
                tensor_outputs[output_name] = helpers.tensor_to_proto(tensor)
            except Exception as exc:
                # Non-serializable outputs (artifacts/custom objects/metadata)
                # are dropped — but loudly, so an empty response is never a
                # silent mystery.
                logger.warning(
                    f"Dropping output {output_name!r} "
                    f"(type={type(value).__name__}): {exc}"
                )
                # An array-like output (tensor/ndarray) that failed to serialize is
                # a real error when the client explicitly requested it; record it so
                # inference() can fail loudly instead of substituting unrelated
                # outputs. Non-array metadata stays a silent skip.
                if specs and isinstance(value, (torch.Tensor, np.ndarray)):
                    dropped.append(output_name)

        return tensor_outputs, metrics, dropped

    def _parse_input_batch(
        self,
        inputs: cuvis_ai_pb2.InputBatch,
        copy_tensors: bool = True,
        stack: ExitStack | None = None,
    ) -> dict[str, Any]:
        """Convert InputBatch proto to dict for CuvisPipeline.

        Converts proto messages to Python types. The pipeline determines
        which inputs are required and validates shapes/types.
        """
        batch: dict[str, Any] = {}

        def parse_tensor(tensor_proto: cuvis_ai_pb2.Tensor) -> torch.Tensor:
            if stack is not None:
                return stack.enter_context(
                    helpers.proto_to_tensor(tensor_proto, copy=copy_tensors)
                )
            with helpers.proto_to_tensor(tensor_proto, copy=True) as t:
                return t.clone()

        # Parse tensor inputs (if provided)
        if inputs.HasField("cube"):
            batch["cube"] = parse_tensor(inputs.cube)

        if inputs.HasField("wavelengths"):
            batch["wavelengths"] = parse_tensor(inputs.wavelengths)

        if inputs.HasField("mask"):
            batch["mask"] = parse_tensor(inputs.mask)

        if inputs.HasField("rgb_image"):
            batch["rgb_image"] = parse_tensor(inputs.rgb_image)

        if inputs.HasField("frame_id"):
            frame_id = parse_tensor(inputs.frame_id)
            if frame_id.numel() > 0:
                batch["frame_id"] = frame_id

        if inputs.HasField("mesu_index"):
            mesu_index = parse_tensor(inputs.mesu_index)
            if mesu_index.numel() > 0:
                batch["mesu_index"] = mesu_index

        # Parse structured inputs (if provided)
        if inputs.HasField("bboxes"):
            batch["bboxes"] = self._parse_bounding_boxes(inputs.bboxes)

        if inputs.HasField("points"):
            batch["points"] = self._parse_points(inputs.points)

        if inputs.text_prompt:
            batch["text_prompt"] = inputs.text_prompt

        # Parse extra tensor inputs (node-specific dynamic inputs).
        for key, tensor_proto in inputs.extra_inputs.items():
            batch[key] = parse_tensor(tensor_proto)

        return batch

    def _move_batch_to_pipeline_device(
        self,
        batch: dict[str, Any],
        pipeline,
    ) -> dict[str, Any]:
        """Move tensor inputs to the same device as the pipeline.

        This mirrors the Python API pattern where the pipeline is moved to a device
        (e.g., via ``pipeline.to('cuda')``) and dataloader batches are produced on
        that device. Here, gRPC deserialization always yields CPU tensors, so we
        align them with the pipeline device before forwarding.

        Uses the same robust device detection pattern as StatisticalTrainer,
        iterating through all nodes to find one with parameters or buffers.
        """
        if pipeline is None:
            return batch

        device = self._get_pipeline_device(pipeline)

        moved: dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(device)
            else:
                moved[key] = value

        return moved

    def _get_pipeline_device(self, pipeline) -> torch.device:
        """Get the device of the pipeline from its parameters/buffers.

        Iterates through all nodes in the pipeline to find one with
        parameters or buffers, returning its device. Falls back to CPU
        if no device information is found.

        This follows the same pattern as StatisticalTrainer._get_pipeline_device
        to ensure consistent device detection across the codebase.
        """
        # Iterate through all torch-backed layers (not just the first one)
        for layer in pipeline.torch_layers:
            # Check parameters first (preferred for device detection)
            for param in layer.parameters():
                return param.device
            # Fall back to buffers if no parameters
            for buf in layer.buffers():
                return buf.device
        # Explicit fallback to CPU if no device information found
        return torch.device("cpu")

    def _parse_bounding_boxes(
        self, bboxes_proto: cuvis_ai_pb2.BoundingBoxes
    ) -> list[dict]:
        """Parse bounding boxes from proto into dictionaries."""
        parsed_boxes: list[dict] = []
        for box in bboxes_proto.boxes:
            parsed = {
                "element_id": box.element_id,
                "x_min": box.x_min,
                "y_min": box.y_min,
                "x_max": box.x_max,
                "y_max": box.y_max,
            }
            if box.HasField("object_id"):
                parsed["object_id"] = box.object_id
            parsed_boxes.append(parsed)
        return parsed_boxes

    def _parse_points(self, points_proto: cuvis_ai_pb2.Points) -> list[dict]:
        """Parse points from proto into dictionaries."""
        return [
            {
                "element_id": point.element_id,
                "x": point.x,
                "y": point.y,
                "type": (
                    "neutral"
                    if point.type == cuvis_ai_pb2.POINT_TYPE_UNSPECIFIED
                    else helpers.point_type_to_string(point.type).lower()
                ),
            }
            for point in points_proto.points
        ]

    def _format_output_key(self, key: Any) -> str:
        """Normalize pipeline output keys (tuple -> 'node.port')."""
        if isinstance(key, tuple) and len(key) == 2:
            node_name, port = key
            return f"{node_name}.{port}"
        return str(key)

    def _should_return(self, output_name: str, specs: set[str]) -> bool:
        if not specs:
            return True
        port_name = output_name.split(".", maxsplit=1)[-1]
        return output_name in specs or port_name in specs

    def _to_tensor(self, value: Any) -> torch.Tensor:
        """Coerce supported outputs to torch.Tensor."""
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value)
        if isinstance(value, Iterable):
            return torch.tensor(list(value))
        # Last resort for scalars (avoid bool/metric routing)
        return torch.tensor(value)


__all__ = ["InferenceService"]
