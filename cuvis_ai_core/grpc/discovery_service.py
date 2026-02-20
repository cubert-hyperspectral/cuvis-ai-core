"""Pipeline discovery service component."""

from __future__ import annotations

import grpc

from . import helpers
from .error_handling import grpc_handler
from .v1 import cuvis_ai_pb2


class DiscoveryService:
    """Discover available pipelines."""

    @grpc_handler("Failed to list pipelines")
    def list_available_pipelines(
        self,
        request: cuvis_ai_pb2.ListAvailablePipelinesRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.ListAvailablePipelinesResponse:
        """List all available pipeline configurations."""
        filter_tag = request.filter_tag if request.HasField("filter_tag") else None

        pipelines_list = helpers.list_available_pipelines(filter_tag=filter_tag)

        pipeline_infos = []
        for pipeline_dict in pipelines_list:
            metadata = pipeline_dict["metadata"]
            pipeline_info = cuvis_ai_pb2.PipelineInfo(
                name=pipeline_dict["name"],
                path=pipeline_dict["path"],
                metadata=cuvis_ai_pb2.PipelineMetadata(
                    name=metadata["name"],
                    description=metadata["description"],
                    created=metadata["created"],
                    cuvis_ai_version=metadata["cuvis_ai_version"],
                    tags=metadata["tags"],
                    author=metadata["author"],
                ),
                tags=pipeline_dict["tags"],
                has_weights=pipeline_dict["has_weights"],
                weights_path=pipeline_dict["weights_path"],
            )
            pipeline_infos.append(pipeline_info)

        return cuvis_ai_pb2.ListAvailablePipelinesResponse(pipelines=pipeline_infos)

    @grpc_handler("Failed to get pipeline info")
    def get_pipeline_info(
        self,
        request: cuvis_ai_pb2.GetPipelineInfoRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.GetPipelineInfoResponse:
        """Get detailed information about a specific pipeline."""
        pipeline_dict = helpers.get_pipeline_info(
            pipeline_name=request.pipeline_name,
            include_yaml_content=True,
        )

        metadata = pipeline_dict["metadata"]
        pipeline_info = cuvis_ai_pb2.PipelineInfo(
            name=pipeline_dict["name"],
            path=pipeline_dict["path"],
            metadata=cuvis_ai_pb2.PipelineMetadata(
                name=metadata["name"],
                description=metadata["description"],
                created=metadata["created"],
                cuvis_ai_version=metadata["cuvis_ai_version"],
                tags=metadata["tags"],
                author=metadata["author"],
            ),
            tags=pipeline_dict["tags"],
            has_weights=pipeline_dict["has_weights"],
            weights_path=pipeline_dict["weights_path"],
            yaml_content=pipeline_dict.get("yaml_content", ""),
        )

        return cuvis_ai_pb2.GetPipelineInfoResponse(pipeline_info=pipeline_info)


__all__ = ["DiscoveryService"]
