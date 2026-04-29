from cuvis_ai_core.grpc.v1 import cuvis_ai_pb2


def test_config_messages_use_bytes():
    """Verify all config messages use bytes config_bytes field."""
    configs = [
        cuvis_ai_pb2.PipelineConfig,
        cuvis_ai_pb2.DataConfig,
        cuvis_ai_pb2.TrainingConfig,
        cuvis_ai_pb2.TrainRunConfig,
    ]

    for config_cls in configs:
        instance = config_cls()
        assert hasattr(instance, "config_bytes")


def test_trainrun_naming():
    """Verify Experiment renamed to TrainRun."""
    assert hasattr(cuvis_ai_pb2, "TrainRunConfig")
    assert hasattr(cuvis_ai_pb2, "SaveTrainRunRequest")
    assert hasattr(cuvis_ai_pb2, "RestoreTrainRunRequest")
    assert not hasattr(cuvis_ai_pb2, "ExperimentConfig")
    assert not hasattr(cuvis_ai_pb2, "SaveExperimentRequest")


def test_new_session_rpcs_exist():
    """Verify new session management RPCs defined."""
    assert hasattr(cuvis_ai_pb2, "CreateSessionRequest")
    assert hasattr(cuvis_ai_pb2, "SetSessionSearchPathsRequest")
    assert hasattr(cuvis_ai_pb2, "CloseSessionRequest")


def test_new_config_rpcs_exist():
    """Verify new config resolution RPCs defined."""
    assert hasattr(cuvis_ai_pb2, "ResolveConfigRequest")
    assert hasattr(cuvis_ai_pb2, "GetParameterSchemaRequest")
    assert hasattr(cuvis_ai_pb2, "ValidateConfigRequest")


def test_new_pipeline_rpcs_exist():
    """Verify new pipeline building RPCs defined."""
    assert hasattr(cuvis_ai_pb2, "LoadPipelineRequest")
    assert hasattr(cuvis_ai_pb2, "LoadPipelineWeightsRequest")
    assert hasattr(cuvis_ai_pb2, "SetTrainRunConfigRequest")
    assert not hasattr(cuvis_ai_pb2, "BuildPipelineRequest")


def test_load_pipeline_uses_wrapper():
    """LoadPipelineRequest should expose typed PipelineConfig wrapper."""
    request = cuvis_ai_pb2.LoadPipelineRequest()
    assert hasattr(request, "pipeline")
    assert isinstance(request.pipeline, cuvis_ai_pb2.PipelineConfig)
    assert not hasattr(request, "pipeline_bytes")


def test_load_weights_oneof():
    """Verify LoadPipelineWeightsRequest supports both weight sources."""
    request = cuvis_ai_pb2.LoadPipelineWeightsRequest()
    assert hasattr(request, "weights_path")
    assert hasattr(request, "weights_bytes")


def test_node_info_carries_metadata_fields():
    """NodeInfo must expose icon_svg / category / tags."""
    field_names = {f.name for f in cuvis_ai_pb2.NodeInfo.DESCRIPTOR.fields}
    assert {"icon_svg", "category", "tags"}.issubset(field_names), field_names


def test_node_category_proto_enum_exposed():
    """All 12 NodeCategory values plus UNSPECIFIED are present on the pb2 module."""
    expected = [
        "NODE_CATEGORY_UNSPECIFIED",
        "NODE_CATEGORY_SOURCE",
        "NODE_CATEGORY_SINK",
        "NODE_CATEGORY_TRANSFORM",
        "NODE_CATEGORY_MODEL",
        "NODE_CATEGORY_LOSS",
        "NODE_CATEGORY_METRIC",
        "NODE_CATEGORY_OPTIMIZER",
        "NODE_CATEGORY_SCHEDULER",
        "NODE_CATEGORY_REGULARIZER",
        "NODE_CATEGORY_RUNNER",
        "NODE_CATEGORY_VISUALIZER",
        "NODE_CATEGORY_CONTROL",
    ]
    for name in expected:
        assert hasattr(cuvis_ai_pb2, name), name


def test_node_tag_proto_enum_exposed_for_each_namespace():
    """Spot-check one tag per ID-range namespace (modality / task / lifecycle / property / backend)."""
    for name in (
        "NODE_TAG_UNSPECIFIED",
        "NODE_TAG_HYPERSPECTRAL",
        "NODE_TAG_SEGMENTATION",
        "NODE_TAG_PREPROCESSING",
        "NODE_TAG_LEARNABLE",
        "NODE_TAG_TORCH",
    ):
        assert hasattr(cuvis_ai_pb2, name), name


def test_proto_conversions_module_importable():
    """`cuvis-ai-schemas[proto]` extra exposes hand-written proto-int helpers."""
    from cuvis_ai_schemas.grpc.conversions import (
        node_category_to_proto,
        node_tag_to_proto,
        proto_to_node_category,
        proto_to_node_tag,
    )
    from cuvis_ai_schemas.enums import NodeCategory, NodeTag

    assert (
        node_category_to_proto(NodeCategory.MODEL) == cuvis_ai_pb2.NODE_CATEGORY_MODEL
    )
    assert (
        proto_to_node_category(cuvis_ai_pb2.NODE_CATEGORY_MODEL) is NodeCategory.MODEL
    )
    assert (
        node_tag_to_proto(NodeTag.HYPERSPECTRAL) == cuvis_ai_pb2.NODE_TAG_HYPERSPECTRAL
    )
    assert (
        proto_to_node_tag(cuvis_ai_pb2.NODE_TAG_HYPERSPECTRAL) is NodeTag.HYPERSPECTRAL
    )
