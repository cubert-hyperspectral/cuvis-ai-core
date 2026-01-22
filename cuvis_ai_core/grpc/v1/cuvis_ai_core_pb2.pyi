from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ProcessingMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    PROCESSING_MODE_UNSPECIFIED: _ClassVar[ProcessingMode]
    PROCESSING_MODE_RAW: _ClassVar[ProcessingMode]
    PROCESSING_MODE_REFLECTANCE: _ClassVar[ProcessingMode]
    PROCESSING_MODE_DARKSUBTRACT: _ClassVar[ProcessingMode]
    PROCESSING_MODE_SPECTRAL_RADIANCE: _ClassVar[ProcessingMode]

class ExecutionStage(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    EXECUTION_STAGE_UNSPECIFIED: _ClassVar[ExecutionStage]
    EXECUTION_STAGE_TRAIN: _ClassVar[ExecutionStage]
    EXECUTION_STAGE_VAL: _ClassVar[ExecutionStage]
    EXECUTION_STAGE_TEST: _ClassVar[ExecutionStage]
    EXECUTION_STAGE_INFERENCE: _ClassVar[ExecutionStage]

class DType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    D_TYPE_UNSPECIFIED: _ClassVar[DType]
    D_TYPE_FLOAT32: _ClassVar[DType]
    D_TYPE_FLOAT64: _ClassVar[DType]
    D_TYPE_INT32: _ClassVar[DType]
    D_TYPE_INT64: _ClassVar[DType]
    D_TYPE_UINT8: _ClassVar[DType]
    D_TYPE_BOOL: _ClassVar[DType]
    D_TYPE_FLOAT16: _ClassVar[DType]
    D_TYPE_UINT16: _ClassVar[DType]

class TrainerType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TRAINER_TYPE_UNSPECIFIED: _ClassVar[TrainerType]
    TRAINER_TYPE_STATISTICAL: _ClassVar[TrainerType]
    TRAINER_TYPE_GRADIENT: _ClassVar[TrainerType]

class TrainStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TRAIN_STATUS_UNSPECIFIED: _ClassVar[TrainStatus]
    TRAIN_STATUS_RUNNING: _ClassVar[TrainStatus]
    TRAIN_STATUS_COMPLETE: _ClassVar[TrainStatus]
    TRAIN_STATUS_ERROR: _ClassVar[TrainStatus]

class PointType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    POINT_TYPE_UNSPECIFIED: _ClassVar[PointType]
    POINT_TYPE_POSITIVE: _ClassVar[PointType]
    POINT_TYPE_NEGATIVE: _ClassVar[PointType]
    POINT_TYPE_NEUTRAL: _ClassVar[PointType]

PROCESSING_MODE_UNSPECIFIED: ProcessingMode
PROCESSING_MODE_RAW: ProcessingMode
PROCESSING_MODE_REFLECTANCE: ProcessingMode
PROCESSING_MODE_DARKSUBTRACT: ProcessingMode
PROCESSING_MODE_SPECTRAL_RADIANCE: ProcessingMode
EXECUTION_STAGE_UNSPECIFIED: ExecutionStage
EXECUTION_STAGE_TRAIN: ExecutionStage
EXECUTION_STAGE_VAL: ExecutionStage
EXECUTION_STAGE_TEST: ExecutionStage
EXECUTION_STAGE_INFERENCE: ExecutionStage
D_TYPE_UNSPECIFIED: DType
D_TYPE_FLOAT32: DType
D_TYPE_FLOAT64: DType
D_TYPE_INT32: DType
D_TYPE_INT64: DType
D_TYPE_UINT8: DType
D_TYPE_BOOL: DType
D_TYPE_FLOAT16: DType
D_TYPE_UINT16: DType
TRAINER_TYPE_UNSPECIFIED: TrainerType
TRAINER_TYPE_STATISTICAL: TrainerType
TRAINER_TYPE_GRADIENT: TrainerType
TRAIN_STATUS_UNSPECIFIED: TrainStatus
TRAIN_STATUS_RUNNING: TrainStatus
TRAIN_STATUS_COMPLETE: TrainStatus
TRAIN_STATUS_ERROR: TrainStatus
POINT_TYPE_UNSPECIFIED: PointType
POINT_TYPE_POSITIVE: PointType
POINT_TYPE_NEGATIVE: PointType
POINT_TYPE_NEUTRAL: PointType

class Tensor(_message.Message):
    __slots__ = ("shape", "dtype", "raw_data")
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    RAW_DATA_FIELD_NUMBER: _ClassVar[int]
    shape: _containers.RepeatedScalarFieldContainer[int]
    dtype: DType
    raw_data: bytes
    def __init__(
        self,
        shape: _Optional[_Iterable[int]] = ...,
        dtype: _Optional[_Union[DType, str]] = ...,
        raw_data: _Optional[bytes] = ...,
    ) -> None: ...

class Context(_message.Message):
    __slots__ = ("stage", "epoch", "batch_idx", "global_step")
    STAGE_FIELD_NUMBER: _ClassVar[int]
    EPOCH_FIELD_NUMBER: _ClassVar[int]
    BATCH_IDX_FIELD_NUMBER: _ClassVar[int]
    GLOBAL_STEP_FIELD_NUMBER: _ClassVar[int]
    stage: ExecutionStage
    epoch: int
    batch_idx: int
    global_step: int
    def __init__(
        self,
        stage: _Optional[_Union[ExecutionStage, str]] = ...,
        epoch: _Optional[int] = ...,
        batch_idx: _Optional[int] = ...,
        global_step: _Optional[int] = ...,
    ) -> None: ...

class PipelineConfig(_message.Message):
    __slots__ = ("config_bytes",)
    CONFIG_BYTES_FIELD_NUMBER: _ClassVar[int]
    config_bytes: bytes
    def __init__(self, config_bytes: _Optional[bytes] = ...) -> None: ...

class DataConfig(_message.Message):
    __slots__ = ("config_bytes",)
    CONFIG_BYTES_FIELD_NUMBER: _ClassVar[int]
    config_bytes: bytes
    def __init__(self, config_bytes: _Optional[bytes] = ...) -> None: ...

class OptimizerConfig(_message.Message):
    __slots__ = ("config_bytes",)
    CONFIG_BYTES_FIELD_NUMBER: _ClassVar[int]
    config_bytes: bytes
    def __init__(self, config_bytes: _Optional[bytes] = ...) -> None: ...

class SchedulerConfig(_message.Message):
    __slots__ = ("config_bytes",)
    CONFIG_BYTES_FIELD_NUMBER: _ClassVar[int]
    config_bytes: bytes
    def __init__(self, config_bytes: _Optional[bytes] = ...) -> None: ...

class CallbacksConfig(_message.Message):
    __slots__ = ("config_bytes",)
    CONFIG_BYTES_FIELD_NUMBER: _ClassVar[int]
    config_bytes: bytes
    def __init__(self, config_bytes: _Optional[bytes] = ...) -> None: ...

class PipelineMetadata(_message.Message):
    __slots__ = ("name", "description", "created", "cuvis_ai_version", "tags", "author")
    NAME_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    CREATED_FIELD_NUMBER: _ClassVar[int]
    CUVIS_AI_VERSION_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    AUTHOR_FIELD_NUMBER: _ClassVar[int]
    name: str
    description: str
    created: str
    cuvis_ai_version: str
    tags: _containers.RepeatedScalarFieldContainer[str]
    author: str
    def __init__(
        self,
        name: _Optional[str] = ...,
        description: _Optional[str] = ...,
        created: _Optional[str] = ...,
        cuvis_ai_version: _Optional[str] = ...,
        tags: _Optional[_Iterable[str]] = ...,
        author: _Optional[str] = ...,
    ) -> None: ...

class PipelineInfo(_message.Message):
    __slots__ = (
        "name",
        "path",
        "metadata",
        "tags",
        "has_weights",
        "weights_path",
        "yaml_content",
    )
    NAME_FIELD_NUMBER: _ClassVar[int]
    PATH_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    HAS_WEIGHTS_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_PATH_FIELD_NUMBER: _ClassVar[int]
    YAML_CONTENT_FIELD_NUMBER: _ClassVar[int]
    name: str
    path: str
    metadata: PipelineMetadata
    tags: _containers.RepeatedScalarFieldContainer[str]
    has_weights: bool
    weights_path: str
    yaml_content: str
    def __init__(
        self,
        name: _Optional[str] = ...,
        path: _Optional[str] = ...,
        metadata: _Optional[_Union[PipelineMetadata, _Mapping]] = ...,
        tags: _Optional[_Iterable[str]] = ...,
        has_weights: _Optional[bool] = ...,
        weights_path: _Optional[str] = ...,
        yaml_content: _Optional[str] = ...,
    ) -> None: ...

class TrainingConfig(_message.Message):
    __slots__ = ("config_bytes",)
    CONFIG_BYTES_FIELD_NUMBER: _ClassVar[int]
    config_bytes: bytes
    def __init__(self, config_bytes: _Optional[bytes] = ...) -> None: ...

class TrainRunConfig(_message.Message):
    __slots__ = ("config_bytes",)
    CONFIG_BYTES_FIELD_NUMBER: _ClassVar[int]
    config_bytes: bytes
    def __init__(self, config_bytes: _Optional[bytes] = ...) -> None: ...

class BoundingBox(_message.Message):
    __slots__ = ("element_id", "x_min", "y_min", "x_max", "y_max")
    ELEMENT_ID_FIELD_NUMBER: _ClassVar[int]
    X_MIN_FIELD_NUMBER: _ClassVar[int]
    Y_MIN_FIELD_NUMBER: _ClassVar[int]
    X_MAX_FIELD_NUMBER: _ClassVar[int]
    Y_MAX_FIELD_NUMBER: _ClassVar[int]
    element_id: int
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    def __init__(
        self,
        element_id: _Optional[int] = ...,
        x_min: _Optional[float] = ...,
        y_min: _Optional[float] = ...,
        x_max: _Optional[float] = ...,
        y_max: _Optional[float] = ...,
    ) -> None: ...

class BoundingBoxes(_message.Message):
    __slots__ = ("boxes",)
    BOXES_FIELD_NUMBER: _ClassVar[int]
    boxes: _containers.RepeatedCompositeFieldContainer[BoundingBox]
    def __init__(
        self, boxes: _Optional[_Iterable[_Union[BoundingBox, _Mapping]]] = ...
    ) -> None: ...

class Point(_message.Message):
    __slots__ = ("element_id", "x", "y", "type")
    ELEMENT_ID_FIELD_NUMBER: _ClassVar[int]
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    element_id: int
    x: float
    y: float
    type: PointType
    def __init__(
        self,
        element_id: _Optional[int] = ...,
        x: _Optional[float] = ...,
        y: _Optional[float] = ...,
        type: _Optional[_Union[PointType, str]] = ...,
    ) -> None: ...

class Points(_message.Message):
    __slots__ = ("points",)
    POINTS_FIELD_NUMBER: _ClassVar[int]
    points: _containers.RepeatedCompositeFieldContainer[Point]
    def __init__(
        self, points: _Optional[_Iterable[_Union[Point, _Mapping]]] = ...
    ) -> None: ...

class InputBatch(_message.Message):
    __slots__ = (
        "wavelengths",
        "cube",
        "mask",
        "bboxes",
        "points",
        "text_prompt",
        "extra_inputs",
    )
    class ExtraInputsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Tensor
        def __init__(
            self,
            key: _Optional[str] = ...,
            value: _Optional[_Union[Tensor, _Mapping]] = ...,
        ) -> None: ...

    WAVELENGTHS_FIELD_NUMBER: _ClassVar[int]
    CUBE_FIELD_NUMBER: _ClassVar[int]
    MASK_FIELD_NUMBER: _ClassVar[int]
    BBOXES_FIELD_NUMBER: _ClassVar[int]
    POINTS_FIELD_NUMBER: _ClassVar[int]
    TEXT_PROMPT_FIELD_NUMBER: _ClassVar[int]
    EXTRA_INPUTS_FIELD_NUMBER: _ClassVar[int]
    wavelengths: Tensor
    cube: Tensor
    mask: Tensor
    bboxes: BoundingBoxes
    points: Points
    text_prompt: str
    extra_inputs: _containers.MessageMap[str, Tensor]
    def __init__(
        self,
        wavelengths: _Optional[_Union[Tensor, _Mapping]] = ...,
        cube: _Optional[_Union[Tensor, _Mapping]] = ...,
        mask: _Optional[_Union[Tensor, _Mapping]] = ...,
        bboxes: _Optional[_Union[BoundingBoxes, _Mapping]] = ...,
        points: _Optional[_Union[Points, _Mapping]] = ...,
        text_prompt: _Optional[str] = ...,
        extra_inputs: _Optional[_Mapping[str, Tensor]] = ...,
    ) -> None: ...

class TensorSpec(_message.Message):
    __slots__ = ("name", "shape", "dtype", "required")
    NAME_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    REQUIRED_FIELD_NUMBER: _ClassVar[int]
    name: str
    shape: _containers.RepeatedScalarFieldContainer[int]
    dtype: DType
    required: bool
    def __init__(
        self,
        name: _Optional[str] = ...,
        shape: _Optional[_Iterable[int]] = ...,
        dtype: _Optional[_Union[DType, str]] = ...,
        required: _Optional[bool] = ...,
    ) -> None: ...

class TrainResponse(_message.Message):
    __slots__ = ("context", "losses", "metrics", "status", "message")
    class LossesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(
            self, key: _Optional[str] = ..., value: _Optional[float] = ...
        ) -> None: ...

    class MetricsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(
            self, key: _Optional[str] = ..., value: _Optional[float] = ...
        ) -> None: ...

    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    LOSSES_FIELD_NUMBER: _ClassVar[int]
    METRICS_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    context: Context
    losses: _containers.ScalarMap[str, float]
    metrics: _containers.ScalarMap[str, float]
    status: TrainStatus
    message: str
    def __init__(
        self,
        context: _Optional[_Union[Context, _Mapping]] = ...,
        losses: _Optional[_Mapping[str, float]] = ...,
        metrics: _Optional[_Mapping[str, float]] = ...,
        status: _Optional[_Union[TrainStatus, str]] = ...,
        message: _Optional[str] = ...,
    ) -> None: ...

class ParamSpec(_message.Message):
    __slots__ = (
        "name",
        "type",
        "required",
        "default_value",
        "description",
        "validation",
    )
    NAME_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    REQUIRED_FIELD_NUMBER: _ClassVar[int]
    DEFAULT_VALUE_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    VALIDATION_FIELD_NUMBER: _ClassVar[int]
    name: str
    type: str
    required: bool
    default_value: str
    description: str
    validation: str
    def __init__(
        self,
        name: _Optional[str] = ...,
        type: _Optional[str] = ...,
        required: _Optional[bool] = ...,
        default_value: _Optional[str] = ...,
        description: _Optional[str] = ...,
        validation: _Optional[str] = ...,
    ) -> None: ...

class CallbackTypeInfo(_message.Message):
    __slots__ = ("type", "description", "parameters")
    TYPE_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    type: str
    description: str
    parameters: _containers.RepeatedCompositeFieldContainer[ParamSpec]
    def __init__(
        self,
        type: _Optional[str] = ...,
        description: _Optional[str] = ...,
        parameters: _Optional[_Iterable[_Union[ParamSpec, _Mapping]]] = ...,
    ) -> None: ...

class OptimizerParamsSchema(_message.Message):
    __slots__ = ("parameters",)
    PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    parameters: _containers.RepeatedCompositeFieldContainer[ParamSpec]
    def __init__(
        self, parameters: _Optional[_Iterable[_Union[ParamSpec, _Mapping]]] = ...
    ) -> None: ...

class SchedulerParamsSchema(_message.Message):
    __slots__ = ("parameters",)
    PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    parameters: _containers.RepeatedCompositeFieldContainer[ParamSpec]
    def __init__(
        self, parameters: _Optional[_Iterable[_Union[ParamSpec, _Mapping]]] = ...
    ) -> None: ...

class ListAvailablePipelineesRequest(_message.Message):
    __slots__ = ("filter_tag",)
    FILTER_TAG_FIELD_NUMBER: _ClassVar[int]
    filter_tag: str
    def __init__(self, filter_tag: _Optional[str] = ...) -> None: ...

class ListAvailablePipelineesResponse(_message.Message):
    __slots__ = ("pipelinees",)
    PIPELINEES_FIELD_NUMBER: _ClassVar[int]
    pipelinees: _containers.RepeatedCompositeFieldContainer[PipelineInfo]
    def __init__(
        self, pipelinees: _Optional[_Iterable[_Union[PipelineInfo, _Mapping]]] = ...
    ) -> None: ...

class GetPipelineInfoRequest(_message.Message):
    __slots__ = ("pipeline_name",)
    PIPELINE_NAME_FIELD_NUMBER: _ClassVar[int]
    pipeline_name: str
    def __init__(self, pipeline_name: _Optional[str] = ...) -> None: ...

class GetPipelineInfoResponse(_message.Message):
    __slots__ = ("pipeline_info",)
    PIPELINE_INFO_FIELD_NUMBER: _ClassVar[int]
    pipeline_info: PipelineInfo
    def __init__(
        self, pipeline_info: _Optional[_Union[PipelineInfo, _Mapping]] = ...
    ) -> None: ...

class CreateSessionRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class CreateSessionResponse(_message.Message):
    __slots__ = ("session_id",)
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: _Optional[str] = ...) -> None: ...

class SetSessionSearchPathsRequest(_message.Message):
    __slots__ = ("session_id", "search_paths", "append")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    SEARCH_PATHS_FIELD_NUMBER: _ClassVar[int]
    APPEND_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    search_paths: _containers.RepeatedScalarFieldContainer[str]
    append: bool
    def __init__(
        self,
        session_id: _Optional[str] = ...,
        search_paths: _Optional[_Iterable[str]] = ...,
        append: _Optional[bool] = ...,
    ) -> None: ...

class SetSessionSearchPathsResponse(_message.Message):
    __slots__ = ("success", "current_paths", "rejected_paths")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    CURRENT_PATHS_FIELD_NUMBER: _ClassVar[int]
    REJECTED_PATHS_FIELD_NUMBER: _ClassVar[int]
    success: bool
    current_paths: _containers.RepeatedScalarFieldContainer[str]
    rejected_paths: _containers.RepeatedScalarFieldContainer[str]
    def __init__(
        self,
        success: _Optional[bool] = ...,
        current_paths: _Optional[_Iterable[str]] = ...,
        rejected_paths: _Optional[_Iterable[str]] = ...,
    ) -> None: ...

class CloseSessionRequest(_message.Message):
    __slots__ = ("session_id",)
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: _Optional[str] = ...) -> None: ...

class CloseSessionResponse(_message.Message):
    __slots__ = ("success",)
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    success: bool
    def __init__(self, success: _Optional[bool] = ...) -> None: ...

class ResolveConfigRequest(_message.Message):
    __slots__ = ("session_id", "config_type", "path", "overrides")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    CONFIG_TYPE_FIELD_NUMBER: _ClassVar[int]
    PATH_FIELD_NUMBER: _ClassVar[int]
    OVERRIDES_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    config_type: str
    path: str
    overrides: _containers.RepeatedScalarFieldContainer[str]
    def __init__(
        self,
        session_id: _Optional[str] = ...,
        config_type: _Optional[str] = ...,
        path: _Optional[str] = ...,
        overrides: _Optional[_Iterable[str]] = ...,
    ) -> None: ...

class ResolveConfigResponse(_message.Message):
    __slots__ = ("config_bytes",)
    CONFIG_BYTES_FIELD_NUMBER: _ClassVar[int]
    config_bytes: bytes
    def __init__(self, config_bytes: _Optional[bytes] = ...) -> None: ...

class GetParameterSchemaRequest(_message.Message):
    __slots__ = ("config_type",)
    CONFIG_TYPE_FIELD_NUMBER: _ClassVar[int]
    config_type: str
    def __init__(self, config_type: _Optional[str] = ...) -> None: ...

class GetParameterSchemaResponse(_message.Message):
    __slots__ = ("json_schema",)
    JSON_SCHEMA_FIELD_NUMBER: _ClassVar[int]
    json_schema: str
    def __init__(self, json_schema: _Optional[str] = ...) -> None: ...

class ValidateConfigRequest(_message.Message):
    __slots__ = ("config_type", "config_bytes")
    CONFIG_TYPE_FIELD_NUMBER: _ClassVar[int]
    CONFIG_BYTES_FIELD_NUMBER: _ClassVar[int]
    config_type: str
    config_bytes: bytes
    def __init__(
        self, config_type: _Optional[str] = ..., config_bytes: _Optional[bytes] = ...
    ) -> None: ...

class ValidateConfigResponse(_message.Message):
    __slots__ = ("valid", "errors", "warnings")
    VALID_FIELD_NUMBER: _ClassVar[int]
    ERRORS_FIELD_NUMBER: _ClassVar[int]
    WARNINGS_FIELD_NUMBER: _ClassVar[int]
    valid: bool
    errors: _containers.RepeatedScalarFieldContainer[str]
    warnings: _containers.RepeatedScalarFieldContainer[str]
    def __init__(
        self,
        valid: _Optional[bool] = ...,
        errors: _Optional[_Iterable[str]] = ...,
        warnings: _Optional[_Iterable[str]] = ...,
    ) -> None: ...

class LoadPipelineWeightsRequest(_message.Message):
    __slots__ = ("session_id", "weights_path", "weights_bytes", "strict")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_PATH_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_BYTES_FIELD_NUMBER: _ClassVar[int]
    STRICT_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    weights_path: str
    weights_bytes: bytes
    strict: bool
    def __init__(
        self,
        session_id: _Optional[str] = ...,
        weights_path: _Optional[str] = ...,
        weights_bytes: _Optional[bytes] = ...,
        strict: _Optional[bool] = ...,
    ) -> None: ...

class LoadPipelineWeightsResponse(_message.Message):
    __slots__ = ("success", "resolved_path")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    RESOLVED_PATH_FIELD_NUMBER: _ClassVar[int]
    success: bool
    resolved_path: str
    def __init__(
        self, success: _Optional[bool] = ..., resolved_path: _Optional[str] = ...
    ) -> None: ...

class SetTrainRunConfigRequest(_message.Message):
    __slots__ = ("session_id", "config")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    config: TrainRunConfig
    def __init__(
        self,
        session_id: _Optional[str] = ...,
        config: _Optional[_Union[TrainRunConfig, _Mapping]] = ...,
    ) -> None: ...

class SetTrainRunConfigResponse(_message.Message):
    __slots__ = ("success", "pipeline_from_config")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    PIPELINE_FROM_CONFIG_FIELD_NUMBER: _ClassVar[int]
    success: bool
    pipeline_from_config: bool
    def __init__(
        self,
        success: _Optional[bool] = ...,
        pipeline_from_config: _Optional[bool] = ...,
    ) -> None: ...

class TrainRequest(_message.Message):
    __slots__ = ("session_id", "trainer_type", "data", "training")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    TRAINER_TYPE_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    TRAINING_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    trainer_type: TrainerType
    data: DataConfig
    training: TrainingConfig
    def __init__(
        self,
        session_id: _Optional[str] = ...,
        trainer_type: _Optional[_Union[TrainerType, str]] = ...,
        data: _Optional[_Union[DataConfig, _Mapping]] = ...,
        training: _Optional[_Union[TrainingConfig, _Mapping]] = ...,
    ) -> None: ...

class GetTrainStatusRequest(_message.Message):
    __slots__ = ("session_id",)
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: _Optional[str] = ...) -> None: ...

class GetTrainStatusResponse(_message.Message):
    __slots__ = ("latest_progress",)
    LATEST_PROGRESS_FIELD_NUMBER: _ClassVar[int]
    latest_progress: TrainResponse
    def __init__(
        self, latest_progress: _Optional[_Union[TrainResponse, _Mapping]] = ...
    ) -> None: ...

class GetTrainingCapabilitiesRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class GetTrainingCapabilitiesResponse(_message.Message):
    __slots__ = (
        "supported_optimizers",
        "supported_schedulers",
        "supported_callbacks",
        "optimizer_params",
        "scheduler_params",
    )
    SUPPORTED_OPTIMIZERS_FIELD_NUMBER: _ClassVar[int]
    SUPPORTED_SCHEDULERS_FIELD_NUMBER: _ClassVar[int]
    SUPPORTED_CALLBACKS_FIELD_NUMBER: _ClassVar[int]
    OPTIMIZER_PARAMS_FIELD_NUMBER: _ClassVar[int]
    SCHEDULER_PARAMS_FIELD_NUMBER: _ClassVar[int]
    supported_optimizers: _containers.RepeatedScalarFieldContainer[str]
    supported_schedulers: _containers.RepeatedScalarFieldContainer[str]
    supported_callbacks: _containers.RepeatedCompositeFieldContainer[CallbackTypeInfo]
    optimizer_params: OptimizerParamsSchema
    scheduler_params: SchedulerParamsSchema
    def __init__(
        self,
        supported_optimizers: _Optional[_Iterable[str]] = ...,
        supported_schedulers: _Optional[_Iterable[str]] = ...,
        supported_callbacks: _Optional[
            _Iterable[_Union[CallbackTypeInfo, _Mapping]]
        ] = ...,
        optimizer_params: _Optional[_Union[OptimizerParamsSchema, _Mapping]] = ...,
        scheduler_params: _Optional[_Union[SchedulerParamsSchema, _Mapping]] = ...,
    ) -> None: ...

class SavePipelineRequest(_message.Message):
    __slots__ = ("session_id", "pipeline_path", "metadata")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    PIPELINE_PATH_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    pipeline_path: str
    metadata: PipelineMetadata
    def __init__(
        self,
        session_id: _Optional[str] = ...,
        pipeline_path: _Optional[str] = ...,
        metadata: _Optional[_Union[PipelineMetadata, _Mapping]] = ...,
    ) -> None: ...

class SavePipelineResponse(_message.Message):
    __slots__ = ("success", "pipeline_path", "weights_path")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    PIPELINE_PATH_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_PATH_FIELD_NUMBER: _ClassVar[int]
    success: bool
    pipeline_path: str
    weights_path: str
    def __init__(
        self,
        success: _Optional[bool] = ...,
        pipeline_path: _Optional[str] = ...,
        weights_path: _Optional[str] = ...,
    ) -> None: ...

class LoadPipelineRequest(_message.Message):
    __slots__ = ("session_id", "pipeline")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    PIPELINE_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    pipeline: PipelineConfig
    def __init__(
        self,
        session_id: _Optional[str] = ...,
        pipeline: _Optional[_Union[PipelineConfig, _Mapping]] = ...,
    ) -> None: ...

class LoadPipelineResponse(_message.Message):
    __slots__ = ("success", "metadata")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    success: bool
    metadata: PipelineMetadata
    def __init__(
        self,
        success: _Optional[bool] = ...,
        metadata: _Optional[_Union[PipelineMetadata, _Mapping]] = ...,
    ) -> None: ...

class SaveTrainRunRequest(_message.Message):
    __slots__ = ("session_id", "trainrun_path", "save_weights")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    TRAINRUN_PATH_FIELD_NUMBER: _ClassVar[int]
    SAVE_WEIGHTS_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    trainrun_path: str
    save_weights: bool
    def __init__(
        self,
        session_id: _Optional[str] = ...,
        trainrun_path: _Optional[str] = ...,
        save_weights: _Optional[bool] = ...,
    ) -> None: ...

class SaveTrainRunResponse(_message.Message):
    __slots__ = ("success", "trainrun_path", "pipeline_path", "weights_path")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    TRAINRUN_PATH_FIELD_NUMBER: _ClassVar[int]
    PIPELINE_PATH_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_PATH_FIELD_NUMBER: _ClassVar[int]
    success: bool
    trainrun_path: str
    pipeline_path: str
    weights_path: str
    def __init__(
        self,
        success: _Optional[bool] = ...,
        trainrun_path: _Optional[str] = ...,
        pipeline_path: _Optional[str] = ...,
        weights_path: _Optional[str] = ...,
    ) -> None: ...

class RestoreTrainRunRequest(_message.Message):
    __slots__ = ("trainrun_path", "weights_path", "strict")
    TRAINRUN_PATH_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_PATH_FIELD_NUMBER: _ClassVar[int]
    STRICT_FIELD_NUMBER: _ClassVar[int]
    trainrun_path: str
    weights_path: str
    strict: bool
    def __init__(
        self,
        trainrun_path: _Optional[str] = ...,
        weights_path: _Optional[str] = ...,
        strict: _Optional[bool] = ...,
    ) -> None: ...

class RestoreTrainRunResponse(_message.Message):
    __slots__ = ("session_id", "trainrun")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    TRAINRUN_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    trainrun: TrainRunConfig
    def __init__(
        self,
        session_id: _Optional[str] = ...,
        trainrun: _Optional[_Union[TrainRunConfig, _Mapping]] = ...,
    ) -> None: ...

class GetPipelineInputsRequest(_message.Message):
    __slots__ = ("session_id",)
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: _Optional[str] = ...) -> None: ...

class GetPipelineInputsResponse(_message.Message):
    __slots__ = ("input_names", "input_specs")
    class InputSpecsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: TensorSpec
        def __init__(
            self,
            key: _Optional[str] = ...,
            value: _Optional[_Union[TensorSpec, _Mapping]] = ...,
        ) -> None: ...

    INPUT_NAMES_FIELD_NUMBER: _ClassVar[int]
    INPUT_SPECS_FIELD_NUMBER: _ClassVar[int]
    input_names: _containers.RepeatedScalarFieldContainer[str]
    input_specs: _containers.MessageMap[str, TensorSpec]
    def __init__(
        self,
        input_names: _Optional[_Iterable[str]] = ...,
        input_specs: _Optional[_Mapping[str, TensorSpec]] = ...,
    ) -> None: ...

class GetPipelineOutputsRequest(_message.Message):
    __slots__ = ("session_id",)
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: _Optional[str] = ...) -> None: ...

class GetPipelineOutputsResponse(_message.Message):
    __slots__ = ("output_names", "output_specs")
    class OutputSpecsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: TensorSpec
        def __init__(
            self,
            key: _Optional[str] = ...,
            value: _Optional[_Union[TensorSpec, _Mapping]] = ...,
        ) -> None: ...

    OUTPUT_NAMES_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_SPECS_FIELD_NUMBER: _ClassVar[int]
    output_names: _containers.RepeatedScalarFieldContainer[str]
    output_specs: _containers.MessageMap[str, TensorSpec]
    def __init__(
        self,
        output_names: _Optional[_Iterable[str]] = ...,
        output_specs: _Optional[_Mapping[str, TensorSpec]] = ...,
    ) -> None: ...

class GetPipelineVisualizationRequest(_message.Message):
    __slots__ = ("session_id", "format")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    format: str
    def __init__(
        self, session_id: _Optional[str] = ..., format: _Optional[str] = ...
    ) -> None: ...

class GetPipelineVisualizationResponse(_message.Message):
    __slots__ = ("image_data", "format")
    IMAGE_DATA_FIELD_NUMBER: _ClassVar[int]
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    image_data: bytes
    format: str
    def __init__(
        self, image_data: _Optional[bytes] = ..., format: _Optional[str] = ...
    ) -> None: ...

class InferenceRequest(_message.Message):
    __slots__ = ("session_id", "inputs", "output_specs")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    INPUTS_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_SPECS_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    inputs: InputBatch
    output_specs: _containers.RepeatedScalarFieldContainer[str]
    def __init__(
        self,
        session_id: _Optional[str] = ...,
        inputs: _Optional[_Union[InputBatch, _Mapping]] = ...,
        output_specs: _Optional[_Iterable[str]] = ...,
    ) -> None: ...

class InferenceResponse(_message.Message):
    __slots__ = ("outputs", "metrics")
    class OutputsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Tensor
        def __init__(
            self,
            key: _Optional[str] = ...,
            value: _Optional[_Union[Tensor, _Mapping]] = ...,
        ) -> None: ...

    class MetricsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(
            self, key: _Optional[str] = ..., value: _Optional[float] = ...
        ) -> None: ...

    OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    METRICS_FIELD_NUMBER: _ClassVar[int]
    outputs: _containers.MessageMap[str, Tensor]
    metrics: _containers.ScalarMap[str, float]
    def __init__(
        self,
        outputs: _Optional[_Mapping[str, Tensor]] = ...,
        metrics: _Optional[_Mapping[str, float]] = ...,
    ) -> None: ...

class PluginManifest(_message.Message):
    __slots__ = ("config_bytes",)
    CONFIG_BYTES_FIELD_NUMBER: _ClassVar[int]
    config_bytes: bytes
    def __init__(self, config_bytes: _Optional[bytes] = ...) -> None: ...

class PluginInfo(_message.Message):
    __slots__ = ("name", "type", "source", "ref", "provides")
    NAME_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    REF_FIELD_NUMBER: _ClassVar[int]
    PROVIDES_FIELD_NUMBER: _ClassVar[int]
    name: str
    type: str
    source: str
    ref: str
    provides: _containers.RepeatedScalarFieldContainer[str]
    def __init__(
        self,
        name: _Optional[str] = ...,
        type: _Optional[str] = ...,
        source: _Optional[str] = ...,
        ref: _Optional[str] = ...,
        provides: _Optional[_Iterable[str]] = ...,
    ) -> None: ...

class NodeInfo(_message.Message):
    __slots__ = ("class_name", "full_path", "source", "plugin_name")
    CLASS_NAME_FIELD_NUMBER: _ClassVar[int]
    FULL_PATH_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    PLUGIN_NAME_FIELD_NUMBER: _ClassVar[int]
    class_name: str
    full_path: str
    source: str
    plugin_name: str
    def __init__(
        self,
        class_name: _Optional[str] = ...,
        full_path: _Optional[str] = ...,
        source: _Optional[str] = ...,
        plugin_name: _Optional[str] = ...,
    ) -> None: ...

class LoadPluginsRequest(_message.Message):
    __slots__ = ("session_id", "manifest")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    MANIFEST_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    manifest: PluginManifest
    def __init__(
        self,
        session_id: _Optional[str] = ...,
        manifest: _Optional[_Union[PluginManifest, _Mapping]] = ...,
    ) -> None: ...

class LoadPluginsResponse(_message.Message):
    __slots__ = ("loaded_plugins", "failed_plugins")
    class FailedPluginsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(
            self, key: _Optional[str] = ..., value: _Optional[str] = ...
        ) -> None: ...

    LOADED_PLUGINS_FIELD_NUMBER: _ClassVar[int]
    FAILED_PLUGINS_FIELD_NUMBER: _ClassVar[int]
    loaded_plugins: _containers.RepeatedScalarFieldContainer[str]
    failed_plugins: _containers.ScalarMap[str, str]
    def __init__(
        self,
        loaded_plugins: _Optional[_Iterable[str]] = ...,
        failed_plugins: _Optional[_Mapping[str, str]] = ...,
    ) -> None: ...

class ListLoadedPluginsRequest(_message.Message):
    __slots__ = ("session_id",)
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: _Optional[str] = ...) -> None: ...

class ListLoadedPluginsResponse(_message.Message):
    __slots__ = ("plugins",)
    PLUGINS_FIELD_NUMBER: _ClassVar[int]
    plugins: _containers.RepeatedCompositeFieldContainer[PluginInfo]
    def __init__(
        self, plugins: _Optional[_Iterable[_Union[PluginInfo, _Mapping]]] = ...
    ) -> None: ...

class GetPluginInfoRequest(_message.Message):
    __slots__ = ("session_id", "plugin_name")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    PLUGIN_NAME_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    plugin_name: str
    def __init__(
        self, session_id: _Optional[str] = ..., plugin_name: _Optional[str] = ...
    ) -> None: ...

class GetPluginInfoResponse(_message.Message):
    __slots__ = ("plugin",)
    PLUGIN_FIELD_NUMBER: _ClassVar[int]
    plugin: PluginInfo
    def __init__(
        self, plugin: _Optional[_Union[PluginInfo, _Mapping]] = ...
    ) -> None: ...

class ListAvailableNodesRequest(_message.Message):
    __slots__ = ("session_id",)
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: _Optional[str] = ...) -> None: ...

class ListAvailableNodesResponse(_message.Message):
    __slots__ = ("nodes",)
    NODES_FIELD_NUMBER: _ClassVar[int]
    nodes: _containers.RepeatedCompositeFieldContainer[NodeInfo]
    def __init__(
        self, nodes: _Optional[_Iterable[_Union[NodeInfo, _Mapping]]] = ...
    ) -> None: ...

class ClearPluginCacheRequest(_message.Message):
    __slots__ = ("plugin_name",)
    PLUGIN_NAME_FIELD_NUMBER: _ClassVar[int]
    plugin_name: str
    def __init__(self, plugin_name: _Optional[str] = ...) -> None: ...

class ClearPluginCacheResponse(_message.Message):
    __slots__ = ("cleared_count",)
    CLEARED_COUNT_FIELD_NUMBER: _ClassVar[int]
    cleared_count: int
    def __init__(self, cleared_count: _Optional[int] = ...) -> None: ...
