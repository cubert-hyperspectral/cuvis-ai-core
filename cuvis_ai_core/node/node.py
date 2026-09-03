from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from types import SimpleNamespace
from collections.abc import Iterable
from typing import Any, Mapping

from torch import nn

from cuvis_ai_schemas.enums import ExecutionStage, NodeCategory, NodeTag
from cuvis_ai_schemas.execution import InputStream
from cuvis_ai_schemas.pipeline import InputPort, OutputPort, PortSpec

from cuvis_ai_core.utils.serializer import Serializable


def _coerce_execution_stages(value: Any, *, owner: str) -> frozenset[ExecutionStage]:
    """Normalize an execution-stage declaration to ``frozenset[ExecutionStage]``.

    Accepts any iterable of ``ExecutionStage`` members or their string values
    (``"train"``, ``"val"``, ``"test"``, ``"inference"``, ``"always"``). A bare
    string or a non-iterable is a ``TypeError``; an unknown stage name is a
    ``ValueError`` naming ``owner``, so a misspelled entry in a pipeline yaml
    fails at load time instead of silently never running the node.
    """
    if isinstance(value, (str, bytes, ExecutionStage)) or not isinstance(
        value, Iterable
    ):
        raise TypeError(
            f"{owner}: execution_stages must be an iterable of ExecutionStage "
            f"members or stage names, got {value!r}."
        )
    stages: set[ExecutionStage] = set()
    for item in value:
        if isinstance(item, ExecutionStage):
            stages.add(item)
        elif isinstance(item, str):
            try:
                stages.add(ExecutionStage(item))
            except ValueError:
                valid = ", ".join(sorted(stage.value for stage in ExecutionStage))
                raise ValueError(
                    f"{owner}: unknown execution stage {item!r}; "
                    f"valid stages are {valid}."
                ) from None
        else:
            raise TypeError(
                f"{owner}: execution stage entries must be ExecutionStage members "
                f"or stage names, got {item!r}."
            )
    return frozenset(stages)


class Node(nn.Module, ABC, Serializable):
    """
    Abstract class for data preprocessing.

    Node Serialization Requirements
    ================================

    All nodes MUST support serialization for pipeline save/load:

    1. Nodes inheriting from Node (nn.Module):
       - Already have state_dict() and load_state_dict() from PyTorch
       - Ensure all learnable parameters are registered properly
       - Use register_buffer() for non-trainable state

    2. Stateless nodes:
       - state_dict() returns empty dict {} (default PyTorch behavior)
       - load_state_dict() is no-op (default PyTorch behavior)

    3. Statistical nodes (requires_initial_fit=True):
       - Store fitted parameters as buffers (register_buffer())
       - Unfreeze converts buffers to nn.Parameters
       - Both states serialize correctly via state_dict()

    4. Custom serialization needs:
       - Override state_dict() to include custom state
       - Override load_state_dict() to restore custom state
       - Must call super().state_dict() / super().load_state_dict()

    Example - Custom State:
        >>> class CustomNode(Node):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.custom_data = {"key": "value"}
        ...
        ...     def state_dict(self):
        ...         state = super().state_dict()
        ...         state["custom_data"] = self.custom_data
        ...         return state
        ...
        ...     def load_state_dict(self, state_dict):
        ...         self.custom_data = state_dict.pop("custom_data", {})
        ...         super().load_state_dict(state_dict)
    """

    INPUT_SPECS: dict[str, PortSpec] = {}
    OUTPUT_SPECS: dict[str, PortSpec] = {}
    TRAINABLE_BUFFERS: tuple[str, ...] = ()

    # Node taxonomy metadata. Subclasses override on the class body; populated
    # values are surfaced over gRPC via NodeInfo and consumed by the palette /
    # visualizer. Defaults are deliberately permissive so unannotated nodes
    # still compile — runtime warns at pipeline-add time.
    _category: NodeCategory = NodeCategory.UNSPECIFIED
    _tags: frozenset[NodeTag] = frozenset()
    _icon_name: str | None = None

    # When the node runs. A class-level declaration like ``_category`` / ``_tags``:
    # subclasses override ``EXECUTION_STAGES`` on the class body (a set literal of
    # ``ExecutionStage`` members or stage names is normalized to a frozenset at
    # class creation). Instances read it through ``execution_stages``; a
    # constructor ``execution_stages=`` argument overrides it for that instance
    # (which is how a pipeline yaml's ``hparams`` can move a node), and assignment
    # after construction is refused unless the subclass opts in with
    # ``EXECUTION_STAGES_MUTABLE = True``.
    EXECUTION_STAGES: frozenset[ExecutionStage] = frozenset({ExecutionStage.ALWAYS})
    EXECUTION_STAGES_MUTABLE: bool = False

    @classmethod
    def get_category(cls) -> NodeCategory:
        return cls._category

    @classmethod
    def get_tags(cls) -> frozenset[NodeTag]:
        return cls._tags

    @classmethod
    def get_icon_name(cls) -> str | None:
        return cls._icon_name

    @classmethod
    def get_execution_stages(cls) -> frozenset[ExecutionStage]:
        """Stages the class runs in by default (before any per-instance override)."""
        return cls.EXECUTION_STAGES

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Validate class-level declarations at class definition time.

        ``TRAINABLE_BUFFERS`` must be a tuple of strings; ``EXECUTION_STAGES`` is
        normalized to a ``frozenset[ExecutionStage]`` (a set literal or stage
        names are accepted, anything else raises); ``EXECUTION_STAGES_MUTABLE``
        must be a bool.
        """
        super().__init_subclass__(**kwargs)
        if "TRAINABLE_BUFFERS" in cls.__dict__:
            tb = cls.__dict__["TRAINABLE_BUFFERS"]
            if (
                tb is None
                or not isinstance(tb, tuple)
                or not all(isinstance(n, str) for n in tb)
            ):
                raise TypeError(
                    f"{cls.__name__}.TRAINABLE_BUFFERS must be a tuple of strings."
                )
        if "EXECUTION_STAGES" in cls.__dict__:
            cls.EXECUTION_STAGES = _coerce_execution_stages(
                cls.__dict__["EXECUTION_STAGES"],
                owner=f"{cls.__name__}.EXECUTION_STAGES",
            )
        if "EXECUTION_STAGES_MUTABLE" in cls.__dict__ and not isinstance(
            cls.__dict__["EXECUTION_STAGES_MUTABLE"], bool
        ):
            raise TypeError(f"{cls.__name__}.EXECUTION_STAGES_MUTABLE must be a bool.")

    def __init__(
        self,
        name: str | None = None,
        execution_stages: Iterable[ExecutionStage | str] | None = None,
        *args,
        **kwargs,
    ) -> None:
        """Initialize node with execution stage control.

        Parameters
        ----------
        name : str, optional
            Custom name for the node. If not provided, uses class name.
            Useful for loss/metric nodes to enable semantic logging names.
        execution_stages : iterable of ExecutionStage or stage name, optional
            Per-instance override of the class-level ``EXECUTION_STAGES``. Omit
            it (the default) to run in the stages the class declares. Members or
            their string values are accepted, so a pipeline yaml can carry
            ``hparams: {execution_stages: [inference]}``; an unknown name raises
            ``ValueError`` naming the node. Stages:
            - ExecutionStage.ALWAYS: Execute in all stages (base-class default)
            - ExecutionStage.TRAIN: Only during training
            - ExecutionStage.VAL: Only during validation
            - ExecutionStage.TEST: Only during testing
            - ExecutionStage.INFERENCE: Only during inference
            - {ExecutionStage.TRAIN, ExecutionStage.VAL}: Multiple stages
            The value is popped before hparams are captured, so ``serialize()``
            never writes it: an override lives in the yaml text that carried it.
        """
        # Allow subclasses to forward name/execution_stages inside kwargs without duplication
        if kwargs:
            name = kwargs.pop("name", name)
            execution_stages = kwargs.pop("execution_stages", execution_stages)
        # Initialize Serializable first to capture hparams
        Serializable.__init__(self, *args, **kwargs)
        # Then initialize nn.Module without any args/kwargs
        nn.Module.__init__(self)

        if name is None:
            name = type(self).__name__
        # Store custom name
        self._name = name
        self._pipeline_counter: int | None = None

        # Execution stages: the class declaration unless this instance overrides it.
        if execution_stages is None:
            self._execution_stages = type(self).EXECUTION_STAGES
        else:
            self._execution_stages = _coerce_execution_stages(
                execution_stages, owner=f"{type(self).__name__} {name!r}"
            )

        self._statistically_initialized = False
        self._frozen = False
        self._input_ports: dict[str, InputPort] = {}
        self._output_ports: dict[str, OutputPort] = {}
        self._create_ports()

    @property
    def frozen(self) -> bool:
        """Whether this node is frozen (no gradient computation)."""
        return self._frozen

    @property
    def name(self) -> str:
        """Get node name (base name with optional counter suffix)."""
        if self._pipeline_counter is None:
            return self._name
        if self._pipeline_counter == 0:
            return self._name
        return f"{self._name}-{self._pipeline_counter}"

    @name.setter
    def name(self, value: str) -> None:  # noqa: D401
        """Disallow mutation of node names to avoid breaking graph keys."""
        raise AttributeError(
            f"Node name is immutable. Cannot change '{self.name}' to '{value}'."
        )

    @property
    def execution_stages(self) -> frozenset[ExecutionStage]:
        """Stages this instance runs in (class default or constructor override)."""
        return self._execution_stages

    @execution_stages.setter
    def execution_stages(self, value: Iterable[ExecutionStage | str]) -> None:
        """Refuse reassignment unless the class opted in with EXECUTION_STAGES_MUTABLE."""
        if not type(self).EXECUTION_STAGES_MUTABLE:
            raise AttributeError(
                f"{type(self).__name__}.execution_stages is declared on the class "
                "(EXECUTION_STAGES) and cannot be reassigned on an instance. Pass "
                "execution_stages=... to the constructor for a per-instance override, "
                "or set EXECUTION_STAGES_MUTABLE = True on the subclass."
            )
        self._execution_stages = _coerce_execution_stages(
            value, owner=f"{type(self).__name__} {self.name!r}"
        )

    @property
    def requires_initial_fit(self) -> bool:
        """Auto-detect whether the node implements statistical initialization."""
        override = getattr(self, "_requires_initial_fit_override", None)
        if override is not None:
            return bool(override)

        impl = getattr(self.__class__, "statistical_initialization", None)
        return callable(impl) and impl is not Node.statistical_initialization

    def statistical_initialization(self, input_stream: InputStream) -> None:
        """Statistical initialization from a port-based input stream."""
        return None

    def fit(self, input_stream: InputStream) -> None:
        """Backward-compatible alias for `statistical_initialization`."""
        warnings.warn(
            "Node.fit() is deprecated; use statistical_initialization() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        impl = getattr(self.__class__, "statistical_initialization", None)
        if self.requires_initial_fit and (
            impl is None or impl is Node.statistical_initialization
        ):
            raise NotImplementedError(
                f"{self.__class__.__name__} requires statistical_initialization() implementation"
            )
        self.statistical_initialization(input_stream)

    def unfreeze(self) -> None:
        """Enable gradient computation for this node's parameters.

        Buffers listed in TRAINABLE_BUFFERS are automatically converted to
        nn.Parameters. Subclasses with non-standard patterns (e.g. nn.Conv2d
        layers) should override both freeze() and unfreeze() instead.
        """
        for name in self.TRAINABLE_BUFFERS:
            if name not in self._buffers and name not in self._parameters:
                raise AttributeError(
                    f"{type(self).__name__}.TRAINABLE_BUFFERS declares '{name}' "
                    f"but it is not registered as a buffer or parameter."
                )
            if name in self._buffers:
                buf = self._buffers[name]
                delattr(self, name)
                setattr(self, name, nn.Parameter(buf.clone()))
        self._frozen = False
        self.requires_grad_(True)

    def freeze(self) -> None:
        """Disable gradient computation for this node's parameters.

        nn.Parameters listed in TRAINABLE_BUFFERS are automatically converted
        back to buffers. Subclasses with non-standard patterns should override
        both freeze() and unfreeze() instead.
        """
        for name in self.TRAINABLE_BUFFERS:
            if name not in self._buffers and name not in self._parameters:
                raise AttributeError(
                    f"{type(self).__name__}.TRAINABLE_BUFFERS declares '{name}' "
                    f"but it is not registered as a buffer or parameter."
                )
            if name in self._parameters:
                data = self._parameters[name].detach().clone()
                delattr(self, name)
                self.register_buffer(name, data)
        self._frozen = True
        self.requires_grad_(False)

    def should_execute(self, stage: ExecutionStage | str) -> bool:
        """Check if node should execute in given stage.

        Parameters
        ----------
        stage : ExecutionStage | str
            Execution stage (enum or string): "train", "val", "test", "inference"

        Returns
        -------
        bool
            True if node should execute in this stage
        """
        # Convert string to enum if needed
        if isinstance(stage, str):
            try:
                stage = ExecutionStage(stage)
            except ValueError:
                return False

        return (
            ExecutionStage.ALWAYS in self.execution_stages
            or stage in self.execution_stages
        )

    def validate_serialization_support(self) -> tuple[bool, str]:
        """Validate that node can be serialized.

        Checks:
        - Has state_dict() method
        - Has load_state_dict() method
        - state_dict() returns a dict
        - No exceptions during state_dict() call

        Returns
        -------
        tuple[bool, str]
            Tuple of (is_valid, error_message)

        Example
        -------
        >>> node = MinMaxNormalizer()
        >>> is_valid, message = node.validate_serialization_support()
        >>> assert is_valid, message
        """
        # Check if state_dict exists
        if not hasattr(self, "state_dict"):
            return False, f"Node {self.name} missing state_dict() method"

        # Check if load_state_dict exists
        if not hasattr(self, "load_state_dict"):
            return False, f"Node {self.name} missing load_state_dict() method"

        # Try to call state_dict (should not raise)
        try:
            state = self.state_dict()
            if not isinstance(state, dict):
                return (
                    False,
                    f"Node {self.name} state_dict() must return dict, got {type(state)}",
                )
        except Exception as e:
            return False, f"Node {self.name} state_dict() failed: {e}"

        return True, "OK"

    def cleanup(self) -> None:
        """Release runtime resources held by the node.

        Nodes with external handles or large cached state can override this hook.
        The default implementation is intentionally a no-op.
        """
        return None

    @abstractmethod
    def forward(self, **inputs: Any) -> dict[str, Any]:
        """Execute node computation returning a dictionary of named outputs."""
        raise NotImplementedError

    @staticmethod
    def consume_base_kwargs(
        kwargs: dict[str, Any], default_stages: set[ExecutionStage] | None = None
    ) -> tuple[str | None, set[ExecutionStage] | None]:
        """Extract base Node kwargs centrally to avoid double-passing.

        Subclasses can pop name/execution_stages here before calling super().__init__.
        Kept for plugins written before ``EXECUTION_STAGES`` existed; new code
        declares its default stages on the class body instead and lets
        ``Node.__init__`` handle the optional per-instance override.
        """
        name = kwargs.pop("name", None)
        execution_stages = kwargs.pop("execution_stages", default_stages)
        return name, execution_stages

    def _create_ports(self) -> None:
        """Create port proxy objects from class-level specifications."""
        cls_name = type(self).__name__
        for port_name, port_spec in self.INPUT_SPECS.items():
            if isinstance(port_spec, list):
                raise TypeError(
                    f"{cls_name}.INPUT_SPECS['{port_name}'] is a list. List-form specs "
                    "are no longer supported — declare a single PortSpec and set "
                    "PortSpec(..., variadic=True) for fan-in ports."
                )
            if port_name in self._input_ports:
                raise AttributeError(
                    f"Cannot create input port '{port_name}'; attribute already exists."
                )
            input_port = InputPort(self, port_name, port_spec)
            self._input_ports[port_name] = input_port

        for port_name, port_spec in self.OUTPUT_SPECS.items():
            if isinstance(port_spec, list):
                raise TypeError(
                    f"{cls_name}.OUTPUT_SPECS['{port_name}'] is a list. List-form specs "
                    "are no longer supported — declare a single PortSpec."
                )
            if getattr(port_spec, "variadic", False):
                raise TypeError(
                    f"{cls_name}.OUTPUT_SPECS['{port_name}'] sets variadic=True. "
                    "variadic applies to input ports only."
                )
            if port_name in self._output_ports:
                raise AttributeError(
                    f"Cannot create output port '{port_name}'; attribute already exists."
                )
            output_port = OutputPort(self, port_name, port_spec)
            self._output_ports[port_name] = output_port

    @property
    def inputs(self) -> SimpleNamespace:
        """Access input ports: node.inputs.portname"""
        return SimpleNamespace(**self._input_ports)

    @property
    def outputs(self) -> SimpleNamespace:
        """Access output ports: node.outputs.portname"""
        return SimpleNamespace(**self._output_ports)

    # def __getattr__(self, name):
    #     # Prevent infinite recursion during initialization
    #     if name.startswith('_'):
    #         raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    #     # Check if it's a unique port name
    #     in_inputs = name in self._input_ports
    #     in_outputs = name in self._output_ports

    #     if in_inputs and in_outputs:
    #         # Conflict - require explicit namespace
    #         raise AttributeError(
    #             f"Port '{name}' exists in both inputs and outputs. "
    #             f"Use {self.name}.inputs.{name} or {self.name}.outputs.{name}"
    #         )
    #     elif in_inputs:
    #         return self._input_ports[name]
    #     elif in_outputs:
    #         return self._output_ports[name]
    #     else:
    #         # Not a port, let normal AttributeError propagate
    #         raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def __getattr__(self, name: str) -> Any:
        # First, let PyTorch's nn.Module handle its own attributes
        # (parameters, buffers, modules, etc.) before checking for ports.
        # This is critical because nn.Module stores parameters in _parameters dict
        # and expects __getattr__ to not interfere with that lookup.

        # Try to get from nn.Module's dictionaries first (parameters, buffers, modules)
        # This avoids interfering with PyTorch's parameter management
        try:
            # Check _parameters, _buffers, _modules in order (like nn.Module does)
            modules = object.__getattribute__(self, "__dict__")
            if "_parameters" in modules and name in modules["_parameters"]:
                return modules["_parameters"][name]
            if "_buffers" in modules and name in modules["_buffers"]:
                return modules["_buffers"][name]
            if "_modules" in modules and name in modules["_modules"]:
                return modules["_modules"][name]
        except AttributeError:
            pass

        # Prevent infinite recursion for non-PyTorch underscore attributes
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

        # Now check if it's a port
        try:
            input_ports = object.__getattribute__(self, "_input_ports")
            output_ports = object.__getattribute__(self, "_output_ports")
        except AttributeError as err:
            # Ports not initialized yet
            raise AttributeError(
                f"'{type(self).__name__}' has no attribute '{name}'"
            ) from err

        # Check if it's a unique port name
        in_inputs = name in input_ports
        in_outputs = name in output_ports

        if in_inputs and in_outputs:
            # Conflict - require explicit namespace
            raise AttributeError(
                f"Port '{name}' exists in both inputs and outputs. "
                f"Use {self.name}.inputs.{name} or {self.name}.outputs.{name}"
            )
        elif in_inputs:
            return input_ports[name]
        elif in_outputs:
            return output_ports[name]
        else:
            # Not a port or parameter, let normal AttributeError propagate
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ) -> Any:
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self._infer_fitted_state_from_loaded_weights(state_dict)
        return result

    def _infer_fitted_state_from_loaded_weights(self, state_dict) -> None:
        """Infer statistical initialization status from loaded weights."""
        if not state_dict:
            return

        if self.requires_initial_fit:
            if hasattr(self, "_statistically_initialized"):
                self._statistically_initialized = True
