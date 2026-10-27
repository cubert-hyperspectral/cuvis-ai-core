"""Export of a pipeline span: the nodes between declared input and output ports.

A span is cut by port. :func:`select_span` walks back from the output ports to the
declared input ports and refuses anything the exported graph could not stand in
for: an input the pipeline would read from the batch, a node that does not run
at inference, a port consumed outside the span but not declared as an output, and
a cut whose contraction into one node would make the deploy graph cyclic.

:class:`PipelineSpanModule` is the span as one module with a static schedule, the
shape :func:`export_deployment` traces. The export writes a deploy folder next to
nothing else: the ONNX file, a sidecar with the ports, shapes, constants and
fingerprints, a deploy yaml in which one node stands in for the span, and the
weights of the nodes that stay outside it. The canonical yaml and ``.pt`` are not
touched. The export fails unless the live nodes, their export adapters and ONNX
Runtime agree on every example.
"""

from __future__ import annotations

import copy
import json
import tempfile
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import networkx as nx
import torch
import yaml
from loguru import logger
from torch import Tensor, nn

from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.execution import Context

if TYPE_CHECKING:
    from cuvis_ai_core.node.node import Node
    from cuvis_ai_core.pipeline.pipeline import CuvisPipeline

#: The node that runs an exported span in the Python runtime, and the manifest
#: that brings ONNX Runtime into a composed environment (both live in cuvis-ai).
DEFAULT_SPAN_CLASS = "cuvis_ai.node.exported_span.ExportedSpan"
DEFAULT_SPAN_PLUGIN = "cuvis_ai_onnx"
DEFAULT_SPAN_NODE_NAME = "exported_span"
SIDECAR_FORMAT = 1
#: Above this many bytes of span state the ONNX weights go to a ``.onnx.data``
#: file next to the graph (a single protobuf cannot exceed 2 GiB).
EXTERNAL_DATA_THRESHOLD = 1 << 30


class SpanSelectionError(ValueError):
    """The requested cut cannot be exported as one graph."""


class SpanExportError(RuntimeError):
    """The span did not export, or the exported graph does not reproduce it."""


@dataclass(frozen=True, order=True)
class PortRef:
    """A port of a pipeline node, written ``node.port``."""

    node: str
    port: str

    def __str__(self) -> str:
        """``node.port``."""
        return f"{self.node}.{self.port}"

    @classmethod
    def parse(cls, value: PortRef | tuple[str, str] | str) -> PortRef:
        """Accept a ``PortRef``, a ``(node, port)`` pair or a ``node.port`` string."""
        if isinstance(value, PortRef):
            return value
        if isinstance(value, tuple) and len(value) == 2:
            return cls(str(value[0]), str(value[1]))
        if isinstance(value, str) and "." in value:
            node, port = value.rsplit(".", 1)
            if node and port:
                return cls(node, port)
        raise SpanSelectionError(f"Port reference {value!r} must read 'node.port'.")


Wiring = dict[str, dict[str, "PortRef | list[PortRef]"]]


@dataclass
class SpanSelection:
    """The nodes of a span in execution order and how their ports are fed.

    ``wiring[node][port]`` names the span output port that feeds an input port (a
    list of them for a variadic port, in the pipeline's own gather order). Declared
    inputs and unconnected optional ports are not in the wiring.
    """

    nodes: list[Node]
    inputs: list[PortRef]
    outputs: list[PortRef]
    wiring: Wiring = field(default_factory=dict)

    @property
    def node_names(self) -> list[str]:
        """The span node names in execution order."""
        return [node.name for node in self.nodes]

    def node(self, name: str) -> Node:
        """The span node called ``name``."""
        for node in self.nodes:
            if node.name == name:
                return node
        raise KeyError(name)


def _parse_ports(values: Iterable, what: str) -> list[PortRef]:
    """Parse a non-empty list of distinct port references."""
    refs = [PortRef.parse(value) for value in values]
    if not refs:
        raise SpanSelectionError(f"At least one {what} port is required.")
    duplicates = {ref for ref in refs if refs.count(ref) > 1}
    if duplicates:
        raise SpanSelectionError(
            f"Duplicate {what} port(s): {', '.join(str(ref) for ref in sorted(duplicates))}."
        )
    return refs


def _inbound_edges(pipeline: CuvisPipeline, node: Node) -> list[tuple[Node, str, str]]:
    """``(predecessor, from_port, to_port)`` in the order ``forward`` gathers them."""
    graph = pipeline._graph
    edges: list[tuple[Node, str, str]] = []
    for predecessor in graph.predecessors(node):
        for _, edge in graph[predecessor][node].items():
            edges.append((predecessor, edge["from_port"], edge["to_port"]))
    return edges


def select_span(
    pipeline: CuvisPipeline,
    inputs: Iterable[PortRef | tuple[str, str] | str],
    outputs: Iterable[PortRef | tuple[str, str] | str],
) -> SpanSelection:
    """Cut the span between ``inputs`` (input ports) and ``outputs`` (output ports).

    Parameters
    ----------
    pipeline : CuvisPipeline
        The pipeline to cut.
    inputs : iterable of ``node.port``
        Input ports where the span starts; the exported graph takes a tensor for
        each. Every other input port of a span node must be fed by a span node or
        be an unconnected optional port.
    outputs : iterable of ``node.port``
        Output ports the span produces; every span port read outside the span
        must be among them.

    Returns
    -------
    SpanSelection
        The span nodes in execution order with their wiring.

    Raises
    ------
    SpanSelectionError
        When the cut is not exportable; the message names the port or node.
    """
    input_refs = _parse_ports(inputs, "input")
    output_refs = _parse_ports(outputs, "output")
    by_name: dict[str, Node] = {node.name: node for node in pipeline.nodes}

    for ref in input_refs:
        node = by_name.get(ref.node)
        if node is None or ref.port not in node.INPUT_SPECS:
            raise SpanSelectionError(f"Unknown input port {ref}.")
        if getattr(node.INPUT_SPECS[ref.port], "variadic", False):
            raise SpanSelectionError(
                f"Input port {ref} is variadic; a span input takes one tensor. Cut the "
                "span at the ports that feed it instead."
            )
    for ref in output_refs:
        node = by_name.get(ref.node)
        if node is None or ref.port not in node.OUTPUT_SPECS:
            raise SpanSelectionError(f"Unknown output port {ref}.")

    declared_inputs = set(input_refs)
    span: dict[str, Node] = {}
    wiring: Wiring = {}
    pending = [by_name[ref.node] for ref in output_refs]
    while pending:
        node = pending.pop()
        if node.name in span:
            continue
        if not node.should_execute(ExecutionStage.INFERENCE):
            raise SpanSelectionError(
                f"Node {node.name!r} does not run at inference "
                f"(stages {sorted(str(s) for s in node.execution_stages)}); it cannot be "
                "part of an exported span."
            )
        span[node.name] = node
        feeds: dict[str, PortRef | list[PortRef]] = {}
        edges = _inbound_edges(pipeline, node)
        for port_name, spec in node.INPUT_SPECS.items():
            ref = PortRef(node.name, port_name)
            if ref in declared_inputs:
                continue
            sources = [
                PortRef(pred.name, from_port)
                for pred, from_port, to_port in edges
                if to_port == port_name
            ]
            if not sources:
                if getattr(spec, "optional", False):
                    continue
                raise SpanSelectionError(
                    f"Input port {ref} has no connection inside the span and is not a "
                    "declared input; the pipeline would read it from the batch. Declare "
                    "it as a span input."
                )
            if getattr(spec, "variadic", False):
                feeds[port_name] = sources
            else:
                feeds[port_name] = sources[-1]
            for source in sources:
                if source.node not in span:
                    pending.append(by_name[source.node])
        wiring[node.name] = feeds

    unreached = [ref for ref in input_refs if ref.node not in span]
    if unreached:
        raise SpanSelectionError(
            "Declared input port(s) not reached from the outputs: "
            f"{', '.join(str(ref) for ref in unreached)}. Every span input must feed a "
            "node on a path to a span output."
        )

    declared_outputs = set(output_refs)
    graph = pipeline._graph
    leaks: list[str] = []
    for node in span.values():
        for successor in graph.successors(node):
            if successor.name in span:
                continue
            for _, edge in graph[node][successor].items():
                ref = PortRef(node.name, edge["from_port"])
                if ref not in declared_outputs:
                    leaks.append(f"{ref} feeds {successor.name}.{edge['to_port']}")
    if leaks:
        raise SpanSelectionError(
            "Span port(s) read outside the span but not declared as outputs: "
            f"{'; '.join(leaks)}. Add them to the span outputs."
        )

    span_nodes = set(span.values())
    outside_below: set[Node] = set()
    outside_above: set[Node] = set()
    for node in span_nodes:
        outside_below |= nx.descendants(graph, node) - span_nodes
        outside_above |= nx.ancestors(graph, node) - span_nodes
    loop = sorted(n.name for n in outside_below & outside_above)
    if loop:
        raise SpanSelectionError(
            f"Contracting the span into one node creates a cycle through {loop}: "
            "these nodes read a span output and feed a span input. Move the cut so "
            "they are inside or entirely outside the span."
        )

    ordered = [node for node in pipeline._sorted_nodes if node.name in span]
    return SpanSelection(
        nodes=ordered, inputs=input_refs, outputs=output_refs, wiring=wiring
    )


# ---------------------------------------------------------------------------
# The span as one module
# ---------------------------------------------------------------------------


def _key(ref: PortRef | str) -> str:
    """A module or buffer name for a node or port (no dots)."""
    return str(ref).replace(".", "__").replace("-", "_")


class PipelineSpanModule(nn.Module):
    """A span as one module with a static schedule, the shape the exporter traces.

    The span nodes are deep-copied and put in eval mode, so the live pipeline is
    never touched; a node's :meth:`Node.export_module` adapter replaces its
    ``forward`` when it returns one. Constants are span inputs baked in as buffers
    (a wavelength grid, token ids): they are not graph inputs. ``forward`` takes
    the remaining span inputs as positional tensors in ``input_names`` order and
    returns the span outputs as a tuple in ``output_names`` order.

    ``copy_nodes=False, use_adapters=False`` runs the live nodes' own ``forward``
    through the same schedule: the exporter's eager reference.
    """

    def __init__(
        self,
        selection: SpanSelection,
        constants: Mapping[Any, Tensor] | None = None,
        *,
        copy_nodes: bool = True,
        use_adapters: bool = True,
        context: Context | None = None,
    ) -> None:
        super().__init__()
        self.selection = selection
        constant_refs = {
            PortRef.parse(ref): value for ref, value in (constants or {}).items()
        }
        stray = [ref for ref in constant_refs if ref not in selection.inputs]
        if stray:
            raise SpanSelectionError(
                "Constant(s) are not span inputs: "
                f"{', '.join(str(ref) for ref in stray)}. A constant is a declared input "
                "port whose value is baked into the graph."
            )
        self.input_refs = [ref for ref in selection.inputs if ref not in constant_refs]
        self.constant_refs = [ref for ref in selection.inputs if ref in constant_refs]
        self.output_refs = list(selection.outputs)
        self._context = context or Context(stage=ExecutionStage.INFERENCE)

        self.constants = nn.Module()
        for ref in self.constant_refs:
            self.constants.register_buffer(
                _key(ref), constant_refs[ref].detach().clone()
            )

        self.modules_by_node = nn.ModuleDict()
        self._adapted: set[str] = set()
        for node in selection.nodes:
            if copy_nodes:
                node = copy.deepcopy(node)
                node.eval()
            adapter = node.export_module() if use_adapters else None
            module: nn.Module = node
            if adapter is not None:
                adapter.eval()
                module = adapter
                self._adapted.add(node.name)
            self.modules_by_node[_key(node.name)] = module
        if copy_nodes:
            # The module itself in eval mode too (the exporter checks the flag);
            # with live nodes inside, their mode is theirs to keep.
            self.eval()

    @property
    def input_names(self) -> list[str]:
        """Graph inputs, in ``forward`` order."""
        return [str(ref) for ref in self.input_refs]

    @property
    def constant_names(self) -> list[str]:
        """Span inputs baked in as buffers."""
        return [str(ref) for ref in self.constant_refs]

    @property
    def output_names(self) -> list[str]:
        """Graph outputs, in the order ``forward`` returns them."""
        return [str(ref) for ref in self.output_refs]

    def is_adapted(self, node_name: str) -> bool:
        """Whether ``node_name`` runs through its export adapter."""
        return node_name in self._adapted

    def module_for(self, node_name: str) -> nn.Module:
        """The copied node, or its export adapter, that runs for ``node_name``."""
        return self.modules_by_node[_key(node_name)]

    def constant(self, ref: PortRef | str) -> Tensor:
        """The buffer baked in for the constant input ``ref``."""
        return getattr(self.constants, _key(PortRef.parse(ref)))

    def forward(self, *inputs: Tensor) -> tuple[Tensor, ...]:
        """Run the span: inputs in ``input_names`` order, outputs in ``output_names`` order."""
        return self._run(inputs, record=None)

    def run_recording(
        self, *inputs: Tensor
    ) -> tuple[tuple[Tensor, ...], dict[str, tuple[dict[str, Any], dict[str, Any]]]]:
        """Run the schedule and also return every node's ``(kwargs, outputs)``."""
        record: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        return self._run(inputs, record=record), record

    def _run(
        self,
        inputs: tuple[Tensor, ...],
        record: dict[str, tuple[dict[str, Any], dict[str, Any]]] | None,
    ) -> tuple[Tensor, ...]:
        """The static schedule; ``record`` collects every node's kwargs and outputs."""
        if len(inputs) != len(self.input_refs):
            raise TypeError(
                f"The span takes {len(self.input_refs)} input(s) {self.input_names}, "
                f"got {len(inputs)}."
            )
        fed: dict[PortRef, Tensor] = dict(zip(self.input_refs, inputs))
        for ref in self.constant_refs:
            fed[ref] = self.constant(ref)
        produced: dict[PortRef, Any] = {}
        for node in self.selection.nodes:
            feeds = self.selection.wiring[node.name]
            kwargs: dict[str, Any] = {}
            for port in node.INPUT_SPECS:
                ref = PortRef(node.name, port)
                if ref in fed:
                    kwargs[port] = fed[ref]
                elif port in feeds:
                    feed = feeds[port]
                    if isinstance(feed, list):
                        kwargs[port] = [produced[source] for source in feed]
                    else:
                        kwargs[port] = produced[feed]
            outputs = self._call(node.name, kwargs)
            if record is not None:
                record[node.name] = (kwargs, outputs)
            for port, value in outputs.items():
                produced[PortRef(node.name, port)] = value
        return tuple(produced[ref] for ref in self.output_refs)

    def _call(self, node_name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Run one node: its adapter with the ports, or its forward with the context."""
        module = self.module_for(node_name)
        if node_name in self._adapted:
            return module(**kwargs)
        return module(**kwargs, context=self._context)


class _NodeProbe(nn.Module):
    """One span node (or its adapter) with its recorded inputs flattened to tensors,
    so the exporter can find the node that fails on its own."""

    def __init__(
        self, span: PipelineSpanModule, node_name: str, kwargs: dict[str, Any]
    ):
        super().__init__()
        self.span = span
        self.node_name = node_name
        self.layout = [
            (name, len(value) if isinstance(value, list) else None)
            for name, value in kwargs.items()
        ]
        self.eval()

    def flat_inputs(self, kwargs: dict[str, Any]) -> tuple[Tensor, ...]:
        """The node's kwargs as the positional tensors ``forward`` takes."""
        flat: list[Tensor] = []
        for name, count in self.layout:
            flat.extend(kwargs[name] if count is not None else [kwargs[name]])
        return tuple(flat)

    def forward(self, *tensors: Tensor) -> tuple[Tensor, ...]:
        """Rebuild the kwargs and run the node; its tensor outputs come back as a tuple."""
        kwargs: dict[str, Any] = {}
        cursor = 0
        for name, count in self.layout:
            if count is None:
                kwargs[name] = tensors[cursor]
                cursor += 1
            else:
                kwargs[name] = list(tensors[cursor : cursor + count])
                cursor += count
        outputs = self.span._call(self.node_name, kwargs)
        return tuple(v for v in outputs.values() if isinstance(v, Tensor))


# ---------------------------------------------------------------------------
# Exporting
# ---------------------------------------------------------------------------


def _export_dynamo(
    module: nn.Module,
    args: tuple[Tensor, ...],
    path: Path,
    *,
    input_names: list[str],
    output_names: list[str],
    opset: int,
    external_data: bool,
) -> None:
    """``torch.onnx.export`` with the dynamo (``torch.export``) exporter."""
    torch.onnx.export(
        module,
        args,
        str(path),
        dynamo=True,
        opset_version=opset,
        input_names=input_names,
        output_names=output_names,
        external_data=external_data,
        verbose=False,
    )


def _export_legacy(
    module: nn.Module,
    args: tuple[Tensor, ...],
    path: Path,
    *,
    input_names: list[str],
    output_names: list[str],
    opset: int,
    external_data: bool,
) -> None:
    """The TorchScript exporter, with every ``TracerWarning`` treated as an error.

    The tracer turns a ``.item()`` or a numpy value into a constant with only a
    warning, so a graph that traced "successfully" can be wrong for other inputs.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=torch.jit.TracerWarning)
        torch.onnx.export(
            module,
            args,
            str(path),
            dynamo=False,
            opset_version=opset,
            input_names=input_names,
            output_names=output_names,
            verbose=False,
        )


def _remove_export_files(path: Path) -> None:
    """Drop the ONNX file and its external data after a failed export or parity check."""
    for candidate in (path, path.with_name(path.name + ".data")):
        if candidate.exists():
            candidate.unlink()


def _export_graph(
    module: nn.Module,
    args: tuple[Tensor, ...],
    path: Path,
    *,
    input_names: list[str],
    output_names: list[str] | None,
    opset: int,
    external_data: bool,
) -> tuple[str, Exception | None]:
    """Export with the dynamo exporter, then the legacy one.

    Returns ``(kind, dynamo_error)`` where ``kind`` is ``"dynamo"`` or ``"legacy"``;
    raises the legacy error (with the dynamo error as its cause) when both fail.
    """
    options: dict[str, Any] = dict(
        input_names=input_names,
        output_names=output_names,
        opset=opset,
        external_data=external_data,
    )
    try:
        _export_dynamo(module, args, path, **options)
        return "dynamo", None
    except Exception as dynamo_error:  # noqa: BLE001 - every exporter failure is recorded
        _remove_export_files(path)
        try:
            _export_legacy(module, args, path, **options)
        except Exception as legacy_error:  # noqa: BLE001
            _remove_export_files(path)
            raise legacy_error from dynamo_error
        return "legacy", dynamo_error


def _dtype_name(tensor: Tensor) -> str:
    """``float32`` for ``torch.float32``."""
    return str(tensor.dtype).removeprefix("torch.")


def _port_entries(
    refs: Sequence[PortRef], tensors: Sequence[Tensor]
) -> list[dict[str, Any]]:
    """The sidecar entries of ``refs`` at the shapes and dtypes of ``tensors``."""
    names = _span_port_names(refs)
    return [
        {
            "port": str(ref),
            "name": name,
            "tensor": str(ref),
            "dtype": _dtype_name(tensor),
            "shape": list(tensor.shape),
        }
        for ref, name, tensor in zip(refs, names, tensors)
    ]


def _span_port_names(refs: Sequence[PortRef]) -> list[str]:
    """Port names of the stand-in node: the bare port name when it is unique among
    ``refs``, else ``<node>_<port>``."""
    counts: dict[str, int] = {}
    for ref in refs:
        counts[ref.port] = counts.get(ref.port, 0) + 1
    return [
        ref.port
        if counts[ref.port] == 1
        else f"{ref.node}_{ref.port}".replace("-", "_")
        for ref in refs
    ]


def _max_abs_diff(a: Tensor, b: Tensor) -> float:
    """The largest difference; infinite on a shape or dtype mismatch."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return float("inf")
    if a.numel() == 0:
        return 0.0
    if not a.is_floating_point():
        return float((a != b).any())
    return float((a.double() - b.double()).abs().max())


def _mismatch(a: Tensor, b: Tensor, atol: float, rtol: float) -> bool:
    """Whether two outputs disagree: shape, dtype, or values beyond the tolerance."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return True
    if not a.is_floating_point():
        return bool((a != b).any())
    return not torch.allclose(
        a.double(), b.double(), atol=atol, rtol=rtol, equal_nan=True
    )


@dataclass
class Deployment:
    """What :func:`export_deployment` wrote."""

    onnx_path: Path
    sidecar_path: Path
    yaml_path: Path
    weights_path: Path
    sidecar: dict[str, Any]


def _check_examples(
    examples: Sequence[Mapping[Any, Tensor]], inputs: list[PortRef]
) -> list[dict[PortRef, Tensor]]:
    """Every example carries a tensor for every span input and nothing else."""
    if not examples:
        raise SpanExportError(
            "At least one example is required: the span is exported at "
            "the example shapes and checked on every example."
        )
    checked: list[dict[PortRef, Tensor]] = []
    for index, example in enumerate(examples):
        parsed = {PortRef.parse(key): value for key, value in example.items()}
        missing = [str(ref) for ref in inputs if ref not in parsed]
        if missing:
            raise SpanExportError(
                f"Example {index} has no value for span input(s) {', '.join(missing)}."
            )
        unknown = [str(ref) for ref in parsed if ref not in inputs]
        if unknown:
            raise SpanExportError(
                f"Example {index} carries value(s) for {', '.join(unknown)}, which are not "
                "span inputs."
            )
        for ref, value in parsed.items():
            if not isinstance(value, Tensor):
                raise SpanExportError(
                    f"Example {index}: the value for {ref} must be a tensor, got "
                    f"{type(value).__name__}."
                )
        checked.append(parsed)
    return checked


def _constant_values(
    examples: list[dict[PortRef, Tensor]], constants: list[PortRef]
) -> dict[PortRef, Tensor]:
    """The value of every constant, which must be the same in every example."""
    values: dict[PortRef, Tensor] = {}
    for ref in constants:
        first = examples[0][ref]
        for index, example in enumerate(examples[1:], start=1):
            if not torch.equal(example[ref].cpu(), first.cpu()):
                raise SpanExportError(
                    f"Constant {ref} differs between example 0 and example {index}; a "
                    "constant is baked into the graph, so every example must carry the "
                    "same value."
                )
        values[ref] = first
    return values


def _first_failing_node(
    span: PipelineSpanModule,
    record: dict[str, tuple[dict[str, Any], dict[str, Any]]],
    *,
    opset: int,
) -> tuple[str, Exception] | None:
    """Export every span node alone, in order; the first that fails, with its error."""
    with tempfile.TemporaryDirectory() as tmp:
        for node_name, (kwargs, _) in record.items():
            probe = _NodeProbe(span, node_name, kwargs)
            args = probe.flat_inputs(kwargs)
            try:
                _export_graph(
                    probe,
                    args,
                    Path(tmp) / f"{_key(node_name)}.onnx",
                    input_names=[f"in{i}" for i in range(len(args))],
                    output_names=None,
                    opset=opset,
                    external_data=False,
                )
            except Exception as error:  # noqa: BLE001
                return node_name, error
    return None


def _node_sidecar(selection: SpanSelection) -> list[dict[str, str]]:
    """Name and class of every span node, in execution order."""
    return [
        {
            "name": node.name,
            "class_name": f"{type(node).__module__}.{type(node).__name__}",
        }
        for node in selection.nodes
    ]


def _wiring_json(selection: SpanSelection) -> dict[str, dict[str, Any]]:
    """The wiring with ``node.port`` strings, for the fingerprint."""
    return {
        node: {
            port: [str(s) for s in feed] if isinstance(feed, list) else str(feed)
            for port, feed in feeds.items()
        }
        for node, feeds in selection.wiring.items()
    }


def _deploy_config(
    pipeline: CuvisPipeline,
    selection: SpanSelection,
    *,
    onnx_name: str,
    span_node_name: str,
    span_class_name: str,
    span_plugin: str | None,
    input_names: list[str],
    output_names: list[str],
) -> dict[str, Any]:
    """The deploy yaml: the pipeline with the span replaced by one node."""
    config = pipeline.serialize().to_dict()
    span_names = set(selection.node_names)
    if span_node_name in {n.name for n in pipeline.nodes} - span_names:
        raise SpanExportError(
            f"The pipeline already has a node named {span_node_name!r}; pass another "
            "span_node_name."
        )
    span_node = {
        "name": span_node_name,
        "class_name": span_class_name,
        "hparams": {"onnx_path": onnx_name, "backend": "onnxruntime"},
    }
    nodes: list[dict[str, Any]] = []
    inserted = False
    for node_cfg in config["nodes"]:
        if node_cfg["name"] in span_names:
            if not inserted:
                nodes.append(span_node)
                inserted = True
            continue
        nodes.append(node_cfg)
    config["nodes"] = nodes

    in_port = {str(ref): name for ref, name in zip(selection.inputs, input_names)}
    out_port = {str(ref): name for ref, name in zip(selection.outputs, output_names)}
    connections: list[dict[str, str]] = []
    for conn in config["connections"]:
        src_node, _, src_port = conn["source"].split(".")
        dst_node, _, dst_port = conn["target"].split(".")
        src_in, dst_in = src_node in span_names, dst_node in span_names
        if src_in and dst_in:
            continue
        if dst_in:
            conn = {
                "source": conn["source"],
                "target": f"{span_node_name}.inputs.{in_port[f'{dst_node}.{dst_port}']}",
            }
        elif src_in:
            conn = {
                "source": f"{span_node_name}.outputs.{out_port[f'{src_node}.{src_port}']}",
                "target": conn["target"],
            }
        connections.append(conn)
    config["connections"] = connections

    plugins = list(config.get("plugins") or [])
    if span_plugin and span_plugin not in plugins:
        plugins.append(span_plugin)
    config["plugins"] = plugins or None
    return config


def export_deployment(
    pipeline: CuvisPipeline,
    out_dir: str | Path,
    *,
    inputs: Iterable[PortRef | tuple[str, str] | str],
    outputs: Iterable[PortRef | tuple[str, str] | str],
    examples: Sequence[Mapping[Any, Tensor]],
    constants: Iterable[PortRef | tuple[str, str] | str] = (),
    name: str | None = None,
    opset: int = 18,
    atol: float = 1e-5,
    rtol: float = 1e-4,
    span_node_name: str = DEFAULT_SPAN_NODE_NAME,
    span_class_name: str = DEFAULT_SPAN_CLASS,
    span_plugin: str | None = DEFAULT_SPAN_PLUGIN,
) -> Deployment:
    """Export the span between ``inputs`` and ``outputs`` and write its deploy folder.

    Parameters
    ----------
    pipeline : CuvisPipeline
        The loaded pipeline (with its weights). It is not modified.
    out_dir : path
        The deploy folder; created when missing. Receives ``<name>.onnx`` (plus
        ``<name>.onnx.data`` for a span above :data:`EXTERNAL_DATA_THRESHOLD`),
        ``<name>.onnx.json``, ``<name>_deploy.yaml`` and ``<name>_deploy.pt``.
    inputs, outputs : iterables of ``node.port``
        The cut; see :func:`select_span`.
    examples : sequence of mappings
        Real samples, each mapping every span input (constants included) to a
        tensor. The graph is exported at the shapes of the first example, and the
        live nodes, their adapters and ONNX Runtime must agree on all of them.
    constants : iterable of ``node.port``
        Span inputs baked into the graph as buffers (a wavelength grid, token ids).
        Their value must be the same in every example; the sidecar records it.
    name : str, optional
        The file stem; the pipeline name by default.
    opset : int
        The ONNX opset.
    atol, rtol : float
        The parity tolerance.
    span_node_name, span_class_name, span_plugin : str
        The node that stands in for the span in the deploy yaml and the plugin
        manifest that provides it (``cuvis_ai_onnx`` brings ONNX Runtime into a
        composed environment; ``None`` adds no plugin).

    Returns
    -------
    Deployment
        The paths and the sidecar contents.

    Raises
    ------
    SpanSelectionError
        When the cut is not exportable.
    SpanExportError
        When an example is incomplete, a constant varies, an output is not a
        tensor, an adapter differs from its node, neither exporter carries the
        span (the message names the first node that fails on its own and keeps
        the span's own error as the cause) or ONNX Runtime differs from the span.
    """
    selection = select_span(pipeline, inputs, outputs)
    constant_refs = [PortRef.parse(ref) for ref in constants]
    checked = _check_examples(examples, selection.inputs)
    constant_values = _constant_values(checked, constant_refs)

    eager = PipelineSpanModule(
        selection, constant_values, copy_nodes=False, use_adapters=False
    )
    span = PipelineSpanModule(selection, constant_values)
    input_refs = span.input_refs
    device = checked[0][input_refs[0]].device if input_refs else torch.device("cpu")
    span.to(device)

    name = name or pipeline.name
    if not name or Path(name).name != name:
        raise SpanExportError(f"Deployment name {name!r} must be a bare file stem.")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / f"{name}.onnx"

    # 1. Eager nodes against the span module (adapters, copies in eval mode).
    example_args = [tuple(ex[ref] for ref in input_refs) for ex in checked]
    span_outputs: list[tuple[Tensor, ...]] = []
    record0: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    with torch.no_grad():
        for index, args in enumerate(example_args):
            _, eager_record = eager.run_recording(*args)
            span_out, span_record = span.run_recording(*args)
            if index == 0:
                record0 = span_record
            for node_name, (_, eager_out) in eager_record.items():
                span_node_out = span_record[node_name][1]
                for port, value in eager_out.items():
                    other = span_node_out.get(port)
                    if not isinstance(value, Tensor) or not isinstance(other, Tensor):
                        continue
                    if _mismatch(value.cpu(), other.cpu(), atol, rtol):
                        what = (
                            "its export adapter differs from its forward"
                            if span.is_adapted(node_name)
                            else "its copy in eval mode differs from the live node (is the "
                            "live node in training mode?)"
                        )
                        raise SpanExportError(
                            f"Node {node_name!r}: {what} on example {index}, port "
                            f"{port!r} (max abs diff {_max_abs_diff(value.cpu(), other.cpu()):.3e}, "
                            f"atol={atol}, rtol={rtol})."
                        )
            for ref, value in zip(selection.outputs, span_out):
                if not isinstance(value, Tensor):
                    raise SpanExportError(
                        f"Span output {ref} is not a tensor ({type(value).__name__}); only "
                        "tensor ports can leave an exported span."
                    )
            span_outputs.append(tuple(v.detach() for v in span_out))

    # 2. The graph.
    state_bytes = sum(t.numel() * t.element_size() for t in span.state_dict().values())
    external_data = state_bytes > EXTERNAL_DATA_THRESHOLD
    input_names, output_names = span.input_names, span.output_names
    try:
        kind, dynamo_error = _export_graph(
            span,
            example_args[0],
            onnx_path,
            input_names=input_names,
            output_names=output_names,
            opset=opset,
            external_data=external_data,
        )
    except Exception as span_error:  # noqa: BLE001
        culprit = _first_failing_node(span, record0, opset=opset)
        if culprit is None:
            raise SpanExportError(
                f"The span {selection.node_names} did not export with the dynamo exporter "
                f"nor the legacy one, and every node exports on its own: {span_error}"
            ) from span_error
        node_name, node_error = culprit
        raise SpanExportError(
            f"Node {node_name!r} fails to export on its own ({type(node_error).__name__}: "
            f"{str(node_error).splitlines()[0] if str(node_error) else ''}); the span "
            f"{selection.node_names} did not export with the dynamo exporter nor the "
            f"legacy one."
        ) from span_error
    if dynamo_error is not None:
        logger.warning(
            "Span {} exported with the legacy exporter; the dynamo exporter failed: {}",
            selection.node_names,
            str(dynamo_error).splitlines()[0] if str(dynamo_error) else dynamo_error,
        )

    # 3. ONNX Runtime against the span module on every example.
    try:
        import onnxruntime as ort
    except ImportError as exc:
        _remove_export_files(onnx_path)
        raise SpanExportError(
            "onnxruntime is not installed; the parity check needs it "
            "(uv sync --extra onnx)."
        ) from exc
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    max_abs: dict[str, float] = {str(ref): 0.0 for ref in selection.outputs}
    for index, (args, expected) in enumerate(zip(example_args, span_outputs)):
        feeds = {n: t.detach().cpu().numpy() for n, t in zip(input_names, args)}
        got = session.run(output_names, feeds)
        for ref, want, have in zip(selection.outputs, expected, got):
            have_t = torch.from_numpy(have)
            want_t = want.cpu()
            diff = _max_abs_diff(want_t, have_t)
            max_abs[str(ref)] = max(max_abs[str(ref)], diff)
            if _mismatch(want_t, have_t, atol, rtol):
                _remove_export_files(onnx_path)
                raise SpanExportError(
                    f"ONNX Runtime differs from the span on example {index}, output {ref} "
                    f"(max abs diff {diff:.3e}, atol={atol}, rtol={rtol}); the exported "
                    f"graph ({kind} exporter) does not reproduce the span."
                )

    # 4. The sidecar.
    from cuvis_ai_core import __version__
    from cuvis_ai_core.utils.trt_engine import file_md5, tensor_fingerprint

    span_port_in = _span_port_names(selection.inputs)
    in_name_by_ref = dict(zip(selection.inputs, span_port_in))
    hparams = {
        node.name: pipeline._convert_numpy_to_native(node.hparams or {})
        for node in selection.nodes
    }
    fingerprint_extra = json.dumps(
        {
            "nodes": _node_sidecar(selection),
            "wiring": _wiring_json(selection),
            "hparams": hparams,
            "inputs": input_names,
            "outputs": output_names,
            "constants": span.constant_names,
        },
        sort_keys=True,
        default=str,
    )
    sidecar: dict[str, Any] = {
        "format": SIDECAR_FORMAT,
        "pipeline": pipeline.name,
        "span": {
            "node_name": span_node_name,
            "nodes": _node_sidecar(selection),
            "inputs": [
                {**entry, "name": in_name_by_ref[ref]}
                for entry, ref in zip(
                    _port_entries(input_refs, list(example_args[0])), input_refs
                )
            ],
            "outputs": _port_entries(selection.outputs, list(span_outputs[0])),
            "constants": [
                {
                    "port": str(ref),
                    "name": in_name_by_ref[ref],
                    "dtype": _dtype_name(constant_values[ref]),
                    "shape": list(constant_values[ref].shape),
                    "value": constant_values[ref].detach().cpu().tolist(),
                }
                for ref in span.constant_refs
            ],
        },
        "exporter": {
            "kind": kind,
            "opset": opset,
            "torch": torch.__version__,
            "cuvis_ai_core": __version__,
        },
        "fingerprint": tensor_fingerprint(span.state_dict(), extra=fingerprint_extra),
        "onnx_md5": file_md5(onnx_path),
        "parity": {
            "examples": len(checked),
            "atol": atol,
            "rtol": rtol,
            "max_abs_diff": max_abs,
        },
    }
    data_path = onnx_path.with_name(onnx_path.name + ".data")
    if data_path.exists():
        sidecar["onnx_data"] = data_path.name
        sidecar["onnx_data_md5"] = file_md5(data_path)
    sidecar_path = onnx_path.with_name(onnx_path.name + ".json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

    # 5. The deploy yaml and the weights of the nodes outside the span.
    config = _deploy_config(
        pipeline,
        selection,
        onnx_name=onnx_path.name,
        span_node_name=span_node_name,
        span_class_name=span_class_name,
        span_plugin=span_plugin,
        input_names=span_port_in,
        output_names=_span_port_names(selection.outputs),
    )
    yaml_path = out_dir / f"{name}_deploy.yaml"
    with yaml_path.open("w", encoding="utf-8") as handle:
        yaml.dump(config, handle, default_flow_style=False, sort_keys=False)
    weights_path = out_dir / f"{name}_deploy.pt"
    span_names = set(selection.node_names)
    torch.save(
        {
            "state_dict": {
                node.name: node.state_dict()
                for node in pipeline.nodes
                if node.name not in span_names
            },
            "metadata": config["metadata"],
        },
        weights_path,
    )
    logger.info(
        "Exported span {} ({} exporter) to {}; deploy yaml {}",
        selection.node_names,
        kind,
        onnx_path,
        yaml_path,
    )
    return Deployment(
        onnx_path=onnx_path,
        sidecar_path=sidecar_path,
        yaml_path=yaml_path,
        weights_path=weights_path,
        sidecar=sidecar,
    )


# ---------------------------------------------------------------------------
# Examples from pipeline runs
# ---------------------------------------------------------------------------


def collect_examples(
    pipeline: CuvisPipeline,
    inputs: Iterable[PortRef | tuple[str, str] | str],
    batches: Iterable[Mapping[str, Any] | None],
    *,
    limit: int | None = None,
) -> list[dict[str, Tensor]]:
    """Run the pipeline on ``batches`` and keep what it feeds into the span inputs.

    A span input fed by a connection takes the value of the port on the other end
    of it; a span input with no connection takes the batch value of the same
    name, the way ``forward`` does. The whole pipeline runs for every batch, so
    sink nodes write what they always write.

    Parameters
    ----------
    pipeline : CuvisPipeline
        The loaded pipeline.
    inputs : iterable of ``node.port``
        The span inputs (constants included).
    batches : iterable of batch dicts
        For example a predict dataloader; ``None`` or ``{}`` for a pipeline whose
        source node needs nothing.
    limit : int, optional
        Stop after this many examples.

    Returns
    -------
    list of dict
        One ``{"node.port": tensor}`` mapping per batch, ready for
        :func:`export_deployment`.
    """
    refs = [PortRef.parse(value) for value in inputs]
    by_name: dict[str, Node] = {node.name: node for node in pipeline.nodes}
    feeds: dict[PortRef, PortRef | None] = {}
    for ref in refs:
        node = by_name.get(ref.node)
        if node is None or ref.port not in node.INPUT_SPECS:
            raise SpanSelectionError(f"Unknown input port {ref}.")
        sources = [
            PortRef(pred.name, from_port)
            for pred, from_port, to_port in _inbound_edges(pipeline, node)
            if to_port == ref.port
        ]
        feeds[ref] = sources[-1] if sources else None

    examples: list[dict[str, Tensor]] = []
    with torch.no_grad():
        for batch in batches:
            if limit is not None and len(examples) >= limit:
                break
            batch = dict(batch or {})
            for ref, source in feeds.items():
                if source is None and ref.port not in batch:
                    raise SpanExportError(
                        f"Span input {ref} has no connection and the batch carries no "
                        f"{ref.port!r} value; it has {sorted(batch)}."
                    )
            ports = pipeline.forward(batch=batch, stage=ExecutionStage.INFERENCE)
            example: dict[str, Tensor] = {}
            for ref, source in feeds.items():
                if source is None:
                    example[str(ref)] = batch[ref.port]
                else:
                    example[str(ref)] = ports[(source.node, source.port)]
            examples.append(example)
    return examples
