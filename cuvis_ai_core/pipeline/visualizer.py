from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from loguru import logger

from cuvis_ai_core.node.node import Node
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.utils.node_registry import NodeRegistry
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.extensions.ui.node_display import is_plugin, resolve_display
from cuvis_ai_schemas.pipeline import PortSpec

try:
    from cuvis_ai_schemas.extensions.ui.port_display import (
        DEFAULT_COLOR as _DTYPE_DEFAULT_RGB,
    )
    from cuvis_ai_schemas.extensions.ui.port_display import (
        DTYPE_COLORS as _DTYPE_COLORS,
    )
except ImportError:  # schemas extension is optional
    _DTYPE_COLORS = {}
    _DTYPE_DEFAULT_RGB = (200, 200, 200)

NodeTypeResolver = Callable[[Node], str]

DEFAULT_STAGE_LABELS: dict[str, str] = {
    ExecutionStage.TRAIN.value: "Train",
    ExecutionStage.VAL.value: "Validation",
    ExecutionStage.TEST.value: "Test",
    ExecutionStage.INFERENCE.value: "Inference",
    ExecutionStage.ALWAYS.value: "All Stages",
}


class PipelineVisualizer:
    """Generate Graphviz and Mermaid representations of a CuvisPipeline."""

    def __init__(self, pipeline: CuvisPipeline) -> None:
        self.pipeline = pipeline
        self._graph = pipeline._graph

    def to_graphviz(
        self,
        *,
        graph_name: str | None = None,
        rankdir: str = "LR",
        group_by_stage: bool = False,
        stage_labels: Mapping[str, str] | None = None,
        show_port_types: bool = False,
        show_execution_stage: bool = False,
        graph_attributes: Mapping[str, Any] | None = None,
        node_attributes: Mapping[str, Any] | None = None,
        edge_attributes: Mapping[str, Any] | None = None,
        show_node_name: bool = False,
        node_registry: NodeRegistry | None = None,
    ) -> str:
        """Return a DOT string describing the pipeline graph as category-coloured cards.

        Each node renders as a rounded HTML-table card with category colour, emoji,
        per-port dtype dots, and an optional "Plugin" pill (when ``node_registry``
        identifies the node as plugin-sourced). Edges are dtype-coloured and skip
        labels when both ports share a name (set ``show_port_types=True`` to force
        ``port: dtype`` labels and disable dtype edge colouring).
        """

        title = self._sanitize_identifier(
            graph_name or self.pipeline.name or "CuvisPipeline"
        )
        lines: list[str] = [f"digraph {title} {{"]

        lines.append(f"    rankdir={rankdir};")
        lines.append("    node [shape=plaintext];")

        if graph_attributes:
            lines.extend(self._format_graphviz_attributes(graph_attributes))
        if edge_attributes:
            attr_line = self._compose_attribute_list(edge_attributes)
            lines.append(f"    edge [{attr_line}];")

        stage_labels = {**DEFAULT_STAGE_LABELS, **(stage_labels or {})}
        node_entries: dict[Node, str] = {}

        for index, node in enumerate(self._graph.nodes, start=1):
            identifier = self._dot_identifier(node)
            stage_text = (
                self._format_execution_stage_text(node, stage_labels)
                if show_execution_stage
                else None
            )
            html_label = self._format_card_label(
                node,
                index=index,
                show_node_name=show_node_name,
                stage_text=stage_text,
                registry=node_registry,
            )
            attrs = [
                f"label=<{html_label}>",
                'shape="plaintext"',
                'margin="0"',
            ]
            if node_attributes:
                attrs.extend(self._format_inline_attributes(node_attributes))
            node_entries[node] = f'"{identifier}" [{", ".join(attrs)}];'

        if group_by_stage:
            stage_groups = self._group_nodes_by_stage()
            for stage_key, nodes in stage_groups.items():
                cluster_name = self._sanitize_identifier(stage_key, allow_dash=False)
                label = self._escape_label(
                    stage_labels.get(stage_key, stage_key.title())
                )
                lines.append(f'    subgraph "cluster_{cluster_name}" {{')
                lines.append(f'        label="{label}";')
                for node in nodes:
                    lines.append(f"        {node_entries[node]}")
                lines.append("    }")
        else:
            for entry in node_entries.values():
                lines.append(f"    {entry}")

        for source, target, edge_data in self._graph.edges(data=True):
            src_id = self._dot_identifier(source)
            dst_id = self._dot_identifier(target)
            from_port = edge_data.get("from_port") or ""
            to_port = edge_data.get("to_port") or ""

            edge_label = self._format_edge_label(
                source,
                target,
                edge_data,
                include_port_types=show_port_types,
                dedupe_matching_ports=not show_port_types,
            )

            src_anchor = (
                f':"{self._port_anchor_id(from_port, is_output=True)}":e'
                if from_port
                else ""
            )
            dst_anchor = (
                f':"{self._port_anchor_id(to_port, is_output=False)}":w'
                if to_port
                else ""
            )

            attrs: list[str] = []
            if edge_label:
                attrs.append(f'label="{self._escape_label(edge_label)}"')
            if not show_port_types:
                spec = self._resolve_port_spec(source, from_port, is_output=True)
                color = self._dtype_hex_color(spec.dtype if spec else None)
                attrs.extend([f'color="{color}"', "penwidth=1.5"])

            attr_str = f" [{', '.join(attrs)}]" if attrs else ""
            lines.append(
                f'    "{src_id}"{src_anchor} -> "{dst_id}"{dst_anchor}{attr_str};'
            )

        lines.append("}")
        return "\n".join(lines)

    def to_mermaid(
        self,
        *,
        direction: str = "LR",
        include_node_class: bool = True,
        node_type_resolver: NodeTypeResolver | None = None,
        node_colors: Mapping[str, str] | None = None,
        group_by_stage: bool = False,
        stage_labels: Mapping[str, str] | None = None,
        show_port_types: bool = False,
        show_execution_stage: bool = False,
    ) -> str:
        """Return Mermaid flowchart syntax describing the pipeline graph."""

        lines = [f"flowchart {direction}"]
        node_type_resolver = node_type_resolver or (
            lambda node: node.__class__.__name__
        )
        node_colors = node_colors or {}
        stage_labels = {**DEFAULT_STAGE_LABELS, **(stage_labels or {})}
        node_classes: dict[str, str] = {}
        class_styles: dict[str, str] = {}

        node_defs: dict[Node, str] = {}
        for node in self._graph.nodes:
            identifier = self._mermaid_identifier(node)
            stage_text = (
                self._format_execution_stage_text(node, stage_labels)
                if show_execution_stage
                else None
            )
            label = self._escape_mermaid_label(
                self._format_node_label(
                    node,
                    include_node_class,
                    mermaid=True,
                    stage_text=stage_text,
                )
            )
            node_defs[node] = f"{identifier}[{label}]"

            node_type = node_type_resolver(node)
            class_name = (
                self._sanitize_identifier(node_type, allow_dash=False) or "type"
            )
            node_classes[identifier] = class_name

            color = node_colors.get(node_type)
            if color:
                class_styles[class_name] = color

        if group_by_stage:
            stage_groups = self._group_nodes_by_stage()
            for stage_key, nodes in stage_groups.items():
                label = stage_labels.get(stage_key, stage_key.title())
                lines.append(
                    f"    subgraph {self._sanitize_identifier(stage_key, allow_dash=False)}[{label}]"
                )
                lines.append(f"        direction {direction}")
                for node in nodes:
                    lines.append(f"        {node_defs[node]}")
                lines.append("    end")
        else:
            for definition in node_defs.values():
                lines.append(f"    {definition}")

        for source, target, edge_data in self._graph.edges(data=True):
            src = self._mermaid_identifier(source)
            dst = self._mermaid_identifier(target)
            edge_label = self._format_edge_label(
                source,
                target,
                edge_data,
                include_port_types=show_port_types,
                mermaid=True,
            )
            if edge_label:
                sanitized = self._sanitize_mermaid_pipe(edge_label)
                lines.append(f"    {src} -->|{sanitized}| {dst}")
            else:
                lines.append(f"    {src} --> {dst}")

        if class_styles:
            lines.append("    %% Node styling")
            for class_name, color in class_styles.items():
                lines.append(
                    f"    classDef {class_name} fill:{color},stroke:#333,stroke-width:1px;"
                )
            for node_id, class_name in node_classes.items():
                if class_name in class_styles:
                    lines.append(f"    class {node_id} {class_name};")

        return "\n".join(lines)

    def render_graphviz(
        self,
        output_path: str | Path,
        *,
        format: str = "png",
        rankdir: str = "LR",
        engine: str = "dot",
        **graphviz_kwargs: Any,
    ) -> Path:
        """Render the Graphviz output to an image file."""

        from graphviz import Source

        dot_source = self.to_graphviz(
            rankdir=rankdir,
            **graphviz_kwargs,
        )

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        source = Source(
            dot_source,
            filename=path.stem,
            directory=str(path.parent),
            engine=engine,
        )
        rendered = Path(source.render(format=format, cleanup=True))
        logger.success("Graphviz visualization saved to {}", rendered)
        return rendered

    def render_mermaid(
        self,
        output_path: str | Path,
        *,
        direction: str = "LR",
        include_node_class: bool = True,
        wrap_markdown: bool = True,
        **mermaid_kwargs: Any,
    ) -> Path:
        """Write the Mermaid diagram to disk (defaults to markdown fenced block)."""

        mermaid_source = self.to_mermaid(
            direction=direction,
            include_node_class=include_node_class,
            **mermaid_kwargs,
        )
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if wrap_markdown:
            content = f"```mermaid\n{mermaid_source}\n```"
        else:
            content = mermaid_source

        path.write_text(content, encoding="utf-8")
        logger.success("Mermaid visualization saved to {}", path)
        return path

    def _dot_identifier(self, node: Node) -> str:
        return self._sanitize_identifier(node.name)

    def _mermaid_identifier(self, node: Node) -> str:
        return self._sanitize_identifier(node.name, allow_dash=False)

    def _format_node_label(
        self,
        node: Node,
        include_node_class: bool,
        mermaid: bool = False,
        stage_text: str | None = None,
    ) -> str:
        parts: list[str] = [node.name]
        if include_node_class and node.__class__.__name__ != node.name:
            parts.append(node.__class__.__name__)
        if stage_text:
            parts.append(f"Stage: {stage_text}")
        separator = "<br/>" if mermaid else "\n"
        return separator.join(parts)

    def _format_edge_label(
        self,
        source: Node,
        target: Node,
        edge_data: dict,
        *,
        include_port_types: bool,
        mermaid: bool = False,
        dedupe_matching_ports: bool = False,
    ) -> str:
        from_port = edge_data.get("from_port") or ""
        to_port = edge_data.get("to_port") or ""

        # In card mode the port names are already rendered beside each dot
        # inside the node card, so no edge label is needed.
        if dedupe_matching_ports and not include_port_types:
            return ""

        from_spec = (
            self._resolve_port_spec(source, from_port, is_output=True)
            if include_port_types
            else None
        )
        to_spec = (
            self._resolve_port_spec(target, to_port, is_output=False)
            if include_port_types
            else None
        )

        left = self._format_port_segment(from_port, from_spec)
        right = self._format_port_segment(to_port, to_spec)

        if not (left or right):
            return ""

        connector = "-->" if mermaid else "->"
        return f"{left} {connector} {right}".strip()

    def _format_port_segment(self, port_name: str, spec: PortSpec | None) -> str:
        if not port_name:
            return ""
        if not spec:
            return port_name
        detail = self._format_port_spec(spec)
        return f"{port_name} ({detail})" if detail else port_name

    def _resolve_port_spec(
        self,
        node: Node,
        port_name: str | None,
        *,
        is_output: bool,
    ) -> PortSpec | None:
        if not port_name:
            return None
        ports = getattr(node, "_output_ports" if is_output else "_input_ports", None)
        if not ports:
            return None
        port = ports.get(port_name)
        if port is None:
            return None
        return getattr(port, "spec", None)

    def _format_port_spec(self, spec: PortSpec | None) -> str:
        if spec is None:
            return ""
        dtype = self._format_dtype(spec.dtype)
        shape = ", ".join(str(dim) for dim in spec.shape)
        if dtype and shape:
            return f"{dtype} [{shape}]"
        return dtype or shape

    @staticmethod
    def _format_dtype(dtype: Any) -> str:
        if hasattr(dtype, "__name__"):
            return dtype.__name__
        return str(dtype)

    @staticmethod
    def _sanitize_identifier(value: str, *, allow_dash: bool = True) -> str:
        safe = []
        for char in value:
            if char.isalnum() or char == "_":
                safe.append(char)
            elif allow_dash and char == "-":
                safe.append(char)
            else:
                safe.append("_")
        return "".join(safe) or "node"

    def _escape_label(self, value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"')

    def _escape_mermaid_label(self, value: str) -> str:
        return value.replace("]", ")")

    def _sanitize_mermaid_pipe(self, value: str) -> str:
        return value.replace("|", "/")

    def _group_nodes_by_stage(self) -> dict[str, list[Node]]:
        groups: dict[str, list[Node]] = defaultdict(list)
        for node in self._graph.nodes:
            groups[self._stage_bucket(node)].append(node)
        return groups

    def _stage_bucket(self, node: Node) -> str:
        stages = getattr(node, "execution_stages", None)
        if not stages:
            return ExecutionStage.ALWAYS.value
        normalized = {self._normalize_stage(stage) for stage in stages}
        normalized.discard(ExecutionStage.ALWAYS.value)
        if not normalized:
            return ExecutionStage.ALWAYS.value
        if len(normalized) > 1:
            return ExecutionStage.ALWAYS.value
        return next(iter(normalized))

    def _normalize_stage(self, stage: ExecutionStage | str) -> str:
        if isinstance(stage, ExecutionStage):
            return stage.value
        return str(stage).lower()

    def _format_graphviz_attributes(self, attrs: Mapping[str, Any]) -> list[str]:
        return [
            f"    {key}={self._quote_graphviz_value(value)};"
            for key, value in attrs.items()
        ]

    def _compose_attribute_list(self, attrs: Mapping[str, Any]) -> str:
        return ", ".join(
            f"{key}={self._quote_graphviz_value(value)}" for key, value in attrs.items()
        )

    def _format_inline_attributes(self, attrs: Mapping[str, Any]) -> list[str]:
        return [
            f"{key}={self._quote_graphviz_value(value)}" for key, value in attrs.items()
        ]

    def _quote_graphviz_value(self, value: Any) -> str:
        if isinstance(value, (int, float)):
            return str(value)
        if value is None:
            return '""'
        return f'"{self._escape_label(str(value))}"'

    def _format_execution_stage_text(
        self, node: Node, stage_labels: Mapping[str, str]
    ) -> str | None:
        stages = self._node_stage_values(node)
        if not stages:
            return None
        labels = sorted({stage_labels.get(stage, stage.title()) for stage in stages})
        if not labels:
            return None
        if labels == [stage_labels.get(ExecutionStage.ALWAYS.value, "All Stages")]:
            return labels[0]
        return " / ".join(labels)

    def _node_stage_values(self, node: Node) -> set[str]:
        raw_stages = getattr(node, "execution_stages", None)
        if not raw_stages:
            return {ExecutionStage.ALWAYS.value}
        normalized = {self._normalize_stage(stage) for stage in raw_stages if stage}
        normalized.discard(None)
        if not normalized or ExecutionStage.ALWAYS.value in normalized:
            return {ExecutionStage.ALWAYS.value}
        return normalized

    def _format_card_label(
        self,
        node: Node,
        *,
        index: int,
        show_node_name: bool,
        stage_text: str | None,
        registry: NodeRegistry | None = None,
    ) -> str:
        display = resolve_display(node)
        title = self._escape_html(display.get("label") or node.__class__.__name__)
        emoji = self._escape_html(display.get("emoji", ""))
        fill = display["fill"]
        border = display["border"]

        plugin_cell = (
            self._pill_html("Plugin", "#E8A33D")
            if is_plugin(node, registry=registry)
            else ""
        )

        # Header spans the full card width via COLSPAN, laying out "Node N"
        # and the optional Plugin pill inside a nested 2-cell table so the
        # pill only takes as much width as its text.
        header_inner = (
            '<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0">'
            "<TR>"
            '<TD ALIGN="LEFT">'
            f'<FONT COLOR="#888888" POINT-SIZE="9">Node {index}</FONT>'
            "</TD>"
            f'<TD ALIGN="RIGHT">{plugin_cell}</TD>'
            "</TR></TABLE>"
        )
        header_row = f'<TR><TD COLSPAN="3" CELLPADDING="6">{header_inner}</TD></TR>'

        body_rows: list[str] = [
            f'<TR><TD CELLPADDING="4"><FONT POINT-SIZE="18">{emoji}</FONT></TD></TR>',
            f'<TR><TD CELLPADDING="2"><B><FONT POINT-SIZE="12">{title}</FONT></B></TD></TR>',
        ]
        if show_node_name and node.name != node.__class__.__name__:
            body_rows.append(
                f'<TR><TD CELLPADDING="2">'
                f'<FONT COLOR="#888888" POINT-SIZE="9">{self._escape_html(node.name)}</FONT>'
                "</TD></TR>"
            )
        if stage_text:
            body_rows.append(
                f'<TR><TD CELLPADDING="2">'
                f'<FONT COLOR="#888888" POINT-SIZE="8">Stage: {self._escape_html(stage_text)}</FONT>'
                "</TD></TR>"
            )

        inner_body = (
            '<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">'
            + "".join(body_rows)
            + "</TABLE>"
        )

        input_dots = self._port_dots_html(node, is_output=False)
        output_dots = self._port_dots_html(node, is_output=True)

        # CELLPADDING="0" on the outer port TDs pushes the dot knots flush
        # against the card's outer border so wires appear to attach AT the
        # edge rather than inset inside the card body.
        body_row = (
            "<TR>"
            f'<TD CELLPADDING="0" VALIGN="MIDDLE">{input_dots}</TD>'
            f'<TD CELLPADDING="10" VALIGN="MIDDLE" ALIGN="CENTER">{inner_body}</TD>'
            f'<TD CELLPADDING="0" VALIGN="MIDDLE">{output_dots}</TD>'
            "</TR>"
        )

        return (
            f'<TABLE BORDER="1" COLOR="{border}" CELLBORDER="0" CELLSPACING="0" '
            f'CELLPADDING="0" BGCOLOR="{fill}" STYLE="ROUNDED">'
            f"{header_row}{body_row}</TABLE>"
        )

    def _port_dots_html(self, node: Node, *, is_output: bool) -> str:
        ports = getattr(node, "_output_ports" if is_output else "_input_ports", None)
        if not ports:
            return "&nbsp;"

        rows: list[str] = []
        for port_name, port in ports.items():
            spec = getattr(port, "spec", None)
            color = self._dtype_hex_color(spec.dtype if spec else None)
            safe_name = self._escape_html(port_name)
            port_id = self._port_anchor_id(port_name, is_output=is_output)

            dot_td = (
                f'<TD BGCOLOR="{color}" CELLPADDING="2" '
                f'PORT="{port_id}" TITLE="{safe_name}">'
                '<FONT POINT-SIZE="6">  </FONT>'
                "</TD>"
            )
            name_align = "RIGHT" if is_output else "LEFT"
            name_td = (
                f'<TD CELLPADDING="2" ALIGN="{name_align}">'
                f'<FONT POINT-SIZE="9" COLOR="#555555">{safe_name}</FONT>'
                "</TD>"
            )
            if is_output:
                rows.append(f"<TR>{name_td}{dot_td}</TR>")
            else:
                rows.append(f"<TR>{dot_td}{name_td}</TR>")

        return (
            '<TABLE BORDER="0" CELLBORDER="0" '
            'CELLSPACING="4" CELLPADDING="0">' + "".join(rows) + "</TABLE>"
        )

    @staticmethod
    def _port_anchor_id(port_name: str, *, is_output: bool) -> str:
        prefix = "out_" if is_output else "in_"
        return prefix + PipelineVisualizer._sanitize_identifier(
            port_name, allow_dash=False
        )

    @staticmethod
    def _pill_html(text: str, color: str) -> str:
        safe = PipelineVisualizer._escape_html(text)
        return (
            f'<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">'
            f'<TR><TD BGCOLOR="{color}" CELLPADDING="3">'
            f'<FONT COLOR="white" POINT-SIZE="8"><B>{safe}</B></FONT>'
            f"</TD></TR></TABLE>"
        )

    @staticmethod
    def _dtype_hex_color(dtype: Any) -> str:
        key = PipelineVisualizer._format_dtype(dtype) if dtype is not None else ""
        rgb = _DTYPE_COLORS.get(key, _DTYPE_DEFAULT_RGB)
        return "#{:02x}{:02x}{:02x}".format(*rgb)

    @staticmethod
    def _escape_html(value: str) -> str:
        return (
            str(value)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )


def visualize_pipeline(
    pipeline: CuvisPipeline,
    *,
    format: str = "graphviz",
    output_path: str | Path | None = None,
    **kwargs,
) -> str | Path:
    """Convenience helper to generate or render pipeline visualizations."""

    visualizer = PipelineVisualizer(pipeline)
    format_key = format.lower()

    if format_key in {"graphviz", "dot", "dot_string"}:
        return visualizer.to_graphviz(**kwargs)
    if format_key in {"mermaid", "mermaid_string"}:
        return visualizer.to_mermaid(**kwargs)
    if format_key in {"render", "render_graphviz"}:
        if output_path is None:
            raise ValueError("output_path is required for render_graphviz formats.")
        return visualizer.render_graphviz(output_path, **kwargs)
    if format_key == "render_mermaid":
        if output_path is None:
            raise ValueError("output_path is required for render_mermaid format.")
        return visualizer.render_mermaid(output_path, **kwargs)

    raise ValueError(f"Unsupported visualization format: {format}")
