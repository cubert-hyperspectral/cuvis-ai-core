"""Sessionless pipeline-config visualization.

Render a pipeline's node graph straight from its YAML config, without building the
pipeline: no node instantiation, no plugin environment, no model load. This backs the
config-preview path of ``GetPipelineVisualization`` so a client can show a pipeline's
graph the moment it is selected in a picker, before it is ever loaded into a session.

The graph is derived purely from the config's declared ``nodes`` and ``connections``
(the ports live in the connection strings), so it needs neither the node classes nor the
plugin catalog. Rendering reuses the ``graphviz`` dependency already required by
:class:`~cuvis_ai_core.pipeline.visualizer.PipelineVisualizer`; if the graphviz binary is
unavailable the DOT source is returned so the caller still has something to display.
"""

from __future__ import annotations

import yaml
from loguru import logger

from cuvis_ai_schemas.pipeline.config import PipelineConfig

_IMAGE_FORMATS = frozenset({"png", "svg"})
_DOT_FORMATS = frozenset({"dot", "graphviz"})


def _short_class(class_name: str) -> str:
    """Return the bare class name from a fully-qualified ``pkg.mod.Class`` path."""
    return class_name.rsplit(".", 1)[-1] if class_name else class_name


def _escape(text: str) -> str:
    """Escape a string for use inside a double-quoted DOT label or identifier."""
    return text.replace("\\", "\\\\").replace('"', '\\"')


def config_to_dot(config: PipelineConfig, *, rankdir: str = "LR") -> str:
    """Build a Graphviz DOT graph of a pipeline config's nodes and connections.

    Each node is a rounded box labelled with its name and short class; each connection is a
    port-labelled edge. The layout is structural only (no port dtypes or category colours,
    which would need the plugin catalog), which is enough to preview a pipeline's shape.

    Args:
        config: Parsed pipeline configuration.
        rankdir: Graphviz layout direction ("LR" left-to-right, "TB" top-to-bottom).

    Returns:
        A DOT source string.
    """
    graph_name = (
        config.metadata.name if config.metadata and config.metadata.name else "pipeline"
    )
    lines: list[str] = [f'digraph "{_escape(graph_name)}" {{']
    lines.append(f"    rankdir={rankdir};")
    lines.append('    bgcolor="transparent";')
    lines.append(
        '    node [shape=box, style="rounded,filled", fillcolor="#2b2b2b", '
        'color="#8a8a8a", fontcolor="#e6e6e6", fontname="Helvetica", fontsize=11, '
        'margin="0.16,0.09"];'
    )
    lines.append(
        '    edge [color="#7aa2d6", fontcolor="#b0b0b0", fontname="Helvetica", '
        "fontsize=9, penwidth=1.4];"
    )

    declared: set[str] = set()
    for node in config.nodes:
        label = f"{_escape(node.name)}\\n{_escape(_short_class(node.class_name))}"
        lines.append(f'    "{_escape(node.name)}" [label="{label}"];')
        declared.add(node.name)

    # A well-formed config only references declared nodes, but never emit a dangling edge:
    # give any endpoint missing from nodes[] a minimal box so the graph still renders.
    for conn in config.connections:
        for endpoint in (conn.from_node, conn.to_node):
            if endpoint not in declared:
                lines.append(
                    f'    "{_escape(endpoint)}" [label="{_escape(endpoint)}"];'
                )
                declared.add(endpoint)

    for conn in config.connections:
        label = (
            conn.from_port
            if conn.from_port == conn.to_port
            else f"{conn.from_port} → {conn.to_port}"
        )
        lines.append(
            f'    "{_escape(conn.from_node)}" -> "{_escape(conn.to_node)}" '
            f'[label="{_escape(label)}"];'
        )

    if not config.nodes and not config.connections:
        lines.append('    "empty" [label="(empty pipeline)", color="#8a8a8a"];')

    lines.append("}")
    return "\n".join(lines)


def render_pipeline_config(
    yaml_content: str, fmt: str = "png", *, rankdir: str = "LR"
) -> tuple[bytes, str]:
    """Render a pipeline YAML config to an image (or DOT) without building the pipeline.

    Args:
        yaml_content: The pipeline configuration YAML text.
        fmt: Output format, one of "png", "svg", "dot"/"graphviz".
        rankdir: Graphviz layout direction passed through to :func:`config_to_dot`.

    Returns:
        ``(data, actual_format)`` where ``data`` is the encoded image bytes for an image
        format, or UTF-8 DOT bytes for a DOT request or when graphviz is unavailable.
    """
    requested = (fmt or "png").lower()
    config = PipelineConfig.model_validate(yaml.safe_load(yaml_content) or {})
    dot = config_to_dot(config, rankdir=rankdir)

    if requested in _DOT_FORMATS:
        return dot.encode("utf-8"), "dot"

    image_format = requested if requested in _IMAGE_FORMATS else "png"
    try:
        import graphviz

        data = graphviz.Source(dot).pipe(format=image_format)
        return bytes(data), image_format
    except Exception as exc:  # graphviz binary missing or render failure
        logger.warning(
            "Config visualization render failed ({}); returning DOT source.", exc
        )
        return dot.encode("utf-8"), "dot"


__all__ = ["config_to_dot", "render_pipeline_config"]
