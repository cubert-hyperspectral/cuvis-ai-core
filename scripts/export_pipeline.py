"""CLI: export a span of a pipeline to ONNX and write its deploy folder.

Loads the pipeline yaml and its weights the way ``restore-pipeline`` does, cuts
the span between the ``--input`` and ``--output`` ports, exports it at the shapes
of the examples and writes ``<name>.onnx``, ``<name>.onnx.json``,
``<name>_deploy.yaml`` and ``<name>_deploy.pt`` into ``--out-dir``. The export
fails unless the live nodes, their export adapters and ONNX Runtime agree on
every example.

Examples come from files (``--examples``, one per file, arrays keyed by
``node.port``) or from a data module (``--data-module``), in which case the
pipeline runs on its predict split and the values that reach the span inputs are
kept.
"""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import torch

from cuvis_ai_core.pipeline.export import (
    DEFAULT_SPAN_NODE_NAME,
    SpanExportError,
    SpanSelectionError,
    collect_examples,
    export_deployment,
)
from cuvis_ai_core.utils.restore import predict_batches, restore_pipeline


def _load_example(path: Path, device: str | None) -> dict[str, torch.Tensor]:
    """One example: an ``.npz`` (``np.savez``) or a ``.pt`` dict of tensors."""
    if path.suffix == ".npz":
        with np.load(path) as archive:
            example = {key: torch.from_numpy(archive[key]) for key in archive.files}
    elif path.suffix == ".pt":
        loaded = torch.load(path, weights_only=True)
        if not isinstance(loaded, dict):
            raise click.ClickException(f"{path} must hold a dict of tensors.")
        example = dict(loaded)
    else:
        raise click.ClickException(f"{path}: examples are .npz or .pt files.")
    if device:
        example = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in example.items()
        }
    return example


@click.command(name="export-pipeline")
@click.option(
    "--pipeline-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="The pipeline yaml.",
)
@click.option(
    "--weights-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="The weights (.pt); defaults to the pipeline path with .pt.",
)
@click.option(
    "--out-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="The deploy folder (created when missing).",
)
@click.option(
    "--input",
    "inputs",
    multiple=True,
    required=True,
    metavar="NODE.PORT",
    help="A span input port. Repeatable.",
)
@click.option(
    "--output",
    "outputs",
    multiple=True,
    required=True,
    metavar="NODE.PORT",
    help="A span output port. Repeatable.",
)
@click.option(
    "--constant",
    "constants",
    multiple=True,
    metavar="NODE.PORT",
    help="A span input baked into the graph (a wavelength grid, token ids). Repeatable.",
)
@click.option(
    "--examples",
    "example_files",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="An example: an .npz or .pt file with one array per span input, keyed "
    "NODE.PORT. Repeatable.",
)
@click.option(
    "--data-module",
    default=None,
    help="A data module name (its DATA_MODULE_NAME, e.g. cu3s) whose predict split "
    "provides the examples; its plugin is loaded from --plugins-dir.",
)
@click.option(
    "--data-arg",
    "data_args",
    multiple=True,
    metavar="KEY=VALUE",
    help="An argument of the selected data module. Repeatable.",
)
@click.option(
    "--num-examples",
    default=2,
    show_default=True,
    type=click.IntRange(min=1),
    help="How many batches of the data module to keep as examples.",
)
@click.option(
    "--plugins-dir",
    "plugins_dirs",
    multiple=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="A directory of plugin manifests. Repeatable.",
)
@click.option(
    "--override",
    "config_overrides",
    multiple=True,
    help="A config override in dot notation. Repeatable.",
)
@click.option(
    "--device",
    default="auto",
    show_default=True,
    type=click.Choice(["auto", "cpu", "cuda"]),
    help="Where the pipeline and the examples live during the export.",
)
@click.option(
    "--name", default=None, help="The file stem; the pipeline name by default."
)
@click.option(
    "--opset", default=18, show_default=True, type=int, help="The ONNX opset."
)
@click.option(
    "--atol", default=1e-5, show_default=True, type=float, help="Parity tolerance."
)
@click.option(
    "--rtol", default=1e-4, show_default=True, type=float, help="Parity tolerance."
)
@click.option(
    "--span-node-name",
    default=DEFAULT_SPAN_NODE_NAME,
    show_default=True,
    help="The name of the node that stands in for the span in the deploy yaml.",
)
def main(
    pipeline_path: Path,
    weights_path: Path | None,
    out_dir: Path,
    inputs: tuple[str, ...],
    outputs: tuple[str, ...],
    constants: tuple[str, ...],
    example_files: tuple[Path, ...],
    data_module: str | None,
    data_args: tuple[str, ...],
    num_examples: int,
    plugins_dirs: tuple[Path, ...],
    config_overrides: tuple[str, ...],
    device: str,
    name: str | None,
    opset: int,
    atol: float,
    rtol: float,
    span_node_name: str,
) -> None:
    """Export the span between the input and output ports of a pipeline to ONNX."""
    if not example_files and not data_module:
        raise click.UsageError(
            "Pass --examples FILE (repeatable) or --data-module NAME to provide samples "
            "for the span inputs."
        )
    parsed_args: dict[str, str] = {}
    for pair in data_args:
        if "=" not in pair:
            raise click.UsageError(f"--data-arg must be KEY=VALUE, got {pair!r}")
        key, value = pair.split("=", 1)
        parsed_args[key.strip()] = value

    load_device = None if device == "auto" else device
    plugin_dirs: list[str | Path] | None = [str(p) for p in plugins_dirs] or None
    pipeline = restore_pipeline(
        pipeline_path=pipeline_path,
        weights_path=weights_path,
        device=device,
        config_overrides=list(config_overrides) or None,
        plugins_dirs=plugin_dirs,
    )
    try:
        if example_files:
            examples = [_load_example(path, load_device) for path in example_files]
        else:
            batches = predict_batches(
                pipeline_path,
                data_module or "",
                data_args=parsed_args,
                plugins_dirs=plugin_dirs,
                device=load_device,
            )
            examples = collect_examples(pipeline, inputs, batches, limit=num_examples)
        deployment = export_deployment(
            pipeline,
            out_dir,
            inputs=inputs,
            outputs=outputs,
            examples=examples,
            constants=constants,
            name=name,
            opset=opset,
            atol=atol,
            rtol=rtol,
            span_node_name=span_node_name,
        )
    except (SpanSelectionError, SpanExportError) as exc:
        raise click.ClickException(str(exc)) from exc

    sidecar = deployment.sidecar
    click.echo(f"Span nodes: {[n['name'] for n in sidecar['span']['nodes']]}")
    click.echo(
        f"Exporter: {sidecar['exporter']['kind']} (opset {sidecar['exporter']['opset']}); "
        f"parity on {sidecar['parity']['examples']} example(s), max abs diff "
        + ", ".join(
            f"{k} {v:.2e}" for k, v in sidecar["parity"]["max_abs_diff"].items()
        )
    )
    click.echo(f"ONNX:    {deployment.onnx_path}")
    click.echo(f"Sidecar: {deployment.sidecar_path}")
    click.echo(f"Deploy:  {deployment.yaml_path}")
    click.echo(f"Weights: {deployment.weights_path}")


if __name__ == "__main__":
    main()
