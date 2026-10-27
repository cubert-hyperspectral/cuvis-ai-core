# Exporting a pipeline span to ONNX

`export-pipeline` takes a trained pipeline (its yaml and `.pt`), cuts the span between
the input and output ports you name, exports that span as one ONNX graph and writes a
deploy folder. The deploy folder is an ordinary pipeline: one `ExportedSpan` node runs the
graph with ONNX Runtime, every node outside the span runs as before. The canonical yaml
and `.pt` are not touched, so training and weight reloads keep working from them.

```
out_dir/
  <name>.onnx            the span (plus <name>.onnx.data above 1 GiB of weights)
  <name>.onnx.json       the sidecar: ports, shapes, constants, fingerprints, parity
  <name>_deploy.yaml     the pipeline with the span replaced by one ExportedSpan node
  <name>_deploy.pt       the weights of the nodes that stay outside the span
```

## Cutting the span

A span is cut by port. The selection walks back from the output ports to the declared
input ports and takes every node on the way. It refuses, with the port or node named:

- an input port of a span node that is neither declared nor fed by a span node (the
  pipeline would read it from the batch);
- a node that does not run at inference;
- a span port that an outside node reads but that is not declared as an output (the
  deploy graph would starve that node);
- a cut whose contraction into one node makes the graph cyclic (an outside node reads a
  span output and feeds a span input);
- a variadic port as a span input (cut at the ports that feed it instead).

An unconnected optional port stays absent, as in the eager pipeline. A declared input
that the outputs do not reach is an error, not silently ignored.

Constants are declared inputs whose value is baked into the graph as a buffer: the
wavelength grid a channel selector reads, the token ids of a text prompt. Their value
must be the same in every example; the sidecar records it, and the runtime node refuses
a frame that carries another value.

## Running the export

```bash
uv run export-pipeline \
  --pipeline-path outputs/dinomaly/dinomaly_custom_trained.yaml \
  --out-dir outputs/dinomaly/deploy \
  --input MinMaxNormalizer.data --input custom_selector.wavelengths \
  --constant custom_selector.wavelengths \
  --output dinomaly_detector.scores --output dinomaly_detector.anomaly_score \
  --data-module cu3s --data-arg cu3s_file_path=data/sample.cu3s --num-examples 3 \
  --plugins-dir ../cuvis-ai/cuvis_ai/configs/plugins
```

Examples are real samples for the span inputs. With `--data-module` the pipeline runs on
the module's predict split and the values that reach the span inputs are kept, `--num-examples`
batches of them. With `--examples` each file is one example, an `.npz` (`numpy.savez`) or a
`.pt` dict with one array per span input keyed `node.port`. The graph is exported at the
shapes of the first example (batch size 1 is the convention; the runtime node loops over a
batch), and the sidecar records those shapes.

The export fails unless three things agree on every example, within `--atol` and `--rtol`:
the live nodes' own `forward`, the span module (deep copies in eval mode, export adapters
in place of `forward`), and ONNX Runtime on the written file. A mismatch names the node
and port. When neither exporter carries the span, the error names the first node that
fails to export on its own and keeps the span's own error as the cause.

The dynamo exporter (`torch.export`, opset 18) runs first. When it fails, the TorchScript
exporter runs with every `TracerWarning` turned into an error, because the tracer turns a
`.item()` or a numpy value into a constant with only a warning; the sidecar records which
exporter wrote the file.

From Python:

```python
from cuvis_ai_core.pipeline.export import collect_examples, export_deployment

examples = collect_examples(pipeline, ["norm.data", "sel.wavelengths"], batches, limit=3)
deployment = export_deployment(
    pipeline,
    "deploy",
    inputs=["norm.data", "sel.wavelengths"],
    outputs=["det.scores", "det.anomaly_score"],
    constants=["sel.wavelengths"],
    examples=examples,
)
```

## The sidecar

`<name>.onnx.json` is what the runtime node reads to rebuild its ports and check the file:

| Field | Content |
|---|---|
| `span.nodes` | name and class of every span node, in execution order |
| `span.inputs`, `span.outputs` | per port: the pipeline port, the port name on the `ExportedSpan` node, the ONNX tensor name, dtype and shape |
| `span.constants` | per baked input: port, name, dtype, shape and the value |
| `exporter` | `dynamo` or `legacy`, the opset, the torch and cuvis-ai-core versions |
| `fingerprint` | a digest of the span weights, wiring and hparams (`utils.trt_engine.tensor_fingerprint`), the key for an engine built from this file |
| `onnx_md5`, `onnx_data_md5` | the hash of the written file(s); the runtime refuses an edited file |
| `parity` | the number of examples, the tolerance and the largest difference per output |

The port names on the `ExportedSpan` node are the bare port names when they are unique
among the span's inputs (and, separately, outputs), else `<node>_<port>`.

## Nodes that need an export adapter

The exporter traces a node's `forward` as it is. A `forward` that reads a value with
`.item()`, looks something up with numpy, branches on a Python float or compiles lazily
cannot be traced, and the error names the node. Such a node overrides
`Node.export_module()` and returns a module that computes the same outputs with tensor
ops:

```python
class MyDetector(Node):
    def forward(self, rgb_image, **_):
        scale = float(rgb_image.max())  # a Python float: not traceable
        ...

    def export_module(self):
        return _MyDetectorExport(self)


class _MyDetectorExport(torch.nn.Module):
    def __init__(self, node):
        super().__init__()
        self.model = node.model  # shares the weights of the copy it was built from

    def forward(self, rgb_image):
        scale = rgb_image.max()  # a tensor: traced
        ...
        return {"scores": scores, "anomaly_score": anomaly_score}
```

The contract: the exporter calls `export_module()` once, outside tracing, on a deep copy
of the node that is already in eval mode; the module takes the node's input ports as
keyword arguments (no `context`) and returns a dict of output port name to tensor; it
reads the node's weights and buffers and never mutates the node. The parity check
compares it with the node's own `forward` on every example before anything is written,
so an adapter that drifts from the node fails the export instead of shipping.

Nodes that export carry the `onnx` tag (`NodeTag.ONNX`), the marker the palette and the
export tooling read.

## Limits

- Fixed shapes: the graph takes the shapes of the first example. Another resolution needs
  another export.
- Tensor ports only: a span output that is not a tensor is refused. Keep such nodes
  outside the span.
- The runtime node and the `cuvis_ai_onnx` plugin manifest live in cuvis-ai; a deploy yaml
  lists `cuvis_ai_onnx` under `plugins:` so a composed environment installs ONNX Runtime.
- TensorRT: the sidecar fingerprint and `utils.trt_engine.build_engine_once` are the hooks
  for an engine backend; the deploy folder itself is runtime-agnostic.
