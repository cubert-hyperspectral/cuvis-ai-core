# Running the gRPC server safely

A caller that reaches the server's port can do a lot: register a plugin and have the server install
and import it, load checkpoints, and read and write files through the path fields of the API
(`SavePipeline`, `RestoreTrainRun` and others). Plugin code runs with the rights of the account that
started the server. Treat access to the port as access to that account.

The defaults keep the server on the local machine. The settings below let a launcher, such as
CuvisNEXT, limit what a local caller can do, and let an operator expose the server on a network
with a token and TLS.

## Settings

All settings are environment variables of the server process.

| Variable | Default | Effect |
|---|---|---|
| `GRPC_BIND_HOST` | `localhost` | Address to listen on. `localhost` covers 127.0.0.1 and ::1. |
| `GRPC_PORT` | `50051` | Port to listen on. |
| `CUVIS_GRPC_TOKEN` | unset | Shared secret. When set, every call except the gRPC health check must carry it. |
| `GRPC_USE_TLS` | `false` | `true` serves TLS with `GRPC_TLS_CERT_PATH` and `GRPC_TLS_KEY_PATH`. |
| `GRPC_ALLOW_INSECURE_REMOTE` | unset | `1` starts on a network address without a token or TLS, with a warning. |
| `CUVIS_ALLOWED_ROOTS` | unset | Folders client paths must stay in, separated by `;` on Windows and `:` elsewhere. |
| `CUVIS_PLUGINS_DIR` | unset | Folder of plugin manifests. When set, only these plugins can be registered. |
| `GRPC_MAX_MSG_SIZE` | 300 MB | Largest message, in bytes, the server and its child runtimes accept. |

The server reads `CUVIS_GRPC_TOKEN` once at start and removes it from its environment, so the
processes it starts later (uv, plugin builds, child runtimes) never see it.

## Local use

Without any of these variables the server listens on `localhost` and checks nothing else. Any
process of any user on the machine can call it. That is fine on a single-user workstation and for
notebooks and tests. On a shared machine, set a token.

## A launcher on the same machine

A launcher that starts the server for its own use (CuvisNEXT does) should set all three of
`CUVIS_GRPC_TOKEN`, `CUVIS_ALLOWED_ROOTS` and `CUVIS_PLUGINS_DIR`.

### Token

Generate a fresh random value for every launch and send it with every call as `authorization`
metadata, either `Bearer <token>` or the bare value. A call without it gets `UNAUTHENTICATED`. Send
it as plain metadata on the channel: gRPC call credentials are dropped on an insecure channel, and a
loopback channel is insecure.

A Python client sends the token with `authorized_channel`:

```python
import grpc

from cuvis_ai_core.grpc import cuvis_ai_pb2_grpc
from cuvis_ai_core.utils.grpc_auth import authorized_channel

channel = authorized_channel(grpc.insecure_channel("localhost:50051"), token)
stub = cuvis_ai_pb2_grpc.CuvisAIServiceStub(channel)
```

`call_metadata(token)` returns the metadata list for a single call instead.

### Allowed roots

List the folders the launcher works in: its data, its models and its output. A path in a request is
then refused with `PERMISSION_DENIED` when it is a network or device path (`\\host\share`,
`//host/share`, `\\?\...`, `\??\...`, a URL such as `http://...` or `s3://...`, and on Windows a
device name such as `NUL`, `CON`, `C:NUL` or `COM1.txt`), starts with `~`, contains a `..`
segment, or resolves outside every root. Network and device paths are refused on their text,
before any filesystem call, so the server never opens an SMB session to a host a client named
through these fields; the data paths listed under "Not covered" are the exception. The splits file
of a training run (`data.splits.splits_path`) is checked the same way before training starts.
A trainrun's `pipeline:` entry that leads outside the roots is answered as not found, whether or
not the file exists. Session search paths outside the roots come back in
`rejected_paths`. Files the server writes next to a requested one (the `.pt` beside a saved
pipeline, the pipeline and weights beside a saved trainrun) have to lie inside the roots too, and so
do the folders gradient training writes to: the checkpoint folder
(`training.callbacks.checkpoint.dirpath`) and `default_root_dir`. A relative folder, or a missing
`default_root_dir`, lands under the runtime's working directory, which then has to lie inside a
root, and the checkpoint file name (`filename`) has to stay inside its folder, also through a
subfolder that is a link. Saved pipelines and trainruns are written to a staging file beside the
target and moved into place, so a link put at the target after the check is replaced, not followed;
a replaced file keeps its permissions. `ResolveConfig` refuses Hydra's own settings
(`hydra.searchpath` and the like), from overrides or from the config file, and every file Hydra
pulls in while composing (a `defaults:` entry, a group override such as `+data=name`) has to lie
inside the roots as well.

The roots cover the locations a client chooses. The server's own pipeline library
(`CUVIS_CONFIGS_DIR`) can be read without being a root: `ResolveConfig`, the session search paths,
weight lookups, `RestoreTrainRun`, `ListAvailablePipelines` and `GetPipelineInfo` may read from it.
Writes still need a root, so a client cannot change the library through `SavePipeline` or
`SaveTrainRun` unless the launcher lists it among the roots.

### Plugins directory

Point it at the folder of plugin manifests the launcher ships. `LoadPlugin` then accepts a manifest
only when that folder holds one of the same name with the same source: the same `repo` and `tag`, or
the same local `path`. The server registers its own copy, so a client cannot add capabilities or
change where the code comes from.

## Remote use

An address other than loopback in `GRPC_BIND_HOST` starts only with both a token and TLS:

```powershell
$env:GRPC_BIND_HOST = "0.0.0.0"
$env:CUVIS_GRPC_TOKEN = "<a long random value>"
$env:GRPC_USE_TLS = "true"
$env:GRPC_TLS_CERT_PATH = "C:\certs\server.pem"
$env:GRPC_TLS_KEY_PATH = "C:\certs\server.key"
uv run python -m cuvis_ai_core.grpc.production_server
```

Without one of them the server stops at start and names what is missing. The token keeps out
callers that do not hold it; TLS keeps the token from being read off the wire. The server does not
check client certificates yet, so every holder of the token has full access.

`GRPC_ALLOW_INSECURE_REMOTE=1` starts the server on a network address without them and logs a
warning. Use it only on a network where every machine is trusted.

## Child runtimes

The server runs each pipeline in a child process that listens on its own loopback port. Every child
gets a fresh token from the server and refuses calls without it, so another local process that finds
the child's port cannot use it. Nothing needs to be configured for this. A child started by hand,
without `CUVIS_RUNTIME_TOKEN`, checks nothing and logs a warning.

Both servers refuse a call without the token before they read its request, so a caller without
the token cannot make either process receive or parse a message. A child accepts no larger message
than the server does (`GRPC_MAX_MSG_SIZE`; `-1`, no limit, applies to both).

## Checkpoints

Pipeline checkpoints load with `torch.load(weights_only=True)`, whether they come as bytes from a
client, from a path, or through `restore-pipeline` and `restore-trainrun`. A checkpoint that holds
anything but tensors and plain containers is refused with `INVALID_ARGUMENT` instead of being
unpickled. Lightning checkpoints given to `GradientTrainer.validate` and `test` load the same way.

## Not covered

- A registered plugin runs with the rights of the server account. There is no sandbox; the plugins
  directory decides which plugins can run at all.
- Plugins load their own pretrained weights in their own code, and some of them unpickle those files
  in full. The checkpoint loading described above does not apply there.
- `CUVIS_ALLOWED_ROOTS` covers the path fields of the API, the trainrun file and the pipeline it
  names, and the splits file. The other data paths of a run (`data_dir`, a data module's
  parameters, the files a split lists) are not checked, network paths included.
- There are no permissions per call: a token holder can call every RPC.
- A save checks its target and then writes it. A process that can write inside an allowed folder
  and swaps a parent folder of the target for a link in between can still redirect the save.
