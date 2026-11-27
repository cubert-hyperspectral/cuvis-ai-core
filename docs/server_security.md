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
| `GRPC_HEALTH_PORT` | unset | A second port that answers only the gRPC health check, without a token. See "Health probes". |
| `CUVIS_GRPC_TOKEN` | unset | Shared secret. When set, every call must carry it, the health check included. |
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

List the folders the launcher works in: its data, its models and its output. Keep them writable by
the server account only; a link placed inside one can redirect the server's writes (see "Not
covered"). A path in a request is then refused with `PERMISSION_DENIED` when it is a network or device path (`\\host\share`,
`//host/share`, `\\?\...`, `\??\...`, a URL such as `http://...` or `s3://...` anywhere in the
path, an fsspec chain such as `simplecache::file:///...`, a scheme with one colon at the start
such as `file:/...` or `local:/...` (fsspec drops it), and on Windows a device name such as
`NUL`, `CON`, `C:NUL` or `COM1.txt`, a colon after the drive (an NTFS stream such as
`pipeline.yaml:hidden`), or a path from the drive root that does not name its drive,
such as `\runs`), starts with `~`, contains a `..` segment, or resolves outside every root. Network and device paths are refused on their text,
before any filesystem call, so the server never opens an SMB session to a host a client named
through these fields; the data paths listed under "Not covered" are the exception. The splits file
of a training run (`data.splits.splits_path`) is checked the same way when `RestoreTrainRun` reads
it and again before training starts.
A trainrun's `pipeline:` entry that leads outside the roots is answered as not found, whether or
not the file exists. Session search paths outside the roots come back in
`rejected_paths`. Files the server writes next to a requested one (the `.pt` beside a saved
pipeline, the pipeline and weights beside a saved trainrun) have to lie inside the roots too, and so
do the folders gradient training writes to: the checkpoint folder
(`training.callbacks.checkpoint.dirpath`) and `default_root_dir`. A relative folder, or a missing
`default_root_dir`, lands under the runtime's working directory, which then has to lie inside a
root. These folders may not hold `$` or `%`: Lightning's TensorBoard logger expands variables
in them after the check. Lightning fills in the placeholders of the checkpoint file name
(`filename`) and joins the
result to the folder, so the name has to be one plain name: letters, digits, `_.=+-` and
`{metric}` or `{metric:format}` placeholders whose format has no fill character and no `c`, `x`
or `X` type (`c` turns a number into any character, and hex digits are letters that can spell a
device name such as `COM1`). A name with a folder in it, such as `{epoch}/model`, is
refused. Saved pipelines and trainruns are written to a staging file beside the target and moved
into place, so a link put at the target after the check is replaced, not followed. The save writes
into the staging file it created and never opens it again by name, so its content never reaches
another file. On POSIX the save checks just before the move that the staging name still names the
file it wrote: a swap made before that check fails the save and leaves the target as it was. A swap
in the instant between the check and the move puts the other account's file at the target, and the
save still reports success; an account that can swap the staging file can also replace the target
directly. On POSIX only the owner can read the staging file. A replaced file keeps its group, its
mode and, on Linux, its access ACL. It keeps its owner when the server runs as root; under any other account that account
becomes its owner. When the owner, the group or the ACL cannot be carried over, only its owner can
read it. A new file stays readable by its owner only. On Windows see "Not covered". `ResolveConfig` refuses Hydra's own settings (`hydra.searchpath` and
the like), from overrides or from the config file, and every file Hydra pulls in while composing (a
`defaults:` entry, a group override such as `+data=name`) has to lie inside the roots as well,
whichever config source Hydra asks: Hydra's own package source does not look outside its package.

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

## Health probes

With a token set, the health check on the main port needs the token too. A probe that holds none,
such as a Docker `HEALTHCHECK` or a Kubernetes gRPC probe, uses `GRPC_HEALTH_PORT` instead: a second
port on the same host that answers `grpc.health.v1.Health/Check` and nothing else. It needs no
token and runs without TLS even when the main port uses it, so anyone who reaches it learns whether
the server is up. It answers without reading the request, keeps at most 2 calls per connection and
closes a connection after about a minute. It says NOT_SERVING until the main port runs and while it
shuts down.

The health port reports that the process is up, not that the main port can serve: when every worker
of the main port is busy it still answers SERVING. A probe that has to know whether the main port
can take a call calls `Check` there, with the token.

## Child runtimes

The server runs each pipeline in a child process that listens on its own loopback port. Every child
gets a fresh token from the server and refuses calls without it, so another local process that finds
the child's port cannot use it. Nothing needs to be configured for this. A child started by hand,
without `CUVIS_RUNTIME_TOKEN`, checks nothing and logs a warning.

Gradient training through the server runs on one device. An unset or `auto` `devices` becomes `1`,
and a training config that asks for more than one device (`devices` above 1, `-1`, or several
indices such as `0,1`) answers `INVALID_ARGUMENT` before the fit starts: Lightning would start the
child's command once more per extra device, and the run would die. Multi-GPU training through the
server needs a worker mode the runtime does not have yet. A saved trainrun keeps the `devices` the
client set. A copy of a child started with `LOCAL_RANK` set and not `0` has no token and exits
without serving.

Both servers refuse a call without the token before they read its request, so a caller without
the token cannot make either process receive or parse a message. A child accepts no larger message
than the server does (`GRPC_MAX_MSG_SIZE`; `-1`, no limit, applies to both). A server started from
Python with its own `max_msg_size` passes that value on to its children; one without a cap passes
none, and its children use the default.

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
- Links inside an allowed folder. A save checks its target and then writes it, and a read checks its
  file and then opens it. A process that can write inside an allowed folder can still send the
  server's writes or reads outside the roots through a link: by swapping a parent folder for one in
  between, or by placing one where gradient training writes (a `lightning_logs` folder under
  `default_root_dir`, a file in the checkpoint folder). Keep every
  allowed folder writable by the server account only.
- On Windows a save gets the permissions its folder passes on, while it is written and after: a file
  that was given tighter permissions than its folder loses them when it is saved again. Keep files
  that need tighter permissions in a folder that has them.
