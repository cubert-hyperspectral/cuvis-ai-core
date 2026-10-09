"""TensorRT engines for nodes with a TensorRT backend: where an engine lives, how it is built
once per machine, and how it runs on torch CUDA tensors.

A TensorRT engine is compiled for one GPU, one TensorRT version and one set of weights. A node
with a TensorRT backend finds or builds its engine in ``Node.prepare_inference()`` (the pipeline
loaders call it once the device and the weights are final) with :func:`find_engine` and
:func:`build_engine_once`, and runs it with :class:`TensorRTEngine`. The plugin keeps what is
specific to its network: the ONNX export, the tensor it feeds, the file name parts beyond the
precision, the GPU and the TensorRT version, and the command that builds an engine by hand.

The rules every engine follows:

- File name: ``<precision>_<parts>_<GPU>-sm<capability>_trt<TensorRT version>.engine``
  (:func:`engine_file_name`), with a JSON build record next to it (``<engine>.json``).
- Folder: keyed by a fingerprint of the weights (:func:`file_fingerprint` for a checkpoint
  file, :func:`tensor_fingerprint` for tensors in memory), so retrained weights never pick up
  a stale engine. A node may prefer a folder of its own (next to its checkpoint, or an explicit
  ``engine_dir``); the fallback, and the default, is :func:`engine_root` in the cuvis-ai model
  cache, which a CuvisNEXT child environment keeps across runs. Engines in the flat folders of
  earlier releases are still found.
- Build: once per engine under a file lock (``<engine>.lock``), so concurrent loads build it
  once and the others load the result. The engine and its record are written under temporary
  names and moved into place, the record first and the engine last, so a half-written engine is
  never visible and a visible engine always has its record. Start and end are logged; a failed
  build raises one error that names the command to build the engine by hand.
- An engine whose build record disagrees with the weights the node holds is refused, never
  loaded (:func:`check_engine_record`).

``tensorrt`` is imported lazily (:func:`import_tensorrt`), so core has no dependency on it: a
plugin lists it in an optional extra, and a pipeline that runs a TensorRT backend lists a plugin
manifest with that extra.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from torch import Tensor

#: Overrides the engine root (``<value>/<namespace>``), as in the plugins' 0.3.0 releases.
ENGINE_DIR_ENV = "CUVIS_AI_TRT_ENGINE_DIR"
#: Hex digits of a fingerprint (an MD5 prefix) that name an engine folder.
FINGERPRINT_CHARS = 16
#: Seconds a load waits for another process that builds the same engine.
LOCK_TIMEOUT_S = 1800.0

_LOGGER: Any = None
#: Full MD5 per (absolute path, size, modification time) of the files fingerprinted so far.
_MD5_CACHE: dict[tuple[str, int, int], str] = {}


# ---------------------------------------------------------------------------
# tensorrt, the GPU, the file name
# ---------------------------------------------------------------------------


def import_tensorrt(consumer: str = "A TensorRT backend") -> Any:
    """Import the ``tensorrt`` package (version 10), or explain how to install it.

    Parameters
    ----------
    consumer : str
        Who needs TensorRT, for the error message (for example
        ``"RFDETRSegmenter backend='tensorrt'"``).

    Returns
    -------
    module
        The ``tensorrt`` module.

    Raises
    ------
    ImportError
        If TensorRT is not installed or older than version 10.
    """
    try:
        import tensorrt
    except ImportError as exc:
        raise ImportError(
            f"{consumer} needs the TensorRT Python package (version 10) matching "
            "torch's CUDA: pip install tensorrt-cu12 (CUDA 12 torch) or tensorrt-cu13 "
            "(CUDA 13 torch), or the plugin's 'tensorrt' extra; building engines also "
            "needs onnx."
        ) from exc
    if int(str(tensorrt.__version__).split(".")[0]) < 10:
        raise ImportError(f"TensorRT >= 10 is required, found {tensorrt.__version__}.")
    return tensorrt


def trt_logger(trt: Any) -> Any:
    """One TensorRT logger per process (TensorRT keeps the first and warns about others).

    Parameters
    ----------
    trt : module
        The ``tensorrt`` module.

    Returns
    -------
    tensorrt.Logger
        A logger at WARNING level, created once per ``tensorrt`` module.
    """
    global _LOGGER
    if _LOGGER is None or _LOGGER[0] is not trt:
        _LOGGER = (trt, trt.Logger(trt.Logger.WARNING))
    return _LOGGER[1]


def gpu_tag(device: torch.device | str | int | None = None) -> str:
    """``<device name>-sm<capability>``, filesystem-safe (for example ``NVIDIA-Thor-sm110``).

    Parameters
    ----------
    device : torch.device or str or int, optional
        The CUDA device; the current one when omitted.

    Returns
    -------
    str
        The GPU part of an engine file name.
    """
    name = torch.cuda.get_device_name(device)
    major, minor = torch.cuda.get_device_capability(device)
    return f"{re.sub(r'[^A-Za-z0-9]+', '-', name).strip('-')}-sm{major}{minor}"


def engine_file_name(
    precision: str,
    *parts: str,
    device: torch.device | str | int | None = None,
    trt: Any = None,
) -> str:
    """``<precision>_<parts>_<gpu tag>_trt<TensorRT version>.engine``.

    Parameters
    ----------
    precision : str
        The plugin's precision name (for example ``"fp16"``).
    *parts : str
        What else the engine is specific to, in the plugin's order (a fingerprint,
        ``"b4"`` for a batch of four, ``"r504"`` for a resolution); empty parts are
        left out.
    device : torch.device or str or int, optional
        The CUDA device the engine runs on.
    trt : module, optional
        The ``tensorrt`` module; imported when omitted.

    Returns
    -------
    str
        The engine file name.
    """
    trt = trt if trt is not None else import_tensorrt()
    middle = "".join(f"_{part}" for part in parts if part)
    return f"{precision}{middle}_{gpu_tag(device)}_trt{trt.__version__}.engine"


# ---------------------------------------------------------------------------
# fingerprints
# ---------------------------------------------------------------------------


def file_md5(path: str | os.PathLike) -> str:
    """MD5 of a file's bytes (an identity check of local weights, not a security measure).

    Parameters
    ----------
    path : str or os.PathLike
        The file.

    Returns
    -------
    str
        32 hex digits.
    """
    digest = hashlib.md5()  # noqa: S324
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def cached_file_md5(path: str | os.PathLike) -> str:
    """:func:`file_md5`, computed once per path, size and modification time.

    A pipeline that is prepared again (after a weights load) does not read its
    checkpoint again.

    Parameters
    ----------
    path : str or os.PathLike
        The file.

    Returns
    -------
    str
        32 hex digits.
    """
    st = os.stat(path)
    key = (os.path.abspath(path), st.st_size, st.st_mtime_ns)
    if key not in _MD5_CACHE:
        _MD5_CACHE[key] = file_md5(path)
    return _MD5_CACHE[key]


def file_fingerprint(path: str | os.PathLike) -> str:
    """The fingerprint of a weights file: the first :data:`FINGERPRINT_CHARS` hex digits
    of its MD5.

    Parameters
    ----------
    path : str or os.PathLike
        The weights file (for example a checkpoint).

    Returns
    -------
    str
        16 hex digits.
    """
    return cached_file_md5(path)[:FINGERPRINT_CHARS]


def tensor_fingerprint(tensors: Mapping[str, Tensor], extra: str = "") -> str:
    """The fingerprint of named tensors: names, dtypes, shapes and values.

    The first :data:`FINGERPRINT_CHARS` hex digits of an MD5 that starts with
    ``extra`` (what else shapes the exported graph, such as architecture options or
    the input size) and then covers every tensor in name order. The same digest as
    the cuvis-ai-steervit and cuvis-ai-efficientad 0.3.0 engine names carry.

    Parameters
    ----------
    tensors : Mapping[str, Tensor]
        For example a module's ``state_dict()``.
    extra : str
        Further text that changes the engine.

    Returns
    -------
    str
        16 hex digits.
    """
    digest = hashlib.md5(extra.encode("utf-8"))  # noqa: S324
    for name, tensor in sorted(tensors.items()):
        t = tensor.detach().to("cpu").contiguous()
        digest.update(f"{name}|{t.dtype}|{tuple(t.shape)}|".encode())
        digest.update(
            t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b""
        )
    return digest.hexdigest()[:FINGERPRINT_CHARS]


# ---------------------------------------------------------------------------
# where an engine lives
# ---------------------------------------------------------------------------


def engine_root(namespace: str) -> Path:
    """The engine folder of one plugin when the node prefers none of its own.

    ``$CUVIS_AI_TRT_ENGINE_DIR/<namespace>`` when that variable is set, else
    ``<model cache>/trt_engines/<namespace>`` in the cuvis-ai model cache
    (``$CUVIS_MODEL_CACHE_DIR``, else ``~/.cuvis_runs/model_cache``). The model cache
    outlives a CuvisNEXT run, unlike the home folder of its child environment.

    Parameters
    ----------
    namespace : str
        The plugin's short name (for example ``"steervit"``).

    Returns
    -------
    Path
        The folder; it may not exist yet.
    """
    override = os.environ.get(ENGINE_DIR_ENV)
    if override:
        return Path(override) / namespace
    from cuvis_ai_core.orchestrator.model_cache import model_cache_dir

    return Path(model_cache_dir()) / "trt_engines" / namespace


def legacy_engine_dirs(namespace: str) -> tuple[Path, ...]:
    """The default engine folder of the plugins' 0.3.0 releases, searched but never built in.

    Parameters
    ----------
    namespace : str
        The plugin's short name.

    Returns
    -------
    tuple of Path
        ``~/.cache/cuvis-ai/tensorrt/<namespace>``.
    """
    return (Path.home() / ".cache" / "cuvis-ai" / "tensorrt" / namespace,)


@dataclass(frozen=True)
class EngineLocation:
    """Where one engine is looked up, and where a missing one is built.

    Attributes
    ----------
    candidates : tuple of Path
        Engine paths in lookup order.
    build_dirs : tuple of Path
        Folders a missing engine is built in, the first writable one wins. Empty when
        there is no fingerprint to key a folder by: such an engine is found, never
        built.
    """

    candidates: tuple[Path, ...]
    build_dirs: tuple[Path, ...]


def engine_location(
    file_name: str,
    namespace: str,
    fingerprint: str | None,
    *,
    preferred_dir: str | os.PathLike | None = None,
    legacy_dirs: Iterable[str | os.PathLike] = (),
) -> EngineLocation:
    """The :class:`EngineLocation` of an engine.

    Lookup order: ``<preferred_dir>/<fingerprint>/``, then ``<preferred_dir>/`` (the
    flat layout of earlier releases), then ``<engine root>/<fingerprint>/``, then
    ``<engine root>/`` and each of ``legacy_dirs`` (flat). A build goes to
    ``<preferred_dir>/<fingerprint>/``, or to ``<engine root>/<fingerprint>/`` when that
    is not writable or there is no preferred folder.

    Parameters
    ----------
    file_name : str
        From :func:`engine_file_name`.
    namespace : str
        The plugin's short name (see :func:`engine_root`).
    fingerprint : str or None
        Of the weights the engine is built from; None when there is none (nothing is
        built then).
    preferred_dir : str or os.PathLike, optional
        The node's own engine folder (next to its checkpoint, or an explicit
        ``engine_dir``).
    legacy_dirs : iterable of str or os.PathLike
        Further flat folders to search, such as :func:`legacy_engine_dirs`.

    Returns
    -------
    EngineLocation
        The candidates and the build folders.
    """
    root = engine_root(namespace)
    candidates: list[Path] = []
    build_dirs: list[Path] = []
    if preferred_dir is not None:
        preferred = Path(preferred_dir)
        if fingerprint:
            candidates.append(preferred / fingerprint / file_name)
            build_dirs.append(preferred / fingerprint)
        candidates.append(preferred / file_name)
    if fingerprint:
        candidates.append(root / fingerprint / file_name)
        build_dirs.append(root / fingerprint)
    candidates.append(root / file_name)
    candidates.extend(Path(folder) / file_name for folder in legacy_dirs)
    return EngineLocation(
        tuple(dict.fromkeys(candidates)), tuple(dict.fromkeys(build_dirs))
    )


def check_engine_record(
    engine_path: str | os.PathLike, expected: Mapping[str, Any]
) -> None:
    """Refuse an engine whose build record disagrees with ``expected``.

    Only keys present in both the record and ``expected`` (with a value other than
    None) are compared, so an engine without a record, or a record from a release
    that did not store a key, is accepted.

    Parameters
    ----------
    engine_path : str or os.PathLike
        The engine; its record is ``<engine>.json``.
    expected : Mapping[str, Any]
        What the record must say, for example ``{"checkpoint_md5": ...}`` or
        ``{"fingerprint": ...}``.

    Raises
    ------
    RuntimeError
        Naming each disagreeing key.
    """
    record_path = Path(f"{engine_path}.json")
    if not expected or not record_path.is_file():
        return
    record = json.loads(record_path.read_text(encoding="utf-8"))
    wrong = [
        f"{key}: the engine says {record[key]!r}, the node holds {value!r}"
        for key, value in expected.items()
        if value is not None and record.get(key) is not None and record[key] != value
    ]
    if wrong:
        raise RuntimeError(
            f"TensorRT engine {engine_path} was built from other weights than this node "
            f"holds ({'; '.join(wrong)}); rebuild it (the plugin's build command with "
            "--force) or delete it."
        )


def find_engine(
    location: EngineLocation, expected: Mapping[str, Any] | None = None
) -> Path | None:
    """The first existing engine of ``location``, checked against ``expected``.

    Parameters
    ----------
    location : EngineLocation
        From :func:`engine_location`.
    expected : Mapping[str, Any], optional
        Passed to :func:`check_engine_record`: an engine whose record disagrees is
        refused (an error), not skipped.

    Returns
    -------
    Path or None
        The engine, or None when no candidate exists.
    """
    for path in location.candidates:
        if path.is_file():
            check_engine_record(path, expected or {})
            return path
    return None


# ---------------------------------------------------------------------------
# building
# ---------------------------------------------------------------------------


def build_serialized_network(
    onnx_path: str | os.PathLike,
    *,
    fp16: bool = False,
    tf32: bool = True,
    trt: Any = None,
) -> tuple[Any, dict[str, Any]]:
    """Compile an ONNX file with the installed TensorRT for the current GPU.

    Parameters
    ----------
    onnx_path : str or os.PathLike
        The exported network (static shapes).
    fp16 : bool
        Set TensorRT's FP16 builder flag (mixed precision; TensorRT 10 only, TensorRT
        11 dropped the flag).
    tf32 : bool
        Leave TF32 tensor-core math allowed (TensorRT's default float build); False
        clears the flag for IEEE float32.
    trt : module, optional
        The ``tensorrt`` module; imported when omitted.

    Returns
    -------
    tuple
        The serialized engine (``tensorrt.IHostMemory``, written with ``file.write``)
        and ``{"fp16": ..., "tf32": ...}`` for the build record.

    Raises
    ------
    RuntimeError
        If the network does not parse, the FP16 flag is missing or the build fails.
    """
    trt = trt if trt is not None else import_tensorrt()
    if fp16 and not hasattr(trt.BuilderFlag, "FP16"):
        raise RuntimeError(
            f"TensorRT {trt.__version__} has no FP16 builder flag (dropped in TensorRT 11): "
            "build fp16 engines with TensorRT 10."
        )
    log = trt_logger(trt)
    builder = trt.Builder(log)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, log)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT could not parse {Path(onnx_path).name}: {errors}")
    config = builder.create_builder_config()
    has_tf32 = hasattr(trt.BuilderFlag, "TF32")
    if has_tf32 and not tf32:
        config.clear_flag(trt.BuilderFlag.TF32)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise RuntimeError("TensorRT engine build failed.")
    info = {
        "fp16": bool(fp16),
        "tf32": config.get_flag(trt.BuilderFlag.TF32) if has_tf32 else None,
    }
    return blob, info


def write_engine_and_record(
    engine_path: str | os.PathLike, blob: Any, record: Mapping[str, Any]
) -> None:
    """Write an engine and its build record so that neither is ever seen half-written.

    Both files are written under temporary names in the target folder; then the
    record and, last, the engine are moved into place. A visible engine always has
    its record.

    Parameters
    ----------
    engine_path : str or os.PathLike
        The engine; the record goes to ``<engine>.json``.
    blob : bytes-like
        The serialized engine.
    record : Mapping[str, Any]
        The build record (JSON-serializable).
    """
    engine_path = Path(engine_path)
    folder = engine_path.resolve().parent
    folder.mkdir(parents=True, exist_ok=True)
    contents = (
        (".json", json.dumps(dict(record), indent=1).encode("utf-8")),
        ("", blob),
    )
    moves: list[tuple[str, Path]] = []
    try:
        for suffix, data in contents:
            fd, tmp = tempfile.mkstemp(
                dir=folder, prefix=f"{engine_path.name}{suffix}.", suffix=".tmp"
            )
            moves.append((tmp, Path(f"{engine_path}{suffix}")))
            with os.fdopen(fd, "wb") as f:
                f.write(data)
        for tmp, final in moves:  # the record first, the engine last
            os.replace(tmp, final)
    finally:
        for tmp, _ in moves:
            if os.path.exists(tmp):
                os.remove(tmp)


def _not_writable(folder: Path) -> str | None:
    """None when ``folder`` exists (or can be created) and takes a new file, else why not."""
    try:
        folder.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=folder):
            pass
    except OSError as exc:
        return f"{folder}: {exc}"
    return None


def _first_line(exc: BaseException) -> str:
    """The first line of an exception's message (empty when it has none)."""
    lines = str(exc).strip().splitlines()
    return lines[0] if lines else ""


def build_engine_once(
    location: EngineLocation,
    file_name: str,
    build: Callable[[], tuple[Any, Mapping[str, Any]]],
    *,
    command: str,
    expected: Mapping[str, Any] | None = None,
    force: bool = False,
    what: str = "TensorRT engine",
) -> Path:
    """Build a missing engine where :func:`find_engine` looks first, once per machine.

    The build runs in the first writable folder of ``location.build_dirs`` under a
    file lock per engine: a second process that needs the same engine waits for the
    first and then loads its result. ``build`` returns the serialized engine and the
    plugin's part of the build record; this function adds the engine name, the
    TensorRT, torch and GPU fields and the build time, and writes both with
    :func:`write_engine_and_record`. Start and end are logged.

    Parameters
    ----------
    location : EngineLocation
        From :func:`engine_location`.
    file_name : str
        From :func:`engine_file_name`.
    build : Callable
        Exports and compiles the network; called at most once, under the lock.
    command : str
        The command that builds this engine by hand, named in every error.
    expected : Mapping[str, Any], optional
        What an engine found under the lock must record (see
        :func:`check_engine_record`).
    force : bool
        Build even when an engine exists (the plugin's ``--force``).
    what : str
        How the logs and errors call the engine.

    Returns
    -------
    Path
        The engine.

    Raises
    ------
    RuntimeError
        If the build fails, no build folder is writable or another process holds
        the lock for longer than :data:`LOCK_TIMEOUT_S`; each names ``command``.
    """
    from filelock import FileLock, Timeout

    trt = import_tensorrt()
    reasons: list[str] = []
    for folder in location.build_dirs:
        reason = _not_writable(folder)
        if reason is not None:
            reasons.append(reason)
            continue
        path = folder / file_name
        try:
            with FileLock(f"{path}.lock", timeout=LOCK_TIMEOUT_S):
                if not force:  # built by another process while this one waited
                    found = find_engine(location, expected)
                    if found is not None:
                        return found
                logger.info(
                    f"Building {what} {path}; this takes a few minutes, once per "
                    "machine and weights."
                )
                t0 = time.perf_counter()
                try:
                    blob, plugin_record = build()
                except Exception as exc:
                    raise RuntimeError(
                        f"could not build {what} {path} ({type(exc).__name__}: "
                        f"{_first_line(exc)}). Build it by hand with: {command}"
                    ) from exc
                seconds = round(time.perf_counter() - t0, 1)
                record = {
                    "engine": file_name,
                    **dict(plugin_record),
                    "tensorrt": trt.__version__,
                    "torch": torch.__version__,
                    "gpu": torch.cuda.get_device_name(),
                    "gpu_tag": gpu_tag(),
                    "build_seconds": seconds,
                    "built_at": datetime.datetime.now(datetime.UTC).isoformat(
                        timespec="seconds"
                    ),
                }
                write_engine_and_record(path, blob, record)
                logger.info(f"Built {what} {path} in {seconds:.0f} s.")
                return path
        except Timeout as exc:
            raise RuntimeError(
                f"waited {LOCK_TIMEOUT_S:.0f} s for another process that builds {path} "
                f"(lock file {path}.lock). Build it by hand with: {command}"
            ) from exc
    detail = "; ".join(reasons) if reasons else "no fingerprint to key a folder by"
    raise RuntimeError(
        f"no folder to build {what} {file_name} in ({detail}). Build it by hand with: "
        f"{command}"
    )


# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------


class TensorRTEngine:
    """Run a serialized TensorRT engine (static shapes, one input) on torch CUDA tensors.

    The engine runs on its own CUDA stream (TensorRT adds a host synchronisation to
    every call on the default stream), ordered after the work already queued on
    torch's current stream, and torch's current stream waits for it, so torch ops
    that follow see the results in order. The output tensors are reused buffers,
    overwritten by the next call: consume or copy them before calling again.

    Parameters
    ----------
    path : str or os.PathLike
        The engine file.
    device : torch.device or str
        The CUDA device it runs on.
    rebuild_hint : str
        Appended to the error when the engine does not load on this machine (for
        example the plugin's build command).
    """

    def __init__(
        self,
        path: str | os.PathLike,
        device: torch.device | str = "cuda",
        rebuild_hint: str = "",
    ) -> None:
        trt = import_tensorrt()
        self.path = str(path)
        self.device = torch.device(device)
        dtypes = {
            getattr(trt, name): dtype
            for name, dtype in (
                ("float32", torch.float32),
                ("float16", torch.float16),
                ("bfloat16", torch.bfloat16),
                ("int32", torch.int32),
                ("int64", torch.int64),
                ("int8", torch.int8),
                ("uint8", torch.uint8),
                ("bool", torch.bool),
            )
            if hasattr(trt, name)
        }
        self._runtime = trt.Runtime(trt_logger(trt))
        with open(self.path, "rb") as f, torch.cuda.device(self.device):
            self._engine = self._runtime.deserialize_cuda_engine(f.read())
            self._stream = torch.cuda.Stream(self.device)
        if self._engine is None:
            raise RuntimeError(
                f"TensorRT could not load {self.path}: an engine only runs on the GPU and "
                f"TensorRT version it was built with; rebuild it on this machine. "
                f"{rebuild_hint}".strip()
            )
        self._context = self._engine.create_execution_context()
        inputs: dict[str, tuple[torch.dtype, tuple[int, ...]]] = {}
        self.outputs: dict[str, Tensor] = {}
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            dtype = dtypes[self._engine.get_tensor_dtype(name)]
            shape = tuple(self._engine.get_tensor_shape(name))
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                inputs[name] = (dtype, shape)
            else:
                buf = torch.empty(shape, dtype=dtype, device=self.device)
                self.outputs[name] = buf
                self._context.set_tensor_address(name, buf.data_ptr())
        if len(inputs) != 1:
            raise RuntimeError(
                f"{self.path}: expected one engine input, found {sorted(inputs)}."
            )
        self.input_name, (self.input_dtype, self.input_shape) = next(
            iter(inputs.items())
        )
        self._held: Tensor | None = None

    def __call__(self, x: Tensor) -> dict[str, Tensor]:
        """Run the engine on ``x`` (cast to the engine's input dtype on its device).

        Parameters
        ----------
        x : Tensor
            The input batch, of the engine's static shape.

        Returns
        -------
        dict of str to Tensor
            The output buffers by output name (reused by the next call).
        """
        x = x.to(device=self.device, dtype=self.input_dtype).contiguous()
        self._context.set_tensor_address(self.input_name, x.data_ptr())
        current = torch.cuda.current_stream(self.device)
        self._stream.wait_stream(current)  # x written, previous outputs consumed
        with torch.cuda.device(self.device):
            if not self._context.execute_async_v3(self._stream.cuda_stream):
                raise RuntimeError(f"TensorRT execution failed for {self.path}.")
        current.wait_stream(
            self._stream
        )  # everything queued after this sees the outputs
        self._held = (
            x  # the engine reads x asynchronously; keep it alive until the next call
        )
        return self.outputs
