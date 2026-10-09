"""``utils.trt_engine``: where a TensorRT engine lives, how it is built once, how it runs.

Mocked: a fake ``tensorrt`` module and a fake build, so the lookup, the folder choice,
the lock, the atomic write and the errors run as in production without TensorRT or a
GPU. The ``slow`` test builds and runs a real engine where TensorRT and CUDA exist.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

from cuvis_ai_core.utils import trt_engine as te

pytestmark = pytest.mark.unit


def _fake_tensorrt(
    monkeypatch, version="10.15.1.29", fp16_flag=True, parse_ok=True, blob=b"plan"
):
    """Minimal ``tensorrt`` module: records builder flags; parse and build are configurable."""
    trt = types.ModuleType("tensorrt")
    trt.__version__ = version
    flags = ["TF32"] + (["FP16"] if fp16_flag else [])
    trt.BuilderFlag = SimpleNamespace(**{f: f for f in flags})
    trt.Logger = type(
        "Logger", (), {"WARNING": 2, "__init__": lambda self, level: None}
    )
    trt.events = []

    class Config:
        def __init__(self):
            self.flags = {"TF32"}

        def set_flag(self, flag):
            trt.events.append(("set", flag))
            self.flags.add(flag)

        def clear_flag(self, flag):
            trt.events.append(("clear", flag))
            self.flags.discard(flag)

        def get_flag(self, flag):
            return flag in self.flags

    class Builder:
        def __init__(self, logger):
            pass

        def create_network(self, flags):
            return "network"

        def create_builder_config(self):
            return Config()

        def build_serialized_network(self, network, config):
            return blob

    class Parser:
        num_errors = 1

        def __init__(self, network, logger):
            pass

        def parse_from_file(self, path):
            trt.parsed = path
            return parse_ok

        def get_error(self, i):
            return "bad node"

    trt.Builder, trt.OnnxParser = Builder, Parser
    monkeypatch.setitem(sys.modules, "tensorrt", trt)
    monkeypatch.setattr(te, "_LOGGER", None)
    monkeypatch.setattr(
        torch.cuda, "get_device_name", lambda device=None: "NVIDIA Thor (x)"
    )
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: (11, 0)
    )
    return trt


@pytest.fixture
def env(monkeypatch, tmp_path):
    """A fake TensorRT, the model cache and the home folder in tmp_path, a log capture."""
    trt = _fake_tensorrt(monkeypatch)
    monkeypatch.delenv(te.ENGINE_DIR_ENV, raising=False)
    monkeypatch.setenv("CUVIS_MODEL_CACHE_DIR", str(tmp_path / "model_cache"))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(m.record["message"]), level="INFO")
    yield SimpleNamespace(trt=trt, root=tmp_path, messages=messages)
    logger.remove(sink)


def _files(folder: Path) -> list[str]:
    """The files in ``folder`` apart from lock files (filelock removes them on Windows only)."""
    return sorted(p.name for p in folder.iterdir() if not p.name.endswith(".lock"))


class Builds:
    """A ``build`` callable: counts calls, optionally sleeps or fails, returns a plan."""

    def __init__(self, seconds: float = 0.0, error: Exception | None = None):
        self.calls = 0
        self.seconds, self.error = seconds, error
        self.lock = threading.Lock()

    def __call__(self):
        with self.lock:
            self.calls += 1
        time.sleep(self.seconds)
        if self.error is not None:
            raise self.error
        return b"plan", {"precision": "fp16", "fingerprint": "f" * 16}


# ---------------------------------------------------------------------------
# tensorrt, names, fingerprints
# ---------------------------------------------------------------------------


def test_import_tensorrt_errors(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "tensorrt", None)  # import tensorrt -> ImportError
    with pytest.raises(ImportError, match="Seg backend.*tensorrt-cu12"):
        te.import_tensorrt("Seg backend")
    _fake_tensorrt(monkeypatch, version="8.6.1")
    with pytest.raises(ImportError, match=">= 10"):
        te.import_tensorrt()


def test_gpu_tag_and_engine_file_name(env) -> None:
    assert te.gpu_tag() == "NVIDIA-Thor-x-sm110"
    assert (
        te.engine_file_name("fp16", "r504")
        == "fp16_r504_NVIDIA-Thor-x-sm110_trt10.15.1.29.engine"
    )
    assert (
        te.engine_file_name("tf32", "0123456789abcdef", "b4", "", "r336")
        == "tf32_0123456789abcdef_b4_r336_NVIDIA-Thor-x-sm110_trt10.15.1.29.engine"
    )
    assert (
        te.engine_file_name("fp32") == "fp32_NVIDIA-Thor-x-sm110_trt10.15.1.29.engine"
    )


def _plugin_fingerprint(tensors: dict, extra: str = "") -> str:
    """The digest cuvis-ai-steervit's ``fingerprint`` and cuvis-ai-efficientad's
    ``weights_fingerprint`` compute in their 0.3.0 releases."""
    digest = hashlib.md5(extra.encode("utf-8"))  # noqa: S324
    for name, tensor in sorted(tensors.items()):
        t = tensor.detach().to("cpu").contiguous()
        digest.update(f"{name}|{t.dtype}|{tuple(t.shape)}|".encode())
        digest.update(
            t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b""
        )
    return digest.hexdigest()[:16]


def test_tensor_fingerprint_matches_the_plugins_and_tracks_every_input() -> None:
    torch.manual_seed(0)
    state = torch.nn.Sequential(
        torch.nn.Linear(4, 3), torch.nn.LayerNorm(3)
    ).state_dict()
    state["count"] = torch.tensor(5)
    state["empty"] = torch.zeros(0)
    fp = te.tensor_fingerprint(state, extra="arch=small|256x256")
    assert (
        fp == _plugin_fingerprint(state, extra="arch=small|256x256") and len(fp) == 16
    )
    assert (
        te.tensor_fingerprint(dict(reversed(list(state.items()))), "arch=small|256x256")
        == fp
    )
    assert te.tensor_fingerprint(state) != fp  # extra counts
    changed = dict(state)
    changed["0.bias"] = state["0.bias"] + 1e-3
    assert te.tensor_fingerprint(changed, "arch=small|256x256") != fp
    as_half = {k: v.half() if v.is_floating_point() else v for k, v in state.items()}
    assert te.tensor_fingerprint(as_half, "arch=small|256x256") != fp  # dtype counts


def test_file_fingerprint_is_the_md5_prefix_read_once_per_file_version(
    monkeypatch, tmp_path
) -> None:
    path = tmp_path / "w.pth"
    path.write_bytes(b"weights v1")
    full = te.file_md5(path)
    reads = []
    real = te.file_md5
    monkeypatch.setattr(te, "file_md5", lambda p: reads.append(p) or real(p))
    assert te.file_fingerprint(path) == full[:16]
    assert te.cached_file_md5(path) == full and len(reads) == 1
    path.write_bytes(b"retrained weights")  # a new size and modification time
    assert te.file_fingerprint(path) != full[:16] and len(reads) == 2


# ---------------------------------------------------------------------------
# where an engine lives
# ---------------------------------------------------------------------------


def test_engine_root_override_and_model_cache(env, monkeypatch) -> None:
    assert (
        te.engine_root("steervit")
        == env.root / "model_cache" / "trt_engines" / "steervit"
    )
    monkeypatch.setenv(te.ENGINE_DIR_ENV, str(env.root / "engines"))
    assert te.engine_root("steervit") == env.root / "engines" / "steervit"
    assert te.legacy_engine_dirs("steervit") == (
        env.root / "home" / ".cache" / "cuvis-ai" / "tensorrt" / "steervit",
    )


def test_engine_location_order(env) -> None:
    root = te.engine_root("rfdetr")
    pref = env.root / "w.pth.trt"
    loc = te.engine_location(
        "e.engine", "rfdetr", "abc", preferred_dir=pref, legacy_dirs=["/old"]
    )
    assert loc.candidates == (
        pref / "abc" / "e.engine",
        pref / "e.engine",
        root / "abc" / "e.engine",
        root / "e.engine",
        Path("/old") / "e.engine",
    )
    assert loc.build_dirs == (pref / "abc", root / "abc")
    plain = te.engine_location("e.engine", "steervit", "abc")
    assert plain.candidates == (
        root.parent / "steervit" / "abc" / "e.engine",
        root.parent / "steervit" / "e.engine",
    )
    assert plain.build_dirs == (root.parent / "steervit" / "abc",)
    nothing = te.engine_location("e.engine", "rfdetr", None, preferred_dir=pref)
    assert (
        nothing.candidates == (pref / "e.engine", root / "e.engine")
        and nothing.build_dirs == ()
    )


def test_check_engine_record(tmp_path) -> None:
    engine = tmp_path / "e.engine"
    te.check_engine_record(engine, {"checkpoint_md5": "a" * 32})  # no record: accepted
    (tmp_path / "e.engine.json").write_text(
        json.dumps({"checkpoint_md5": "b" * 32, "fingerprint": None})
    )
    te.check_engine_record(engine, {})  # nothing expected
    te.check_engine_record(engine, {"fingerprint": "f" * 16})  # not recorded: accepted
    te.check_engine_record(engine, {"checkpoint_md5": None})  # nothing to compare
    with pytest.raises(
        RuntimeError,
        match=r"checkpoint_md5: the engine says 'bbbb.*--force\) or delete it",
    ):
        te.check_engine_record(engine, {"checkpoint_md5": "a" * 32})
    te.check_engine_record(engine, {"checkpoint_md5": "b" * 32})


def test_find_engine_takes_the_first_and_refuses_a_mismatch(env) -> None:
    pref = env.root / "pref"
    loc = te.engine_location("e.engine", "rfdetr", "abc", preferred_dir=pref)
    assert te.find_engine(loc) is None
    flat = pref / "e.engine"
    flat.parent.mkdir(parents=True)
    flat.write_bytes(b"old")
    (pref / "e.engine.json").write_text(json.dumps({"checkpoint_md5": "0" * 32}))
    with pytest.raises(RuntimeError, match="other weights"):
        te.find_engine(loc, {"checkpoint_md5": "1" * 32})  # refused, not skipped
    keyed = pref / "abc" / "e.engine"
    keyed.parent.mkdir()
    keyed.write_bytes(b"new")
    assert (
        te.find_engine(loc, {"checkpoint_md5": "1" * 32}) == keyed
    )  # first candidate wins


# ---------------------------------------------------------------------------
# building
# ---------------------------------------------------------------------------


def test_build_engine_once_writes_engine_and_record_and_logs(env) -> None:
    loc = te.engine_location(
        "e.engine", "rfdetr", "abc", preferred_dir=env.root / "pref"
    )
    builds = Builds()
    path = te.build_engine_once(
        loc, "e.engine", builds, command="build-it --checkpoint w.pth"
    )
    assert (
        path == env.root / "pref" / "abc" / "e.engine" and path.read_bytes() == b"plan"
    )
    record = json.loads(Path(f"{path}.json").read_text())
    assert record["engine"] == "e.engine" and record["precision"] == "fp16"
    assert record["fingerprint"] == "f" * 16 and record["tensorrt"] == "10.15.1.29"
    assert record["gpu_tag"] == "NVIDIA-Thor-x-sm110" and record["build_seconds"] >= 0
    assert _files(path.parent) == [
        "e.engine",
        "e.engine.json",
    ]  # no temporary file left
    assert any(m.startswith(f"Building TensorRT engine {path}") for m in env.messages)
    assert any(m.startswith(f"Built TensorRT engine {path}") for m in env.messages)
    assert te.build_engine_once(loc, "e.engine", builds, command="x") == path
    assert builds.calls == 1  # found under the lock, not built again
    assert (
        te.build_engine_once(loc, "e.engine", builds, command="x", force=True) == path
    )
    assert builds.calls == 2


def test_unwritable_preferred_folder_falls_back_to_the_engine_root(
    env, monkeypatch
) -> None:
    pref = env.root / "read_only"
    loc = te.engine_location("e.engine", "rfdetr", "abc", preferred_dir=pref)
    real = te._not_writable
    monkeypatch.setattr(
        te,
        "_not_writable",
        lambda f: f"{f}: read-only" if f == pref / "abc" else real(f),
    )
    path = te.build_engine_once(loc, "e.engine", Builds(), command="x")
    assert path == te.engine_root("rfdetr") / "abc" / "e.engine"
    assert not (pref / "abc").exists()
    monkeypatch.setattr(te, "_not_writable", real)
    assert te.find_engine(loc) == path  # the next load finds it there


def test_build_errors_name_the_command(env, monkeypatch) -> None:
    loc = te.engine_location(
        "e.engine", "rfdetr", "abc", preferred_dir=env.root / "pref"
    )
    failing = Builds(error=RuntimeError("TensorRT engine build failed.\nmore lines"))
    with pytest.raises(RuntimeError) as err:
        te.build_engine_once(
            loc, "e.engine", failing, command="build-it --checkpoint w.pth"
        )
    msg = str(err.value)
    assert (
        "could not build TensorRT engine" in msg
        and "RuntimeError: TensorRT engine build failed.)" in msg
    )
    assert msg.endswith("Build it by hand with: build-it --checkpoint w.pth")
    assert isinstance(err.value.__cause__, RuntimeError)
    assert _files(env.root / "pref" / "abc") == []  # nothing half-written
    monkeypatch.setattr(te, "_not_writable", lambda f: f"{f}: read-only")
    with pytest.raises(
        RuntimeError, match="no folder to build.*read-only.*Build it by hand with: x$"
    ):
        te.build_engine_once(loc, "e.engine", Builds(), command="x")
    nothing = te.engine_location("e.engine", "rfdetr", None)
    with pytest.raises(RuntimeError, match="no fingerprint"):
        te.build_engine_once(nothing, "e.engine", Builds(), command="x")


def test_concurrent_builds_happen_once(env) -> None:
    loc = te.engine_location("e.engine", "steervit", "abc")
    builds = Builds(seconds=0.5)
    paths: list[Path] = []
    errors: list[BaseException] = []

    def load() -> None:
        try:
            paths.append(te.build_engine_once(loc, "e.engine", builds, command="x"))
        except BaseException as exc:  # noqa: BLE001 (reported below)
            errors.append(exc)

    threads = [threading.Thread(target=load) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert errors == [] and builds.calls == 1
    assert paths == [te.engine_root("steervit") / "abc" / "e.engine"] * 3


def test_write_engine_and_record_is_one_unit(tmp_path) -> None:
    path = tmp_path / "e.engine"
    te.write_engine_and_record(path, b"plan", {"k": 1})
    assert path.read_bytes() == b"plan" and json.loads(
        Path(f"{path}.json").read_text()
    ) == {"k": 1}
    with pytest.raises(
        TypeError
    ):  # the engine bytes cannot be written: nothing appears
        te.write_engine_and_record(tmp_path / "broken.engine", object(), {"k": 2})
    assert sorted(p.name for p in tmp_path.iterdir()) == ["e.engine", "e.engine.json"]


@pytest.mark.parametrize(
    ("fp16", "tf32", "events", "info"),
    [
        (False, True, [], {"fp16": False, "tf32": True}),
        (False, False, [("clear", "TF32")], {"fp16": False, "tf32": False}),
        (True, True, [("set", "FP16")], {"fp16": True, "tf32": True}),
    ],
)
def test_build_serialized_network_flags(
    env, tmp_path, fp16, tf32, events, info
) -> None:
    blob, got = te.build_serialized_network(tmp_path / "net.onnx", fp16=fp16, tf32=tf32)
    assert blob == b"plan" and env.trt.events == events and got == info
    assert env.trt.parsed.endswith("net.onnx")


def test_build_serialized_network_failures(monkeypatch, tmp_path) -> None:
    _fake_tensorrt(monkeypatch, fp16_flag=False)
    with pytest.raises(RuntimeError, match="FP16 builder flag"):
        te.build_serialized_network(tmp_path / "n.onnx", fp16=True)
    _fake_tensorrt(monkeypatch, parse_ok=False)
    with pytest.raises(RuntimeError, match="could not parse n.onnx: bad node"):
        te.build_serialized_network(tmp_path / "n.onnx")
    _fake_tensorrt(monkeypatch, blob=None)
    with pytest.raises(RuntimeError, match="build failed"):
        te.build_serialized_network(tmp_path / "n.onnx")


# ---------------------------------------------------------------------------
# real TensorRT
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize(("fp16", "atol"), [(False, 1e-3), (True, 2e-2)])
def test_real_engine_built_once_and_run(monkeypatch, tmp_path, fp16, atol) -> None:
    pytest.importorskip("tensorrt")
    pytest.importorskip("onnx")
    monkeypatch.setenv("CUVIS_MODEL_CACHE_DIR", str(tmp_path / "model_cache"))

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 4, 3, padding=1)

        def forward(self, x):
            y = self.conv(x)
            return y.mean((2, 3)), torch.sigmoid(y)

    torch.manual_seed(0)
    net = Net().eval()
    fp = te.tensor_fingerprint(net.state_dict(), extra="16x16")
    name = te.engine_file_name("fp16" if fp16 else "fp32", fp, "16x16")
    loc = te.engine_location(name, "test", fp)

    def build():
        onnx_path = tmp_path / "net.onnx"
        torch.onnx.export(
            net,
            (torch.rand(1, 3, 16, 16),),
            str(onnx_path),
            input_names=["input"],
            output_names=["mean", "map"],
            opset_version=17,
            dynamo=False,
        )
        blob, info = te.build_serialized_network(onnx_path, fp16=fp16)
        return blob, {"fingerprint": fp, **info}

    path = te.build_engine_once(loc, name, build, command="pytest")
    assert te.find_engine(loc, {"fingerprint": fp}) == path
    engine = te.TensorRTEngine(path, "cuda")
    assert tuple(engine.input_shape) == (1, 3, 16, 16)
    x = torch.rand(1, 3, 16, 16, device="cuda")
    out = engine(x)
    with torch.no_grad():
        ref = net.cuda()(x)
    for key, r in zip(("mean", "map"), ref, strict=True):
        torch.testing.assert_close(out[key].float(), r, atol=atol, rtol=0)
    assert os.path.getsize(path) > 0
