"""Import a CUDA device buffer exported by the producer over CUDA IPC.

Sibling implementation: ``cuvis_ipc.py`` in the cuvis SDK repo does the same job for
consumers that receive a raw 184-byte descriptor. cuvis-ai-core cannot depend on the
cuvis Python wrapper, so the logic is duplicated here. The two must stay in step; the
proto ``CudaIpcRef`` message is the contract, not the C struct either of them mirrors.

Scope is the two backends the producer emits. Pool is unsupported on Windows and is
excluded by the producer's backend policy, so it raises rather than being half-handled.

The exporting process must outlive this importer: CUDA IPC has no cross-process
refcount. The producer holds its export open for the duration of the RPC, which is why
the mapping must be released before the response is built.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Generator

if TYPE_CHECKING:
    import torch

    from cuvis_ai_schemas.grpc.v1 import cuvis_ai_pb2

BACKEND_NONE, BACKEND_POOL, BACKEND_LEGACY, BACKEND_VMM = 0, 1, 2, 3
HANDLE_NONE, HANDLE_WIN32, HANDLE_WIN32_KMT, HANDLE_POSIX_FD = 0, 1, 2, 3

_LEGACY_BLOB_LEN = 64  # sizeof(cudaIpcMemHandle_t)

# Typestrings for __cuda_array_interface__, keyed by torch dtype name.
_TYPESTR = {
    "uint8": "|u1",
    "uint16": "<u2",
    "int32": "<i4",
    "int64": "<i8",
    "float16": "<f2",
    "float32": "<f4",
    "float64": "<f8",
}


def _check(ret, what: str):
    """cuda-python returns (error, *values); raise on a non-success error code."""
    err, *values = ret if isinstance(ret, tuple) else (ret,)
    if int(err) != 0:
        raise RuntimeError(f"{what} failed: {err}")
    return values


class _CudaArray:
    """Minimal __cuda_array_interface__ holder, enough for torch.as_tensor."""

    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
        }


def _open_legacy(ref: "cuvis_ai_pb2.CudaIpcRef"):
    """Legacy cudaIpc: the blob is a self-contained cudaIpcMemHandle_t."""
    from cuda.bindings import runtime

    handle = runtime.cudaIpcMemHandle_t()
    handle.reserved = bytes(ref.handle_blob).ljust(_LEGACY_BLOB_LEN, b"\x00")
    (ptr,) = _check(
        runtime.cudaIpcOpenMemHandle(handle, runtime.cudaIpcMemLazyEnablePeerAccess),
        "cudaIpcOpenMemHandle",
    )

    def close() -> None:
        runtime.cudaIpcCloseMemHandle(ptr)

    return int(ptr), close


def _open_vmm(ref: "cuvis_ai_pb2.CudaIpcRef"):
    """VMM: import the generic handle, then reserve, map and grant access."""
    from cuda.bindings import driver

    driver.cuInit(0)
    if ref.handle_type == HANDLE_WIN32_KMT:
        handle_type = driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_WIN32_KMT
    elif ref.handle_type == HANDLE_POSIX_FD:
        handle_type = (
            driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        )
    else:
        raise NotImplementedError(
            f"vmm handle_type {ref.handle_type} is not importable here"
        )

    handle_value = int.from_bytes(bytes(ref.handle_blob), "little")
    (generic,) = _check(
        driver.cuMemImportFromShareableHandle(handle_value, handle_type),
        "cuMemImportFromShareableHandle",
    )
    # alloc_size, not byte_size: VMM rounds the allocation up to a granularity
    # boundary and the mapping must cover the whole reservation.
    size = ref.alloc_size
    (ptr,) = _check(driver.cuMemAddressReserve(size, 0, 0, 0), "cuMemAddressReserve")
    _check(driver.cuMemMap(ptr, size, 0, generic, 0), "cuMemMap")

    access = driver.CUmemAccessDesc()
    access.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    access.location.id = ref.device_ordinal
    access.flags = driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    _check(driver.cuMemSetAccess(ptr, size, [access], 1), "cuMemSetAccess")

    def close() -> None:
        driver.cuMemUnmap(ptr, size)
        driver.cuMemAddressFree(ptr, size)
        driver.cuMemRelease(generic)

    return int(ptr), close


def validate_ref(
    ref: "cuvis_ai_pb2.CudaIpcRef",
    shape: tuple[int, ...],
    dtype: "torch.dtype",
) -> None:
    """Reject a reference this process cannot or should not import.

    The sizes are untrusted uint64 from the wire and feed straight into
    cuMemAddressReserve, so they get the same scrutiny an untrusted ShmRef gets.
    """
    if ref.backend == BACKEND_POOL:
        raise NotImplementedError(
            "CUDA IPC pool backend is not supported (unsupported on Windows and "
            "excluded by the producer's backend policy)"
        )
    if ref.backend not in (BACKEND_LEGACY, BACKEND_VMM):
        raise ValueError(f"CUDA IPC backend {ref.backend} is not importable")
    if ref.handle_type == HANDLE_WIN32:
        raise ValueError(
            "CUDA IPC handle_type win32 is an NT handle that would need "
            "DuplicateHandle into this process; the producer must emit win32_kmt"
        )
    if not ref.handle_blob:
        raise ValueError("CUDA IPC reference carries an empty handle blob")

    itemsize = _itemsize(dtype)
    if ref.byte_size % itemsize != 0:
        raise ValueError(
            f"CUDA IPC byte_size {ref.byte_size} is not divisible by dtype {dtype}"
        )
    if shape:
        expected = math.prod(shape) * itemsize
        if ref.byte_size != expected:
            raise ValueError(
                f"CUDA IPC byte_size {ref.byte_size} does not match shape {shape} "
                f"x dtype {dtype} ({expected} bytes)"
            )
    if ref.backend == BACKEND_VMM and ref.alloc_size < ref.byte_size + ref.byte_offset:
        raise ValueError(
            f"CUDA IPC alloc_size {ref.alloc_size} cannot hold byte_size "
            f"{ref.byte_size} at offset {ref.byte_offset}"
        )


def _itemsize(dtype: "torch.dtype") -> int:
    import torch

    return torch.empty(0, dtype=dtype).element_size()


@contextmanager
def open_ref(
    ref: "cuvis_ai_pb2.CudaIpcRef",
    shape: tuple[int, ...],
    dtype: "torch.dtype",
) -> Generator["torch.Tensor", None, Any]:
    """Map an exported device buffer and yield it as a zero-copy CUDA tensor.

    The tensor is valid only inside the block: leaving it unmaps the buffer in this
    process (it does not free the exporter's memory).
    """
    import torch
    from cuda.bindings import runtime

    validate_ref(ref, shape, dtype)
    _check(runtime.cudaSetDevice(ref.device_ordinal), "cudaSetDevice")

    if ref.backend == BACKEND_LEGACY:
        base, close = _open_legacy(ref)
    else:
        base, close = _open_vmm(ref)

    try:
        ptr = base + ref.byte_offset
        flat = torch.as_tensor(
            _CudaArray(ptr, ref.byte_size), device=f"cuda:{ref.device_ordinal}"
        )
        tensor = flat if dtype == torch.uint8 else flat.view(dtype)
        yield tensor.reshape(shape) if shape else tensor
    finally:
        close()


def typestr_for(dtype: "torch.dtype") -> str:
    """__cuda_array_interface__ typestr for a torch dtype."""
    name = str(dtype).removeprefix("torch.")
    if name not in _TYPESTR:
        raise ValueError(f"dtype {dtype} has no __cuda_array_interface__ typestr")
    return _TYPESTR[name]
