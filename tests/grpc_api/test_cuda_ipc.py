"""CUDA IPC reader tests that need no GPU.

Everything here exercises validation and dispatch. The mapping itself needs a producer
process holding a live export, which is covered by the end-to-end run instead.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from cuvis_ai_core.grpc import cuvis_ai_pb2, helpers
from cuvis_ai_core.grpc import cuda_ipc


def _ref(**overrides) -> cuvis_ai_pb2.CudaIpcRef:
    """A well-formed legacy reference for a 2x3 uint16 tensor."""
    fields = dict(
        backend=cuda_ipc.BACKEND_LEGACY,
        device_ordinal=0,
        handle_type=cuda_ipc.HANDLE_NONE,
        handle_blob=b"\x01" * 64,
        byte_size=12,
        alloc_size=12,
        byte_offset=0,
        exporter_pid=4321,
    )
    fields.update(overrides)
    return cuvis_ai_pb2.CudaIpcRef(**fields)


class TestValidateRef:
    def test_accepts_a_well_formed_legacy_reference(self):
        cuda_ipc.validate_ref(_ref(), (2, 3), torch.uint16)

    def test_accepts_vmm_with_alloc_size_larger_than_byte_size(self):
        # The real producer rounds VMM allocations up to a granularity boundary, so
        # alloc_size > byte_size is the normal case and must not be rejected.
        ref = _ref(
            backend=cuda_ipc.BACKEND_VMM,
            handle_type=cuda_ipc.HANDLE_WIN32_KMT,
            handle_blob=b"\x08" * 8,
            alloc_size=4096,
        )
        cuda_ipc.validate_ref(ref, (2, 3), torch.uint16)

    def test_rejects_the_pool_backend_by_name(self):
        with pytest.raises(NotImplementedError, match="pool"):
            cuda_ipc.validate_ref(_ref(backend=cuda_ipc.BACKEND_POOL), (2, 3), torch.uint16)

    def test_rejects_an_unknown_backend(self):
        with pytest.raises(ValueError, match="not importable"):
            cuda_ipc.validate_ref(_ref(backend=99), (2, 3), torch.uint16)

    def test_rejects_an_nt_handle(self):
        # An NT handle would need DuplicateHandle into this process; the producer
        # refuses to emit one, so this is defence against a mismatched client.
        ref = _ref(backend=cuda_ipc.BACKEND_VMM, handle_type=cuda_ipc.HANDLE_WIN32)
        with pytest.raises(ValueError, match="win32"):
            cuda_ipc.validate_ref(ref, (2, 3), torch.uint16)

    def test_rejects_an_empty_handle_blob(self):
        with pytest.raises(ValueError, match="empty handle blob"):
            cuda_ipc.validate_ref(_ref(handle_blob=b""), (2, 3), torch.uint16)

    def test_rejects_a_byte_size_that_contradicts_the_shape(self):
        # byte_size is an untrusted uint64 that reaches cuMemAddressReserve.
        with pytest.raises(ValueError, match="does not match shape"):
            cuda_ipc.validate_ref(_ref(byte_size=1000, alloc_size=1000), (2, 3), torch.uint16)

    def test_rejects_a_byte_size_not_divisible_by_the_item_size(self):
        with pytest.raises(ValueError, match="not divisible"):
            cuda_ipc.validate_ref(_ref(byte_size=11, alloc_size=11), (), torch.uint16)

    def test_rejects_a_vmm_allocation_too_small_for_the_offset(self):
        ref = _ref(
            backend=cuda_ipc.BACKEND_VMM,
            handle_type=cuda_ipc.HANDLE_WIN32_KMT,
            handle_blob=b"\x08" * 8,
            byte_offset=8,
            alloc_size=12,
        )
        with pytest.raises(ValueError, match="cannot hold"):
            cuda_ipc.validate_ref(ref, (2, 3), torch.uint16)


class TestDtypeMapping:
    def test_proto_to_torch_inverts_torch_to_proto(self):
        for dtype, proto in helpers.DTYPE_TORCH_TO_PROTO.items():
            assert helpers.DTYPE_PROTO_TO_TORCH[proto] == dtype

    def test_every_mapped_dtype_has_a_cuda_array_typestr(self):
        # torch.bool has no meaningful typestr here and is not a cube dtype; the rest
        # must be describable, or a valid tensor could not be wrapped.
        for dtype in helpers.DTYPE_TORCH_TO_PROTO:
            if dtype is torch.bool:
                continue
            assert cuda_ipc.typestr_for(dtype)


class TestProtoToTensorDispatch:
    def test_cuda_payload_never_reaches_the_numpy_path(self):
        # numpy cannot describe device memory, so proto_to_numpy must not be consulted.
        proto = cuvis_ai_pb2.Tensor(
            shape=[2, 3], dtype=cuvis_ai_pb2.D_TYPE_UINT16, cuda_ipc=_ref()
        )
        sentinel = torch.zeros(2, 3, dtype=torch.uint16)

        with patch.object(helpers, "proto_to_numpy") as numpy_path:
            with patch(
                "cuvis_ai_core.grpc.cuda_ipc.open_ref"
            ) as open_ref:
                open_ref.return_value.__enter__.return_value = sentinel
                with helpers.proto_to_tensor(proto) as tensor:
                    assert tensor is sentinel

        numpy_path.assert_not_called()
        open_ref.assert_called_once()
        _, shape, dtype = open_ref.call_args.args
        assert shape == (2, 3)
        assert dtype == torch.uint16

    def test_raw_data_payload_still_uses_the_numpy_path(self):
        arr = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
        proto = cuvis_ai_pb2.Tensor(
            shape=[2, 2],
            dtype=cuvis_ai_pb2.D_TYPE_INT32,
            raw_data=arr.numpy().tobytes(),
        )
        with helpers.proto_to_tensor(proto) as tensor:
            assert torch.equal(tensor, arr)

    def test_cuda_helper_refuses_a_non_cuda_payload(self):
        proto = cuvis_ai_pb2.Tensor(
            shape=[1], dtype=cuvis_ai_pb2.D_TYPE_UINT8, raw_data=b"\x01"
        )
        with pytest.raises(ValueError, match="does not carry a cuda_ipc payload"):
            with helpers.proto_to_cuda_tensor(proto):
                pass


class TestDeviceMoveGuard:
    def test_a_cuda_resident_tensor_is_not_moved_back_to_the_host(self):
        # The whole point of the transport: _get_pipeline_device reports CPU for a
        # parameterless pipeline, and moving would copy the cube off the GPU.
        from cuvis_ai_core.grpc.inference_service import InferenceService

        service = InferenceService.__new__(InferenceService)
        cuda_tensor = MagicMock(spec=torch.Tensor)
        cuda_tensor.is_cuda = True
        host_tensor = torch.zeros(2, 2)

        with patch.object(
            InferenceService, "_get_pipeline_device", return_value=torch.device("cpu")
        ):
            moved = service._move_batch_to_pipeline_device(
                {"cube": cuda_tensor, "mask": host_tensor}, object()
            )

        assert moved["cube"] is cuda_tensor
        cuda_tensor.to.assert_not_called()
        assert moved["mask"].device.type == "cpu"
