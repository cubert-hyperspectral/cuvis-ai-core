import mmap
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from cuvis_ai_schemas.pipeline import PortSpec

from cuvis_ai_core.grpc import cuvis_ai_pb2, helpers
from cuvis_ai_core.grpc.helpers import ShmBufferOwner
from cuvis_ai_core.grpc.plugin_service import _convert_port_spec_to_proto


def _make_owner(data: bytes) -> ShmBufferOwner:
    """Create a ShmBufferOwner backed by an anonymous mmap pre-filled with data."""
    size = max(len(data), 1)
    mm = mmap.mmap(-1, size)
    mm.write(data)
    mm.seek(0)
    return ShmBufferOwner(mm)


class TestProtoToNumpy:
    """Test proto → numpy conversion"""

    def test_proto_to_numpy_float32(self):
        """Test converting float32 tensor proto to numpy"""
        # Arrange
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        tensor_proto = cuvis_ai_pb2.Tensor(
            shape=[2, 2], dtype=cuvis_ai_pb2.D_TYPE_FLOAT32, raw_data=arr.tobytes()
        )

        # Act
        with helpers.proto_to_numpy(tensor_proto) as result:
            # Assert
            assert isinstance(result, np.ndarray)
            np.testing.assert_array_equal(result, arr)
            assert result.dtype == np.float32

    def test_proto_to_numpy_int32(self):
        """Test converting int32 tensor proto to numpy"""
        arr = np.array([1, 2, 3, 4], dtype=np.int32)
        tensor_proto = cuvis_ai_pb2.Tensor(
            shape=[4], dtype=cuvis_ai_pb2.D_TYPE_INT32, raw_data=arr.tobytes()
        )

        with helpers.proto_to_numpy(tensor_proto) as result:
            np.testing.assert_array_equal(result, arr)
            assert result.dtype == np.int32

    def test_proto_to_numpy_invalid_dtype(self):
        """Test error handling for unsupported dtype"""
        tensor_proto = cuvis_ai_pb2.Tensor(
            shape=[2, 2],
            dtype=999,  # Invalid
            raw_data=b"invalid",
        )

        with pytest.raises(ValueError, match="Unsupported dtype"):
            with helpers.proto_to_numpy(tensor_proto) as _:
                pass

    def test_proto_to_numpy_empty_tensor(self):
        """Test handling empty tensors"""
        arr = np.array([], dtype=np.float32)
        tensor_proto = cuvis_ai_pb2.Tensor(
            shape=[0], dtype=cuvis_ai_pb2.D_TYPE_FLOAT32, raw_data=arr.tobytes()
        )

        with helpers.proto_to_numpy(tensor_proto) as result:
            assert result.shape == (0,)
            assert result.dtype == np.float32

    def test_proto_to_numpy_writable_by_default(self):
        """Test that proto_to_numpy returns writable arrays by default"""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        tensor_proto = cuvis_ai_pb2.Tensor(
            shape=[2, 2], dtype=cuvis_ai_pb2.D_TYPE_FLOAT32, raw_data=arr.tobytes()
        )

        with helpers.proto_to_numpy(tensor_proto) as result:
            # Should be writable
            assert result.flags.writeable
            # Verify we can modify it
            result[0, 0] = 999.0
            assert result[0, 0] == 999.0

    def test_proto_to_numpy_copy_true(self):
        """Test proto_to_numpy with copy=True returns writable array"""
        arr = np.array([1, 2, 3, 4], dtype=np.int32)
        tensor_proto = cuvis_ai_pb2.Tensor(
            shape=[4], dtype=cuvis_ai_pb2.D_TYPE_INT32, raw_data=arr.tobytes()
        )

        with helpers.proto_to_numpy(tensor_proto, copy=True) as result:
            # Should be writable
            assert result.flags.writeable
            result[0] = 999
            assert result[0] == 999

    def test_proto_to_numpy_copy_false(self):
        """Test proto_to_numpy with copy=False returns read-only view"""
        arr = np.array([1, 2, 3, 4], dtype=np.int32)
        tensor_proto = cuvis_ai_pb2.Tensor(
            shape=[4], dtype=cuvis_ai_pb2.D_TYPE_INT32, raw_data=arr.tobytes()
        )

        with helpers.proto_to_numpy(tensor_proto, copy=False) as result:
            # Should be read-only
            assert not result.flags.writeable
            # Verify modification raises error
            with pytest.raises(ValueError, match="assignment destination is read-only"):
                result[0] = 999


class TestNumpyToProto:
    """Test numpy → proto conversion"""

    def test_numpy_to_proto_float32(self):
        """Test converting numpy array to proto"""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        tensor_proto = helpers.numpy_to_proto(arr)

        assert list(tensor_proto.shape) == [2, 2]
        assert tensor_proto.dtype == cuvis_ai_pb2.D_TYPE_FLOAT32
        assert tensor_proto.raw_data == arr.tobytes()

    def test_numpy_to_proto_roundtrip(self):
        """Test numpy → proto → numpy preserves data"""
        arr = np.random.randn(3, 4, 5).astype(np.float32)

        proto = helpers.numpy_to_proto(arr)
        with helpers.proto_to_numpy(proto) as result:
            np.testing.assert_array_almost_equal(result, arr)
            assert result.shape == arr.shape
            assert result.dtype == arr.dtype


class TestTorchConversion:
    """Test torch tensor conversion"""

    def test_proto_to_tensor(self):
        """Test proto → torch tensor conversion"""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        tensor_proto = helpers.numpy_to_proto(arr)

        with helpers.proto_to_tensor(tensor_proto) as tensor:
            assert isinstance(tensor, torch.Tensor)
            assert tensor.shape == torch.Size([2, 2])
            assert tensor.dtype == torch.float32
            torch.testing.assert_close(tensor, torch.tensor(arr))

    def test_tensor_to_proto(self):
        """Test torch tensor → proto conversion"""
        tensor = torch.randn(2, 3, 4)

        proto = helpers.tensor_to_proto(tensor)

        assert list(proto.shape) == [2, 3, 4]
        assert proto.dtype == cuvis_ai_pb2.D_TYPE_FLOAT32

    def test_tensor_to_proto_roundtrip(self):
        """Test tensor → proto → tensor preserves data"""
        tensor = torch.randn(3, 4, 5)

        proto = helpers.tensor_to_proto(tensor)
        with helpers.proto_to_tensor(proto) as result:
            torch.testing.assert_close(result, tensor)

    def test_tensor_to_proto_rejects_numpy_input_with_actionable_error(self):
        """Test tensor_to_proto fails fast on non-torch input with guidance."""
        arr = np.zeros((2, 2), dtype=np.float32)

        with pytest.raises(ValueError) as exc:
            helpers.tensor_to_proto(arr)  # type: ignore[arg-type]

        message = str(exc.value)
        assert "torch.Tensor" in message
        assert "numpy.ndarray" in message
        assert "numpy_to_proto" in message

    def test_tensor_to_proto_lists_supported_dtypes_for_torch_dtype_mismatch(self):
        """Test unsupported torch dtype errors include supported alternatives."""
        tensor = torch.ones((2, 2), dtype=torch.complex64)

        with pytest.raises(ValueError) as exc:
            helpers.tensor_to_proto(tensor)

        message = str(exc.value)
        assert "Unsupported torch dtype: torch.complex64" in message
        assert "Supported dtypes:" in message
        assert "torch.float32" in message

    def test_proto_to_tensor_copy_false_raw_data(self):
        """copy=False on raw_data path yields tensor with correct values.

        PyTorch does not support non-writable tensors (warns but allows); just
        verify the values are correct.
        """
        arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        tensor_proto = helpers.numpy_to_proto(arr)

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with helpers.proto_to_tensor(tensor_proto, copy=False) as tensor:
                torch.testing.assert_close(tensor, torch.tensor(arr))


class TestProcessingModeMapping:
    """Test ProcessingMode enum mapping"""

    def test_processing_mode_to_string(self):
        """Test proto ProcessingMode → mode-name string"""
        proto_raw = cuvis_ai_pb2.PROCESSING_MODE_RAW
        proto_refl = cuvis_ai_pb2.PROCESSING_MODE_REFLECTANCE

        assert helpers.proto_to_processing_mode(proto_raw) == "Raw"
        assert helpers.proto_to_processing_mode(proto_refl) == "Reflectance"

    def test_processing_mode_invalid(self):
        """Test error handling for invalid processing mode"""
        with pytest.raises(ValueError, match="Unsupported ProcessingMode"):
            helpers.proto_to_processing_mode(999)

    def test_processing_mode_new_values(self):
        """Test new ProcessingMode enum values (DarkSubtract, SpectralRadiance)"""
        # Test DarkSubtract mapping
        proto_darksubtract = cuvis_ai_pb2.PROCESSING_MODE_DARKSUBTRACT
        assert helpers.proto_to_processing_mode(proto_darksubtract) == "DarkSubtract"

        # Test SpectralRadiance mapping
        proto_spectral_radiance = cuvis_ai_pb2.PROCESSING_MODE_SPECTRAL_RADIANCE
        assert (
            helpers.proto_to_processing_mode(proto_spectral_radiance)
            == "SpectralRadiance"
        )

    def test_processing_mode_enum_values(self):
        """Test that all ProcessingMode enum values are correctly defined"""
        # Verify the enum values match expected numbers
        assert cuvis_ai_pb2.PROCESSING_MODE_UNSPECIFIED == 0
        assert cuvis_ai_pb2.PROCESSING_MODE_RAW == 1
        assert cuvis_ai_pb2.PROCESSING_MODE_REFLECTANCE == 2
        assert cuvis_ai_pb2.PROCESSING_MODE_DARKSUBTRACT == 3
        assert cuvis_ai_pb2.PROCESSING_MODE_SPECTRAL_RADIANCE == 4


class TestDtypeToProto:
    """Test the shared dtype → proto enum dispatch."""

    def test_concrete_torch_dtype(self):
        assert helpers.dtype_to_proto(torch.float32) == cuvis_ai_pb2.D_TYPE_FLOAT32
        assert helpers.dtype_to_proto(torch.int64) == cuvis_ai_pb2.D_TYPE_INT64
        assert helpers.dtype_to_proto(torch.bool) == cuvis_ai_pb2.D_TYPE_BOOL

    def test_torch_tensor_is_unspecified(self):
        """torch.Tensor (the class) maps to UNSPECIFIED, not raising."""
        assert helpers.dtype_to_proto(torch.Tensor) == cuvis_ai_pb2.D_TYPE_UNSPECIFIED

    def test_numpy_dtype_instance(self):
        assert (
            helpers.dtype_to_proto(np.dtype("float32")) == cuvis_ai_pb2.D_TYPE_FLOAT32
        )
        assert helpers.dtype_to_proto(np.dtype("int32")) == cuvis_ai_pb2.D_TYPE_INT32

    def test_numpy_scalar_class(self):
        assert helpers.dtype_to_proto(np.int32) == cuvis_ai_pb2.D_TYPE_INT32
        assert helpers.dtype_to_proto(np.float64) == cuvis_ai_pb2.D_TYPE_FLOAT64
        assert helpers.dtype_to_proto(np.uint16) == cuvis_ai_pb2.D_TYPE_UINT16

    def test_python_builtin_types_are_unspecified(self):
        assert helpers.dtype_to_proto(dict) == cuvis_ai_pb2.D_TYPE_UNSPECIFIED
        assert helpers.dtype_to_proto(str) == cuvis_ai_pb2.D_TYPE_UNSPECIFIED
        assert helpers.dtype_to_proto(list) == cuvis_ai_pb2.D_TYPE_UNSPECIFIED

    def test_unsupported_torch_dtype_raises(self):
        # complex64 is not in the torch→proto mapping
        with pytest.raises(ValueError, match="Unsupported torch dtype"):
            helpers.dtype_to_proto(torch.complex64)

    def test_unsupported_numpy_dtype_raises(self):
        """Numpy scalar class not in the mapping raises (e.g. np.complex64)."""
        with pytest.raises(ValueError, match="Unsupported numpy dtype"):
            helpers.dtype_to_proto(np.complex64)

    def test_unsupported_numpy_dtype_instance_raises(self):
        """np.dtype instance not in the mapping raises (distinct branch from
        the numpy scalar class path)."""
        with pytest.raises(ValueError, match="Unsupported numpy dtype"):
            helpers.dtype_to_proto(np.dtype("complex64"))

    def test_non_type_non_dtype_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            helpers.dtype_to_proto("not-a-dtype")


class TestConvertPortSpecToProto:
    """Test PortSpec → proto PortSpec conversion"""

    def test_concrete_torch_dtype_float32(self):
        """torch.float32 dtype maps to D_TYPE_FLOAT32"""
        spec = PortSpec(dtype=torch.float32, shape=(-1, -1))

        result = _convert_port_spec_to_proto(spec, name="cube")

        assert result.name == "cube"
        assert result.dtype == cuvis_ai_pb2.D_TYPE_FLOAT32
        assert list(result.shape) == [-1, -1]

    def test_concrete_torch_dtype_int64_scalar_shape(self):
        """torch.int64 with scalar shape maps to D_TYPE_INT64, empty shape list"""
        spec = PortSpec(dtype=torch.int64, shape=())

        result = _convert_port_spec_to_proto(spec, name="index")

        assert result.dtype == cuvis_ai_pb2.D_TYPE_INT64
        assert list(result.shape) == []

    def test_generic_torch_tensor_dtype_is_unspecified(self):
        """torch.Tensor (the class) as a generic-tensor marker maps to UNSPECIFIED.

        Regression: the torch.Tensor branch used to sit after a
        hasattr(spec.dtype, "dtype") check. torch.Tensor exposes a `dtype`
        descriptor at the class level, so the hasattr branch captured it
        first and raised "Unsupported numpy dtype: <class 'torch.Tensor'>".
        """
        spec = PortSpec(dtype=torch.Tensor, shape=(-1, -1, -1, -1))

        result = _convert_port_spec_to_proto(spec, name="cube")

        assert result.dtype == cuvis_ai_pb2.D_TYPE_UNSPECIFIED
        assert list(result.shape) == [-1, -1, -1, -1]

    def test_numpy_scalar_class_int32(self):
        """np.int32 (a numpy scalar class) maps to D_TYPE_INT32"""
        spec = PortSpec(dtype=np.int32, shape=(-1,))

        result = _convert_port_spec_to_proto(spec, name="wavelengths")

        assert result.dtype == cuvis_ai_pb2.D_TYPE_INT32
        assert list(result.shape) == [-1]

    def test_python_builtin_type_is_unspecified(self):
        """Python builtin types (dict, str, list) map to UNSPECIFIED"""
        spec = PortSpec(dtype=dict, shape=())

        result = _convert_port_spec_to_proto(spec, name="metadata")

        assert result.dtype == cuvis_ai_pb2.D_TYPE_UNSPECIFIED

    def test_unsupported_dtype_raises(self):
        """Non-type, non-dtype values raise ValueError"""
        spec = PortSpec(dtype="not-a-dtype", shape=())

        with pytest.raises(ValueError, match="Unsupported"):
            _convert_port_spec_to_proto(spec, name="bad")

    def test_symbolic_string_shape_dim_coerced_to_minus_one(self):
        """Symbolic shape dimensions (str, e.g. 'batch') are coerced to -1"""
        spec = PortSpec(dtype=torch.float32, shape=(-1, "batch", 10))

        result = _convert_port_spec_to_proto(spec, name="features")

        assert list(result.shape) == [-1, -1, 10]

    def test_optional_and_description_passthrough(self):
        """optional and description fields are copied onto the proto message"""
        spec = PortSpec(
            dtype=torch.float32,
            shape=(-1,),
            optional=True,
            description="frame index",
        )

        result = _convert_port_spec_to_proto(spec, name="mesu_index")

        assert result.optional is True
        assert result.description == "frame index"


class TestShmBufferOwner:
    """Test ShmBufferOwner lifecycle."""

    def test_close_sets_closed_flag(self):
        owner = _make_owner(b"\x00" * 16)
        assert not owner.closed
        owner.close()
        assert owner.closed

    def test_close_idempotent(self):
        owner = _make_owner(b"\x00" * 16)
        owner.close()
        owner.close()  # must not raise

    def test_context_manager_closes_on_exit(self):
        owner = _make_owner(b"\x00" * 16)
        with owner:
            assert not owner.closed
        assert owner.closed

    def test_buffer_error_on_close_does_not_set_closed(self):
        mock_mm = MagicMock()
        mock_mm.close.side_effect = BufferError("view still active")
        owner = ShmBufferOwner(mock_mm)
        owner.close()  # must not raise
        assert not owner.closed

    def test_file_obj_closed_on_normal_close(self):
        mock_mm = MagicMock()
        mock_file = MagicMock()
        owner = ShmBufferOwner(mock_mm, file_obj=mock_file)
        owner.close()
        mock_file.close.assert_called_once()
        assert owner.closed


class TestProtoToNumpyShmRef:
    """Test proto_to_numpy with shm_ref payload (SHM path)."""

    def _make_tensor_proto(
        self, arr: np.ndarray, byte_offset: int = 0
    ) -> cuvis_ai_pb2.Tensor:
        data = arr.tobytes()
        return cuvis_ai_pb2.Tensor(
            shape=list(arr.shape),
            dtype=cuvis_ai_pb2.D_TYPE_FLOAT32,
            shm_ref=cuvis_ai_pb2.ShmRef(
                name="/test_seg",
                byte_offset=byte_offset,
                byte_size=len(data),
            ),
        )

    def test_shm_ref_copy_true(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        tensor_proto = self._make_tensor_proto(arr)
        with patch(
            "cuvis_ai_core.grpc.helpers._map_shm",
            return_value=_make_owner(arr.tobytes()),
        ):
            with helpers.proto_to_numpy(tensor_proto, copy=True) as result:
                assert isinstance(result, np.ndarray)
                assert result.flags.writeable
                np.testing.assert_array_equal(result, arr)

    def test_shm_ref_copy_false(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        tensor_proto = self._make_tensor_proto(arr)
        with patch(
            "cuvis_ai_core.grpc.helpers._map_shm",
            return_value=_make_owner(arr.tobytes()),
        ):
            with helpers.proto_to_numpy(tensor_proto, copy=False) as result:
                np.testing.assert_array_equal(result, arr)

    def test_shm_ref_with_byte_offset(self):
        arr = np.array([10.0, 20.0], dtype=np.float32)
        padding = b"\xff" * 8
        raw = padding + arr.tobytes()
        tensor_proto = cuvis_ai_pb2.Tensor(
            shape=[2],
            dtype=cuvis_ai_pb2.D_TYPE_FLOAT32,
            shm_ref=cuvis_ai_pb2.ShmRef(name="/test_seg", byte_offset=8, byte_size=8),
        )
        with patch(
            "cuvis_ai_core.grpc.helpers._map_shm", return_value=_make_owner(raw)
        ):
            with helpers.proto_to_numpy(tensor_proto) as result:
                np.testing.assert_array_equal(result, arr)

    def test_shm_ref_unaligned_byte_size_raises(self):
        tensor_proto = cuvis_ai_pb2.Tensor(
            shape=[],
            dtype=cuvis_ai_pb2.D_TYPE_FLOAT32,
            shm_ref=cuvis_ai_pb2.ShmRef(name="/test_seg", byte_offset=0, byte_size=5),
        )
        with patch(
            "cuvis_ai_core.grpc.helpers._map_shm", return_value=_make_owner(b"\x00" * 5)
        ):
            with pytest.raises(ValueError, match="not divisible"):
                with helpers.proto_to_numpy(tensor_proto) as _:
                    pass

    def test_shm_ref_close_called_after_context_exit(self):
        """owner.close() is called in the finally block even if the mmap has live views."""
        arr = np.array([1.0, 2.0], dtype=np.float32)
        owner = _make_owner(arr.tobytes())
        tensor_proto = cuvis_ai_pb2.Tensor(
            shape=[2],
            dtype=cuvis_ai_pb2.D_TYPE_FLOAT32,
            shm_ref=cuvis_ai_pb2.ShmRef(name="/test_seg", byte_offset=0, byte_size=8),
        )
        with patch("cuvis_ai_core.grpc.helpers._map_shm", return_value=owner):
            with patch.object(owner, "close") as mock_close:
                with helpers.proto_to_numpy(tensor_proto) as _:
                    pass
        mock_close.assert_called_once()


class TestProtoToTensorShmRef:
    """Test proto_to_tensor with shm_ref and raw_data payloads."""

    def test_shm_ref_yields_tensor(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        tensor_proto = cuvis_ai_pb2.Tensor(
            shape=[2, 2],
            dtype=cuvis_ai_pb2.D_TYPE_FLOAT32,
            shm_ref=cuvis_ai_pb2.ShmRef(name="/test_seg", byte_offset=0, byte_size=16),
        )
        with patch(
            "cuvis_ai_core.grpc.helpers._map_shm",
            return_value=_make_owner(arr.tobytes()),
        ):
            with helpers.proto_to_tensor(tensor_proto) as tensor:
                assert isinstance(tensor, torch.Tensor)
                assert tensor.shape == torch.Size([2, 2])
                assert tensor.dtype == torch.float32
                torch.testing.assert_close(tensor, torch.tensor(arr))

    def test_shm_ref_copy_true_writable(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tensor_proto = cuvis_ai_pb2.Tensor(
            shape=[3],
            dtype=cuvis_ai_pb2.D_TYPE_FLOAT32,
            shm_ref=cuvis_ai_pb2.ShmRef(name="/test_seg", byte_offset=0, byte_size=12),
        )
        with patch(
            "cuvis_ai_core.grpc.helpers._map_shm",
            return_value=_make_owner(arr.tobytes()),
        ):
            with helpers.proto_to_tensor(tensor_proto, copy=True) as tensor:
                tensor[0] = 99.0  # must not raise
                assert tensor[0].item() == 99.0

    def test_shm_ref_copy_false_shares_memory(self):
        """copy=False yields tensor backed by the mmap; modifications write through."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        owner = _make_owner(arr.tobytes())
        tensor_proto = cuvis_ai_pb2.Tensor(
            shape=[3],
            dtype=cuvis_ai_pb2.D_TYPE_FLOAT32,
            shm_ref=cuvis_ai_pb2.ShmRef(name="/test_seg", byte_offset=0, byte_size=12),
        )
        with patch("cuvis_ai_core.grpc.helpers._map_shm", return_value=owner):
            with helpers.proto_to_tensor(tensor_proto, copy=False) as tensor:
                assert isinstance(tensor, torch.Tensor)
                torch.testing.assert_close(tensor, torch.tensor(arr))
                # Modifying the tensor writes through to the mmap (zero-copy semantics)
                tensor[0] = 99.0
                owner.mmap_obj.seek(0)
                raw = owner.mmap_obj.read(12)
                assert np.frombuffer(raw, dtype=np.float32)[0] == pytest.approx(99.0)

    def test_raw_data_copy_false(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        tensor_proto = helpers.numpy_to_proto(arr)
        with helpers.proto_to_tensor(tensor_proto, copy=False) as tensor:
            torch.testing.assert_close(tensor, torch.tensor(arr))


class TestMapShmDispatch:
    """Test _map_shm platform dispatch."""

    def test_dispatches_to_windows_on_win32(self):
        mock_owner = MagicMock()
        with patch(
            "cuvis_ai_core.grpc.helpers._map_shm_windows", return_value=mock_owner
        ) as mock_win:
            with patch("sys.platform", "win32"):
                result = helpers._map_shm("test_name", 16)
        mock_win.assert_called_once_with("test_name", 16)
        assert result is mock_owner

    def test_dispatches_to_posix_on_linux(self):
        mock_owner = MagicMock()
        with patch(
            "cuvis_ai_core.grpc.helpers._map_shm_posix", return_value=mock_owner
        ) as mock_posix:
            with patch("sys.platform", "linux"):
                result = helpers._map_shm("test_name", 16)
        mock_posix.assert_called_once_with("test_name", 16)
        assert result is mock_owner


class TestShmNameValidation:
    """SHM names from the wire must not escape the shared-memory namespace."""

    @pytest.mark.parametrize("bad", ["", "../etc/passwd", "a/../../b", "foo/..", ".."])
    def test_map_shm_rejects_traversal_or_empty(self, bad):
        with pytest.raises(ValueError):
            helpers._map_shm(bad, 16)

    @pytest.mark.parametrize("bad", ["foo/bar", "a\\b", "", "with/sep"])
    def test_map_shm_posix_rejects_embedded_separator(self, bad):
        # The single-segment check runs before any open(), so it raises on every
        # platform without touching the filesystem.
        with pytest.raises(ValueError):
            helpers._map_shm_posix(bad, 16)

    def test_map_shm_posix_accepts_leading_slash_name(self):
        # The legitimate POSIX form "/cuvis_<pid>_<n>" must pass validation; it only
        # fails later at open() because no such segment exists here.
        with pytest.raises((FileNotFoundError, OSError)):
            helpers._map_shm_posix("/cuvis_123_0", 16)


class TestProtoToNumpyShmValidation:
    """proto_to_numpy validates untrusted ShmRef sizes before mapping."""

    def test_byte_size_shape_mismatch_raises(self):
        proto = cuvis_ai_pb2.Tensor(
            shape=[2, 2],
            dtype=cuvis_ai_pb2.D_TYPE_FLOAT32,
            shm_ref=cuvis_ai_pb2.ShmRef(name="cuvis_1_0", byte_offset=0, byte_size=8),
        )
        with pytest.raises(ValueError):
            with helpers.proto_to_numpy(proto):
                pass

    def test_byte_size_not_divisible_raises(self):
        proto = cuvis_ai_pb2.Tensor(
            shape=[],
            dtype=cuvis_ai_pb2.D_TYPE_FLOAT32,
            shm_ref=cuvis_ai_pb2.ShmRef(name="cuvis_1_0", byte_offset=0, byte_size=7),
        )
        with pytest.raises(ValueError):
            with helpers.proto_to_numpy(proto):
                pass

    def test_zero_byte_size_yields_empty_without_mapping(self):
        proto = cuvis_ai_pb2.Tensor(
            shape=[0],
            dtype=cuvis_ai_pb2.D_TYPE_FLOAT32,
            shm_ref=cuvis_ai_pb2.ShmRef(name="cuvis_1_0", byte_offset=0, byte_size=0),
        )
        with helpers.proto_to_numpy(proto) as arr:
            assert arr.shape == (0,)
            assert arr.dtype == np.float32
