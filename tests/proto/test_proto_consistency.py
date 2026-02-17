import importlib.resources


def test_no_legacy_validate_training_config_rpc() -> None:
    """Ensure old ValidateTrainingConfig RPC is not reintroduced.

    Proto definitions are owned by cuvis-ai-schemas; this test reads
    the generated stub file from the schemas package.
    """
    stub_ref = (
        importlib.resources.files("cuvis_ai_schemas")
        / "grpc"
        / "v1"
        / "cuvis_ai_pb2.pyi"
    )
    stub_content = stub_ref.read_text(encoding="utf-8")

    assert "ValidateTrainingConfigRequest" not in stub_content
    assert "ValidateTrainingConfigResponse" not in stub_content
