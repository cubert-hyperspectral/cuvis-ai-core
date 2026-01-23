# cuvis-ai-core Test Suite

This directory contains the test suite for cuvis-ai-core. Tests are organized by component and use pytest for test execution.

## Running Tests

### Run All Tests
```bash
uv run python -m pytest
```

### Run with Coverage
```bash
uv run python -m pytest --cov=cuvis_ai_core tests/
```

### Run Specific Test File
```bash
uv run python -m pytest tests/pipeline/test_pipeline.py
```

### Run Tests by Marker
```bash
# Run only unit tests
uv run python -m pytest -m unit

# Run only integration tests
uv run python -m pytest -m integration

# Run only gRPC tests
uv run python -m pytest -m grpc
```

### Run with Verbose Output
```bash
uv run python -m pytest -v
```

## Test Structure

```
tests/
├── conftest.py              # Root fixtures and configuration
├── fixtures/                # Reusable test fixtures
│   ├── basic_nodes.py       # Basic node fixtures for testing
│   ├── basic_pipelines.py   # Pipeline fixtures
│   ├── mock_models.py       # Mock .pt file fixtures
│   └── data_factory.py      # Data generation fixtures
├── utils/                   # Tests for utility modules
├── pipeline/                # Tests for pipeline framework
├── grpc_api/                # Tests for gRPC services
├── node_registry/           # Tests for node registry
├── training/                # Tests for training infrastructure
├── integration/             # Integration tests
└── config/                  # Configuration tests
```

## Available Fixtures

### Basic Nodes (`tests/fixtures/basic_nodes.py`)

- `simple_input_node` - Simple input node that generates synthetic data
- `simple_transform_node` - Transform node that scales input
- `simple_output_node` - Output node that accepts input
- `trainable_node` - Node with learnable parameters
- `statistically_initializable_node` - Node that can be statistically initialized
- `multi_output_node` - Node with multiple outputs
- `data_node` - Data node that emulates a dataset loader

### Pipeline Fixtures (`tests/fixtures/basic_pipelines.py`)

- `simple_two_node_pipeline` - Two-node pipeline (Input → Output)
- `simple_three_node_pipeline` - Three-node pipeline (Input → Transform → Output)
- `pipeline_factory` - Factory for creating customizable pipelines
- `trainable_pipeline` - Pipeline with trainable node
- `data_pipeline` - Pipeline with data node
- `minimal_pipeline_dict` - Minimal pipeline configuration dict
- `pipeline_dict_factory` - Factory for creating pipeline config dicts

### Model Fixtures (`tests/fixtures/mock_models.py`)

- `mock_pt_file` - Simple .pt file for testing
- `mock_checkpoint_file` - Full checkpoint with metadata
- `mock_pt_file_factory` - Factory for creating customizable .pt files
- `mock_pipeline_weights` - Pipeline weights file
- `mock_statistical_params` - Statistical parameters file
- `mock_mismatched_weights` - Weights with wrong keys
- `mock_empty_weights` - Empty weights file
- `mock_corrupted_weights` - Corrupted file for error testing

### Data Fixtures (`tests/fixtures/data_factory.py`)

- `sample_batch` - Simple image batch
- `hyperspectral_batch` - Hyperspectral data batch
- `batch_factory` - Factory for creating customizable batches

### Directory Fixtures (`tests/conftest.py`)

- `tmp_config_dir` - Temporary config directory
- `tmp_weights_dir` - Temporary weights directory
- `tmp_data_dir` - Temporary data directory

## Test Markers

Tests can be marked with pytest markers for categorization:

- `@pytest.mark.unit` - Unit tests for individual components
- `@pytest.mark.integration` - Integration tests across components
- `@pytest.mark.grpc` - Tests for gRPC services
- `@pytest.mark.pipeline` - Tests for pipeline functionality
- `@pytest.mark.slow` - Tests that take significant time
- `@pytest.mark.requires_gpu` - Tests requiring GPU hardware

Example:
```python
@pytest.mark.unit
@pytest.mark.pipeline
def test_pipeline_creation():
    pipeline = CuvisPipeline("test")
    assert pipeline.name == "test"
```

## Writing Tests

### Using Basic Node Fixtures

```python
def test_simple_pipeline(simple_input_node, simple_output_node):
    """Test pipeline with simple nodes."""
    pipeline = CuvisPipeline("test")
    
    input_node = simple_input_node()
    output_node = simple_output_node()
    
    pipeline.connect(input_node.outputs.output, output_node.inputs.input)
    
    # Test pipeline
    result = pipeline.forward(...)
    assert result is not None
```

### Using Mock Models

```python
def test_weight_loading(mock_pt_file):
    """Test loading weights from file."""
    weights = torch.load(mock_pt_file)
    assert "layer1.weight" in weights
```

### Using Batch Factory

```python
def test_with_custom_data(batch_factory):
    """Test with custom batch configuration."""
    batch = batch_factory(
        batch_size=8,
        height=128,
        width=128,
        channels=10,
        include_labels=True
    )
    
    assert batch["cube"].shape == (8, 128, 128, 10)
```

## Dependencies

Test dependencies are managed via pyproject.toml:

```toml
[project.optional-dependencies]
test = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "pytest-mock>=3.10.0",
    "pytest-asyncio>=0.21.0",
]
```

Install test dependencies:
```bash
uv sync --extra test
```

## Test Isolation

- Each test runs in isolation with its own temporary directory (`tmp_path`)
- Global state is reset between tests via `reset_global_state` fixture
- Use fixtures instead of global state for test data

## Best Practices

1. **Use fixtures for shared setup** - Don't duplicate setup code
2. **Keep tests small and focused** - One assertion per test when possible
3. **Use descriptive names** - Test names should describe what they test
4. **Use markers appropriately** - Tag tests for easy filtering
5. **Avoid test interdependencies** - Tests should run independently
6. **Use tmp_path for file operations** - Don't write to the repo
7. **Mock external dependencies** - Use fixtures instead of real nodes from cuvis-ai

## Migrated Tests

These tests were migrated from the cuvis-ai repository as part of Phase 3:

- All gRPC API tests (`grpc_api/`)
- Pipeline framework tests (`pipeline/`)
- Node registry tests (`node_registry/`)
- Plugin system tests (`utils/test_plugin_system.py`)
- Core configuration tests (`config/`)
- Core training infrastructure tests (`training/`)
- Core integration tests (`integration/`)

## Future Additions

As cuvis-ai-core grows, add tests in the appropriate directories:
- New node implementations → `tests/node/`
- New utilities → `tests/utils/`
- New gRPC services → `tests/grpc_api/`
- Integration scenarios → `tests/integration/`
