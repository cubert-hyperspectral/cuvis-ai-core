# cuvis-ai-schemas

**Lightweight schema definitions for the cuvis-ai ecosystem**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)

## Overview

`cuvis-ai-schemas` is a small, dependency-light package of Pydantic models used across cuvis-ai services for type-safe configuration and messaging.

## Features

- Lightweight: minimal deps (pydantic, pyyaml)
- Type-safe schemas for pipeline, plugin, training, execution, discovery
- Optional extras: `proto`, `torch`, `numpy`, `lightning`, `full`, `dev`

## Installation

```bash
uv add cuvis-ai-schemas
uv add "cuvis-ai-schemas[proto]"
uv add "cuvis-ai-schemas[full]"
```

## Usage

```python
from cuvis_ai_schemas.pipeline import PipelineConfig, NodeConfig

pipeline = PipelineConfig(
    nodes=[NodeConfig(id="node_1", class_name="DataLoader", params={"batch_size": 32})],
    connections=[],
)

pipeline_json = pipeline.to_json()
pipeline = PipelineConfig.from_json(pipeline_json)
```

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
uv run ruff format .
```

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

## Contact

**Cubert GmbH**
Email: cuvis.ai@cubert-gmbh.com
Website: https://www.cubert-hyperspectral.com/

## Related Projects

- [cuvis-ai-core](https://github.com/cubert-hyperspectral/cuvis-ai-core) - Main processing server
- [cuvis-ai-ui](https://github.com/cubert-hyperspectral/cuvis-ai-ui) - Qt-based UI
- [cuvis-ai](https://github.com/cubert-hyperspectral/cuvis-ai) - Node catalog/library
