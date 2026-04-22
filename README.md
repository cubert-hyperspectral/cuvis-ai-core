# Cuvis.AI Core



![image](https://github.com/cubert-hyperspectral/cuvis.sdk/blob/main/branding/logo/banner.png?raw=true)

[![PyPI version](https://img.shields.io/pypi/v/cuvis-ai-core.svg)](https://pypi.org/project/cuvis-ai-core/)
[![CI Status](https://github.com/cubert-hyperspectral/cuvis-ai-core/actions/workflows/ci.yml/badge.svg)](https://github.com/cubert-hyperspectral/cuvis-ai-core/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/cubert-hyperspectral/cuvis-ai-core/branch/main/graph/badge.svg)](https://codecov.io/gh/cubert-hyperspectral/cuvis-ai-core)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)



## Overview

Cuvis.AI is an opensource and extensible framework for building AI powered processing pipelines for hyperspectral video data.

It allows you to process and structure spectral data, train and apply machine learning models, visualize and interpret results, and deploy applications in real time environments.

Pipelines are built from reusable modular nodes and can be extended with custom plugins or external integrations.

Cuvis.AI bridges the gap between hyperspectral hardware and real world applications and enables faster development, testing, and deployment of new solutions.

**cuvis-ai-core** is the foundational framework underneath Cuvis.AI. While the [cuvis-ai](https://github.com/cubert-hyperspectral/cuvis-ai) repository contains the catalog of domain-specific nodes and pre-built models, **cuvis-ai-core** provides the essential infrastructure that makes those modular pipelines work:

- **Node System**: Base classes for creating processing nodes with typed input/output ports
- **Pipeline Infrastructure**: Graph-based pipeline orchestration, execution, and visualization
- **Plugin System**: Dynamic node loading from Git repositories or local filesystem paths
- **Serialization & Restoration**: Save and restore complete pipeline states and configurations
- **Type Safety**: Strongly-typed port system with runtime validation
- **gRPC Services**: Remote pipeline management, training, and inference APIs
- **Training Framework**: Integration with PyTorch Lightning for model training workflows

This separation allows the core framework to evolve independently while the catalog of domain-specific nodes grows through a plugin architecture.

- **Website:** https://www.cubert-hyperspectral.com/
- **Support:** http://support.cubert-hyperspectral.com/

## Installation

### Prerequisites

If you want to directly work with cubert session files (.cu3s), you need to install cuvis C SDK from 
[here](https://cloud.cubert-gmbh.de/s/qpxkyWkycrmBK9m).

Local development now relies on [uv](https://docs.astral.sh/uv/) for Python and dependency management.
If `uv` is not already available on your system you can install it following their installation instructions.

### Install from PyPI (Recommended)

```bash
# Install cuvis-ai-core from PyPI
uv add cuvis-ai-core
```

PyTorch will be installed automatically with CUDA support if available on your system.

### Local development with uv

Create or refresh a development environment at the repository root with:

```bash
uv sync --all-extras --dev
```

This installs the runtime dependencies declared in `pyproject.toml`. `uv` automatically provisions the Python version declared in the project metadata, so no manual interpreter management is required.

#### Enable Git Hooks (Required)

After cloning the repository, enable the git hooks for code quality enforcement:

```bash
git config core.hooksPath .githooks
```

This configures Git to use the version-controlled hooks in `.githooks/` which automatically enforce code formatting, linting, and testing standards before commits and pushes. See [docs/development/git-hooks.md](docs/development/git-hooks.md) for details.

#### Advanced environment setup

When you need the reproducible development toolchain (JupyterLab, TensorBoard, etc.) from the lock file, run:

```bash
uv sync --locked --extra dev
```

Use `uv run` to execute project tooling without manually activating virtual environments, for example:

```bash
uv run pytest
```

Collect coverage details (the `dev` extra installs `pytest-cov`) with:

```bash
uv run pytest --cov=cuvis_ai --cov-report=term-missing
```

Ruff handles both formatting and linting. Format sources and check style with:

```bash
uv run ruff format .
uv run ruff check .
```

The configuration enforces import ordering, newline hygiene, modern string formatting, safe exception chaining, and practical return type annotations while avoiding noisy `Any` policing.

Validate packaging metadata and build artifacts before publishing:

```bash
uv build
```


To build the documentation, add the `docs` extra:

```bash
uv sync --locked --extra docs
```

Combine extras as needed (e.g. `uv sync --locked --extra dev --extra docs`). Whenever the `pyproject.toml` or `uv.lock` changes, rerun `uv sync --locked` with the extras you need to stay up to date.
