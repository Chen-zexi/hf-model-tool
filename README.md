
# HF-MODEL-TOOL

[![CI/CD Pipeline](https://github.com/Chen-zexi/hf-model-tool/actions/workflows/ci.yml/badge.svg)](https://github.com/Chen-zexi/hf-model-tool/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/hf-model-tool.svg)](https://badge.fury.io/py/hf-model-tool)
[![Python versions](https://img.shields.io/pypi/pyversions/hf-model-tool.svg)](https://pypi.org/project/hf-model-tool/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![codecov](https://codecov.io/gh/Chen-zexi/hf-model-tool/branch/main/graph/badge.svg)](https://codecov.io/gh/Chen-zexi/hf-model-tool)

A CLI tool for managing your locally downloaded HuggingFace models and datasets

## Features

### Core Functionality
*   **Smart Asset Listing:** View models & datasets with comprehensive size and metadata information
*   **Duplicate Detection:** Automatically find and clean duplicate downloads to save space
*   **Asset Details:** View model configurations and dataset documentation with rich formatting

## Installation

### From PyPI (Recommended)
```bash
pip install hf-model-tool
```

### From Source
```bash
git clone https://github.com/Chen-zexi/hf-model-tool.git
cd hf-model-tool
pip install -e .
```

## Usage

### Quick Start
```bash
hf-model-tool
```

This launches the interactive CLI with a professional welcome screen showing:
- System status and cache information
- Feature overview and navigation help
- Intuitive menu-driven interface

### Navigation
- **↑/↓ arrows:** Navigate menu options
- **Enter:** Select current option
- **Back:** Return to previous menu
- **Config:** Access settings and sort options
- **Main Menu:** Quick return to main menu from anywhere
- **Exit:** Clean application shutdown

### Key Workflows

1. **List Assets:** View all models and datasets with size information
2. **Manage Assets:** Access deletion and deduplication tools
3. **View Details:** Inspect model configs and dataset documentation
4. **Configuration:** Change sorting preferences and access help

## Configuration

Access the configuration menu via "Config" from any screen:
- **Sort Options:** Size (default), Date, or Name
- **Future Features:** Cache directory, display preferences
- **Help System:** Navigation and usage guide

## Project Structure

```
hf_model_tool/
├── __main__.py      # Application entry point with welcome screen
├── cache.py         # HuggingFace cache directory scanning
├── ui.py           # Rich terminal interface components
├── utils.py        # Asset grouping and duplicate detection
└── navigation.py   # Unified menu navigation system
```

## 🛠️ Development

### Requirements
- Python ≥ 3.7
- Dependencies: `rich`, `inquirer`, `html2text`

### Logging
Application logs are written to `~/.hf-model-tool.log` for debugging and monitoring.



