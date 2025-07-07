
# HF-MODEL-TOOL

[![CI/CD Pipeline](https://github.com/Chen-zexi/hf-model-tool/actions/workflows/ci.yml/badge.svg)](https://github.com/Chen-zexi/hf-model-tool/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/hf-model-tool.svg)](https://badge.fury.io/py/hf-model-tool)
[![Python versions](https://img.shields.io/pypi/pyversions/hf-model-tool.svg)](https://pypi.org/project/hf-model-tool/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![codecov](https://codecov.io/gh/Chen-zexi/hf-model-tool/branch/main/graph/badge.svg)](https://codecov.io/gh/Chen-zexi/hf-model-tool)

A CLI tool for managing your locally downloaded Huggingface models and datasets

> **Disclaimer:** This tool is not affiliated with or endorsed by Hugging Face. It is an independent, community-developed utility.

## Screenshots

### Welcome Screen
![Welcome Screen](images/welcome-screen.png)

### List All Assets
![List Assets](images/list-asset.png)

## Features

### Core Functionality
*   **Smart Asset Detection:** Detect HuggingFace models, datasets, LoRA adapters, fine-tuned models, and custom formats
*   **Asset Listing:** View all your AI assets with size information and metadata
*   **Duplicate Detection:** Find and clean duplicate downloads to save disk space
*   **Asset Details:** View model configurations and dataset documentation with rich formatting
*   **Directory Management:** Add and manage custom directories containing your AI assets

### Supported Asset Types

- **HuggingFace Models & Datasets:** Standard cached downloads from Hugging Face Hub
- **LoRA Adapters:** Fine-tuned adapters from training frameworks like Unsloth
- **Custom Models:** Fine-tuned models, merged models, and other custom formats

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

### Interactive Mode
```bash
hf-model-tool
```

Launches the interactive CLI with:
- System status showing assets across all configured directories
- Asset management tools for all supported formats
- Easy directory configuration and management

### Command Line Usage

The tool provides comprehensive command-line options for direct operations:

#### Basic Commands
```bash
# Launch interactive mode
hf-model-tool

# List all detected assets
hf-model-tool -l
hf-model-tool --list

# Enter asset management mode
hf-model-tool -m
hf-model-tool --manage

# View detailed asset information
hf-model-tool -v
hf-model-tool --view
hf-model-tool --details

# Show version
hf-model-tool --version

# Show help
hf-model-tool -h
hf-model-tool --help
```

#### Directory Management
```bash
# Add a directory containing LoRA adapters
hf-model-tool -path ~/my-lora-models
hf-model-tool --add-path ~/my-lora-models

# Add a custom model directory
hf-model-tool -path /data/custom-models

# Add current working directory
hf-model-tool -path .

# Add with absolute path
hf-model-tool -path /home/user/ai-projects/models
```

#### Sorting Options
```bash
# List assets sorted by size (default)
hf-model-tool -l --sort size

# List assets sorted by name
hf-model-tool -l --sort name

# List assets sorted by date
hf-model-tool -l --sort date
```

### Interactive Navigation
- **↑/↓ arrows:** Navigate menu options
- **Enter:** Select current option
- **Back:** Select to return to previous menu
- **Config:** Select to access settings and directory management
- **Main Menu:** Select to return to main menu from anywhere
- **Exit:** Select to clean application shutdown
- **Ctrl+C:** Force exit

### Key Workflows

1. **Directory Setup:** Add directories containing your AI assets (HuggingFace cache, LoRA adapters, custom models)
2. **List Assets:** View all detected assets with size information across all directories
3. **Manage Assets:** Delete unwanted files and deduplicate identical assets
4. **View Details:** Inspect model configurations and dataset documentation
5. **Configuration:** Manage directories, change sorting preferences, and access help

## Configuration

### Directory Management
Add custom directories containing your AI assets:
- **HuggingFace Cache:** Standard HF cache with `models--publisher--name` structure
- **Custom Directory:** LoRA adapters, fine-tuned models, or other custom formats  
- **Auto-detect:** Let the tool automatically determine the directory type

#### Interactive Configuration
Access via "Config" from any screen:
- **Directory Management:** Add, remove, and test directories
- **Sort Options:** Size (default), Date, or Name
- **Help System:** Navigation and usage guide


## Project Structure

```
hf_model_tool/
├── __main__.py       # Application entry point with welcome screen
├── cache.py          # Multi-directory asset scanning
├── ui.py             # Rich terminal interface components
├── utils.py          # Asset grouping and duplicate detection
├── navigation.py     # Menu navigation
├── config.py         # Configuration and directory management
└── asset_detector.py # Asset detection (LoRA, custom models, etc.)
```

## Development

### Requirements
- Python ≥ 3.7
- Dependencies: `rich`, `inquirer`, `html2text`

### Logging
Application logs are written to `~/.hf-model-tool.log` for debugging and monitoring.

### Configuration Storage
Settings and directory configurations are stored in `~/.config/hf-model-tool/config.json`

## Contributing

We welcome contributions from the community! Please feel free to:

1. **Open an issue** at [GitHub Issues](https://github.com/Chen-zexi/hf-model-tool/issues)
2. **Submit a pull request** with your improvements
3. **Share feedback** about your experience using the tool

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



