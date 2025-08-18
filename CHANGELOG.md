# Changelog

All notable changes to HF-MODEL-TOOL will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.5] - 2025-08-18

### Added
- **Ollama Model Support**: Full integration with Ollama models
  - Automatic detection of Ollama models in system and user directories
  - Toggle Ollama scanning on/off via configuration
  - Support for both `/home/.ollama` and `/usr/share/ollama` directories
- **Permission Error Handling**: Graceful handling of permission-denied errors
  - No more crashes when attempting to delete system-owned files
  - Helpful hints suggesting sudo commands for system directories
  - Summary statistics showing successful vs failed operations

### Changed
- **Enhanced Path Display**: Improved path information in deletion and deduplication menus
  - Smart path formatting showing meaningful prefixes (e.g., "user: ~/.ollama", "system: /usr/share/ollama")
  - Added modification dates to help identify which version to keep/delete
  - Extracted model names from HuggingFace cache paths for clarity
- **Improved Deduplication**:
  - Registry now uses model names for Ollama deduplication instead of file paths

### Fixed
- Corrected field name from "name" to "served_model_name" for vLLM compatibility

## [0.2.4] - 2025-08-17

### Added
- **Manifest System**: Automatic generation of `models_manifest.json` for custom directories
  - JSON manifests for customizing model names, publishers, and metadata
  - Auto-generated when adding custom directories
  - Primary source of truth for model classification
- **Comprehensive Documentation**:
  - `docs/manifest-system.md` - Complete manifest system guide
  - `docs/custom-directories.md` - Custom directory configuration guide
  - `docs/api-reference.md` - Full API documentation
- **Enhanced API Methods**:
  - `generate_manifest()` - Generate manifest for a directory
  - `load_manifest()` - Load existing manifest
  - `save_manifest()` - Save manifest to disk
  - `update_manifest()` - Update specific model entries
  - `get_models_with_manifest()` - Get models with manifest data merged

### Changed
- Manifest data takes precedence over auto-detected metadata
- Simplified configuration code by removing duplicate imports and logic

### Fixed
- Asset detector now correctly identifies only actual model directories

## [0.2.3] - 2025-08-17

### Added
- Improved LoRA adapter management and detection
- Enhanced API support for vLLM-CLI integration
- Better organization of custom model directories

### Changed
- Optimized model discovery performance
- Improved duplicate detection logic

## [0.2.1] - 2025-07-07

### Added
- Command-line argument parser for direct operations
- Support for LoRA adapter detection
- Custom path detection improvements

### Changed
- Updated documentation and README
- Version display now always matches pyproject.toml

### Fixed
- Bug in asset management workflow
- Cross-platform compatibility issues

## [0.2.0] - 2025-07-06

### Added
- Model detail inspection feature
- Custom directory support for non-HuggingFace models
- GitHub Actions CI/CD pipeline
- Automatic code formatting
- Comprehensive test suite
- Issue templates for GitHub

### Changed
- Significantly improved user interface
- Enhanced navigation system
- Better error handling and reporting

### Fixed
- Cross-platform test compatibility
- Various UI inconsistencies

## [0.1.0] - 2025-07-05

### Added
- Initial release of HF-MODEL-TOOL
- Core functionality for managing HuggingFace models and datasets
- Asset listing with size information
- Duplicate detection and cleanup
- Asset detail viewing
- Interactive CLI with rich terminal interface
- Configuration management
- Directory scanning for HuggingFace cache

### Features
- Smart asset detection for models and datasets
- Interactive menu navigation
- Sorting options (size, name, date)
- Configuration storage in `~/.config/hf-model-tool/`
- Logging to `~/.hf-model-tool.log`

## Development Timeline

### 2025-08-14
- Added API usage for getting downloaded model names
- Enhanced vLLM-CLI integration

### 2025-07-07
- Released version 0.2.1 with bug fixes
- Added command-line argument support
- Improved custom path detection
- Added LoRA adapter support

### 2025-07-06
- Released version 0.2.0 with major improvements
- Added CI/CD pipeline and testing

### 2025-07-05
- Initial commit and version 0.1.0 release
- Basic functionality implementation

[0.2.4]: https://github.com/Chen-zexi/hf-model-tool/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/Chen-zexi/hf-model-tool/compare/v0.2.1...v0.2.3
[0.2.1]: https://github.com/Chen-zexi/hf-model-tool/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/Chen-zexi/hf-model-tool/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Chen-zexi/hf-model-tool/releases/tag/v0.1.0