# API Reference

## Overview

The HF-MODEL-TOOL provides a comprehensive Python API for programmatic access to all features. This reference covers the main `HFModelAPI` class and its methods.

## Installation

```python
from hf_model_tool.api import HFModelAPI

# Initialize the API
api = HFModelAPI()

# Or with custom config directory
api = HFModelAPI(config_dir="/path/to/config")
```

## Core Functions

### Quick Access Functions

These functions provide direct access without instantiating the API class:

#### `get_downloaded_models()`

Get a list of all downloaded models in VLLM-compatible format.

```python
from hf_model_tool import get_downloaded_models

models = get_downloaded_models(
    include_custom_models=True,  # Include custom/merged models
    include_lora_adapters=False, # Include LoRA adapters
    deduplicate=True             # Remove duplicates
)

# Returns: ['bert-base-uncased', 'microsoft/Florence-2-large', ...]
```

#### `get_model_info()`

Get detailed information about a specific model.

```python
from hf_model_tool import get_model_info

info = get_model_info("bert-base-uncased")
# Returns: {
#     'name': 'bert-base-uncased',
#     'path': '/home/user/.cache/huggingface/hub/models--bert-base-uncased',
#     'size': 440473133,
#     'type': 'model',
#     'metadata': {...}
# }
```

## HFModelAPI Class

### Initialization

```python
class HFModelAPI:
    def __init__(self, config_dir: Optional[Path] = None)
```

**Parameters:**
- `config_dir`: Optional custom configuration directory (default: `~/.config/hf-model-tool`)

### Asset Management Methods

#### `list_assets()`

List all discovered assets with filtering and sorting.

```python
def list_assets(
    sort_by: str = "size",
    asset_type: Optional[str] = None,
    include_lora: bool = True,
    include_datasets: bool = False,
    force_refresh: bool = False
) -> List[Dict[str, Any]]
```

**Parameters:**
- `sort_by`: Sort field ('size', 'name', 'date', 'type')
- `asset_type`: Filter by type ('model', 'lora_adapter', 'dataset', 'custom_model')
- `include_lora`: Include LoRA adapters in results
- `include_datasets`: Include datasets in results
- `force_refresh`: Force refresh of cached data

**Example:**
```python
# List all models sorted by size
models = api.list_assets(
    sort_by="size",
    asset_type="model",
    include_lora=False
)

for model in models[:5]:
    print(f"{model['name']}: {model['size'] / 1e9:.2f} GB")
```

#### `get_asset_details()`

Get detailed information about a specific asset.

```python
def get_asset_details(asset_id: str) -> Optional[Dict[str, Any]]
```

**Example:**
```python
details = api.get_asset_details("llama2-7b")
if details:
    print(f"Path: {details['path']}")
    print(f"Size: {details['size_human']}")
    print(f"Type: {details['type']}")
```

#### `delete_asset()`

Delete an asset from disk.

```python
def delete_asset(asset_id: str, confirm: bool = True) -> bool
```

**Example:**
```python
# Delete with confirmation
success = api.delete_asset("old-model-v1", confirm=True)
```

### Directory Management Methods

#### `add_directory()`

Add a new directory for scanning.

```python
def add_directory(path: str, dir_type: str = "auto") -> bool
```

**Parameters:**
- `path`: Directory path to add
- `dir_type`: Directory type ('huggingface', 'custom', 'lora', 'auto')

**Example:**
```python
# Add custom model directory
api.add_directory("/home/user/my-models", "custom")

# Add with auto-detection
api.add_directory("/data/ml-assets", "auto")
```

#### `remove_directory()`

Remove a directory from scanning.

```python
def remove_directory(path: str) -> bool
```

#### `list_directories()`

List all configured directories.

```python
def list_directories() -> List[Dict[str, str]]
```

**Example:**
```python
directories = api.list_directories()
for dir_info in directories:
    print(f"{dir_info['path']} - Type: {dir_info['type']}")
```

#### `scan_directories()`

Scan specific or all configured directories.

```python
def scan_directories(paths: Optional[List[str]] = None) -> List[Dict[str, Any]]
```

**Example:**
```python
# Scan specific directories
items = api.scan_directories(["/path1", "/path2"])

# Scan all configured directories
all_items = api.scan_directories()
```

### Manifest Management Methods

**Important Note**: When you add a custom directory using `add_directory()`, a manifest is automatically generated. This manifest:
- Is saved as `models_manifest.json` in the directory root
- Becomes the **primary source of truth** for model information
- Is **always read first** by both HF-MODEL-TOOL and vLLM-CLI
- Should be reviewed and edited to ensure accurate model names and publishers

#### `generate_manifest()`

Generate a manifest for models in a directory.

```python
def generate_manifest(directory: str) -> Optional[Dict[str, Any]]
```

**Example:**
```python
manifest = api.generate_manifest("/home/user/models")
if manifest:
    print(f"Generated manifest with {len(manifest['models'])} models")
```

#### `load_manifest()`

Load an existing manifest from a directory.

```python
def load_manifest(directory: str) -> Optional[Dict[str, Any]]
```

#### `save_manifest()`

Save a manifest to a directory.

```python
def save_manifest(directory: str, manifest: Dict[str, Any]) -> bool
```

**Example:**
```python
# Generate and customize manifest
manifest = api.generate_manifest("/path/to/models")
manifest["models"][0]["name"] = "Custom Name"
manifest["models"][0]["publisher"] = "MyOrg"

# Save the manifest
api.save_manifest("/path/to/models", manifest)
```

#### `update_manifest()`

Update specific entries in an existing manifest.

```python
def update_manifest(directory: str, updates: Dict[str, Any]) -> bool
```

**Example:**
```python
updates = {
    "/path/to/model1": {
        "name": "Updated Model Name",
        "publisher": "New Publisher",
        "notes": "Version 2.0"
    }
}

api.update_manifest("/path/to/models", updates)
```

#### `get_models_with_manifest()`

Get models from a directory with manifest data merged.

```python
def get_models_with_manifest(directory: str) -> List[Dict[str, Any]]
```

### LoRA Adapter Methods

#### `list_lora_adapters()`

List all discovered LoRA adapters.

```python
def list_lora_adapters() -> List[Dict[str, Any]]
```

#### `get_lora_details()`

Get detailed information about a specific LoRA adapter.

```python
def get_lora_details(lora_id: str) -> Optional[Dict[str, Any]]
```

#### `find_compatible_loras()`

Find LoRA adapters compatible with a specific model.

```python
def find_compatible_loras(model_name: str) -> List[Dict[str, Any]]
```

**Example:**
```python
# Find LoRAs for Llama2
loras = api.find_compatible_loras("llama2-7b")
for lora in loras:
    print(f"{lora['name']} - Base: {lora['base_model']}")
```

### Model Methods

#### `list_models()`

List all model names in VLLM-compatible format.

```python
def list_models(include_custom: bool = True) -> List[str]
```

**Example:**
```python
models = api.list_models(include_custom=True)
# Returns: ['bert-base-uncased', 'meta-llama/Llama-2-7b', ...]
```

#### `get_model_path()`

Get the local path for a model.

```python
def get_model_path(model_name: str) -> Optional[str]
```

### Utility Methods

#### `refresh_cache()`

Force refresh of the asset cache.

```python
def refresh_cache() -> None
```

#### `get_statistics()`

Get statistics about managed assets.

```python
def get_statistics() -> Dict[str, Any]
```

**Example:**
```python
stats = api.get_statistics()
print(f"Total models: {stats['total_models']}")
print(f"Total size: {stats['total_size'] / 1e9:.2f} GB")
print(f"Directories monitored: {stats['directories']}")
```

## Complete Examples

### Example 1: Model Inventory Report

```python
from hf_model_tool.api import HFModelAPI

api = HFModelAPI()

# Get all models
models = api.list_assets(asset_type="model", sort_by="size")

# Generate report
print("Model Inventory Report")
print("=" * 50)

total_size = 0
by_publisher = {}

for model in models:
    size = model.get('size', 0)
    total_size += size
    
    publisher = model.get('publisher', 'unknown')
    if publisher not in by_publisher:
        by_publisher[publisher] = []
    by_publisher[publisher].append(model)

print(f"Total models: {len(models)}")
print(f"Total size: {total_size / 1e9:.2f} GB")
print(f"Publishers: {len(by_publisher)}")

print("\nTop 5 Largest Models:")
for model in models[:5]:
    print(f"  - {model['name']}: {model['size'] / 1e9:.2f} GB")

print("\nModels by Publisher:")
for publisher, pub_models in sorted(by_publisher.items()):
    print(f"  {publisher}: {len(pub_models)} models")
```

### Example 2: Custom Directory Setup

```python
from hf_model_tool.api import HFModelAPI
from pathlib import Path

api = HFModelAPI()

# Setup project directories
project_root = Path("/home/user/ml-project")
directories = {
    project_root / "base-models": "custom",
    project_root / "fine-tuned": "custom",
    project_root / "lora-adapters": "custom"
}

# Add directories
for path, dir_type in directories.items():
    if path.exists():
        success = api.add_directory(str(path), dir_type)
        print(f"Added {path.name}: {success}")

# Generate manifests for each
for path in directories.keys():
    if path.exists():
        manifest = api.generate_manifest(str(path))
        if manifest:
            api.save_manifest(str(path), manifest)
            print(f"Generated manifest for {path.name}")

# List all assets
all_assets = api.list_assets()
print(f"\nTotal assets found: {len(all_assets)}")
```

### Example 3: LoRA Adapter Management

```python
from hf_model_tool.api import HFModelAPI

api = HFModelAPI()

# List all LoRA adapters
loras = api.list_lora_adapters()
print(f"Found {len(loras)} LoRA adapters")

# Group by base model
by_base_model = {}
for lora in loras:
    base = lora.get('metadata', {}).get('base_model', 'unknown')
    if base not in by_base_model:
        by_base_model[base] = []
    by_base_model[base].append(lora)

# Display organized view
for base_model, adapters in by_base_model.items():
    print(f"\nBase Model: {base_model}")
    for adapter in adapters:
        details = api.get_lora_details(adapter['name'])
        if details:
            print(f"  - {details['name']}")
            print(f"    Rank: {details.get('lora_rank', 'unknown')}")
            print(f"    Task: {details.get('task_type', 'unknown')}")
```

### Example 4: Duplicate Detection and Cleanup

```python
from hf_model_tool.api import HFModelAPI

api = HFModelAPI()

# Get all models (including duplicates)
all_models = get_downloaded_models(deduplicate=False)
unique_models = get_downloaded_models(deduplicate=True)

duplicates = len(all_models) - len(unique_models)
print(f"Found {duplicates} duplicate models")

# Find duplicate model paths
from collections import defaultdict
model_paths = defaultdict(list)

assets = api.list_assets(asset_type="model")
for asset in assets:
    name = asset.get('display_name', asset['name'])
    model_paths[name].append(asset['path'])

# Show duplicates
for name, paths in model_paths.items():
    if len(paths) > 1:
        print(f"\nDuplicate: {name}")
        for path in paths:
            asset = api.get_asset_details(path)
            if asset:
                print(f"  - {path}")
                print(f"    Size: {asset['size_human']}")
```

## Error Handling

All API methods include proper error handling and logging. Errors are logged to `~/.hf-model-tool.log`.

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

api = HFModelAPI()

# Methods return None or False on error
result = api.add_directory("/nonexistent/path", "custom")
if not result:
    print("Failed to add directory")

# Check logs for details
# tail -f ~/.hf-model-tool.log
```

## Performance Considerations

1. **Caching**: The API caches results for 60 seconds by default
2. **Force Refresh**: Use `force_refresh=True` to bypass cache
3. **Large Directories**: Scanning may be slow for directories with many files
4. **Network Paths**: Avoid scanning network-mounted directories when possible

## Thread Safety

The API is not thread-safe. For concurrent operations, create separate API instances or use locking:

```python
import threading

lock = threading.Lock()
api = HFModelAPI()

def scan_directory(path):
    with lock:
        return api.scan_directories([path])
```