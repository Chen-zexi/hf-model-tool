# Custom Directories Configuration Guide

## Overview

HF-MODEL-TOOL supports scanning multiple directories for ML assets beyond the default HuggingFace cache. This guide explains how to configure and manage custom directories containing your models, LoRA adapters, and other ML assets.

## Directory Types

### HuggingFace Cache Directories

Standard HuggingFace cache directories with the following structure:
- `models--publisher--model-name/`
- `datasets--publisher--dataset-name/`
- Contains `blobs/` and `snapshots/` subdirectories

### Custom Model Directories

Directories containing:
- Fine-tuned models
- Merged models
- Custom model formats
- Models with `config.json` and safetensors/bin files

### LoRA Adapter Directories

Directories containing:
- LoRA adapters from training frameworks
- Files like `adapter_config.json` and `adapter_model.safetensors`
- Multiple adapters in subdirectories

## Adding Custom Directories

### Automatic Manifest Generation

**Important**: When you add a custom directory (not HuggingFace cache), the tool automatically:
1. Scans for all models in the directory
2. Generates a `models_manifest.json` file
3. Saves it to the directory root

This manifest file:
- Becomes the **primary source** for model information
- Is **always read first** by both HF-MODEL-TOOL and vLLM-CLI
- Can be edited to provide accurate names, publishers, and descriptions
- Ensures consistent model display across all tools

**Recommendation**: After adding a directory, review and edit the generated `models_manifest.json` to ensure model names and publishers are accurate for your use case.

### Method 1: Command Line Interface

Add a directory directly from the command line:

```bash
# Add a directory with auto-detection
hf-model-tool -path /path/to/models

# You'll be prompted to select the directory type:
# 1. HuggingFace Cache
# 2. Custom Directory  
# 3. Auto-detect
```

For custom directories, a manifest will be automatically generated.

### Method 2: Interactive Mode

1. Launch the tool in interactive mode:
```bash
hf-model-tool
```

2. Navigate to Configuration:
   - Select "Config" from any menu
   - Choose "Manage Cache Directories"
   - Select "Add Directory Path" or "Add Current Directory"

3. Choose directory type:
   - **HuggingFace Cache**: For standard HF cache structure
   - **Custom Directory**: For LoRA adapters and custom models
   - **Auto-detect**: Let the tool determine the type

### Method 3: Programmatic API

```python
from hf_model_tool.api import HFModelAPI

api = HFModelAPI()

# Add with auto-detection
api.add_directory("/path/to/models", "auto")

# Add as custom directory
api.add_directory("/path/to/lora-adapters", "custom")

# Add as HuggingFace cache
api.add_directory("/path/to/hf-cache", "huggingface")
```

## Directory Structure Examples

### Example 1: Custom Models Directory

```
/home/user/my-models/
├── models_manifest.json           # Optional manifest file
├── llama2-finance-tuned/         # Fine-tuned model
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── tokenizer_config.json
└── mistral-merged/               # Merged model
    ├── config.json
    └── model.safetensors
```

### Example 2: LoRA Adapters Directory

```
/home/user/lora-adapters/
├── models_manifest.json          # Optional manifest file
├── finance-lora/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
└── medical-lora/
    ├── adapter_config.json
    └── adapter_model.bin
```

### Example 3: Mixed Assets Directory

```
/home/user/ai-assets/
├── models_manifest.json
├── base-models/
│   ├── llama2-7b/
│   └── mistral-7b/
├── lora-adapters/
│   ├── task-specific-v1/
│   └── task-specific-v2/
└── fine-tuned/
    ├── custom-chatbot/
    └── domain-expert/
```

## Configuration File

Custom directories are stored in `~/.config/hf-model-tool/config.json`:

```json
{
  "custom_directories": [
    {
      "path": "/home/user/my-models",
      "type": "custom",
      "added_date": "2024-01-15T10:30:00"
    },
    {
      "path": "/data/shared-models",
      "type": "huggingface",
      "added_date": "2024-01-14T09:00:00"
    }
  ],
  "include_default_cache": true,
  "last_updated": "2024-01-15T10:30:00"
}
```

## Managing Directories

### Viewing Configured Directories

In interactive mode:
1. Go to Config → Manage Cache Directories
2. View the list of configured directories with their types

Via API:
```python
api = HFModelAPI()
directories = api.list_directories()
for dir_info in directories:
    print(f"{dir_info['path']} ({dir_info['type']})")
```

### Removing Directories

In interactive mode:
1. Go to Config → Manage Cache Directories
2. Select "Remove Directory"
3. Choose the directory to remove

Via API:
```python
api.remove_directory("/path/to/remove")
```

### Toggling Default Cache

Enable/disable scanning of the default HuggingFace cache:

```python
# Via configuration menu in interactive mode
# Or programmatically:
from hf_model_tool.config import ConfigManager

config = ConfigManager()
config.toggle_default_cache()
```

## Best Practices

### 1. Directory Organization

- **Group by Type**: Keep models, LoRA adapters, and datasets in separate directories
- **Use Descriptive Names**: Name directories clearly (e.g., `production-models`, `experiment-loras`)
- **Maintain Structure**: Keep consistent directory structures within each type

### 2. Performance Optimization

- **Avoid Deep Nesting**: Limit directory depth for faster scanning
- **Exclude Build Artifacts**: Don't add directories with build/cache files
- **Use Specific Paths**: Add specific model directories rather than entire project roots

### 3. Manifest Management

**Critical for vLLM-CLI Integration**:
- **Always review auto-generated manifests** after adding a directory
- **Edit model names** to be meaningful for your workflow
- **Specify correct publishers** for proper organization in vLLM-CLI
- **Add notes** to document model versions, training details, or usage

Remember: The manifest is read first and determines how models appear in vLLM-CLI's model selection menu.

### 4. Version Control

- **Always include** `models_manifest.json` in version control
- Document directory structure in README files
- Track model versions and changes in manifest notes

## Auto-Detection Logic

When using auto-detection, the tool determines directory type by:

1. **HuggingFace Cache Detection**:
   - Looks for `models--` or `datasets--` prefixed directories
   - Checks for `blobs/` subdirectories
   - >50% of subdirectories follow HF naming pattern

2. **Custom Directory Detection**:
   - Contains `config.json` files
   - Has model files (`.safetensors`, `.bin`, `.pt`)
   - Contains LoRA adapter files

## Troubleshooting

### Directory Not Found

```
Error: Directory does not exist: /path/to/models
```

**Solution**: Verify the path exists and is accessible:
```bash
ls -la /path/to/models
```

### No Assets Detected

```
Warning: Directory doesn't appear to contain custom assets
```

**Possible Causes**:
1. Wrong directory type selected
2. Models in deeper subdirectories than expected
3. Non-standard model format

**Solutions**:
- Try auto-detection instead of manual type selection
- Ensure models have `config.json` files
- Check that model files are in expected formats

### Permission Errors

```
Error: Permission denied accessing directory
```

**Solution**: Ensure read permissions:
```bash
chmod +r /path/to/models
```

### Slow Scanning

**Causes**:
- Very large directories
- Network-mounted filesystems
- Too many nested subdirectories

**Solutions**:
- Add specific model directories instead of parent directories
- Exclude unnecessary subdirectories
- Use local storage when possible

## Examples

### Adding Multiple Research Model Directories

```python
from hf_model_tool.api import HFModelAPI

api = HFModelAPI()

# Add various research directories
research_dirs = [
    "/research/baseline-models",
    "/research/fine-tuned",
    "/research/lora-experiments",
    "/research/production-ready"
]

for dir_path in research_dirs:
    api.add_directory(dir_path, "custom")
    print(f"Added: {dir_path}")

# List all models
models = api.list_models(include_custom=True)
print(f"Total models found: {len(models)}")
```

### Setting Up a Shared Team Directory

```bash
# 1. Add the shared directory
hf-model-tool -path /mnt/shared/team-models

# 2. Generate manifest for team models
python -c "
from hf_model_tool.api import HFModelAPI
api = HFModelAPI()
manifest = api.generate_manifest('/mnt/shared/team-models')
api.save_manifest('/mnt/shared/team-models', manifest)
"

# 3. View the models
hf-model-tool -l
```

### Organizing Personal Model Collection

```python
from pathlib import Path
from hf_model_tool.api import HFModelAPI

api = HFModelAPI()

# Set up personal model directories
home = Path.home()
model_dirs = {
    home / "models" / "base": "custom",
    home / "models" / "fine-tuned": "custom", 
    home / "models" / "lora": "custom",
    home / ".cache" / "huggingface" / "hub": "huggingface"
}

for path, dir_type in model_dirs.items():
    if path.exists():
        api.add_directory(str(path), dir_type)
        
# Get statistics
stats = api.get_statistics()
print(f"Total models: {stats.get('total_models', 0)}")
print(f"Total size: {stats.get('total_size', 0) / 1e9:.2f} GB")
```