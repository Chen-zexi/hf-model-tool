# Manifest System Documentation

## Overview

The manifest system in HF-MODEL-TOOL provides a way to customize and override metadata for models in your directories. It allows you to define custom names, publishers, types, and notes for your models, which will be displayed instead of the automatically detected values.

## What is a Manifest?

A manifest is a JSON file named `models_manifest.json` that resides in the root of your model directory. It contains metadata about the models in that directory and its subdirectories.

## Manifest File Structure

### Basic Structure

```json
{
  "version": "1.0",
  "created_at": "2024-01-15T10:30:00",
  "updated_at": "2024-01-15T10:30:00",
  "models": [
    {
      "path": "/path/to/model/directory",
      "name": "Custom Model Name",
      "publisher": "Organization Name",
      "type": "custom_model",
      "notes": "Optional notes about this model"
    }
  ]
}
```

### Field Descriptions

#### Root Fields
- `version`: Manifest schema version (currently "1.0")
- `created_at`: ISO 8601 timestamp of manifest creation
- `updated_at`: ISO 8601 timestamp of last update
- `models`: Array of model entries

#### Model Entry Fields
- `path`: Absolute or relative path to the model directory
- `name`: Custom display name for the model
- `publisher`: Organization or author name
- `type`: Model type (`model`, `custom_model`, `lora_adapter`)
- `notes`: Optional notes or description

## How the Manifest System Works

### Automatic Generation on Directory Addition

**Important**: When you add a new custom directory to HF-MODEL-TOOL, the system automatically:
1. Scans the directory for all models
2. Generates a `models_manifest.json` file with detected information
3. Saves it to the root of your directory

This manifest becomes the **primary source of truth** for model classification and display in both HF-MODEL-TOOL and vLLM-CLI.

### Discovery and Merging Process

1. **Manifest First**: The tool **always reads the manifest first** if it exists
2. **Model Discovery**: Then performs automatic detection for any new models
3. **Data Merging**: Manifest data takes precedence:
   - Manifest entries override automatic detection
   - Models not in the manifest use detected values
   - Path matching determines which manifest entry applies

### Priority Order

The system uses the following priority for model metadata:
1. **Manifest data** (highest priority - always used when available)
2. **Detected metadata** from model files
3. **Inferred data** from directory structure (lowest priority)

### Why Edit the Manifest?

While the auto-generated manifest works well, you should customize it to:
- **Ensure accurate model names** for display in vLLM-CLI
- **Specify correct publishers** for proper organization
- **Add meaningful descriptions** in the notes field
- **Correct any misclassified model types**

The manifest is the definitive source for how your models appear in all tools.

## Creating a Manifest

### Automatic Generation

The tool can automatically generate a manifest template:

```python
from hf_model_tool.api import HFModelAPI

api = HFModelAPI()
manifest = api.generate_manifest("/path/to/models")
api.save_manifest("/path/to/models", manifest)
```

### Manual Creation

Create a `models_manifest.json` file in your model directory:

```json
{
  "version": "1.0",
  "created_at": "2024-01-15T10:30:00",
  "updated_at": "2024-01-15T10:30:00",
  "models": [
    {
      "path": "test-gemma-custom",
      "name": "Test Gemma Custom",
      "publisher": "google",
      "type": "custom_model",
      "notes": "Fine-tuned Gemma model for testing"
    },
    {
      "path": "test-qwen3-custom",
      "name": "Test Qwen3 Custom",
      "publisher": "Qwen",
      "type": "custom_model",
      "notes": "Custom Qwen3 model variant"
    }
  ]
}
```

## Use Cases

### 1. Custom Naming for Fine-tuned Models

When you have fine-tuned models with generic folder names, use the manifest to give them meaningful names:

```json
{
  "models": [
    {
      "path": "checkpoint-5000",
      "name": "ChatBot v2.1 Final",
      "publisher": "MyCompany",
      "type": "custom_model",
      "notes": "Production-ready chatbot model"
    }
  ]
}
```

### 2. Organizing LoRA Adapters

For directories containing multiple LoRA adapters:

```json
{
  "models": [
    {
      "path": "lora-finance",
      "name": "Financial Domain LoRA",
      "publisher": "FinanceAI",
      "type": "lora_adapter",
      "notes": "Specialized for financial text analysis"
    },
    {
      "path": "lora-medical",
      "name": "Medical Terminology LoRA",
      "publisher": "MedAI",
      "type": "lora_adapter",
      "notes": "Trained on medical literature"
    }
  ]
}
```

### 3. Multi-Model Directories

For directories containing multiple model variants:

```json
{
  "models": [
    {
      "path": ".",
      "name": "Base Model Collection",
      "publisher": "Research Team",
      "type": "model",
      "notes": "Contains multiple model checkpoints"
    },
    {
      "path": "variant-a",
      "name": "Variant A - High Precision",
      "publisher": "Research Team",
      "type": "custom_model"
    },
    {
      "path": "variant-b",
      "name": "Variant B - Fast Inference",
      "publisher": "Research Team",
      "type": "custom_model"
    }
  ]
}
```

## Editing Manifests

### Via API

```python
from hf_model_tool.api import HFModelAPI

api = HFModelAPI()

# Update specific model entries
updates = {
    "/path/to/model": {
        "name": "Updated Model Name",
        "publisher": "New Publisher",
        "notes": "Updated description"
    }
}

api.update_manifest("/path/to/models", updates)
```

### Manual Editing

Simply edit the `models_manifest.json` file with any text editor. The changes will be picked up on the next scan.

## Best Practices

1. **Use Relative Paths**: When possible, use relative paths in the manifest for portability
2. **Keep It Updated**: Update the manifest when adding or removing models
3. **Version Control**: Include the manifest in version control with your models
4. **Consistent Naming**: Use consistent naming conventions across your manifests
5. **Document Changes**: Use the notes field to document model versions and changes

## Troubleshooting

### Manifest Not Being Applied

1. **Check File Name**: Ensure the file is named exactly `models_manifest.json`
2. **Validate JSON**: Ensure the JSON is valid (use a JSON validator)
3. **Check Paths**: Verify that paths in the manifest match actual directories
4. **Clear Cache**: Try clearing the tool's cache if changes aren't reflected

### Path Matching Issues

- Paths can be absolute or relative to the manifest location
- Use forward slashes (`/`) even on Windows
- Paths are case-sensitive on Linux/Mac

## Example: Complete Workflow

1. **Add a custom directory**:
```bash
hf-model-tool -path /home/user/my-models
```

2. **Generate initial manifest**:
```python
from hf_model_tool.api import HFModelAPI

api = HFModelAPI()
manifest = api.generate_manifest("/home/user/my-models")
```

3. **Customize the manifest**:
```python
# Modify the generated manifest
manifest["models"][0]["name"] = "Production Model v1.0"
manifest["models"][0]["publisher"] = "MyCompany"

# Save it back
api.save_manifest("/home/user/my-models", manifest)
```

4. **Verify the results**:
```bash
hf-model-tool -l
```

The models will now display with your custom names and metadata.