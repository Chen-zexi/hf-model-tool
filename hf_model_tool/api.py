#!/usr/bin/env python3
"""
Public API module for HF-MODEL-TOOL.

Provides programmatic access to HuggingFace model management functionality
for integration with other tools like VLLM.
"""
import logging
from typing import List, Optional, Set

from .cache import scan_all_directories

logger = logging.getLogger(__name__)


def get_downloaded_models(
    include_custom_models: bool = True,
    include_lora_adapters: bool = False,
    deduplicate: bool = True,
) -> List[str]:
    """
    Get a list of all downloaded HuggingFace models in VLLM-compatible naming format.
    This function scans all configured directories (default HuggingFace cache and any
    custom directories) and returns model names in the format expected by VLLM and
    other inference frameworks (e.g., "Qwen/Qwen3-30B-A3B-Instruct-2507").

    Args:
        include_custom_models: Whether to include custom/merged models from non-HF directories
        include_lora_adapters: Whether to include LoRA adapters in the results
        deduplicate: Whether to remove duplicate model names (default True)

    Returns:
        List of model names in VLLM-compatible format (e.g., "publisher/model-name")

    Examples:
        >>> from hf_model_tool import get_downloaded_models
        >>> models = get_downloaded_models()
        >>> print(models)
        ['bert-base-uncased', 'microsoft/Florence-2-large', 'facebook/bart-large-cnn']

        >>> # Include LoRA adapters
        >>> all_models = get_downloaded_models(include_lora_adapters=True)
    """
    try:
        # Scan all configured directories
        all_items = scan_all_directories()
        logger.info(f"Found {len(all_items)} total assets across all directories")

        models = []
        seen_models: Set[str] = set()

        for item in all_items:
            # Filter by asset type
            asset_type = item.get("type", "unknown")

            # Skip datasets and unknown types
            if asset_type == "dataset" or asset_type == "unknown":
                continue

            # Skip LoRA adapters if not requested
            if asset_type == "lora_adapter" and not include_lora_adapters:
                continue

            # Skip custom models if not requested
            if (
                asset_type in ["custom_model", "unknown_model"]
                and not include_custom_models
            ):
                continue

            # Extract model name in VLLM format
            model_name = _extract_vllm_model_name(item)

            if model_name:
                # Handle deduplication
                if deduplicate:
                    if model_name not in seen_models:
                        models.append(model_name)
                        seen_models.add(model_name)
                else:
                    models.append(model_name)

        logger.info(f"Returning {len(models)} model names in VLLM format")
        return sorted(models)

    except Exception as e:
        logger.error(f"Error getting downloaded models: {e}")
        return []


def _extract_vllm_model_name(item: dict) -> Optional[str]:
    """
    Extract model name in VLLM-compatible format from asset item.

    Converts HuggingFace cache naming convention to VLLM format:
    - "models--publisher--model-name" -> "publisher/model-name"
    - "models--model-name" -> "model-name" (no publisher)

    Args:
        item: Asset dictionary from cache scanning

    Returns:
        Model name in VLLM format, or None if extraction fails
    """
    name = item.get("name", "")
    source_type = item.get("source_type", "")

    # Handle HuggingFace cache format
    if source_type == "huggingface_cache" and name.startswith("models--"):
        # Remove "models--" prefix
        name_without_prefix = name[8:]  # len("models--") = 8

        # Split by "--" to get publisher and model name
        parts = name_without_prefix.split("--", 1)  # Split only on first "--"

        if len(parts) == 2:
            # Format: models--publisher--model-name
            publisher = parts[0]
            model_name = parts[1]
            # Replace remaining "--" in model name with "-" (some models have "--" in their names)
            model_name = model_name.replace("--", "-")
            return f"{publisher}/{model_name}"
        elif len(parts) == 1:
            # Format: models--model-name (no publisher)
            # Replace "--" with "-" in model name
            return parts[0].replace("--", "-")

    # Handle custom models and LoRA adapters
    elif source_type == "custom_directory":
        # For custom models, try to extract from metadata or use display name
        metadata = item.get("metadata", {})

        # For LoRA adapters, check if base model is specified
        if item.get("type") == "lora_adapter":
            base_model = metadata.get("base_model")
            if base_model and base_model != "unknown":
                # Return the base model name (already in correct format usually)
                return base_model

        # Otherwise use display name
        display_name = item.get("display_name", "")
        if display_name:
            # Clean up display name (remove timestamps for LoRA)
            if " (" in display_name and display_name.endswith(")"):
                # Remove timestamp suffix like " (2024-12-25 10:30)"
                display_name = display_name.split(" (")[0]
            return display_name

    # Fallback: try to use publisher and display_name if available
    publisher = item.get("publisher", "")
    display_name = item.get("display_name", "")

    if publisher and publisher != "unknown" and display_name:
        # Don't duplicate publisher if it's already in display_name
        if display_name.startswith(f"{publisher}/"):
            return display_name
        else:
            return f"{publisher}/{display_name}"
    elif display_name:
        return display_name

    return None


def get_model_info(model_name: str) -> Optional[dict]:
    """
    Get detailed information about a specific downloaded model.

    Args:
        model_name: Model name in VLLM format (e.g., "microsoft/Florence-2-large")

    Returns:
        Dictionary with model information including path, size, metadata, etc.
        Returns None if model not found.

    Examples:
        >>> info = get_model_info("bert-base-uncased")
        >>> print(info['path'])
        /home/user/.cache/huggingface/hub/models--bert-base-uncased
    """
    try:
        all_items = scan_all_directories()

        for item in all_items:
            # Skip non-model assets
            if item.get("type") in ["dataset", "unknown"]:
                continue

            # Check if this item matches the requested model
            extracted_name = _extract_vllm_model_name(item)
            if extracted_name == model_name:
                return {
                    "name": model_name,
                    "path": item.get("path", ""),
                    "size": item.get("size", 0),
                    "type": item.get("type", ""),
                    "subtype": item.get("subtype", ""),
                    "metadata": item.get("metadata", {}),
                    "source_directory": item.get("source_dir", ""),
                    "last_modified": item.get("date"),
                }

        logger.warning(f"Model not found: {model_name}")
        return None

    except Exception as e:
        logger.error(f"Error getting model info for {model_name}: {e}")
        return None
