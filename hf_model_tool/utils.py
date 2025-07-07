#!/usr/bin/env python3
"""
Utility functions for HF-MODEL-TOOL.

Provides asset grouping, duplicate detection, and data processing utilities
for managing HuggingFace models and datasets.
"""
import logging
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Set, FrozenSet, Any

logger = logging.getLogger(__name__)


def group_and_identify_duplicates(
    items: List[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, List[Dict[str, Any]]]], Set[FrozenSet[str]]]:
    """
    Group assets by category/publisher and identify duplicate assets.

    Supports multiple asset types:
    - models--publisher--model-name (HuggingFace models)
    - datasets--publisher--dataset-name (HuggingFace datasets)
    - Custom directory structures (LoRA adapters, custom models)

    Args:
        items: List of asset dictionaries from cache scanning

    Returns:
        Tuple containing:
        - Grouped assets by category and publisher/source
        - Set of duplicate asset name groups

    Raises:
        ValueError: If items list contains invalid asset structure
    """
    if not isinstance(items, list):
        raise ValueError("Items must be a list")

    logger.info(f"Processing {len(items)} assets for grouping and duplicate detection")

    # Group by potential duplicate keys for detection
    grouped_for_dupes = defaultdict(list)

    for item in items:
        if not isinstance(item, dict) or "name" not in item:
            logger.warning(f"Skipping invalid item: {item}")
            continue

        try:
            # Use display name for duplicate detection when available
            display_name = item.get("display_name", item["name"])
            asset_type = item.get("type", "unknown")
            source_type = item.get("source_type", "unknown")
            
            # For LoRA adapters from custom directories, don't treat as duplicates
            # since they may be different versions with timestamps
            if asset_type == "lora_adapter" and source_type == "custom_directory":
                # Use full path as unique key to avoid merging LoRA adapters
                key = (asset_type, item.get("path", item["name"]))
            elif source_type == "huggingface_cache":
                # For HuggingFace cache assets, require matching publisher, model name, AND file size
                publisher = item.get("publisher", "unknown")
                file_size = item.get("size", 0)
                key = (asset_type, publisher, display_name.lower(), file_size)
            else:
                # For other assets, use display name for duplicate detection
                key = (asset_type, display_name.lower())
            
            grouped_for_dupes[key].append(item["name"])
            
        except (AttributeError, IndexError) as e:
            logger.warning(
                f"Error processing asset name '{item.get('name', 'unknown')}': {e}"
            )
            continue

    # Identify sets of duplicates (same key with multiple versions)
    duplicate_sets = {frozenset(v) for v in grouped_for_dupes.values() if len(v) > 1}
    is_duplicate = {name for dup_set in duplicate_sets for name in dup_set}

    logger.info(
        f"Found {len(duplicate_sets)} duplicate sets affecting {len(is_duplicate)} assets"
    )

    # Group for display by category and publisher/source
    grouped_for_display: Dict[str, defaultdict] = {
        "models": defaultdict(list),
        "datasets": defaultdict(list),
        "lora_adapters": defaultdict(list),
        "custom_models": defaultdict(list),
        "unknown_models": defaultdict(list),
        "unknown": defaultdict(list),
    }

    for item in items:
        if not isinstance(item, dict) or "name" not in item or "type" not in item:
            logger.warning(f"Skipping invalid item for display grouping: {item}")
            continue

        try:
            asset_type = item["type"]
            display_name = item.get("display_name", item["name"])
            
            # Set display name and duplicate flag
            item["display_name"] = display_name
            item["is_duplicate"] = item["name"] in is_duplicate

            # Determine category and publisher/source
            category, publisher = _categorize_asset(item)
            
            # Ensure category exists
            if category not in grouped_for_display:
                logger.warning(f"Unknown asset category: {category}")
                category = "unknown"

            grouped_for_display[category][publisher].append(item)
            
        except (AttributeError, IndexError, KeyError) as e:
            logger.warning(
                f"Error processing asset for display: {item.get('name', 'unknown')}: {e}"
            )
            continue

    # Log summary
    total_counts = {}
    for category, publishers in grouped_for_display.items():
        count = sum(len(items) for items in publishers.values())
        total_counts[category] = count
    
    logger.info(f"Grouped assets: {total_counts}")

    # Convert defaultdict to regular dict for return type compatibility
    result_dict: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        category: dict(publishers)
        for category, publishers in grouped_for_display.items()
        if publishers  # Only include categories with items
    }

    return result_dict, duplicate_sets


def _categorize_asset(item: Dict[str, Any]) -> Tuple[str, str]:
    """
    Categorize an asset and determine its publisher/source.
    
    Args:
        item: Asset dictionary
        
    Returns:
        Tuple of (category, publisher/source)
    """
    asset_type = item.get("type", "unknown")
    name = item.get("name", "")
    source_type = item.get("source_type", "unknown")
    
    # Handle HuggingFace format assets with improved publisher extraction
    if source_type == "huggingface_cache" or "--" in name:
        parts = name.split("--")
        if len(parts) >= 2:
            publisher = parts[1]
            
            # Map asset types to categories
            if asset_type == "model":
                return "models", publisher
            elif asset_type == "dataset":
                return "datasets", publisher
        elif item.get("publisher"):
            # Use extracted publisher from HF metadata
            publisher = item["publisher"]
            if asset_type == "model":
                return "models", publisher
            elif asset_type == "dataset":
                return "datasets", publisher
    
    # Handle custom directory structures
    if asset_type == "lora_adapter":
        # For LoRA adapters, use base model as publisher if available
        base_model = item.get("metadata", {}).get("base_model", "unknown")
        if base_model != "unknown" and base_model != "custom":
            # Extract publisher from base model name
            if "/" in base_model:
                publisher = base_model.split("/")[0]
            else:
                publisher = base_model.split("--")[0] if "--" in base_model else "custom"
        else:
            publisher = "custom"
        return "lora_adapters", publisher
    
    elif asset_type == "custom_model":
        # For custom models, use directory parent as publisher
        path = Path(item.get("path", ""))
        publisher = path.parent.name if path.parent.name else "custom"
        return "custom_models", publisher
    
    elif asset_type == "unknown_model":
        # For unknown models, use directory parent as publisher
        path = Path(item.get("path", ""))
        publisher = path.parent.name if path.parent.name else "unknown"
        return "unknown_models", publisher
    
    # Default fallback
    return "unknown", "unknown"
