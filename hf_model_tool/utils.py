#!/usr/bin/env python3
"""
Utility functions for HF-MODEL-TOOL.

Provides asset grouping, duplicate detection, and data processing utilities
for managing HuggingFace models and datasets.
"""
import logging
from collections import defaultdict
from typing import List, Dict, Tuple, Set, FrozenSet, Any

logger = logging.getLogger(__name__)


def group_and_identify_duplicates(
    items: List[Dict[str, Any]]
) -> Tuple[Dict[str, Dict[str, List[Dict[str, Any]]]], Set[FrozenSet[str]]]:
    """
    Group assets by category/publisher and identify duplicate assets.

    HuggingFace assets follow the naming pattern:
    - models--publisher--model-name
    - datasets--publisher--dataset-name

    Args:
        items: List of asset dictionaries from cache scanning

    Returns:
        Tuple containing:
        - Grouped assets by category (models/datasets) and publisher
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
            parts = item["name"].lower().split("--")
            if len(parts) > 2:
                # Create key from type, publisher, and base name for duplicate detection
                key = (parts[0], parts[1], "--".join(parts[2:]))
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

    # Group for display by category and publisher
    grouped_for_display: Dict[str, defaultdict] = {
        "models": defaultdict(list),
        "datasets": defaultdict(list),
    }

    for item in items:
        if not isinstance(item, dict) or "name" not in item or "type" not in item:
            logger.warning(f"Skipping invalid item for display grouping: {item}")
            continue

        try:
            parts = item["name"].split("--")
            if len(parts) > 1:
                publisher = parts[1]

                # Create user-friendly display name
                item["display_name"] = (
                    "--".join(parts[2:]) if len(parts) > 2 else item["name"]
                )
                item["is_duplicate"] = item["name"] in is_duplicate

                # Ensure category exists
                category = (
                    item["type"] + "s"
                )  # "model" -> "models", "dataset" -> "datasets"
                if category not in grouped_for_display:
                    logger.warning(f"Unknown asset type: {item['type']}")
                    continue

                grouped_for_display[category][publisher].append(item)
            else:
                logger.warning(
                    f"Asset name doesn't follow expected pattern: {item['name']}"
                )

        except (AttributeError, IndexError, KeyError) as e:
            logger.warning(
                f"Error processing asset for display: {item.get('name', 'unknown')}: {e}"
            )
            continue

    # Log summary
    models_count = sum(len(items) for items in grouped_for_display["models"].values())
    datasets_count = sum(
        len(items) for items in grouped_for_display["datasets"].values()
    )
    logger.info(f"Grouped assets: {models_count} models, {datasets_count} datasets")

    # Convert defaultdict to regular dict for return type compatibility
    result_dict: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        category: dict(publishers)
        for category, publishers in grouped_for_display.items()
    }

    return result_dict, duplicate_sets
