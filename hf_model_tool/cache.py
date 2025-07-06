#!/usr/bin/env python3
"""
Cache management module for HF-MODEL-TOOL.

Handles scanning and parsing of HuggingFace cache directories,
providing structured access to locally stored models and datasets.
"""
import logging
from datetime import datetime
from typing import List, Dict, Union
from pathlib import Path

logger = logging.getLogger(__name__)


def get_items(
    cache_dir: Union[str, Path]
) -> List[Dict[str, Union[str, int, datetime]]]:
    """
    Scan HuggingFace cache directory and return structured asset information.

    Args:
        cache_dir: Path to the HuggingFace cache directory

    Returns:
        List of dictionaries containing asset metadata:
        - name: Asset directory name
        - size: Total size in bytes
        - date: Last modification date
        - type: 'model' or 'dataset'
        - path: Absolute path to asset directory

    Raises:
        OSError: If cache directory is not accessible
        PermissionError: If insufficient permissions to read cache
    """
    if isinstance(cache_dir, str):
        cache_dir = Path(cache_dir)

    if not cache_dir.exists():
        logger.error(f"Cache directory does not exist: {cache_dir}")
        raise OSError(f"Cache directory not found: {cache_dir}")

    if not cache_dir.is_dir():
        logger.error(f"Cache path is not a directory: {cache_dir}")
        raise OSError(f"Cache path is not a directory: {cache_dir}")

    items: List[Dict[str, Union[str, int, datetime]]] = []
    logger.info(f"Scanning cache directory: {cache_dir}")

    try:
        for item_dir in cache_dir.iterdir():
            if not item_dir.is_dir():
                continue

            try:
                blobs_path = item_dir / "blobs"
                size: int = 0

                if blobs_path.is_dir():
                    try:
                        # Calculate total size of all blobs
                        for blob_file in blobs_path.iterdir():
                            if blob_file.is_file():
                                size += blob_file.stat().st_size
                    except (OSError, PermissionError) as e:
                        logger.warning(f"Error accessing blobs in {item_dir.name}: {e}")
                        continue

                # Only include items with actual content
                if size > 0:
                    try:
                        mod_time: datetime = datetime.fromtimestamp(
                            item_dir.stat().st_mtime
                        )
                    except OSError:
                        mod_time = datetime.now()
                        logger.warning(
                            f"Could not get modification time for {item_dir.name}"
                        )

                    asset_type: str = (
                        "dataset"
                        if item_dir.name.lower().startswith("datasets--")
                        else "model"
                    )

                    item_dict: Dict[str, Union[str, int, datetime]] = {
                        "name": item_dir.name,
                        "size": size,
                        "date": mod_time,
                        "type": asset_type,
                        "path": str(item_dir),
                    }
                    items.append(item_dict)

            except (OSError, PermissionError) as e:
                logger.warning(f"Error processing {item_dir.name}: {e}")
                continue

    except (OSError, PermissionError) as e:
        logger.error(f"Error reading cache directory {cache_dir}: {e}")
        raise

    logger.info(f"Found {len(items)} assets in cache")
    return items
