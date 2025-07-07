#!/usr/bin/env python3
"""
Cache management module for HF-MODEL-TOOL.

Handles scanning and parsing of HuggingFace cache directories,
providing structured access to locally stored models and datasets.
"""
import logging
from datetime import datetime
from typing import List, Dict, Union, Set, Any
from pathlib import Path

from .config import ConfigManager
from .asset_detector import AssetDetector

logger = logging.getLogger(__name__)


def get_huggingface_items(
    cache_dir: Union[str, Path],
) -> List[Dict[str, Union[str, int, datetime]]]:
    """
    Scan HuggingFace cache directory using optimized HF-specific logic.

    Args:
        cache_dir: Path to HuggingFace cache directory

    Returns:
        List of dictionaries containing HuggingFace asset metadata

    Raises:
        OSError: If cache directory is not accessible
        PermissionError: If insufficient permissions to read cache
    """
    if isinstance(cache_dir, str):
        cache_dir = Path(cache_dir)

    if not cache_dir.exists():
        logger.error(f"HuggingFace cache directory does not exist: {cache_dir}")
        raise OSError(f"HuggingFace cache directory not found: {cache_dir}")

    if not cache_dir.is_dir():
        logger.error(f"HuggingFace cache path is not a directory: {cache_dir}")
        raise OSError(f"HuggingFace cache path is not a directory: {cache_dir}")

    items: List[Dict[str, Union[str, int, datetime]]] = []
    logger.info(f"Scanning HuggingFace cache directory: {cache_dir}")

    try:
        for item_dir in cache_dir.iterdir():
            if not item_dir.is_dir():
                continue

            try:
                # Skip non-HuggingFace format directories
                if not item_dir.name.startswith(("models--", "datasets--")):
                    continue
                    
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

                    # Parse HuggingFace naming convention
                    parts = item_dir.name.split("--")
                    asset_type = "dataset" if parts[0] == "datasets" else "model"
                    
                    # Extract publisher and model name
                    if len(parts) >= 3:
                        publisher = parts[1]
                        model_name = "--".join(parts[2:])
                        display_name = model_name
                    elif len(parts) == 2:
                        publisher = parts[1]
                        display_name = publisher
                    else:
                        publisher = "unknown"
                        display_name = item_dir.name

                    # Try to extract additional metadata from config files
                    metadata = _extract_hf_metadata(item_dir, asset_type)
                    
                    item_dict: Dict[str, Union[str, int, datetime]] = {
                        "name": item_dir.name,
                        "size": size,
                        "date": mod_time,
                        "type": asset_type,
                        "subtype": "huggingface",
                        "metadata": metadata,
                        "display_name": display_name,
                        "publisher": publisher,
                        "path": str(item_dir),
                        "source_type": "huggingface_cache"
                    }
                    items.append(item_dict)

            except (OSError, PermissionError) as e:
                logger.warning(f"Error processing {item_dir.name}: {e}")
                continue

    except (OSError, PermissionError) as e:
        logger.error(f"Error reading HuggingFace cache directory {cache_dir}: {e}")
        raise

    logger.info(f"Found {len(items)} HuggingFace assets")
    return items


def get_custom_items(
    custom_dir: Union[str, Path],
) -> List[Dict[str, Union[str, int, datetime]]]:
    """
    Scan custom directory using flexible detection for various model formats.

    Args:
        custom_dir: Path to custom directory containing ML assets

    Returns:
        List of dictionaries containing custom asset metadata

    Raises:
        OSError: If directory is not accessible
        PermissionError: If insufficient permissions to read directory
    """
    if isinstance(custom_dir, str):
        custom_dir = Path(custom_dir)

    if not custom_dir.exists():
        logger.error(f"Custom directory does not exist: {custom_dir}")
        raise OSError(f"Custom directory not found: {custom_dir}")

    if not custom_dir.is_dir():
        logger.error(f"Custom path is not a directory: {custom_dir}")
        raise OSError(f"Custom path is not a directory: {custom_dir}")

    items: List[Dict[str, Union[str, int, datetime]]] = []
    logger.info(f"Scanning custom directory: {custom_dir}")
    
    detector = AssetDetector()

    try:
        for item_dir in custom_dir.iterdir():
            if not item_dir.is_dir():
                continue

            try:
                # Use flexible asset detection for custom directories
                asset_info = detector.detect_asset_type(item_dir)
                
                # Skip if no content found
                if asset_info["size"] == 0:
                    logger.debug(f"Skipping empty directory: {item_dir.name}")
                    continue
                
                # Get modification date
                mod_time = detector.get_modification_date(item_dir)
                
                # Add timestamp to display name for LoRA adapters to help with duplicates
                display_name = asset_info.get("display_name", item_dir.name)
                if asset_info["type"] == "lora_adapter":
                    time_str = mod_time.strftime("%Y-%m-%d %H:%M")
                    display_name = f"{display_name} ({time_str})"
                
                # Create item dictionary with enhanced information
                item_dict: Dict[str, Union[str, int, datetime]] = {
                    "name": asset_info.get("display_name", item_dir.name),
                    "size": asset_info["size"],
                    "date": mod_time,
                    "type": asset_info["type"],
                    "subtype": asset_info.get("subtype", "custom"),
                    "metadata": asset_info.get("metadata", {}),
                    "display_name": display_name,
                    "path": str(item_dir),
                    "files": asset_info.get("files", []),
                    "source_type": "custom_directory"
                }
                
                # Add LoRA-specific path if available
                if "lora_path" in asset_info:
                    item_dict["lora_path"] = asset_info["lora_path"]
                    
                items.append(item_dict)
                
                logger.debug(f"Detected {asset_info['type']} custom asset: {item_dir.name}")

            except (OSError, PermissionError) as e:
                logger.warning(f"Error processing {item_dir.name}: {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error processing {item_dir.name}: {e}")
                continue

    except (OSError, PermissionError) as e:
        logger.error(f"Error reading custom directory {custom_dir}: {e}")
        raise

    logger.info(f"Found {len(items)} custom assets")
    return items


def get_items(
    cache_dir: Union[str, Path],
    path_type: str = "auto"
) -> List[Dict[str, Union[str, int, datetime]]]:
    """
    Scan directory and return structured asset information.
    
    This is a wrapper function that delegates to the appropriate scanner
    based on path type.

    Args:
        cache_dir: Path to directory containing ML assets
        path_type: Type of path - "huggingface", "custom", or "auto"

    Returns:
        List of dictionaries containing asset metadata
    """
    if path_type == "huggingface":
        return get_huggingface_items(cache_dir)
    elif path_type == "custom":
        return get_custom_items(cache_dir)
    else:  # auto-detect
        cache_path = Path(cache_dir)
        # Check if it looks like a HuggingFace cache directory
        if _is_huggingface_cache(cache_path):
            return get_huggingface_items(cache_dir)
        else:
            return get_custom_items(cache_dir)


def _extract_hf_metadata(item_dir: Path, asset_type: str) -> Dict[str, Any]:
    """
    Extract metadata from HuggingFace cache files.
    
    Args:
        item_dir: Path to HuggingFace asset directory
        asset_type: Type of asset ("model" or "dataset")
        
    Returns:
        Dictionary of extracted metadata
    """
    metadata = {}
    
    # Look for config files in snapshots
    snapshots_dir = item_dir / "snapshots"
    if snapshots_dir.exists():
        # Get the most recent snapshot
        try:
            snapshot_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
            if snapshot_dirs:
                # Sort by modification time, get most recent
                latest_snapshot = max(snapshot_dirs, key=lambda d: d.stat().st_mtime)
                
                if asset_type == "model":
                    config_file = latest_snapshot / "config.json"
                    if config_file.exists():
                        try:
                            import json
                            with open(config_file, 'r') as f:
                                config = json.load(f)
                                metadata.update({
                                    "model_type": config.get("model_type", "unknown"),
                                    "architectures": config.get("architectures", []),
                                    "torch_dtype": config.get("torch_dtype"),
                                    "vocab_size": config.get("vocab_size"),
                                    "snapshot_commit": latest_snapshot.name
                                })
                        except (json.JSONDecodeError, OSError) as e:
                            logger.debug(f"Error reading HF model config: {e}")
                            
                elif asset_type == "dataset":
                    readme_file = latest_snapshot / "README.md"
                    dataset_info = latest_snapshot / "dataset_info.json"
                    if dataset_info.exists():
                        try:
                            import json
                            with open(dataset_info, 'r') as f:
                                info = json.load(f)
                                metadata.update({
                                    "dataset_size": info.get("dataset_size"),
                                    "features": info.get("features", {}),
                                    "snapshot_commit": latest_snapshot.name
                                })
                        except (json.JSONDecodeError, OSError) as e:
                            logger.debug(f"Error reading HF dataset info: {e}")
                            
        except OSError as e:
            logger.debug(f"Error accessing snapshots in {item_dir}: {e}")
            
    return metadata


def _is_huggingface_cache(directory: Path) -> bool:
    """
    Determine if a directory is a HuggingFace cache directory.
    
    Args:
        directory: Directory to check
        
    Returns:
        True if it appears to be a HuggingFace cache directory
    """
    if not directory.exists() or not directory.is_dir():
        return False
        
    # Check for HuggingFace naming patterns
    hf_pattern_count = 0
    total_dirs = 0
    
    try:
        for item in directory.iterdir():
            if item.is_dir():
                total_dirs += 1
                if item.name.startswith(("models--", "datasets--")):
                    hf_pattern_count += 1
                    
        # If more than 50% of directories follow HF naming, consider it HF cache
        if total_dirs > 0 and (hf_pattern_count / total_dirs) > 0.5:
            return True
            
    except OSError:
        pass
        
    return False


def scan_all_directories() -> List[Dict[str, Union[str, int, datetime]]]:
    """
    Scan all configured directories for HuggingFace assets.

    Merges results from default cache and custom directories,
    removing duplicates based on asset name.

    Returns:
        Consolidated list of assets from all directories
    """
    config_manager = ConfigManager()
    all_directories = config_manager.get_all_directories_with_types()

    all_items: List[Dict[str, Union[str, int, datetime]]] = []
    seen_names: Set[str] = set()

    logger.info(f"Scanning {len(all_directories)} directories for assets")

    for directory_info in all_directories:
        directory = directory_info["path"]
        path_type = directory_info["type"]
        
        try:
            items = get_items(directory, path_type)

            # Add source directory to each item and handle duplicates differently for custom vs HF
            for item in items:
                item_name = str(item["name"])
                
                # For custom directories (especially LoRA), don't merge duplicates
                # For HuggingFace cache, merge duplicates by name
                should_add = True
                if path_type == "huggingface":
                    if item_name in seen_names:
                        should_add = False
                        logger.debug(f"Skipping duplicate HF asset: {item_name} from {directory}")
                # For custom directories, always add (they have timestamps to differentiate)
                
                if should_add:
                    item["source_dir"] = directory
                    all_items.append(item)
                    if path_type == "huggingface":
                        seen_names.add(item_name)

        except (OSError, PermissionError) as e:
            logger.warning(f"Failed to scan directory {directory}: {e}")
            continue

    logger.info(f"Found {len(all_items)} assets across all directories")
    return all_items
