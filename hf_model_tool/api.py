#!/usr/bin/env python3
"""
Public API module for HF-MODEL-TOOL.

Provides programmatic access to HuggingFace model management functionality
for integration with other tools like VLLM.
"""
import logging
import shutil
from pathlib import Path
from typing import List, Optional, Set, Dict, Any, Union

from .cache import scan_all_directories, get_custom_items
from .config import ConfigManager
from .asset_detector import AssetDetector
from .registry import get_registry, ModelRegistry
from .manifest import ManifestHandler
from .ollama import get_ollama_items, get_ollama_model_info

logger = logging.getLogger(__name__)


def _should_include_asset(
    item: Dict[str, Any],
    include_custom_models: bool,
    include_lora_adapters: bool,
    include_ollama: bool,
) -> bool:
    """
    Determine if an asset should be included based on filter criteria.

    Args:
        item: Asset dictionary from scanning
        include_custom_models: Whether to include custom/merged models
        include_lora_adapters: Whether to include LoRA adapters
        include_ollama: Whether to include Ollama/GGUF models

    Returns:
        True if asset should be included, False otherwise
    """
    asset_type = item.get("type", "unknown")

    # Always skip datasets and unknown types
    if asset_type in ["dataset", "unknown"]:
        return False

    # Check LoRA adapter filter
    if asset_type == "lora_adapter" and not include_lora_adapters:
        return False

    # Check custom model filter
    if asset_type in ["custom_model", "unknown_model"] and not include_custom_models:
        return False

    # Check Ollama/GGUF model filter
    if asset_type in ["ollama_model", "gguf_model"] and not include_ollama:
        return False

    return True


def get_downloaded_models(
    include_custom_models: bool = True,
    include_lora_adapters: bool = False,
    include_ollama: Optional[bool] = None,
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
        include_ollama: Whether to include Ollama models (GGUF format). If None, uses config setting
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
        # If include_ollama is None, check config setting
        if include_ollama is None:
            config_manager = ConfigManager()
            config = config_manager.load_config()
            include_ollama = config.get("scan_ollama", False) or bool(
                config.get("ollama_directories", [])
            )

        # Scan all configured directories
        all_items = scan_all_directories()
        logger.info(f"Found {len(all_items)} total assets across all directories")

        models = []
        seen_models: Set[str] = set()

        for item in all_items:
            # Check if asset should be included based on filters
            if not _should_include_asset(
                item, include_custom_models, include_lora_adapters, include_ollama
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

    # Handle Ollama models
    elif source_type == "ollama":
        # Ollama models use format like "gpt-oss:120b"
        # For vLLM, we return the path to the GGUF file
        return item.get("path", name)

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


def get_ollama_models() -> List[Dict[str, Any]]:
    """
    Get all Ollama models discovered on the system.

    Returns:
        List of Ollama model dictionaries with metadata
    """
    try:
        models = get_ollama_items()
        logger.info(f"Found {len(models)} Ollama models")
        return models
    except Exception as e:
        logger.error(f"Error getting Ollama models: {e}")
        return []


def get_ollama_model_path(model_name: str) -> Optional[str]:
    """
    Get the path to an Ollama model's GGUF file.

    Args:
        model_name: Name of the Ollama model (e.g., "llama2:7b")

    Returns:
        Path to the GGUF file or None if not found
    """
    model_info = get_ollama_model_info(model_name)
    if model_info:
        return model_info.get("path")
    return None


class HFModelAPI:
    """
    Comprehensive API for HuggingFace model management.

    Provides a unified interface for managing models, LoRA adapters,
    and directories for integration with tools like vLLM CLI.
    """

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize the HF Model API.

        Args:
            config_dir: Optional custom config directory
        """
        self.config_manager = ConfigManager(config_dir)
        self.asset_detector = AssetDetector()
        self.registry = get_registry()  # Use shared registry
        self.manifest_handler = ManifestHandler()
        self._cache: Optional[List[Dict[str, Any]]] = None
        self._cache_timestamp: float = 0
        self._cache_ttl: float = 60  # Cache for 60 seconds

    # Asset Management Methods

    def list_assets(
        self,
        sort_by: str = "size",
        asset_type: Optional[str] = None,
        include_lora: bool = True,
        include_datasets: bool = False,
        force_refresh: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        List all discovered assets with filtering and sorting.

        Args:
            sort_by: Sort field ('size', 'name', 'date', 'type')
            asset_type: Filter by type ('model', 'lora_adapter', 'dataset', 'custom_model')
            include_lora: Include LoRA adapters in results
            include_datasets: Include datasets in results
            force_refresh: Force refresh of cached data

        Returns:
            List of asset dictionaries with full metadata
        """
        # Use registry for unified scanning
        self.registry.scan_all(force=force_refresh)

        # Get all assets from registry
        all_items = []

        if asset_type == "model":
            all_items = list(self.registry.models.values())
        elif asset_type == "lora_adapter":
            all_items = list(self.registry.lora_adapters.values())
        elif asset_type == "dataset":
            all_items = list(self.registry.datasets.values())
        elif asset_type == "custom_model":
            all_items = list(self.registry.custom_models.values())
        elif asset_type == "ollama_model":
            all_items = list(self.registry.ollama_models.values())
        elif asset_type == "gguf_model":
            all_items = list(self.registry.gguf_models.values())
        else:
            # Get all types based on filters
            if asset_type is None:
                all_items.extend(self.registry.models.values())
                all_items.extend(self.registry.custom_models.values())
                all_items.extend(self.registry.ollama_models.values())
                all_items.extend(self.registry.gguf_models.values())
                if include_lora:
                    all_items.extend(self.registry.lora_adapters.values())
                if include_datasets:
                    all_items.extend(self.registry.datasets.values())

        # Sort results
        if sort_by == "size":
            all_items.sort(key=lambda x: x.get("size", 0), reverse=True)
        elif sort_by == "name":
            all_items.sort(key=lambda x: x.get("name", ""))
        elif sort_by == "date":
            all_items.sort(key=lambda x: x.get("date", ""), reverse=True)
        elif sort_by == "type":
            all_items.sort(key=lambda x: x.get("type", ""))

        return all_items

    def get_asset_details(self, asset_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific asset.

        Args:
            asset_id: Asset identifier (name or path)

        Returns:
            Dictionary with full asset details or None if not found
        """
        # Use registry to find asset
        self.registry.scan_all()
        asset = self.registry.find_asset(asset_id)

        if asset:
            return self._enrich_asset_details(asset)

        logger.warning(f"Asset not found: {asset_id}")
        return None

    def delete_asset(self, asset_id: str, confirm: bool = True) -> bool:
        """
        Delete an asset from disk.

        Args:
            asset_id: Asset identifier (name or path)
            confirm: Whether to require confirmation (for safety)

        Returns:
            True if asset was deleted successfully
        """
        asset = self.get_asset_details(asset_id)
        if not asset:
            logger.error(f"Cannot delete: asset not found: {asset_id}")
            return False

        asset_path = Path(asset.get("path", ""))
        if not asset_path.exists():
            logger.error(f"Cannot delete: path does not exist: {asset_path}")
            return False

        # Safety check
        if confirm:
            logger.info(f"Delete confirmation required for: {asset_path}")
            # In a real implementation, this would prompt the user
            # For API usage, the caller should handle confirmation

        try:
            if asset_path.is_dir():
                shutil.rmtree(asset_path)
            else:
                asset_path.unlink()

            logger.info(f"Successfully deleted asset: {asset_path}")

            # Clear cache to force refresh
            self._cache = None

            return True
        except Exception as e:
            logger.error(f"Failed to delete asset {asset_path}: {e}")
            return False

    # Directory Management Methods

    def scan_directories(
        self, paths: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Scan specified or all configured directories.

        Args:
            paths: Optional list of specific paths to scan

        Returns:
            List of discovered assets from scanned directories
        """
        if paths:
            # Scan specific paths
            items = []
            for path in paths:
                path_obj = Path(path)
                if not path_obj.exists():
                    logger.warning(f"Path does not exist: {path}")
                    continue

                # Detect and scan the path
                asset_info = self.asset_detector.detect_asset_type(path_obj)
                if asset_info:
                    items.append(asset_info)

                # Also scan subdirectories
                if path_obj.is_dir():
                    for subdir in path_obj.iterdir():
                        if subdir.is_dir():
                            sub_asset = self.asset_detector.detect_asset_type(subdir)
                            if sub_asset and sub_asset.get("type") != "unknown":
                                items.append(sub_asset)

            return items
        else:
            # Scan all configured directories
            return scan_all_directories()

    def add_directory(self, path: str, dir_type: str = "auto") -> bool:
        """
        Add a new directory for scanning.

        Args:
            path: Directory path to add
            dir_type: Directory type ('huggingface', 'custom', 'lora', 'auto')

        Returns:
            True if directory was added successfully
        """
        return self.config_manager.add_directory(path, dir_type)

    def remove_directory(self, path: str) -> bool:
        """
        Remove a directory from scanning.

        Args:
            path: Directory path to remove

        Returns:
            True if directory was removed successfully
        """
        return self.config_manager.remove_directory(path)

    def list_directories(self) -> List[Dict[str, str]]:
        """
        List all configured directories.

        Returns:
            List of directory information dictionaries
        """
        return self.config_manager.get_all_directories_with_types()

    # Ollama-specific convenience methods

    def toggle_ollama_scanning(self) -> bool:
        """
        Toggle Ollama model scanning on/off.

        Returns:
            New state of Ollama scanning (True if enabled, False if disabled)
        """
        return self.config_manager.toggle_ollama_scanning()

    def add_ollama_directory(self, path: str) -> bool:
        """
        Add a custom Ollama directory for scanning.

        Args:
            path: Path to Ollama models directory

        Returns:
            True if directory was added successfully
        """
        return self.config_manager.add_ollama_directory(path)

    def remove_ollama_directory(self, path: str) -> bool:
        """
        Remove a custom Ollama directory from scanning.

        Args:
            path: Path to Ollama directory to remove

        Returns:
            True if directory was removed successfully
        """
        return self.config_manager.remove_ollama_directory(path)

    def get_ollama_status(self) -> Dict[str, Any]:
        """
        Get current Ollama configuration status.

        Returns:
            Dictionary with Ollama scanning status and configured directories
        """
        config = self.config_manager.load_config()
        return {
            "scan_enabled": config.get("scan_ollama", False),
            "custom_directories": config.get("ollama_directories", []),
            "default_directories": [
                str(Path.home() / ".ollama" / "models"),
                "/usr/share/ollama/.ollama/models",
                "/var/lib/ollama/.ollama/models",
            ],
        }

    # LoRA Adapter Methods

    def list_lora_adapters(self) -> List[Dict[str, Any]]:
        """
        List all discovered LoRA adapters.

        Returns:
            List of LoRA adapter dictionaries with metadata
        """
        self.registry.scan_all()
        return list(self.registry.lora_adapters.values())

    def get_lora_details(self, lora_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific LoRA adapter.

        Args:
            lora_id: LoRA adapter identifier

        Returns:
            Dictionary with LoRA details or None if not found
        """
        lora_adapters = self.list_lora_adapters()

        for adapter in lora_adapters:
            if (
                adapter.get("name") == lora_id
                or adapter.get("display_name") == lora_id
                or adapter.get("path") == lora_id
            ):
                return self._enrich_lora_details(adapter)

        return None

    def find_compatible_loras(self, model_name: str) -> List[Dict[str, Any]]:
        """
        Find LoRA adapters compatible with a specific model.

        Args:
            model_name: Base model name to check compatibility for

        Returns:
            List of compatible LoRA adapters
        """
        # Use registry's compatibility checking
        return self.registry.get_loras_for_model(model_name)

    # Model-specific convenience methods

    def list_models(self, include_custom: bool = True) -> List[str]:
        """
        List all model names in VLLM-compatible format.

        Args:
            include_custom: Include custom models

        Returns:
            List of model names (e.g., "publisher/model-name")
        """
        return get_downloaded_models(
            include_custom_models=include_custom, include_lora_adapters=False
        )

    def get_model_path(self, model_name: str) -> Optional[str]:
        """
        Get the local path for a model.

        Args:
            model_name: Model name in VLLM format

        Returns:
            Path to model directory or None if not found
        """
        info = get_model_info(model_name)
        return info.get("path") if info else None

    # Utility Methods

    def refresh_cache(self) -> None:
        """Force refresh of the asset cache."""
        self._cache = None
        self._cache_timestamp = 0
        logger.info("Asset cache cleared, will refresh on next access")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about managed assets.

        Returns:
            Dictionary with statistics (counts, sizes, etc.)
        """
        # Use registry statistics
        stats = self.registry.get_statistics()

        # Add API-specific stats
        stats["dataset_count"] = stats.pop("datasets_count", 0)
        stats["directories"] = stats.pop("directories_monitored", 0)

        return stats

    # Manifest Management Methods

    def generate_manifest(self, directory: str) -> Optional[Dict[str, Any]]:
        """
        Generate a manifest for models in a directory.

        Args:
            directory: Path to directory containing models

        Returns:
            Generated manifest dictionary or None if generation failed
        """
        try:
            dir_path = Path(directory)
            if not dir_path.exists():
                logger.error(f"Directory does not exist: {directory}")
                return None

            # Use get_custom_items to discover models
            items = get_custom_items(dir_path)

            if not items:
                logger.info(f"No models found in {directory}")
                return None

            # Convert to manifest format
            discovered_models = []
            for item in items:
                discovered_models.append(
                    {
                        "path": item.get("path", str(dir_path / item["name"])),
                        "name": item["name"],
                        "type": item.get("type", "custom_model"),
                        "size": item.get("size", 0),
                        "metadata": item.get("metadata", {}),
                        "display_name": item.get("display_name", item["name"]),
                    }
                )

            # Generate manifest
            manifest = self.manifest_handler.generate_manifest(
                dir_path, discovered_models
            )

            logger.info(f"Generated manifest for {directory} with {len(items)} models")
            return manifest

        except Exception as e:
            logger.error(f"Error generating manifest for {directory}: {e}")
            return None

    def load_manifest(self, directory: str) -> Optional[Dict[str, Any]]:
        """
        Load an existing manifest from a directory.

        Args:
            directory: Path to directory containing manifest

        Returns:
            Loaded manifest dictionary or None if not found
        """
        try:
            dir_path = Path(directory)
            manifest = self.manifest_handler.load_manifest(dir_path)

            if manifest:
                logger.info(f"Loaded manifest from {directory}")
            else:
                logger.debug(f"No manifest found in {directory}")

            return manifest

        except Exception as e:
            logger.error(f"Error loading manifest from {directory}: {e}")
            return None

    def save_manifest(self, directory: str, manifest: Dict[str, Any]) -> bool:
        """
        Save a manifest to a directory.

        Args:
            directory: Path to directory for manifest
            manifest: Manifest dictionary to save

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            dir_path = Path(directory)
            success = self.manifest_handler.save_manifest(dir_path, manifest)

            if success:
                logger.info(f"Saved manifest to {directory}")

            return success

        except Exception as e:
            logger.error(f"Error saving manifest to {directory}: {e}")
            return False

    def update_manifest(self, directory: str, updates: Dict[str, Any]) -> bool:
        """
        Update specific entries in an existing manifest.

        Args:
            directory: Path to directory containing manifest
            updates: Dictionary of updates to apply (model paths as keys)

        Returns:
            True if updated successfully, False otherwise
        """
        try:
            dir_path = Path(directory)

            # Load existing manifest
            manifest = self.manifest_handler.load_manifest(dir_path)
            if not manifest:
                logger.error(f"No manifest found to update in {directory}")
                return False

            # Apply updates
            models = manifest.get("models", [])
            for model in models:
                model_path = model.get("path", "")
                if model_path in updates:
                    # Update model entry with provided updates
                    model_updates = updates[model_path]
                    if "name" in model_updates:
                        model["name"] = model_updates["name"]
                    if "publisher" in model_updates:
                        model["publisher"] = model_updates["publisher"]
                    if "type" in model_updates:
                        model["type"] = model_updates["type"]
                    if "notes" in model_updates:
                        model["notes"] = model_updates["notes"]

            # Save updated manifest
            success = self.manifest_handler.save_manifest(dir_path, manifest)

            if success:
                logger.info(f"Updated manifest in {directory}")

            return success

        except Exception as e:
            logger.error(f"Error updating manifest in {directory}: {e}")
            return False

    def get_models_with_manifest(self, directory: str) -> List[Dict[str, Any]]:
        """
        Get models from a directory with manifest data merged.

        Args:
            directory: Path to directory containing models

        Returns:
            List of model dictionaries with manifest data applied
        """
        try:
            dir_path = Path(directory)

            # Get discovered models
            items = get_custom_items(dir_path)

            # This already includes manifest merging in get_custom_items
            logger.info(
                f"Found {len(items)} models in {directory} (with manifest data)"
            )
            return items

        except Exception as e:
            logger.error(f"Error getting models with manifest from {directory}: {e}")
            return []

    # Private helper methods

    def _enrich_asset_details(self, asset: Dict[str, Any]) -> Dict[str, Any]:
        """Add additional details to an asset dictionary."""
        enriched = asset.copy()

        # Add human-readable size
        size_bytes = asset.get("size", 0)
        if size_bytes > 1024 * 1024 * 1024:
            enriched["size_human"] = f"{size_bytes / (1024*1024*1024):.2f} GB"
        elif size_bytes > 1024 * 1024:
            enriched["size_human"] = f"{size_bytes / (1024*1024):.2f} MB"
        else:
            enriched["size_human"] = f"{size_bytes / 1024:.2f} KB"

        # Add VLLM-compatible name if applicable
        if asset.get("type") == "model":
            vllm_name = _extract_vllm_model_name(asset)
            if vllm_name:
                enriched["vllm_name"] = vllm_name

        return enriched

    def _enrich_lora_details(self, adapter: Dict[str, Any]) -> Dict[str, Any]:
        """Add additional details specific to LoRA adapters."""
        enriched = self._enrich_asset_details(adapter)

        metadata = adapter.get("metadata", {})

        # Extract LoRA-specific information
        enriched["lora_rank"] = metadata.get("rank", metadata.get("r", "unknown"))
        enriched["base_model"] = metadata.get("base_model", "unknown")
        enriched["task_type"] = metadata.get("task_type", "unknown")

        # Check if we can find the base model locally
        if enriched["base_model"] != "unknown":
            base_model_info = get_model_info(enriched["base_model"])
            if base_model_info:
                enriched["base_model_available"] = True
                enriched["base_model_path"] = base_model_info.get("path")
            else:
                enriched["base_model_available"] = False

        return enriched
