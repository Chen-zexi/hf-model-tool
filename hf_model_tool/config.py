#!/usr/bin/env python3
"""
Configuration management module for HF-MODEL-TOOL.

Handles persistent storage of user preferences including custom
cache directories and application settings.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from .manifest import ManifestHandler
from .asset_detector import AssetDetector

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages application configuration with persistent storage."""

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize configuration manager.

        Args:
            config_dir: Optional custom config directory. Defaults to ~/.config/hf-model-tool
        """
        if config_dir is None:
            self.config_dir = Path.home() / ".config" / "hf-model-tool"
        else:
            self.config_dir = Path(config_dir)

        self.config_file = self.config_dir / "config.json"
        self._ensure_config_dir()
        self._config_cache: Optional[Dict[str, Any]] = None

    def _validate_directory_path(self, directory: str) -> Path:
        """
        Validate that a directory path exists and is a directory.

        Args:
            directory: Path string to validate

        Returns:
            Resolved Path object

        Raises:
            ValueError: If directory doesn't exist or is not a directory
        """
        directory_path = Path(directory).resolve()

        if not directory_path.exists():
            raise ValueError(f"Directory does not exist: {directory}")

        if not directory_path.is_dir():
            raise ValueError(f"Path is not a directory: {directory}")

        return directory_path

    def _ensure_config_dir(self) -> None:
        """Create configuration directory if it doesn't exist."""
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Configuration directory ensured at: {self.config_dir}")
        except OSError as e:
            logger.error(f"Failed to create config directory: {e}")
            raise

    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from disk.

        Returns:
            Dictionary containing configuration data

        Raises:
            json.JSONDecodeError: If config file is corrupted
        """
        if self._config_cache is not None:
            return self._config_cache

        default_config = {
            "custom_directories": [],
            "include_default_cache": True,
            "scan_ollama": False,  # Disabled by default - user must explicitly enable
            "ollama_directories": [],  # Custom Ollama directories added by user
            "last_updated": datetime.now().isoformat(),
        }

        if not self.config_file.exists():
            logger.info("No config file found, using defaults")
            self._config_cache = default_config
            return default_config

        try:
            with open(self.config_file, "r") as f:
                config = json.load(f)
                logger.info(f"Loaded configuration from {self.config_file}")
                self._config_cache = config
                return config
        except json.JSONDecodeError as e:
            logger.error(f"Config file corrupted: {e}")
            logger.info("Using default configuration")
            self._config_cache = default_config
            return default_config
        except OSError as e:
            logger.error(f"Failed to read config file: {e}")
            self._config_cache = default_config
            return default_config

    def save_config(self, config: Dict[str, Any]) -> None:
        """
        Save configuration to disk.

        Args:
            config: Configuration dictionary to save

        Raises:
            OSError: If unable to write config file
        """
        config["last_updated"] = datetime.now().isoformat()

        try:
            with open(self.config_file, "w") as f:
                json.dump(config, f, indent=2)
            logger.info(f"Saved configuration to {self.config_file}")
            self._config_cache = config
        except OSError as e:
            logger.error(f"Failed to save config: {e}")
            raise

    def add_directory(self, directory: str, path_type: str = "auto") -> bool:
        """
        Add a custom directory to configuration.

        Args:
            directory: Path to directory to add
            path_type: Type of path - "huggingface", "custom", "ollama", "lora", or "auto"

        Returns:
            True if directory was added, False if already exists

        Raises:
            ValueError: If directory doesn't exist or is invalid
        """
        directory_path = self._validate_directory_path(directory)

        config = self.load_config()
        custom_dirs = config.get("custom_directories", [])

        # Convert to string for JSON serialization
        directory_str = str(directory_path)

        # Check if directory already exists (regardless of type)
        existing_entry = next(
            (
                entry
                for entry in custom_dirs
                if (isinstance(entry, str) and entry == directory_str)
                or (isinstance(entry, dict) and entry.get("path") == directory_str)
            ),
            None,
        )

        if existing_entry:
            logger.info(f"Directory already in config: {directory_str}")
            return False

        # Create directory entry with type information
        directory_entry = {
            "path": directory_str,
            "type": path_type,
            "added_date": datetime.now().isoformat(),
        }

        custom_dirs.append(directory_entry)
        config["custom_directories"] = custom_dirs
        self.save_config(config)

        logger.info(f"Added {path_type} directory to config: {directory_str}")

        # Optionally generate manifest for the directory
        if path_type == "custom":
            self._offer_manifest_generation(directory_path)

        return True

    def remove_directory(self, directory: str) -> bool:
        """
        Remove a custom directory from configuration.

        Args:
            directory: Path to directory to remove

        Returns:
            True if directory was removed, False if not found
        """
        config = self.load_config()
        custom_dirs = config.get("custom_directories", [])

        # Normalize path for comparison
        directory_path = str(Path(directory).resolve())

        # Handle both old format (strings) and new format (dicts)
        found_index = -1
        for i, entry in enumerate(custom_dirs):
            if isinstance(entry, str):
                if entry == directory_path or entry == directory:
                    found_index = i
                    break
            elif isinstance(entry, dict):
                entry_path = entry.get("path", "")
                if entry_path == directory_path or entry_path == directory:
                    found_index = i
                    break

        if found_index == -1:
            logger.info(f"Directory not in config: {directory}")
            return False

        custom_dirs.pop(found_index)
        config["custom_directories"] = custom_dirs
        self.save_config(config)

        logger.info(f"Removed directory from config: {directory}")
        return True

    def get_all_directories(self) -> List[str]:
        """
        Get all configured directories including default cache (legacy method).

        Returns:
            List of directory paths to scan
        """
        directory_info = self.get_all_directories_with_types()
        return [info["path"] for info in directory_info]

    def get_all_directories_with_types(self) -> List[Dict[str, str]]:
        """
        Get all configured directories with their types.

        Returns:
            List of dictionaries with path and type information
        """
        config = self.load_config()
        directories = []

        # Add default HuggingFace cache directories if enabled
        if config.get("include_default_cache", True):
            default_hub = Path.home() / ".cache" / "huggingface" / "hub"
            default_datasets = Path.home() / ".cache" / "huggingface" / "datasets"

            if default_hub.exists():
                directories.append(
                    {
                        "path": str(default_hub),
                        "type": "huggingface",
                        "source": "default_cache",
                    }
                )
            if default_datasets.exists():
                directories.append(
                    {
                        "path": str(default_datasets),
                        "type": "huggingface",
                        "source": "default_cache",
                    }
                )

        # Add custom directories (handle both old and new formats)
        custom_dirs = config.get("custom_directories", [])
        for dir_entry in custom_dirs:
            if isinstance(dir_entry, str):
                # Legacy format - assume custom type
                if Path(dir_entry).exists():
                    directories.append(
                        {"path": dir_entry, "type": "custom", "source": "custom_legacy"}
                    )
                else:
                    logger.warning(
                        f"Configured directory no longer exists: {dir_entry}"
                    )
            elif isinstance(dir_entry, dict):
                # New format with type information
                dir_path = dir_entry.get("path")
                if dir_path and Path(dir_path).exists():
                    directories.append(
                        {
                            "path": dir_path,
                            "type": dir_entry.get("type", "custom"),
                            "source": "custom_configured",
                        }
                    )
                else:
                    logger.warning(f"Configured directory no longer exists: {dir_path}")

        # Add Ollama directories if scanning is enabled
        if config.get("scan_ollama", False):
            # Add default Ollama directories
            default_ollama_dirs = [
                Path.home() / ".ollama" / "models",
                Path("/usr/share/ollama/.ollama/models"),
                Path("/var/lib/ollama/.ollama/models"),
            ]

            for ollama_dir in default_ollama_dirs:
                if ollama_dir.exists():
                    directories.append(
                        {
                            "path": str(ollama_dir),
                            "type": "ollama",
                            "source": "default_ollama",
                        }
                    )

        # Add custom Ollama directories
        ollama_dirs = config.get("ollama_directories", [])
        for ollama_path in ollama_dirs:
            if Path(ollama_path).exists():
                directories.append(
                    {
                        "path": ollama_path,
                        "type": "ollama",
                        "source": "custom_ollama",
                    }
                )
            else:
                logger.warning(
                    f"Configured Ollama directory no longer exists: {ollama_path}"
                )

        return directories

    def toggle_default_cache(self) -> bool:
        """
        Toggle whether to include default HuggingFace cache in scans.

        Returns:
            New state of include_default_cache
        """
        config = self.load_config()
        current_state = config.get("include_default_cache", True)
        config["include_default_cache"] = not current_state
        self.save_config(config)

        logger.info(f"Toggled default cache inclusion to: {not current_state}")
        return not current_state

    def toggle_ollama_scanning(self) -> bool:
        """
        Toggle whether to scan default Ollama directories for models.

        Returns:
            New state of scan_ollama
        """
        config = self.load_config()
        current_state = config.get("scan_ollama", False)
        config["scan_ollama"] = not current_state
        self.save_config(config)

        logger.info(f"Toggled Ollama scanning to: {not current_state}")
        return not current_state

    def add_ollama_directory(self, directory: str) -> bool:
        """
        Add a custom Ollama directory to configuration.

        Args:
            directory: Path to Ollama-style directory to add

        Returns:
            True if directory was added, False if already exists

        Raises:
            ValueError: If directory doesn't exist or is invalid
        """
        directory_path = self._validate_directory_path(directory)

        # Check if it has Ollama structure (manifests and blobs directories)
        if (
            not (directory_path / "manifests").exists()
            or not (directory_path / "blobs").exists()
        ):
            logger.warning(
                f"Directory doesn't appear to have Ollama structure: {directory}"
            )

        config = self.load_config()
        ollama_dirs = config.get("ollama_directories", [])

        # Convert to string for JSON serialization
        directory_str = str(directory_path)

        # Check if directory already exists
        if directory_str in ollama_dirs:
            logger.info(f"Ollama directory already in config: {directory_str}")
            return False

        ollama_dirs.append(directory_str)
        config["ollama_directories"] = ollama_dirs
        self.save_config(config)

        logger.info(f"Added Ollama directory to config: {directory_str}")
        return True

    def remove_ollama_directory(self, directory: str) -> bool:
        """
        Remove a custom Ollama directory from configuration.

        Args:
            directory: Path to Ollama directory to remove

        Returns:
            True if directory was removed, False if not found
        """
        config = self.load_config()
        ollama_dirs = config.get("ollama_directories", [])

        # Normalize path for comparison
        directory_path = str(Path(directory).resolve())

        if directory_path in ollama_dirs:
            ollama_dirs.remove(directory_path)
            config["ollama_directories"] = ollama_dirs
            self.save_config(config)
            logger.info(f"Removed Ollama directory from config: {directory_path}")
            return True

        # Also check without normalization
        if directory in ollama_dirs:
            ollama_dirs.remove(directory)
            config["ollama_directories"] = ollama_dirs
            self.save_config(config)
            logger.info(f"Removed Ollama directory from config: {directory}")
            return True

        logger.info(f"Ollama directory not in config: {directory}")
        return False

    def get_ollama_directories(self) -> List[str]:
        """
        Get all configured Ollama directories (custom only, not defaults).

        Returns:
            List of Ollama directory paths
        """
        config = self.load_config()
        return config.get("ollama_directories", [])

    def _offer_manifest_generation(self, directory: Path) -> bool:
        """
        Offer to generate a manifest for a newly added directory.

        Args:
            directory: Path to the directory

        Returns:
            True if manifest was generated, False otherwise
        """
        try:
            manifest_handler = ManifestHandler()

            # Check if manifest already exists
            existing_manifest = manifest_handler.load_manifest(directory)
            if existing_manifest:
                logger.info(f"Manifest already exists for {directory}")
                return False

            # Use the existing get_custom_items function to scan for models
            from .cache import get_custom_items

            items = get_custom_items(directory)

            # Convert items to the format expected by manifest
            discovered_models = []
            for item in items:
                discovered_models.append(
                    {
                        "path": item.get("path", str(directory / item["name"])),
                        "name": item["name"],
                        "type": item.get("type", "custom_model"),
                        "size": item.get("size", 0),
                        "metadata": item.get("metadata", {}),
                        "display_name": item.get("display_name", item["name"]),
                    }
                )

            if discovered_models:
                # Generate manifest
                manifest = manifest_handler.generate_manifest(
                    directory, discovered_models
                )

                # For now, auto-save the manifest
                # In a full implementation, this could be interactive
                if manifest_handler.save_manifest(directory, manifest):
                    logger.info(
                        f"Generated manifest for {directory} with {len(discovered_models)} models"
                    )
                    return True

        except Exception as e:
            logger.error(f"Error generating manifest for {directory}: {e}")

        return False

    def validate_directory(self, directory: str) -> bool:
        """
        Validate if a directory contains ML assets (HuggingFace, LoRA, custom models).

        Args:
            directory: Path to directory to validate

        Returns:
            True if directory appears to contain ML assets
        """
        directory_path = Path(directory)

        if not directory_path.exists() or not directory_path.is_dir():
            return False

        detector = AssetDetector()

        # Check for typical HuggingFace directory structure
        # Look for directories with "models--" or "datasets--" prefix
        # or directories containing "blobs" subdirectory
        for item in directory_path.iterdir():
            if item.is_dir():
                # Check for HuggingFace format
                if item.name.startswith(("models--", "datasets--")):
                    return True
                if (item / "blobs").exists():
                    return True

                # Check for custom model patterns using asset detector
                try:
                    asset_info = detector.detect_asset_type(item)
                    # Consider it valid if we can detect any asset type with content
                    if asset_info["size"] > 0 and asset_info["type"] != "unknown":
                        return True
                except Exception:
                    # Continue checking other directories if one fails
                    continue

        # Also check if this directory itself contains assets
        try:
            asset_info = detector.detect_asset_type(directory_path)
            if asset_info["size"] > 0 and asset_info["type"] != "unknown":
                return True
        except Exception:
            pass

        # Legacy check for blobs directory
        if (directory_path / "blobs").exists():
            return True

        return False
