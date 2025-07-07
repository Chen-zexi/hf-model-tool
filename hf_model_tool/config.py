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

    def add_directory(self, directory: str) -> bool:
        """
        Add a custom directory to configuration.

        Args:
            directory: Path to directory to add

        Returns:
            True if directory was added, False if already exists

        Raises:
            ValueError: If directory doesn't exist or is invalid
        """
        directory_path = Path(directory).resolve()

        if not directory_path.exists():
            raise ValueError(f"Directory does not exist: {directory}")

        if not directory_path.is_dir():
            raise ValueError(f"Path is not a directory: {directory}")

        config = self.load_config()
        custom_dirs = config.get("custom_directories", [])

        # Convert to string for JSON serialization
        directory_str = str(directory_path)

        if directory_str in custom_dirs:
            logger.info(f"Directory already in config: {directory_str}")
            return False

        custom_dirs.append(directory_str)
        config["custom_directories"] = custom_dirs
        self.save_config(config)

        logger.info(f"Added directory to config: {directory_str}")
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

        if directory_path not in custom_dirs:
            # Also try the original string in case it's stored differently
            if directory not in custom_dirs:
                logger.info(f"Directory not in config: {directory}")
                return False
            custom_dirs.remove(directory)
        else:
            custom_dirs.remove(directory_path)

        config["custom_directories"] = custom_dirs
        self.save_config(config)

        logger.info(f"Removed directory from config: {directory}")
        return True

    def get_all_directories(self) -> List[str]:
        """
        Get all configured directories including default cache.

        Returns:
            List of directory paths to scan
        """
        config = self.load_config()
        directories = []

        # Add default HuggingFace cache directories if enabled
        if config.get("include_default_cache", True):
            default_hub = Path.home() / ".cache" / "huggingface" / "hub"
            default_datasets = Path.home() / ".cache" / "huggingface" / "datasets"

            if default_hub.exists():
                directories.append(str(default_hub))
            if default_datasets.exists():
                directories.append(str(default_datasets))

        # Add custom directories
        custom_dirs = config.get("custom_directories", [])
        for dir_path in custom_dirs:
            if Path(dir_path).exists():
                directories.append(dir_path)
            else:
                logger.warning(f"Configured directory no longer exists: {dir_path}")

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

    def validate_directory(self, directory: str) -> bool:
        """
        Validate if a directory contains HuggingFace assets.

        Args:
            directory: Path to directory to validate

        Returns:
            True if directory appears to contain HF assets
        """
        directory_path = Path(directory)

        if not directory_path.exists() or not directory_path.is_dir():
            return False

        # Check for typical HuggingFace directory structure
        # Look for directories with "models--" or "datasets--" prefix
        # or directories containing "blobs" subdirectory
        for item in directory_path.iterdir():
            if item.is_dir():
                if item.name.startswith(("models--", "datasets--")):
                    return True
                if (item / "blobs").exists():
                    return True

        # Also check if this directory itself contains blobs
        if (directory_path / "blobs").exists():
            return True

        return False
