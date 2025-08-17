#!/usr/bin/env python3
"""
Manifest management module for HF-MODEL-TOOL.

Provides functionality to create, load, and manage model manifests
that allow users to customize model metadata in their directories.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

logger = logging.getLogger(__name__)

MANIFEST_VERSION = "1.0"
MANIFEST_FILENAME = "models_manifest.json"


class ManifestHandler:
    """Handles model manifest operations for custom directories."""

    def __init__(self):
        """Initialize the manifest handler."""
        self.manifest_filename = MANIFEST_FILENAME
        self.version = MANIFEST_VERSION

    def load_manifest(self, directory: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """
        Load manifest from a directory if it exists.

        Args:
            directory: Path to directory containing manifest

        Returns:
            Manifest data or None if not found/invalid
        """
        directory = Path(directory)
        manifest_path = directory / self.manifest_filename

        if not manifest_path.exists():
            logger.debug(f"No manifest found at {manifest_path}")
            return None

        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)

            # Validate manifest structure
            if not self._validate_manifest(manifest):
                logger.warning(f"Invalid manifest at {manifest_path}")
                return None

            logger.info(f"Loaded manifest from {manifest_path}")
            return manifest

        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Error loading manifest from {manifest_path}: {e}")
            return None

    def generate_manifest(
        self, directory: Union[str, Path], discovered_models: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate a manifest template from discovered models.

        Args:
            directory: Directory path where manifest will be saved
            discovered_models: List of auto-discovered model information

        Returns:
            Generated manifest dictionary
        """
        directory = Path(directory)

        manifest = {
            "version": self.version,
            "generated": datetime.now().isoformat(),
            "directory": str(directory),
            "models": [],
        }

        for model in discovered_models:
            # Extract relative path from full path if available
            model_path = Path(model.get("path", ""))
            if model_path.is_absolute() and directory in model_path.parents:
                relative_path = model_path.relative_to(directory)
            else:
                relative_path = model_path

            # Create manifest entry with user-customizable fields
            entry = {
                "path": str(relative_path),
                "name": model.get(
                    "display_name", model.get("name", relative_path.name)
                ),
                "publisher": model.get("publisher", "unknown"),
                "type": model.get("type", "custom_model"),
                "notes": model.get("notes", ""),
                "enabled": True,
                # Store auto-detected metadata for reference
                "auto_detected": {
                    "size": model.get("size", 0),
                    "model_type": model.get("metadata", {}).get("model_type"),
                    "architectures": model.get("metadata", {}).get("architectures", []),
                },
            }
            manifest["models"].append(entry)

        logger.info(f"Generated manifest with {len(manifest['models'])} models")
        return manifest

    def save_manifest(
        self, directory: Union[str, Path], manifest: Dict[str, Any]
    ) -> bool:
        """
        Save manifest to a directory.

        Args:
            directory: Directory where manifest will be saved
            manifest: Manifest data to save

        Returns:
            True if saved successfully, False otherwise
        """
        directory = Path(directory)
        manifest_path = directory / self.manifest_filename

        try:
            # Update save timestamp
            manifest["last_modified"] = datetime.now().isoformat()

            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)

            logger.info(f"Saved manifest to {manifest_path}")
            return True

        except OSError as e:
            logger.error(f"Error saving manifest to {manifest_path}: {e}")
            return False

    def merge_with_discovered(
        self,
        manifest_models: List[Dict[str, Any]],
        discovered_models: List[Dict[str, Any]],
        directory: Path,
    ) -> List[Dict[str, Any]]:
        """
        Merge manifest data with auto-discovered models.

        Manifest data takes precedence over auto-discovered data.
        Models in manifest but not discovered are included if enabled.
        Models discovered but not in manifest are added with auto-detected values.

        Args:
            manifest_models: Models from manifest
            discovered_models: Auto-discovered models
            directory: Base directory for resolving paths

        Returns:
            Merged list of model dictionaries
        """
        merged = []
        manifest_by_path = {}

        # Process manifest models
        for entry in manifest_models:
            if not entry.get("enabled", True):
                continue

            # Resolve path relative to directory
            model_path = directory / entry["path"]
            manifest_by_path[str(model_path)] = entry

            # Create model dict with manifest data taking precedence
            model_dict = {
                "path": str(model_path),
                "name": entry.get("name", entry["path"]),
                "display_name": entry.get("name", entry["path"]),
                "publisher": entry.get("publisher", "unknown"),
                "type": entry.get("type", "custom_model"),
                "notes": entry.get("notes", ""),
                "from_manifest": True,
                # Include auto-detected metadata if available
                "size": entry.get("auto_detected", {}).get("size", 0),
                "metadata": {
                    "model_type": entry.get("auto_detected", {}).get("model_type"),
                    "architectures": entry.get("auto_detected", {}).get(
                        "architectures", []
                    ),
                },
            }
            merged.append(model_dict)

        # Process discovered models not in manifest
        for model in discovered_models:
            model_path = model.get("path", "")

            if model_path not in manifest_by_path:
                # Model not in manifest, add with auto-detected values
                model["from_manifest"] = False
                merged.append(model)
            else:
                # Model is in manifest, update size and metadata if needed
                for m in merged:
                    if m["path"] == model_path:
                        # Update dynamic data like size
                        m["size"] = model.get("size", m.get("size", 0))
                        m["date"] = model.get("date", datetime.now())
                        # Preserve other discovered metadata
                        m["subtype"] = model.get("subtype", "custom")
                        m["files"] = model.get("files", [])
                        m["source_type"] = model.get("source_type", "custom_directory")
                        break

        logger.debug(
            f"Merged {len(manifest_models)} manifest models with {len(discovered_models)} discovered models"
        )
        return merged

    def _validate_manifest(self, manifest: Dict[str, Any]) -> bool:
        """
        Validate manifest structure and required fields.

        Args:
            manifest: Manifest data to validate

        Returns:
            True if valid, False otherwise
        """
        required_fields = ["version", "models"]

        for field in required_fields:
            if field not in manifest:
                logger.warning(f"Manifest missing required field: {field}")
                return False

        if not isinstance(manifest.get("models"), list):
            logger.warning("Manifest 'models' field must be a list")
            return False

        # Validate each model entry
        for i, model in enumerate(manifest["models"]):
            if not isinstance(model, dict):
                logger.warning(f"Manifest model {i} is not a dictionary")
                return False

            if "path" not in model:
                logger.warning(f"Manifest model {i} missing 'path' field")
                return False

        return True

    def create_example_manifest(self, directory: Union[str, Path]) -> Dict[str, Any]:
        """
        Create an example manifest with sample entries.

        Args:
            directory: Directory for the manifest

        Returns:
            Example manifest dictionary
        """
        return {
            "version": self.version,
            "generated": datetime.now().isoformat(),
            "directory": str(directory),
            "models": [
                {
                    "path": "example-model",
                    "name": "Example Model Name",
                    "publisher": "YourOrganization",
                    "type": "custom_model",
                    "notes": "This is an example model entry. Customize as needed.",
                    "enabled": True,
                    "auto_detected": {},
                }
            ],
        }
