#!/usr/bin/env python3
"""
Ollama model discovery module for HF-MODEL-TOOL.

Provides detection and management of Ollama-downloaded models in GGUF format,
enabling integration with vLLM and other inference frameworks.
"""
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

logger = logging.getLogger(__name__)

# Default Ollama model directories
DEFAULT_OLLAMA_DIRS = [
    Path.home() / ".ollama" / "models",  # User directory
    Path("/usr/share/ollama/.ollama/models"),  # System directory (common)
    Path("/var/lib/ollama/.ollama/models"),  # System directory (alternative)
]


def get_ollama_items(
    ollama_dirs: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
) -> List[Dict[str, Any]]:
    """
    Scan Ollama model directories for downloaded models.

    Checks multiple locations:
    1. OLLAMA_MODELS environment variable (if set)
    2. User directory: ~/.ollama/models
    3. System directories: /usr/share/ollama/.ollama/models, /var/lib/ollama/.ollama/models

    Args:
        ollama_dirs: Path(s) to Ollama models directories (optional)

    Returns:
        List of dictionaries containing Ollama model metadata
    """
    # Determine directories to scan
    dirs_to_scan: List[Path] = []

    if ollama_dirs is not None:
        # Use provided directories
        if isinstance(ollama_dirs, (str, Path)):
            dirs_to_scan = [Path(ollama_dirs)]
        else:
            dirs_to_scan = [Path(d) for d in ollama_dirs]
    else:
        # Check environment variable first
        env_dir = os.environ.get("OLLAMA_MODELS")
        if env_dir:
            logger.info(f"Using OLLAMA_MODELS environment variable: {env_dir}")
            dirs_to_scan.append(Path(env_dir))

        # Add default directories
        dirs_to_scan.extend(DEFAULT_OLLAMA_DIRS)

    # Scan all directories and merge results
    all_items: List[Dict[str, Any]] = []
    seen_models: Dict[str, Dict[str, Any]] = (
        {}
    )  # Track models by name to handle duplicates

    for ollama_dir in dirs_to_scan:
        if not ollama_dir.exists():
            logger.debug(f"Ollama directory does not exist: {ollama_dir}")
            continue

        if not ollama_dir.is_dir():
            logger.debug(f"Ollama path is not a directory: {ollama_dir}")
            continue

        logger.info(f"Scanning Ollama directory: {ollama_dir}")
        items = _scan_single_ollama_directory(ollama_dir)

        # Merge results, handling duplicates
        for item in items:
            model_name = item.get("name", "")
            if model_name in seen_models:
                # Model already found, update if this one is newer or larger
                existing = seen_models[model_name]
                if item.get("size", 0) > existing.get("size", 0):
                    logger.debug(
                        f"Updating {model_name} with larger version from {ollama_dir}"
                    )
                    seen_models[model_name] = item
                    item["metadata"]["locations"] = existing.get("metadata", {}).get(
                        "locations", []
                    ) + [str(ollama_dir)]
            else:
                item["metadata"]["locations"] = [str(ollama_dir)]
                seen_models[model_name] = item

    # Convert to list
    all_items = list(seen_models.values())

    logger.info(f"Found {len(all_items)} unique Ollama models across all directories")
    return all_items


def _scan_single_ollama_directory(ollama_dir: Path) -> List[Dict[str, Any]]:
    """
    Scan a single Ollama directory for models.

    This function recursively scans the manifests directory within an Ollama
    installation to discover all downloaded models. It parses each manifest
    file to extract model metadata and verifies the corresponding GGUF blobs exist.

    Args:
        ollama_dir: Path to Ollama models directory (e.g., ~/.ollama/models)

    Returns:
        List of model dictionaries from this directory, each containing:
        - name: Model identifier (e.g., "llama2:7b")
        - size: Size in bytes of the model file
        - date: Modification timestamp
        - type: Always "ollama_model"
        - metadata: Additional Ollama-specific metadata
        - path: Direct path to the GGUF file

    Raises:
        No exceptions are raised; errors are logged and empty list returned
    """
    items: List[Dict[str, Any]] = []

    manifests_dir = ollama_dir / "manifests"
    blobs_dir = ollama_dir / "blobs"

    if not manifests_dir.exists() or not blobs_dir.exists():
        logger.debug(f"Ollama manifests or blobs directory not found in {ollama_dir}")
        return []

    # Scan all manifest files
    try:
        for manifest_file in manifests_dir.rglob("*"):
            if not manifest_file.is_file():
                continue

            try:
                model_info = _parse_ollama_manifest(manifest_file, blobs_dir)
                if model_info:
                    model_info["metadata"]["ollama_dir"] = str(ollama_dir)
                    items.append(model_info)
            except Exception as e:
                logger.debug(f"Error parsing manifest {manifest_file}: {e}")
                continue

    except (OSError, PermissionError) as e:
        logger.debug(f"Error scanning Ollama directory {ollama_dir}: {e}")
        return []

    logger.debug(f"Found {len(items)} models in {ollama_dir}")
    return items


def _parse_ollama_manifest(
    manifest_path: Path, blobs_dir: Path
) -> Optional[Dict[str, Any]]:
    """
    Parse an Ollama manifest file to extract model information.

    Ollama stores models as manifest files containing layer information.
    This function reads the manifest, identifies the main model layer (GGUF file),
    verifies the blob exists, and constructs a complete model information dictionary.

    Args:
        manifest_path: Path to the manifest JSON file
        blobs_dir: Path to the blobs directory containing actual model files

    Returns:
        Dictionary with model information including:
        - name: Full model identifier extracted from path
        - size: Model file size in bytes
        - path: Direct path to GGUF blob file
        - metadata: Ollama-specific metadata (digest, format, etc.)
        Returns None if:
        - Manifest cannot be parsed
        - No model layer found
        - GGUF blob doesn't exist
        - File is not GGUF format

    Notes:
        Model names are extracted from the manifest path structure:
        .../library/{model_name}/{tag} -> "{model_name}:{tag}"
    """
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.debug(f"Failed to read manifest {manifest_path}: {e}")
        return None

    # Extract model name from path
    # Format: .ollama/models/manifests/registry.ollama.ai/library/{model_name}/{tag}
    parts = manifest_path.parts

    # Try to extract model name and tag from path
    model_name = None
    tag = None

    # Look for patterns in the path
    if "library" in parts:
        library_idx = parts.index("library")
        if library_idx + 1 < len(parts):
            model_name = parts[library_idx + 1]
            if library_idx + 2 < len(parts):
                tag = parts[library_idx + 2]
    elif len(parts) >= 2:
        # Fallback: use last two parts as model/tag
        model_name = parts[-2]
        tag = parts[-1]

    if not model_name:
        model_name = manifest_path.parent.name
        tag = manifest_path.stem

    # Create full model identifier
    full_name = f"{model_name}:{tag}" if tag and tag != model_name else model_name

    # Find the main model layer (GGUF file)
    model_blob = None
    model_size = 0

    for layer in manifest.get("layers", []):
        if layer.get("mediaType") == "application/vnd.ollama.image.model":
            digest = layer.get("digest", "")
            if digest.startswith("sha256:"):
                model_blob = digest[7:]  # Remove "sha256:" prefix
            else:
                model_blob = digest
            model_size = layer.get("size", 0)
            break

    if not model_blob:
        logger.debug(f"No model layer found in manifest {manifest_path}")
        return None

    # Construct path to actual GGUF blob
    blob_filename = f"sha256-{model_blob}"
    blob_path = blobs_dir / blob_filename

    if not blob_path.exists():
        # Try alternative naming
        blob_path = blobs_dir / f"sha256:{model_blob}"
        if not blob_path.exists():
            logger.debug(f"Model blob not found: {blob_filename}")
            return None

    # Verify it's a GGUF file
    if not _is_gguf_file(blob_path):
        logger.debug(f"Model blob is not GGUF format: {blob_path}")
        return None

    # Get modification time
    try:
        mod_time = datetime.fromtimestamp(blob_path.stat().st_mtime)
    except OSError:
        mod_time = datetime.now()

    # Build metadata
    metadata = {
        "format": "gguf",
        "source": "ollama",
        "manifest_path": str(manifest_path),
        "model_digest": model_blob,
        "blob_path": str(blob_path),
        "experimental_warning": "GGUF support in vLLM is experimental and may have compatibility issues",
        "quantization": "gguf",
    }

    # Extract config from manifest if available
    config_layer = None
    for layer in manifest.get("layers", []):
        if layer.get("mediaType") == "application/vnd.docker.container.image.v1+json":
            config_layer = layer
            break

    if config_layer:
        metadata["config_digest"] = config_layer.get("digest", "unknown")

    return {
        "name": full_name,
        "size": model_size,
        "date": mod_time,
        "type": "ollama_model",
        "subtype": "gguf",
        "metadata": metadata,
        "display_name": full_name,
        "publisher": "ollama",
        "path": str(blob_path),  # Direct path to GGUF file
        "source_type": "ollama",
    }


def _is_gguf_file(file_path: Path) -> bool:
    """
    Check if a file is in GGUF format by reading its magic header.

    Args:
        file_path: Path to the file to check

    Returns:
        True if file is GGUF format, False otherwise
    """
    try:
        with open(file_path, "rb") as f:
            # GGUF files start with magic bytes "GGUF"
            magic = f.read(4)
            return magic == b"GGUF"
    except (OSError, PermissionError):
        return False


def validate_ollama_model(model_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Validate an Ollama model for use with vLLM.

    Args:
        model_path: Path to the GGUF model file

    Returns:
        Validation result dictionary with 'valid' status and details
    """
    if isinstance(model_path, str):
        model_path = Path(model_path)

    if not model_path.exists():
        return {
            "valid": False,
            "error": "Model file does not exist",
        }

    if not model_path.is_file():
        return {
            "valid": False,
            "error": "Path is not a file",
        }

    if not _is_gguf_file(model_path):
        return {
            "valid": False,
            "error": "File is not in GGUF format",
        }

    # Check file size
    try:
        size = model_path.stat().st_size
    except OSError as e:
        return {
            "valid": False,
            "error": f"Cannot access file: {e}",
        }

    if size == 0:
        return {
            "valid": False,
            "error": "Model file is empty",
        }

    # Check vLLM GGUF support
    gguf_support = check_vllm_gguf_support()
    if not gguf_support["supported"]:
        return {
            "valid": False,
            "error": "GGUF format is not supported by current vLLM version",
            "details": gguf_support,
        }

    return {
        "valid": True,
        "warnings": gguf_support.get("notes", []),
        "experimental": True,
        "size": size,
        "format": "gguf",
        "path": str(model_path),
    }


def check_vllm_gguf_support() -> Dict[str, Any]:
    """
    Check if GGUF format is supported by the current vLLM installation.

    GGUF (GPT-Generated Unified Format) support was added to vLLM in version 0.5.0+
    as an experimental feature. This function checks the installed vLLM version
    and determines if GGUF models can be used.

    Returns:
        Dictionary containing:
        - supported: Boolean indicating if GGUF is supported
        - version: Installed vLLM version string
        - experimental: Always True (GGUF support is experimental)
        - notes: List of warnings/limitations for GGUF usage

    Notes:
        - GGUF support requires vLLM >= 0.5.0
        - Support is experimental with potential compatibility issues
        - Performance may be suboptimal compared to safetensors format
    """
    try:
        import vllm

        # Check vLLM version for GGUF support
        # GGUF support was added in vLLM 0.5.0+
        version_str = getattr(vllm, "__version__", "0.0.0")

        # Parse version
        try:
            version_parts = version_str.split(".")
            major = int(version_parts[0]) if version_parts else 0
            minor = int(version_parts[1].split("+")[0]) if len(version_parts) > 1 else 0
        except (ValueError, IndexError):
            # Handle non-standard version formats
            major = 0
            minor = 0

        has_support = major > 0 or (major == 0 and minor >= 5)

        return {
            "supported": has_support,
            "version": version_str,
            "experimental": True,
            "notes": [
                "GGUF support in vLLM is highly experimental",
                "Performance may be suboptimal compared to safetensors format",
                "Some features may not work with GGUF models",
                "Only single-file GGUF models are supported",
                "Custom architectures may not be recognized",
            ],
        }
    except ImportError:
        return {
            "supported": False,
            "version": "unknown",
            "experimental": False,
            "notes": ["vLLM is not installed"],
        }


def get_ollama_model_info(model_name: str) -> Optional[Dict[str, Any]]:
    """
    Get detailed information about a specific Ollama model.

    Searches all configured Ollama directories for a model matching the given name.
    The search is case-sensitive and matches against both the 'name' and 'display_name'
    fields of discovered models.

    Args:
        model_name: Name of the model to find. Can be in format:
            - "model:tag" (e.g., "llama2:7b")
            - "model" (will match any tag)

    Returns:
        Dictionary containing full model information if found:
        - name: Model identifier
        - size: File size in bytes
        - path: Direct path to GGUF file
        - type: "ollama_model"
        - metadata: Ollama-specific metadata
        Returns None if model not found

    Example:
        >>> info = get_ollama_model_info("llama2:7b")
        >>> if info:
        ...     print(f"Model path: {info['path']}")
    """
    models = get_ollama_items()

    for model in models:
        if model.get("name") == model_name or model.get("display_name") == model_name:
            return model

    return None
