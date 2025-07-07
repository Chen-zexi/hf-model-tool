#!/usr/bin/env python3
"""
Asset detection module for HF-MODEL-TOOL.

Provides robust detection and classification of various model types including
LoRA adapters, custom models, merged models, and standard HuggingFace assets.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

logger = logging.getLogger(__name__)


class AssetDetector:
    """Detects and classifies different types of ML assets."""
    
    def __init__(self) -> None:
        """Initialize the asset detector with pattern definitions."""
        self.lora_patterns = {
            "adapter_config": "adapter_config.json",
            "adapter_model": "adapter_model.safetensors",
            "alternatives": ["adapter_model.bin", "pytorch_adapter.bin"]
        }
        
        self.model_patterns = {
            "config": "config.json",
            "safetensors": [".safetensors"],
            "alternatives": [".bin", ".pt", ".pth"]
        }
        
        self.dataset_patterns = {
            "readme": "README.md",
            "dataset_info": "dataset_info.json",
            "config": "config.json"
        }

    def detect_asset_type(self, directory: Union[str, Path]) -> Dict[str, Any]:
        """
        Detect the type and properties of an asset in a directory.
        
        Args:
            directory: Path to directory to analyze
            
        Returns:
            Dictionary containing asset type and metadata:
            - type: Asset type (lora_adapter, custom_model, model, dataset)
            - subtype: More specific classification
            - metadata: Additional properties
            - size: Total size in bytes
            - files: List of relevant files found
            
        Raises:
            OSError: If directory is not accessible
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise OSError(f"Directory does not exist: {directory}")
            
        if not directory.is_dir():
            raise OSError(f"Path is not a directory: {directory}")
            
        logger.debug(f"Analyzing directory: {directory}")
        
        # Get all files in directory and subdirectories
        all_files = self._get_all_files(directory)
        
        # Calculate total size
        total_size = self._calculate_total_size(directory)
        
        # Detect asset type using pattern matching
        asset_info = self._classify_asset(directory, all_files)
        asset_info["size"] = total_size
        asset_info["path"] = str(directory)
        
        return asset_info
        
    def _get_all_files(self, directory: Path) -> List[Path]:
        """Get all files in directory and subdirectories."""
        all_files = []
        try:
            for item in directory.rglob("*"):
                if item.is_file():
                    all_files.append(item)
        except (OSError, PermissionError) as e:
            logger.warning(f"Error accessing files in {directory}: {e}")
        return all_files
    
    def _calculate_total_size(self, directory: Path) -> int:
        """Calculate total size of all files in directory."""
        total_size = 0
        try:
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    try:
                        total_size += file_path.stat().st_size
                    except (OSError, PermissionError):
                        continue
        except (OSError, PermissionError) as e:
            logger.warning(f"Error calculating size for {directory}: {e}")
        return total_size
    
    def _classify_asset(self, directory: Path, all_files: List[Path]) -> Dict[str, Any]:
        """Classify the asset type based on file patterns."""
        file_names = [f.name for f in all_files]
        relative_paths = [f.relative_to(directory) for f in all_files]
        
        # Check for LoRA adapter
        lora_result = self._detect_lora_adapter(directory, file_names, relative_paths)
        if lora_result:
            return lora_result
            
        # Check for custom/merged model
        custom_result = self._detect_custom_model(directory, file_names, relative_paths)
        if custom_result:
            return custom_result
            
        # Check for standard HuggingFace model
        hf_result = self._detect_hf_model(directory, file_names, relative_paths)
        if hf_result:
            return hf_result
            
        # Check for dataset
        dataset_result = self._detect_dataset(directory, file_names, relative_paths)
        if dataset_result:
            return dataset_result
            
        # Fallback to generic classification
        return self._detect_generic_asset(directory, file_names, relative_paths)
    
    def _detect_lora_adapter(self, directory: Path, file_names: List[str], relative_paths: List[Path]) -> Optional[Dict[str, Any]]:
        """Detect LoRA adapter models."""
        # Get files that are directly in the root directory (not in subdirectories)
        root_files = [p.name for p in relative_paths if len(p.parts) == 1]
        
        # Check if adapter files are in root directory
        has_adapter_config_root = self.lora_patterns["adapter_config"] in root_files
        has_adapter_model_root = (
            self.lora_patterns["adapter_model"] in root_files or
            any(alt in root_files for alt in self.lora_patterns["alternatives"])
        )
        
        if has_adapter_config_root and has_adapter_model_root:
            metadata = self._extract_lora_metadata(directory)
            return {
                "type": "lora_adapter",
                "subtype": "fine_tuned",
                "metadata": metadata,
                "files": [f for f in root_files if "adapter" in f.lower()],
                "display_name": self._generate_display_name(directory, "LoRA")
            }
        
        # Check if adapter files are in subdirectories
        adapter_config_paths = [p for p in relative_paths if p.name == self.lora_patterns["adapter_config"] and len(p.parts) > 1]
        adapter_model_paths = [p for p in relative_paths if (p.name == self.lora_patterns["adapter_model"] or 
                              any(p.name == alt for alt in self.lora_patterns["alternatives"])) and len(p.parts) > 1]
        
        if adapter_config_paths and adapter_model_paths:
            # Find the subdirectory containing the LoRA adapter
            for config_path in adapter_config_paths:
                config_dir = directory / config_path.parent
                # Check if this subdirectory also has adapter model files
                subdir_files = [f.name for f in config_dir.iterdir() if f.is_file()]
                has_model_in_subdir = (
                    self.lora_patterns["adapter_model"] in subdir_files or
                    any(alt in subdir_files for alt in self.lora_patterns["alternatives"])
                )
                
                if has_model_in_subdir:
                    metadata = self._extract_lora_metadata(config_dir)
                    return {
                        "type": "lora_adapter",
                        "subtype": "fine_tuned",
                        "metadata": metadata,
                        "files": [f for f in subdir_files if "adapter" in f.lower()],
                        "display_name": config_dir.name,  # Use the actual LoRA directory name
                        "lora_path": str(config_dir)  # Store the actual LoRA directory path
                    }
        
        return None
    
    def _detect_custom_model(self, directory: Path, file_names: List[str], relative_paths: List[Path]) -> Optional[Dict[str, Any]]:
        """Detect custom or merged models."""
        has_config = self.model_patterns["config"] in file_names
        has_safetensors = any(
            any(pattern in f for pattern in self.model_patterns["safetensors"])
            for f in file_names
        )
        
        # Check if it's in a non-standard directory structure (not HF cache format)
        is_non_standard = not directory.name.startswith(("models--", "datasets--"))
        
        if has_config and has_safetensors and is_non_standard:
            metadata = self._extract_model_metadata(directory)
            
            # Determine if it's a merged model or custom model
            subtype = "merged" if "merged" in directory.name.lower() else "custom"
            
            return {
                "type": "custom_model",
                "subtype": subtype,
                "metadata": metadata,
                "files": [f for f in file_names if any(ext in f for ext in [".safetensors", ".bin"])],
                "display_name": self._generate_display_name(directory, "Custom")
            }
        return None
    
    def _detect_hf_model(self, directory: Path, file_names: List[str], relative_paths: List[Path]) -> Optional[Dict[str, Any]]:
        """Detect standard HuggingFace models."""
        has_config = self.model_patterns["config"] in file_names
        has_blobs = any("blobs" in str(p) for p in relative_paths)
        is_hf_format = directory.name.startswith("models--")
        
        if (has_config or has_blobs) and is_hf_format:
            metadata = self._extract_model_metadata(directory)
            return {
                "type": "model",
                "subtype": "huggingface",
                "metadata": metadata,
                "files": [f for f in file_names if any(ext in f for ext in [".safetensors", ".bin", ".json"])],
                "display_name": self._generate_display_name(directory, "Model")
            }
        return None
    
    def _detect_dataset(self, directory: Path, file_names: List[str], relative_paths: List[Path]) -> Optional[Dict[str, Any]]:
        """Detect HuggingFace datasets."""
        has_readme = self.dataset_patterns["readme"] in file_names
        has_dataset_info = self.dataset_patterns["dataset_info"] in file_names
        is_dataset_format = directory.name.startswith("datasets--")
        has_blobs = any("blobs" in str(p) for p in relative_paths)
        
        if (has_readme or has_dataset_info or has_blobs) and is_dataset_format:
            return {
                "type": "dataset",
                "subtype": "huggingface",
                "metadata": {},
                "files": [f for f in file_names if f.endswith((".json", ".md", ".parquet"))],
                "display_name": self._generate_display_name(directory, "Dataset")
            }
        return None
    
    def _detect_generic_asset(self, directory: Path, file_names: List[str], relative_paths: List[Path]) -> Dict[str, Any]:
        """Fallback detection for unrecognized assets."""
        has_safetensors = any(".safetensors" in f for f in file_names)
        has_model_files = any(ext in f for f in file_names for ext in [".bin", ".pt", ".pth"])
        
        if has_safetensors or has_model_files:
            asset_type = "unknown_model"
            subtype = "safetensors" if has_safetensors else "generic"
        else:
            asset_type = "unknown"
            subtype = "generic"
            
        return {
            "type": asset_type,
            "subtype": subtype,
            "metadata": {},
            "files": file_names,
            "display_name": self._generate_display_name(directory, "Unknown")
        }
    
    def _extract_lora_metadata(self, directory: Path) -> Dict[str, Any]:
        """Extract metadata from LoRA adapter configuration."""
        metadata = {}
        adapter_config_path = directory / self.lora_patterns["adapter_config"]
        
        if adapter_config_path.exists():
            try:
                with open(adapter_config_path, 'r') as f:
                    config = json.load(f)
                    metadata.update({
                        "base_model": config.get("base_model_name_or_path", "unknown"),
                        "peft_type": config.get("peft_type", "unknown"),
                        "lora_rank": config.get("r", "unknown"),
                        "lora_alpha": config.get("lora_alpha", "unknown"),
                        "target_modules": config.get("target_modules", []),
                        "task_type": config.get("task_type", "unknown")
                    })
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Error reading LoRA config: {e}")
        
        return metadata
    
    def _extract_model_metadata(self, directory: Path) -> Dict[str, Any]:
        """Extract metadata from model configuration."""
        metadata = {}
        config_path = directory / self.model_patterns["config"]
        
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    metadata.update({
                        "model_type": config.get("model_type", "unknown"),
                        "architectures": config.get("architectures", []),
                        "torch_dtype": config.get("torch_dtype", "unknown"),
                        "vocab_size": config.get("vocab_size", "unknown")
                    })
                    
                    # Check for unsloth or other fine-tuning indicators
                    if config.get("unsloth_version"):
                        metadata["fine_tuning_framework"] = "unsloth"
                        metadata["unsloth_version"] = config.get("unsloth_version")
                        
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Error reading model config: {e}")
        
        return metadata
    
    def _generate_display_name(self, directory: Path, asset_type: str) -> str:
        """Generate a user-friendly display name for the asset."""
        dir_name = directory.name
        
        # For HuggingFace format, extract the model name
        if "--" in dir_name:
            parts = dir_name.split("--")
            if len(parts) >= 3:
                return "--".join(parts[2:])
            elif len(parts) == 2:
                return parts[1]
        
        # For custom directories, use the actual directory name (one level up for scanning)
        # This handles cases like "output/20250629_194910/lora_model_name/"
        # where we want to display "lora_model_name" not "20250629_194910"
        return dir_name
    
    def get_modification_date(self, directory: Path) -> datetime:
        """Get the modification date of the directory."""
        try:
            return datetime.fromtimestamp(directory.stat().st_mtime)
        except OSError:
            return datetime.now()