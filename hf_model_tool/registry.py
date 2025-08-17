#!/usr/bin/env python3
"""
Unified Model Registry for HF-MODEL-TOOL and vLLM CLI.

Provides a singleton registry for all discovered models, LoRA adapters,
and datasets with intelligent caching and incremental updates.
"""
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from threading import Lock

from .cache import scan_all_directories
from .asset_detector import AssetDetector
from .config import ConfigManager

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Singleton registry for all discovered ML assets.
    
    This registry provides:
    - Unified scanning across all configured directories
    - Intelligent caching with TTL
    - Incremental updates based on file changes
    - vLLM compatibility checking
    - Thread-safe operations
    """
    
    _instance: Optional['ModelRegistry'] = None
    _lock = Lock()
    
    def __new__(cls) -> 'ModelRegistry':
        """Ensure singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the registry (only once due to singleton)."""
        if self._initialized:
            return
            
        self.models: Dict[str, Dict[str, Any]] = {}
        self.lora_adapters: Dict[str, Dict[str, Any]] = {}
        self.datasets: Dict[str, Dict[str, Any]] = {}
        self.custom_models: Dict[str, Dict[str, Any]] = {}
        
        # Cache management
        self._cache_file = Path.home() / ".cache" / "hf-model-tool" / "registry_cache.json"
        self._cache_ttl = 300  # 5 minutes default
        self._last_scan_time: float = 0
        self._directory_mtimes: Dict[str, float] = {}
        
        # Configuration
        self.config_manager = ConfigManager()
        self.asset_detector = AssetDetector()
        
        # vLLM compatibility patterns
        self._vllm_compatible_architectures = {
            "LlamaForCausalLM", "MistralForCausalLM", "Qwen2ForCausalLM",
            "GPTNeoXForCausalLM", "GPT2LMHeadModel", "BloomForCausalLM",
            "FalconForCausalLM", "BaichuanForCausalLM", "ChatGLMModel",
            "InternLMForCausalLM", "Yi", "DeepseekForCausalLM"
        }
        
        self._initialized = True
        
        # Load cache if available
        self._load_cache()
    
    def scan_all(self, force: bool = False, incremental: bool = True) -> None:
        """
        Scan all configured directories for assets.
        
        Args:
            force: Force full rescan ignoring cache
            incremental: Only scan changed directories (based on mtime)
        """
        current_time = time.time()
        
        # Check if cache is still valid
        if not force and self._cache_valid(current_time):
            logger.debug("Using cached registry data")
            return
        
        logger.info("Starting registry scan...")
        start_time = time.time()
        
        # Clear current data if forcing full scan
        if force and not incremental:
            self.models.clear()
            self.lora_adapters.clear()
            self.datasets.clear()
            self.custom_models.clear()
        
        # Get directories to scan
        directories_to_scan = self._get_directories_to_scan(incremental)
        
        if not directories_to_scan:
            logger.info("No directories need scanning")
            self._last_scan_time = current_time
            return
        
        # Perform scan
        scanned_items = self._scan_directories(directories_to_scan)
        
        # Process and categorize items
        self._process_scanned_items(scanned_items)
        
        # Update scan metadata
        self._last_scan_time = current_time
        self._update_directory_mtimes(directories_to_scan)
        
        # Save to cache
        self._save_cache()
        
        elapsed = time.time() - start_time
        logger.info(f"Registry scan completed in {elapsed:.2f}s")
        logger.info(f"Found: {len(self.models)} models, {len(self.lora_adapters)} LoRAs, "
                   f"{len(self.datasets)} datasets, {len(self.custom_models)} custom models")
    
    def get_models_for_vllm(self) -> List[Dict[str, Any]]:
        """
        Get models that are compatible with vLLM serving.
        
        Returns:
            List of model dictionaries suitable for vLLM
        """
        self.scan_all()  # Ensure data is current
        
        vllm_models = []
        
        for model_id, model in self.models.items():
            if self._is_vllm_compatible(model):
                # Add vLLM-specific fields
                enriched = model.copy()
                enriched['vllm_compatible'] = True
                enriched['model_id'] = model_id
                vllm_models.append(enriched)
        
        # Also check custom models
        for model_id, model in self.custom_models.items():
            if self._is_vllm_compatible(model):
                enriched = model.copy()
                enriched['vllm_compatible'] = True
                enriched['model_id'] = model_id
                vllm_models.append(enriched)
        
        return vllm_models
    
    def get_all_assets(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all discovered assets.
        
        Returns:
            Dictionary with all assets by ID
        """
        self.scan_all()  # Ensure data is current
        
        all_assets = {}
        all_assets.update(self.models)
        all_assets.update(self.lora_adapters)
        all_assets.update(self.datasets)
        all_assets.update(self.custom_models)
        
        return all_assets
    
    def find_asset(self, identifier: str) -> Optional[Dict[str, Any]]:
        """
        Find an asset by various identifiers.
        
        Args:
            identifier: Asset ID, name, path, or display name
        
        Returns:
            Asset dictionary if found, None otherwise
        """
        self.scan_all()  # Ensure data is current
        
        # Check all collections
        all_collections = [
            self.models,
            self.lora_adapters,
            self.datasets,
            self.custom_models
        ]
        
        for collection in all_collections:
            # Direct ID match
            if identifier in collection:
                return collection[identifier]
            
            # Search by other fields
            for asset_id, asset in collection.items():
                if (asset.get('name') == identifier or
                    asset.get('display_name') == identifier or
                    asset.get('path') == identifier or
                    str(Path(asset.get('path', ''))) == str(Path(identifier))):
                    return asset
        
        return None
    
    def get_loras_for_model(self, model_identifier: str) -> List[Dict[str, Any]]:
        """
        Find LoRA adapters compatible with a specific model.
        
        Args:
            model_identifier: Model ID, name, or path
        
        Returns:
            List of compatible LoRA adapters
        """
        self.scan_all()  # Ensure data is current
        
        # Find the model
        model = self.find_asset(model_identifier)
        if not model:
            logger.warning(f"Model not found: {model_identifier}")
            # Try to match by name anyway
            model_name = model_identifier
        else:
            model_name = model.get('name', model.get('display_name', ''))
        
        compatible_loras = []
        
        for lora_id, lora in self.lora_adapters.items():
            if self._is_lora_compatible(lora, model_name):
                compatible_loras.append(lora)
        
        return compatible_loras
    
    def add_directory(self, directory: str, dir_type: str = 'auto') -> bool:
        """
        Add a directory and scan it immediately.
        
        Args:
            directory: Directory path to add
            dir_type: Directory type ('huggingface', 'custom', 'lora', 'auto')
        
        Returns:
            True if directory was added and scanned successfully
        """
        # Add to config
        success = self.config_manager.add_directory(directory, dir_type)
        
        if success:
            # Scan the new directory immediately
            logger.info(f"Scanning new directory: {directory}")
            items = self._scan_single_directory(Path(directory))
            self._process_scanned_items(items)
            self._save_cache()
        
        return success
    
    def remove_directory(self, directory: str) -> bool:
        """
        Remove a directory and its assets from the registry.
        
        Args:
            directory: Directory path to remove
        
        Returns:
            True if directory was removed successfully
        """
        directory_path = str(Path(directory).resolve())
        
        # Remove from config
        success = self.config_manager.remove_directory(directory)
        
        if success:
            # Remove assets from this directory
            self._remove_assets_from_directory(directory_path)
            self._save_cache()
        
        return success
    
    def invalidate_cache(self) -> None:
        """Force cache invalidation for next scan."""
        self._last_scan_time = 0
        self._directory_mtimes.clear()
        logger.info("Registry cache invalidated")
    
    # Private methods
    
    def _cache_valid(self, current_time: float) -> bool:
        """Check if cache is still valid."""
        if self._last_scan_time == 0:
            return False
        
        elapsed = current_time - self._last_scan_time
        return elapsed < self._cache_ttl
    
    def _get_directories_to_scan(self, incremental: bool) -> List[Path]:
        """Get list of directories that need scanning."""
        all_dirs = self.config_manager.get_all_directories()
        
        if not incremental:
            return [Path(d) for d in all_dirs if Path(d).exists()]
        
        # Check for modified directories
        dirs_to_scan = []
        
        for dir_path in all_dirs:
            path = Path(dir_path)
            if not path.exists():
                continue
            
            # Get current mtime
            try:
                current_mtime = path.stat().st_mtime
            except OSError:
                continue
            
            # Check if directory has changed
            last_mtime = self._directory_mtimes.get(str(path), 0)
            if current_mtime > last_mtime:
                dirs_to_scan.append(path)
        
        return dirs_to_scan
    
    def _scan_directories(self, directories: List[Path]) -> List[Dict[str, Any]]:
        """Scan multiple directories for assets."""
        all_items = []
        
        for directory in directories:
            items = self._scan_single_directory(directory)
            all_items.extend(items)
        
        return all_items
    
    def _scan_single_directory(self, directory: Path) -> List[Dict[str, Any]]:
        """Scan a single directory for assets."""
        items = []
        
        try:
            # Check if it's a HuggingFace cache directory
            if directory.name == "hub" or "huggingface" in str(directory):
                # Use optimized HF scanning
                from .cache import get_huggingface_items
                items = get_huggingface_items(directory)
            else:
                # Use general asset detection
                from .cache import get_custom_items
                items = get_custom_items(directory)
        except Exception as e:
            logger.error(f"Error scanning directory {directory}: {e}")
        
        return items
    
    def _process_scanned_items(self, items: List[Dict[str, Any]]) -> None:
        """Process and categorize scanned items."""
        for item in items:
            asset_type = item.get('type', 'unknown')
            
            # Generate unique ID
            asset_id = self._generate_asset_id(item)
            
            # Enrich with additional metadata
            item['id'] = asset_id
            item['last_seen'] = datetime.now().isoformat()
            
            # Categorize by type
            if asset_type == 'model':
                self.models[asset_id] = item
            elif asset_type == 'lora_adapter':
                self.lora_adapters[asset_id] = item
            elif asset_type == 'dataset':
                self.datasets[asset_id] = item
            elif asset_type in ['custom_model', 'unknown_model']:
                self.custom_models[asset_id] = item
    
    def _generate_asset_id(self, item: Dict[str, Any]) -> str:
        """Generate a unique ID for an asset."""
        # Use path as primary identifier
        path = item.get('path', '')
        if path:
            return str(Path(path).resolve())
        
        # Fallback to name + type
        name = item.get('name', item.get('display_name', 'unknown'))
        asset_type = item.get('type', 'unknown')
        return f"{asset_type}:{name}"
    
    def _is_vllm_compatible(self, model: Dict[str, Any]) -> bool:
        """Check if a model is compatible with vLLM."""
        metadata = model.get('metadata', {})
        
        # Check architecture
        architectures = metadata.get('architectures', [])
        if architectures:
            for arch in architectures:
                if arch in self._vllm_compatible_architectures:
                    return True
        
        # Check model type
        model_type = metadata.get('model_type', '')
        vllm_types = {'llama', 'mistral', 'qwen', 'gpt', 'bloom', 'falcon', 'yi'}
        if any(t in model_type.lower() for t in vllm_types):
            return True
        
        # Check by name patterns
        name = model.get('name', '').lower()
        if any(pattern in name for pattern in ['llama', 'mistral', 'qwen', 'gpt']):
            return True
        
        return False
    
    def _is_lora_compatible(self, lora: Dict[str, Any], model_name: str) -> bool:
        """Check if a LoRA adapter is compatible with a model."""
        metadata = lora.get('metadata', {})
        base_model = metadata.get('base_model', '')
        
        if not base_model:
            return False
        
        # Normalize for comparison
        model_lower = model_name.lower()
        base_lower = base_model.lower()
        
        # Exact match
        if base_lower == model_lower:
            return True
        
        # Partial match
        if model_lower in base_lower or base_lower in model_lower:
            return True
        
        # Extract model family
        model_parts = model_name.split('/')
        if len(model_parts) > 1:
            model_short = model_parts[-1].lower()
            if model_short in base_lower:
                return True
        
        return False
    
    def _remove_assets_from_directory(self, directory: str) -> None:
        """Remove all assets from a specific directory."""
        collections = [
            self.models,
            self.lora_adapters,
            self.datasets,
            self.custom_models
        ]
        
        for collection in collections:
            to_remove = []
            for asset_id, asset in collection.items():
                asset_path = asset.get('path', '')
                if asset_path and str(Path(asset_path).parent) == directory:
                    to_remove.append(asset_id)
            
            for asset_id in to_remove:
                del collection[asset_id]
                logger.debug(f"Removed asset: {asset_id}")
    
    def _update_directory_mtimes(self, directories: List[Path]) -> None:
        """Update modification times for scanned directories."""
        for directory in directories:
            try:
                mtime = directory.stat().st_mtime
                self._directory_mtimes[str(directory)] = mtime
            except OSError:
                pass
    
    def _load_cache(self) -> None:
        """Load cached registry data from disk."""
        if not self._cache_file.exists():
            return
        
        try:
            with open(self._cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Restore data
            self.models = cache_data.get('models', {})
            self.lora_adapters = cache_data.get('lora_adapters', {})
            self.datasets = cache_data.get('datasets', {})
            self.custom_models = cache_data.get('custom_models', {})
            
            # Restore metadata
            self._last_scan_time = cache_data.get('last_scan_time', 0)
            self._directory_mtimes = cache_data.get('directory_mtimes', {})
            
            logger.info(f"Loaded registry cache: {len(self.models)} models, "
                       f"{len(self.lora_adapters)} LoRAs")
        except Exception as e:
            logger.warning(f"Failed to load registry cache: {e}")
    
    def _save_cache(self) -> None:
        """Save registry data to disk cache."""
        try:
            # Ensure cache directory exists
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare cache data
            cache_data = {
                'models': self.models,
                'lora_adapters': self.lora_adapters,
                'datasets': self.datasets,
                'custom_models': self.custom_models,
                'last_scan_time': self._last_scan_time,
                'directory_mtimes': self._directory_mtimes,
                'timestamp': datetime.now().isoformat()
            }
            
            # Write to file
            with open(self._cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2, default=str)
            
            logger.debug(f"Saved registry cache to {self._cache_file}")
        except Exception as e:
            logger.error(f"Failed to save registry cache: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics."""
        self.scan_all()  # Ensure data is current
        
        total_size = 0
        for collection in [self.models, self.lora_adapters, self.datasets, self.custom_models]:
            for asset in collection.values():
                total_size += asset.get('size', 0)
        
        return {
            'models_count': len(self.models),
            'lora_count': len(self.lora_adapters),
            'datasets_count': len(self.datasets),
            'custom_count': len(self.custom_models),
            'total_count': len(self.models) + len(self.lora_adapters) + 
                          len(self.datasets) + len(self.custom_models),
            'total_size': total_size,
            'vllm_compatible_count': len(self.get_models_for_vllm()),
            'cache_age': time.time() - self._last_scan_time if self._last_scan_time else None,
            'directories_monitored': len(self.config_manager.get_all_directories())
        }


# Global registry instance
_global_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """Get the global ModelRegistry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ModelRegistry()
    return _global_registry