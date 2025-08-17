#!/usr/bin/env python3
"""
LoRA Adapter Management for HF-MODEL-TOOL.

Provides comprehensive LoRA adapter management including compatibility checking,
validation, and profile management for saved configurations.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime

from .registry import get_registry

logger = logging.getLogger(__name__)


@dataclass
class LoRAProfile:
    """Represents a saved LoRA configuration."""
    name: str
    base_model: str
    lora_adapters: List[str]
    merged_size_estimate: int
    created_at: str
    last_used: Optional[str] = None
    notes: Optional[str] = None
    vllm_args: Optional[Dict[str, Any]] = None


class LoRAManager:
    """
    Manages LoRA adapters with compatibility checking and profile management.
    
    This manager provides:
    - Model-LoRA compatibility validation
    - Architecture and dimension checking
    - Memory requirement estimation
    - Profile management for tested configurations
    """
    
    def __init__(self):
        """Initialize the LoRA manager."""
        self.registry = get_registry()
        self.profiles_file = Path.home() / ".config" / "hf-model-tool" / "lora_profiles.json"
        self.profiles: Dict[str, LoRAProfile] = {}
        self._compatibility_cache: Dict[Tuple[str, str], bool] = {}
        
        # Load saved profiles
        self._load_profiles()
        
        # Known LoRA-compatible architectures
        self._lora_architectures = {
            "LlamaForCausalLM", "MistralForCausalLM", "Qwen2ForCausalLM",
            "GPTNeoXForCausalLM", "BloomForCausalLM", "FalconForCausalLM",
            "BaichuanForCausalLM", "ChatGLMModel", "InternLMForCausalLM",
            "YiForCausalLM", "DeepseekForCausalLM", "GemmaForCausalLM"
        }
    
    def get_compatible_loras(self, model_identifier: str) -> List[Dict[str, Any]]:
        """
        Find LoRA adapters compatible with a specific model.
        
        Args:
            model_identifier: Model ID, name, or path
        
        Returns:
            List of compatible LoRA adapters with compatibility info
        """
        # Ensure registry is current
        self.registry.scan_all()
        
        # Find the base model
        model = self.registry.find_asset(model_identifier)
        if not model:
            logger.warning(f"Model not found: {model_identifier}")
            return []
        
        compatible_loras = []
        
        for lora_id, lora in self.registry.lora_adapters.items():
            compatibility = self.check_compatibility(model, lora)
            if compatibility['compatible']:
                # Enrich LoRA info with compatibility details
                enriched_lora = lora.copy()
                enriched_lora['compatibility'] = compatibility
                compatible_loras.append(enriched_lora)
        
        # Sort by compatibility score (exact matches first)
        compatible_loras.sort(
            key=lambda x: x['compatibility'].get('score', 0),
            reverse=True
        )
        
        return compatible_loras
    
    def check_compatibility(
        self, 
        model: Dict[str, Any], 
        lora: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check if a LoRA adapter is compatible with a model.
        
        Args:
            model: Model information dictionary
            lora: LoRA adapter information dictionary
        
        Returns:
            Dictionary with compatibility status and details
        """
        # Check cache first
        cache_key = (model.get('id', ''), lora.get('id', ''))
        if cache_key in self._compatibility_cache:
            return {'compatible': self._compatibility_cache[cache_key]}
        
        result = {
            'compatible': False,
            'score': 0,
            'issues': [],
            'warnings': []
        }
        
        # Extract metadata
        model_meta = model.get('metadata', {})
        lora_meta = lora.get('metadata', {})
        
        # Check base model match
        lora_base = lora_meta.get('base_model', '')
        model_name = model.get('name', '')
        
        if not lora_base:
            result['issues'].append("LoRA has no base model specified")
            return result
        
        # Exact match is best
        if lora_base.lower() == model_name.lower():
            result['score'] = 100
        # Partial match is acceptable
        elif lora_base.lower() in model_name.lower() or model_name.lower() in lora_base.lower():
            result['score'] = 70
            result['warnings'].append(f"Partial match: LoRA base '{lora_base}' vs model '{model_name}'")
        else:
            # Check model family
            if self._same_model_family(model_name, lora_base):
                result['score'] = 50
                result['warnings'].append(f"Same model family but different variant")
            else:
                result['issues'].append(f"Base model mismatch: LoRA expects '{lora_base}', got '{model_name}'")
                return result
        
        # Check architecture compatibility
        model_arch = model_meta.get('architectures', [])
        if model_arch:
            if not any(arch in self._lora_architectures for arch in model_arch):
                result['warnings'].append(f"Model architecture {model_arch} may not support LoRA")
                result['score'] -= 10
        
        # Check dimensions if available
        dimension_compat = self._check_dimensions(model_meta, lora_meta)
        if dimension_compat['compatible']:
            result['score'] += 10
        else:
            if dimension_compat.get('error'):
                result['issues'].append(dimension_compat['error'])
                return result
            elif dimension_compat.get('warning'):
                result['warnings'].append(dimension_compat['warning'])
                result['score'] -= 5
        
        # If we got here with a score > 0, it's compatible
        if result['score'] > 0:
            result['compatible'] = True
        
        # Cache the result
        self._compatibility_cache[cache_key] = result['compatible']
        
        return result
    
    def validate_combination(
        self,
        model_identifier: str,
        lora_identifiers: List[str],
        available_memory: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Validate a model + LoRA combination before serving.
        
        Args:
            model_identifier: Base model ID
            lora_identifiers: List of LoRA adapter IDs
            available_memory: Available GPU memory in bytes (optional)
        
        Returns:
            Validation result with details
        """
        self.registry.scan_all()
        
        result = {
            'valid': True,
            'issues': [],
            'warnings': [],
            'estimated_size': 0,
            'memory_ok': True
        }
        
        # Find base model
        model = self.registry.find_asset(model_identifier)
        if not model:
            result['valid'] = False
            result['issues'].append(f"Model not found: {model_identifier}")
            return result
        
        # Start with model size
        model_size = model.get('size', 0)
        result['estimated_size'] = model_size
        
        # Check each LoRA
        for lora_id in lora_identifiers:
            lora = self.registry.find_asset(lora_id)
            if not lora:
                result['issues'].append(f"LoRA not found: {lora_id}")
                result['valid'] = False
                continue
            
            # Check compatibility
            compat = self.check_compatibility(model, lora)
            if not compat['compatible']:
                result['issues'].extend(compat.get('issues', []))
                result['valid'] = False
            else:
                result['warnings'].extend(compat.get('warnings', []))
            
            # Add LoRA size
            lora_size = lora.get('size', 0)
            result['estimated_size'] += lora_size
        
        # Check if multiple LoRAs might conflict
        if len(lora_identifiers) > 1:
            result['warnings'].append(f"Loading {len(lora_identifiers)} LoRAs - ensure they don't conflict")
        
        # Memory check if provided
        if available_memory and result['estimated_size'] > 0:
            # Add 20% overhead for activation memory
            required_memory = int(result['estimated_size'] * 1.2)
            if required_memory > available_memory:
                result['memory_ok'] = False
                result['issues'].append(
                    f"Insufficient memory: need {required_memory/(1024**3):.1f}GB, "
                    f"have {available_memory/(1024**3):.1f}GB"
                )
                result['valid'] = False
        
        return result
    
    def create_profile(
        self,
        name: str,
        model_identifier: str,
        lora_identifiers: List[str],
        notes: Optional[str] = None,
        vllm_args: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Save a tested model+LoRA configuration as a profile.
        
        Args:
            name: Profile name
            model_identifier: Base model ID
            lora_identifiers: List of LoRA adapter IDs
            notes: Optional notes about the configuration
            vllm_args: Optional vLLM-specific arguments
        
        Returns:
            True if profile was created successfully
        """
        # Validate the combination first
        validation = self.validate_combination(model_identifier, lora_identifiers)
        if not validation['valid']:
            logger.error(f"Cannot create profile: {validation['issues']}")
            return False
        
        # Create profile
        profile = LoRAProfile(
            name=name,
            base_model=model_identifier,
            lora_adapters=lora_identifiers,
            merged_size_estimate=validation['estimated_size'],
            created_at=datetime.now().isoformat(),
            notes=notes,
            vllm_args=vllm_args
        )
        
        # Save profile
        self.profiles[name] = profile
        self._save_profiles()
        
        logger.info(f"Created LoRA profile: {name}")
        return True
    
    def get_profile(self, name: str) -> Optional[LoRAProfile]:
        """Get a saved profile by name."""
        return self.profiles.get(name)
    
    def list_profiles(self) -> List[LoRAProfile]:
        """List all saved profiles."""
        return list(self.profiles.values())
    
    def delete_profile(self, name: str) -> bool:
        """Delete a saved profile."""
        if name in self.profiles:
            del self.profiles[name]
            self._save_profiles()
            logger.info(f"Deleted profile: {name}")
            return True
        return False
    
    def use_profile(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Load a profile and return its configuration.
        
        Args:
            name: Profile name
        
        Returns:
            Configuration dictionary ready for vLLM
        """
        profile = self.profiles.get(name)
        if not profile:
            logger.error(f"Profile not found: {name}")
            return None
        
        # Update last used timestamp
        profile.last_used = datetime.now().isoformat()
        self._save_profiles()
        
        # Build configuration
        config = {
            'model': profile.base_model,
            'lora_modules': profile.lora_adapters,
            'profile_name': name
        }
        
        # Add vLLM args if specified
        if profile.vllm_args:
            config.update(profile.vllm_args)
        
        return config
    
    def estimate_memory_requirement(
        self,
        model_identifier: str,
        lora_identifiers: List[str],
        quantization: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Estimate memory requirements for a model+LoRA combination.
        
        Args:
            model_identifier: Base model ID
            lora_identifiers: List of LoRA adapter IDs
            quantization: Quantization type (e.g., 'int4', 'int8')
        
        Returns:
            Memory requirement estimation
        """
        self.registry.scan_all()
        
        result = {
            'base_model_size': 0,
            'lora_sizes': {},
            'total_size': 0,
            'activation_memory': 0,
            'recommended_gpu_memory': 0,
            'quantization_factor': 1.0
        }
        
        # Get base model size
        model = self.registry.find_asset(model_identifier)
        if model:
            result['base_model_size'] = model.get('size', 0)
        
        # Apply quantization factor
        if quantization:
            if 'int4' in quantization.lower() or '4bit' in quantization.lower():
                result['quantization_factor'] = 0.25
            elif 'int8' in quantization.lower() or '8bit' in quantization.lower():
                result['quantization_factor'] = 0.5
        
        # Adjust base model size for quantization
        result['base_model_size'] = int(
            result['base_model_size'] * result['quantization_factor']
        )
        
        # Add LoRA sizes
        total_lora_size = 0
        for lora_id in lora_identifiers:
            lora = self.registry.find_asset(lora_id)
            if lora:
                lora_size = lora.get('size', 0)
                result['lora_sizes'][lora_id] = lora_size
                total_lora_size += lora_size
        
        # Calculate totals
        result['total_size'] = result['base_model_size'] + total_lora_size
        
        # Estimate activation memory (20-30% of model size)
        result['activation_memory'] = int(result['total_size'] * 0.25)
        
        # Recommended GPU memory (model + activation + buffer)
        result['recommended_gpu_memory'] = int(
            (result['total_size'] + result['activation_memory']) * 1.2
        )
        
        return result
    
    # Private helper methods
    
    def _same_model_family(self, name1: str, name2: str) -> bool:
        """Check if two model names are from the same family."""
        # Extract model family identifiers
        families = ['llama', 'mistral', 'qwen', 'yi', 'deepseek', 'gemma', 'phi']
        
        name1_lower = name1.lower()
        name2_lower = name2.lower()
        
        for family in families:
            if family in name1_lower and family in name2_lower:
                return True
        
        return False
    
    def _check_dimensions(
        self,
        model_meta: Dict[str, Any],
        lora_meta: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check dimensional compatibility between model and LoRA."""
        result = {'compatible': True}
        
        # Get model dimensions
        model_hidden = model_meta.get('hidden_size')
        model_layers = model_meta.get('num_hidden_layers')
        
        # Get LoRA rank
        lora_rank = lora_meta.get('rank', lora_meta.get('r', 16))
        
        # Basic sanity checks
        if model_hidden and lora_rank:
            if lora_rank > model_hidden:
                result['compatible'] = False
                result['error'] = f"LoRA rank {lora_rank} exceeds model hidden size {model_hidden}"
            elif lora_rank > model_hidden // 4:
                result['warning'] = f"LoRA rank {lora_rank} is large for hidden size {model_hidden}"
        
        return result
    
    def _load_profiles(self) -> None:
        """Load saved profiles from disk."""
        if not self.profiles_file.exists():
            return
        
        try:
            with open(self.profiles_file, 'r') as f:
                data = json.load(f)
                
            for name, profile_data in data.items():
                self.profiles[name] = LoRAProfile(**profile_data)
            
            logger.info(f"Loaded {len(self.profiles)} LoRA profiles")
        except Exception as e:
            logger.error(f"Error loading LoRA profiles: {e}")
    
    def _save_profiles(self) -> None:
        """Save profiles to disk."""
        try:
            # Ensure directory exists
            self.profiles_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert profiles to dict
            data = {
                name: asdict(profile)
                for name, profile in self.profiles.items()
            }
            
            # Save to file
            with open(self.profiles_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Saved {len(self.profiles)} LoRA profiles")
        except Exception as e:
            logger.error(f"Error saving LoRA profiles: {e}")


# Global LoRA manager instance
_global_lora_manager: Optional[LoRAManager] = None


def get_lora_manager() -> LoRAManager:
    """Get the global LoRAManager instance."""
    global _global_lora_manager
    if _global_lora_manager is None:
        _global_lora_manager = LoRAManager()
    return _global_lora_manager