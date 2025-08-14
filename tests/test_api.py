"""
Unit tests for api.py module.

Tests the public API functions for retrieving HuggingFace models
in VLLM-compatible naming format.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime

from hf_model_tool.api import (
    get_downloaded_models,
    get_model_info,
    _extract_vllm_model_name,
)


@pytest.mark.unit
class TestGetDownloadedModels:
    """Test cases for get_downloaded_models function."""

    @patch("hf_model_tool.api.scan_all_directories")
    def test_get_downloaded_models_basic(self, mock_scan):
        """Test basic functionality of get_downloaded_models."""
        # Mock scan results with various asset types
        mock_scan.return_value = [
            {
                "name": "models--bert-base-uncased",
                "type": "model",
                "source_type": "huggingface_cache",
                "size": 1000000,
                "path": "/cache/models--bert-base-uncased",
            },
            {
                "name": "models--facebook--bart-large-cnn",
                "type": "model",
                "source_type": "huggingface_cache",
                "size": 2000000,
                "path": "/cache/models--facebook--bart-large-cnn",
            },
            {
                "name": "datasets--squad--v1",
                "type": "dataset",
                "source_type": "huggingface_cache",
                "size": 500000,
                "path": "/cache/datasets--squad--v1",
            },
        ]

        models = get_downloaded_models()

        # Should only return models, not datasets
        assert len(models) == 2
        assert "bert-base-uncased" in models
        assert "facebook/bart-large-cnn" in models

    @patch("hf_model_tool.api.scan_all_directories")
    def test_get_downloaded_models_with_custom(self, mock_scan):
        """Test including custom models."""
        mock_scan.return_value = [
            {
                "name": "models--microsoft--Florence-2-large",
                "type": "model",
                "source_type": "huggingface_cache",
                "size": 3000000,
                "path": "/cache/models--microsoft--Florence-2-large",
            },
            {
                "name": "my-custom-model",
                "type": "custom_model",
                "source_type": "custom_directory",
                "display_name": "my-custom-model",
                "size": 1500000,
                "path": "/custom/my-custom-model",
            },
        ]

        # Include custom models
        models = get_downloaded_models(include_custom_models=True)
        assert len(models) == 2
        assert "microsoft/Florence-2-large" in models
        assert "my-custom-model" in models

        # Exclude custom models
        models = get_downloaded_models(include_custom_models=False)
        assert len(models) == 1
        assert "microsoft/Florence-2-large" in models

    @patch("hf_model_tool.api.scan_all_directories")
    def test_get_downloaded_models_with_lora(self, mock_scan):
        """Test including LoRA adapters."""
        mock_scan.return_value = [
            {
                "name": "models--Qwen--Qwen3-30B",
                "type": "model",
                "source_type": "huggingface_cache",
                "size": 5000000,
                "path": "/cache/models--Qwen--Qwen3-30B",
            },
            {
                "name": "qwen-lora-adapter",
                "type": "lora_adapter",
                "source_type": "custom_directory",
                "display_name": "qwen-lora-adapter",
                "metadata": {"base_model": "Qwen/Qwen3-30B-Instruct"},
                "size": 100000,
                "path": "/custom/qwen-lora-adapter",
            },
        ]

        # Include LoRA adapters
        models = get_downloaded_models(include_lora_adapters=True)
        assert len(models) == 2
        assert "Qwen/Qwen3-30B" in models
        assert "Qwen/Qwen3-30B-Instruct" in models  # Base model from LoRA metadata

        # Exclude LoRA adapters
        models = get_downloaded_models(include_lora_adapters=False)
        assert len(models) == 1
        assert "Qwen/Qwen3-30B" in models

    @patch("hf_model_tool.api.scan_all_directories")
    def test_get_downloaded_models_deduplication(self, mock_scan):
        """Test deduplication of model names."""
        mock_scan.return_value = [
            {
                "name": "models--bert-base-uncased",
                "type": "model",
                "source_type": "huggingface_cache",
                "size": 1000000,
                "path": "/cache1/models--bert-base-uncased",
            },
            {
                "name": "models--bert-base-uncased",
                "type": "model",
                "source_type": "huggingface_cache",
                "size": 1000000,
                "path": "/cache2/models--bert-base-uncased",
            },
        ]

        # With deduplication (default)
        models = get_downloaded_models(deduplicate=True)
        assert len(models) == 1
        assert "bert-base-uncased" in models

        # Without deduplication
        models = get_downloaded_models(deduplicate=False)
        assert len(models) == 2
        assert models.count("bert-base-uncased") == 2

    @patch("hf_model_tool.api.scan_all_directories")
    def test_get_downloaded_models_empty(self, mock_scan):
        """Test with no models found."""
        mock_scan.return_value = []

        models = get_downloaded_models()
        assert models == []

    @patch("hf_model_tool.api.scan_all_directories")
    def test_get_downloaded_models_error_handling(self, mock_scan):
        """Test error handling in get_downloaded_models."""
        mock_scan.side_effect = Exception("Scan error")

        models = get_downloaded_models()
        assert models == []  # Should return empty list on error


@pytest.mark.unit
class TestExtractVllmModelName:
    """Test cases for _extract_vllm_model_name function."""

    def test_extract_huggingface_cache_with_publisher(self):
        """Test extraction from HF cache format with publisher."""
        item = {
            "name": "models--facebook--bart-large-cnn",
            "source_type": "huggingface_cache",
        }

        name = _extract_vllm_model_name(item)
        assert name == "facebook/bart-large-cnn"

    def test_extract_huggingface_cache_without_publisher(self):
        """Test extraction from HF cache format without publisher."""
        item = {"name": "models--bert-base-uncased", "source_type": "huggingface_cache"}

        name = _extract_vllm_model_name(item)
        assert name == "bert-base-uncased"

    def test_extract_model_with_double_dash_in_name(self):
        """Test model names containing '--' are handled correctly."""
        item = {
            "name": "models--microsoft--Florence-2-large--ft",
            "source_type": "huggingface_cache",
        }

        name = _extract_vllm_model_name(item)
        assert name == "microsoft/Florence-2-large-ft"

    def test_extract_custom_model(self):
        """Test extraction from custom directory."""
        item = {
            "name": "my-custom-model",
            "source_type": "custom_directory",
            "display_name": "my-custom-model",
            "type": "custom_model",
        }

        name = _extract_vllm_model_name(item)
        assert name == "my-custom-model"

    def test_extract_lora_with_base_model(self):
        """Test extraction of LoRA adapter with base model."""
        item = {
            "name": "qwen-lora",
            "source_type": "custom_directory",
            "type": "lora_adapter",
            "display_name": "qwen-lora (2024-12-25 10:30)",
            "metadata": {"base_model": "Qwen/Qwen3-30B-Instruct"},
        }

        name = _extract_vllm_model_name(item)
        assert name == "Qwen/Qwen3-30B-Instruct"

    def test_extract_lora_without_base_model(self):
        """Test extraction of LoRA adapter without base model."""
        item = {
            "name": "qwen-lora",
            "source_type": "custom_directory",
            "type": "lora_adapter",
            "display_name": "qwen-lora (2024-12-25 10:30)",
            "metadata": {},
        }

        name = _extract_vllm_model_name(item)
        assert name == "qwen-lora"  # Timestamp should be removed

    def test_extract_with_publisher_and_display_name(self):
        """Test extraction using publisher and display_name fallback."""
        item = {
            "name": "some-internal-name",
            "publisher": "openai",
            "display_name": "gpt-model",
        }

        name = _extract_vllm_model_name(item)
        assert name == "openai/gpt-model"

    def test_extract_invalid_item(self):
        """Test extraction from invalid item returns None."""
        item = {}

        name = _extract_vllm_model_name(item)
        assert name is None


@pytest.mark.unit
class TestGetModelInfo:
    """Test cases for get_model_info function."""

    @patch("hf_model_tool.api.scan_all_directories")
    def test_get_model_info_found(self, mock_scan):
        """Test getting info for an existing model."""
        test_date = datetime.now()
        mock_scan.return_value = [
            {
                "name": "models--microsoft--Florence-2-large",
                "type": "model",
                "subtype": "huggingface",
                "source_type": "huggingface_cache",
                "size": 3000000,
                "path": "/cache/models--microsoft--Florence-2-large",
                "date": test_date,
                "metadata": {
                    "model_type": "florence2",
                    "architectures": ["Florence2Model"],
                },
                "source_dir": "/cache",
            }
        ]

        info = get_model_info("microsoft/Florence-2-large")

        assert info is not None
        assert info["name"] == "microsoft/Florence-2-large"
        assert info["path"] == "/cache/models--microsoft--Florence-2-large"
        assert info["size"] == 3000000
        assert info["type"] == "model"
        assert info["subtype"] == "huggingface"
        assert info["metadata"]["model_type"] == "florence2"
        assert info["last_modified"] == test_date

    @patch("hf_model_tool.api.scan_all_directories")
    def test_get_model_info_not_found(self, mock_scan):
        """Test getting info for non-existent model."""
        mock_scan.return_value = [
            {
                "name": "models--bert-base-uncased",
                "type": "model",
                "source_type": "huggingface_cache",
                "path": "/cache/models--bert-base-uncased",
            }
        ]

        info = get_model_info("non-existent-model")
        assert info is None

    @patch("hf_model_tool.api.scan_all_directories")
    def test_get_model_info_error_handling(self, mock_scan):
        """Test error handling in get_model_info."""
        mock_scan.side_effect = Exception("Scan error")

        info = get_model_info("any-model")
        assert info is None
