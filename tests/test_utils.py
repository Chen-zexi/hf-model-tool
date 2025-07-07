"""
Unit tests for utils.py module.

Tests the asset grouping and duplicate detection functionality.
"""

import pytest
from datetime import datetime
from unittest.mock import patch

from hf_model_tool.utils import group_and_identify_duplicates


@pytest.mark.unit
class TestGroupAndIdentifyDuplicates:
    """Test cases for group_and_identify_duplicates function."""

    def test_basic_grouping(self):
        """Test basic asset grouping by type and publisher."""
        items = [
            {
                "name": "models--huggingface--bert-base-uncased",
                "size": 1000,
                "date": datetime(2023, 1, 1),
                "type": "model",
                "path": "/path/to/model",
                "display_name": "bert-base-uncased",  # Provide display_name explicitly
                "source_type": "huggingface_cache",
                "publisher": "huggingface",
            },
            {
                "name": "datasets--squad--v1",
                "size": 2000,
                "date": datetime(2023, 1, 2),
                "type": "dataset",
                "path": "/path/to/dataset",
                "display_name": "v1",  # Provide display_name explicitly
                "source_type": "huggingface_cache",
                "publisher": "squad",
            },
        ]

        grouped, duplicates = group_and_identify_duplicates(items)

        # Check structure
        assert "models" in grouped
        assert "datasets" in grouped

        # Check model grouping
        assert "huggingface" in grouped["models"]
        assert len(grouped["models"]["huggingface"]) == 1
        model_item = grouped["models"]["huggingface"][0]
        assert model_item["display_name"] == "bert-base-uncased"
        assert model_item["is_duplicate"] is False

        # Check dataset grouping
        assert "squad" in grouped["datasets"]
        assert len(grouped["datasets"]["squad"]) == 1
        dataset_item = grouped["datasets"]["squad"][0]
        assert dataset_item["display_name"] == "v1"
        assert dataset_item["is_duplicate"] is False

        # No duplicates
        assert len(duplicates) == 0

    def test_duplicate_detection_huggingface_cache(self):
        """Test detection of duplicate assets for HuggingFace cache (requires same publisher, name, AND size)."""
        items = [
            {
                "name": "models--huggingface--bert-base-uncased",
                "size": 1000,
                "date": datetime(2023, 1, 1),
                "type": "model",
                "path": "/path/to/model1",
                "display_name": "bert-base-uncased",
                "publisher": "huggingface",
                "source_type": "huggingface_cache",
            },
            {
                "name": "models--huggingface--bert-base-uncased",  
                "size": 1000,  # Same size = duplicate
                "date": datetime(2023, 1, 2),
                "type": "model",
                "path": "/path/to/model2",
                "display_name": "bert-base-uncased",
                "publisher": "huggingface",
                "source_type": "huggingface_cache",
            },
        ]

        grouped, duplicates = group_and_identify_duplicates(items)

        # Should detect duplicates (same publisher, name, and size)
        assert len(duplicates) == 1
        duplicate_set = list(duplicates)[0]
        assert len(duplicate_set) == 1  # One unique name appears multiple times
        assert "models--huggingface--bert-base-uncased" in duplicate_set

        # Both items should be marked as duplicates
        for item in grouped["models"]["huggingface"]:
            assert item["is_duplicate"] is True

    def test_no_duplicate_different_sizes_huggingface(self):
        """Test that HuggingFace items with same name but different sizes are NOT duplicates."""
        items = [
            {
                "name": "models--huggingface--bert-base-uncased",
                "size": 1000,
                "date": datetime(2023, 1, 1),
                "type": "model",
                "path": "/path/to/model1",
                "display_name": "bert-base-uncased",
                "publisher": "huggingface",
                "source_type": "huggingface_cache",
            },
            {
                "name": "models--huggingface--bert-base-uncased",  
                "size": 1100,  # Different size = NOT duplicate
                "date": datetime(2023, 1, 2),
                "type": "model",
                "path": "/path/to/model2",
                "display_name": "bert-base-uncased",
                "publisher": "huggingface",
                "source_type": "huggingface_cache",
            },
        ]

        grouped, duplicates = group_and_identify_duplicates(items)

        # Should NOT detect duplicates (different sizes)
        assert len(duplicates) == 0

        # Both items should NOT be marked as duplicates
        for item in grouped["models"]["huggingface"]:
            assert item["is_duplicate"] is False

    def test_lora_adapters_not_duplicates(self):
        """Test that LoRA adapters from custom directories are NOT treated as duplicates."""
        items = [
            {
                "name": "lora_unsloth_Qwen3_4B_unsloth_bnb_4bit",
                "size": 560000000,
                "date": datetime(2023, 1, 1),
                "type": "lora_adapter",
                "path": "/path/to/lora1",
                "display_name": "lora_unsloth_Qwen3_4B_unsloth_bnb_4bit (2025-07-04 12:16)",
                "source_type": "custom_directory",
            },
            {
                "name": "lora_unsloth_Qwen3_4B_unsloth_bnb_4bit",
                "size": 560000000,
                "date": datetime(2023, 1, 2),
                "type": "lora_adapter",
                "path": "/path/to/lora2",  # Different path
                "display_name": "lora_unsloth_Qwen3_4B_unsloth_bnb_4bit (2025-07-04 13:20)",
                "source_type": "custom_directory",
            },
        ]

        grouped, duplicates = group_and_identify_duplicates(items)

        # Should NOT detect duplicates (LoRA adapters use path as unique key)
        assert len(duplicates) == 0

        # Items should NOT be marked as duplicates
        for category in grouped.values():
            for publisher in category.values():
                for item in publisher:
                    assert item["is_duplicate"] is False

    def test_complex_name_patterns(self):
        """Test handling of complex naming patterns."""
        items = [
            {
                "name": "models--microsoft--DialoGPT-medium",
                "size": 1000,
                "date": datetime(2023, 1, 1),
                "type": "model",
                "path": "/path/1",
                "display_name": "DialoGPT-medium",
                "source_type": "huggingface_cache",
                "publisher": "microsoft",
            },
            {
                "name": "datasets--glue--cola",
                "size": 2000,
                "date": datetime(2023, 1, 2),
                "type": "dataset",
                "path": "/path/2",
                "display_name": "cola",
                "source_type": "huggingface_cache",
                "publisher": "glue",
            },
            {
                "name": "models--huggingface--transformers--main--pytorch_model.bin",
                "size": 3000,
                "date": datetime(2023, 1, 3),
                "type": "model",
                "path": "/path/3",
                "display_name": "transformers--main--pytorch_model.bin",
                "source_type": "huggingface_cache",
                "publisher": "huggingface",
            },
        ]

        grouped, duplicates = group_and_identify_duplicates(items)

        # Check complex names are handled correctly
        assert "microsoft" in grouped["models"]
        assert grouped["models"]["microsoft"][0]["display_name"] == "DialoGPT-medium"

        assert "glue" in grouped["datasets"]
        assert grouped["datasets"]["glue"][0]["display_name"] == "cola"

        assert "huggingface" in grouped["models"]
        assert (
            grouped["models"]["huggingface"][0]["display_name"]
            == "transformers--main--pytorch_model.bin"
        )

    def test_invalid_input_handling(self):
        """Test handling of invalid input."""
        # Non-list input
        with pytest.raises(ValueError, match="Items must be a list"):
            group_and_identify_duplicates("not a list")

        # Empty list
        grouped, duplicates = group_and_identify_duplicates([])
        assert grouped == {}  # Empty result since no categories have items
        assert len(duplicates) == 0

    @patch("hf_model_tool.utils.logger")
    def test_malformed_items_handling(self, mock_logger):
        """Test handling of malformed items."""
        items = [
            # Valid item
            {
                "name": "models--huggingface--bert",
                "size": 1000,
                "date": datetime(2023, 1, 1),
                "type": "model",
                "path": "/path/1",
                "display_name": "bert",
                "source_type": "huggingface_cache",
                "publisher": "huggingface",
            },
            # Missing name
            {
                "size": 2000,
                "date": datetime(2023, 1, 2),
                "type": "dataset",
                "path": "/path/2",
            },
            # Missing type
            {
                "name": "datasets--squad--v1",
                "size": 3000,
                "date": datetime(2023, 1, 3),
                "path": "/path/3",
            },
            # Invalid item format
            {
                "name": "invalid-name",
                "size": 4000,
                "date": datetime(2023, 1, 4),
                "type": "unknown",  # This will be categorized as unknown
                "path": "/path/4",
                "display_name": "invalid-name",
                "source_type": "unknown",
            },
        ]

        grouped, duplicates = group_and_identify_duplicates(items)

        # Should only process valid items properly
        if "models" in grouped and "huggingface" in grouped["models"]:
            assert len(grouped["models"]["huggingface"]) == 1
        
        # Should log warnings for malformed items
        assert mock_logger.warning.called

    def test_unknown_asset_type(self):
        """Test handling of unknown asset types."""
        items = [
            {
                "name": "unknown--publisher--asset",
                "size": 1000,
                "date": datetime(2023, 1, 1),
                "type": "unknown_new_type",  # Type not in predefined categories
                "path": "/path/1",
                "display_name": "asset",
                "source_type": "unknown",
            }
        ]

        grouped, duplicates = group_and_identify_duplicates(items)

        # Unknown asset types fall back to "unknown" category without warning
        # (warning only happens if returned category doesn't exist)
        assert "unknown" in grouped
        assert len(grouped["unknown"]) > 0

    def test_single_part_names(self):
        """Test handling of names that don't follow the expected pattern."""
        items = [
            {
                "name": "single-name",
                "size": 1000,
                "date": datetime(2023, 1, 1),
                "type": "model",
                "path": "/path/1",
                "display_name": "single-name",
                "source_type": "custom_directory",  # Custom paths don't require HF naming
            }
        ]

        grouped, duplicates = group_and_identify_duplicates(items)

        # Should be categorized as custom model since it doesn't follow HF pattern
        # and has custom_directory source type
        if "custom_models" in grouped:
            assert len(grouped["custom_models"]) > 0
        elif "unknown_models" in grouped:
            assert len(grouped["unknown_models"]) > 0

    def test_multiple_duplicates(self):
        """Test detection of multiple duplicate sets."""
        items = [
            # First duplicate set - same names
            {
                "name": "models--pub1--model1",
                "size": 1000,
                "date": datetime(2023, 1, 1),
                "type": "model",
                "path": "/p1",
            },
            {
                "name": "models--pub1--model1",
                "size": 1100,
                "date": datetime(2023, 1, 2),
                "type": "model",
                "path": "/p2",
            },
            # Second duplicate set - same names
            {
                "name": "datasets--pub2--data1",
                "size": 2000,
                "date": datetime(2023, 1, 3),
                "type": "dataset",
                "path": "/p3",
            },
            {
                "name": "datasets--pub2--data1",
                "size": 2100,
                "date": datetime(2023, 1, 4),
                "type": "dataset",
                "path": "/p4",
            },
            # Non-duplicate
            {
                "name": "models--pub3--unique",
                "size": 3000,
                "date": datetime(2023, 1, 5),
                "type": "model",
                "path": "/p5",
            },
        ]

        grouped, duplicates = group_and_identify_duplicates(items)

        # Should detect 2 duplicate sets
        assert len(duplicates) == 2

        # Check that duplicates are correctly marked
        for item in grouped["models"]["pub1"]:
            assert item["is_duplicate"] is True

        for item in grouped["datasets"]["pub2"]:
            assert item["is_duplicate"] is True

        for item in grouped["models"]["pub3"]:
            assert item["is_duplicate"] is False
