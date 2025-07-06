"""
Unit tests for utils.py module.

Tests the asset grouping and duplicate detection functionality.
"""

import pytest
from datetime import datetime
from unittest.mock import patch

from hf_model_tool.utils import group_and_identify_duplicates


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
            },
            {
                "name": "datasets--squad--v1",
                "size": 2000,
                "date": datetime(2023, 1, 2),
                "type": "dataset",
                "path": "/path/to/dataset",
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

    def test_duplicate_detection(self):
        """Test detection of duplicate assets."""
        items = [
            {
                "name": "models--huggingface--bert-base-uncased",
                "size": 1000,
                "date": datetime(2023, 1, 1),
                "type": "model",
                "path": "/path/to/model1",
            },
            {
                "name": "models--huggingface--bert-base-uncased",  # Same name = duplicate
                "size": 1100,
                "date": datetime(2023, 1, 2),
                "type": "model",
                "path": "/path/to/model2",
            },
        ]

        grouped, duplicates = group_and_identify_duplicates(items)

        # Should detect duplicates
        assert len(duplicates) == 1
        duplicate_set = list(duplicates)[0]
        assert len(duplicate_set) == 1  # Only one unique name
        assert "models--huggingface--bert-base-uncased" in duplicate_set

        # Both items should be marked as duplicates
        for item in grouped["models"]["huggingface"]:
            assert item["is_duplicate"] is True

    def test_complex_name_patterns(self):
        """Test handling of complex naming patterns."""
        items = [
            {
                "name": "models--microsoft--DialoGPT-medium",
                "size": 1000,
                "date": datetime(2023, 1, 1),
                "type": "model",
                "path": "/path/1",
            },
            {
                "name": "datasets--glue--cola",
                "size": 2000,
                "date": datetime(2023, 1, 2),
                "type": "dataset",
                "path": "/path/2",
            },
            {
                "name": "models--huggingface--transformers--main--pytorch_model.bin",
                "size": 3000,
                "date": datetime(2023, 1, 3),
                "type": "model",
                "path": "/path/3",
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
        assert grouped == {"models": {}, "datasets": {}}
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
            # Invalid name format
            {
                "name": "invalid-name",
                "size": 4000,
                "date": datetime(2023, 1, 4),
                "type": "model",
                "path": "/path/4",
            },
        ]

        grouped, duplicates = group_and_identify_duplicates(items)

        # Should only process valid items
        assert len(grouped["models"]["huggingface"]) == 1
        assert len(grouped["datasets"]) == 0

        # Should log warnings for malformed items
        assert mock_logger.warning.called

    def test_unknown_asset_type(self):
        """Test handling of unknown asset types."""
        items = [
            {
                "name": "unknown--publisher--asset",
                "size": 1000,
                "date": datetime(2023, 1, 1),
                "type": "unknown",  # Unknown type
                "path": "/path/1",
            }
        ]

        with patch("hf_model_tool.utils.logger") as mock_logger:
            grouped, duplicates = group_and_identify_duplicates(items)

            # Should log warning and skip item
            mock_logger.warning.assert_called()
            assert len(grouped["models"]) == 0
            assert len(grouped["datasets"]) == 0

    def test_single_part_names(self):
        """Test handling of names that don't follow the expected pattern."""
        items = [
            {
                "name": "single-name",
                "size": 1000,
                "date": datetime(2023, 1, 1),
                "type": "model",
                "path": "/path/1",
            }
        ]

        with patch("hf_model_tool.utils.logger") as mock_logger:
            grouped, duplicates = group_and_identify_duplicates(items)

            # Should log warning and skip item
            mock_logger.warning.assert_called()
            assert len(grouped["models"]) == 0

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
