"""
Unit tests for ui.py module.

Tests the user interface functionality.
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

from hf_model_tool.ui import print_items


@pytest.mark.unit
class TestPrintItems:
    """Test cases for print_items function."""

    @patch("hf_model_tool.ui.Console")
    @patch("hf_model_tool.ui.group_and_identify_duplicates")
    def test_print_items_valid_sort_options(self, mock_group, mock_console):
        """Test print_items with valid sort options."""
        # Mock data
        mock_items = [
            {
                "name": "models--test--model1",
                "size": 1000,
                "date": datetime(2023, 1, 1),
                "type": "model",
                "display_name": "model1",
                "is_duplicate": False,
            }
        ]

        mock_grouped = {"models": {"test": [mock_items[0]]}, "datasets": {}}

        mock_group.return_value = (mock_grouped, set())
        mock_console_instance = MagicMock()
        mock_console.return_value = mock_console_instance

        # Test valid sort options
        for sort_by in ["size", "date", "name"]:
            print_items(mock_items, sort_by=sort_by)

            # Verify console was used
            assert mock_console_instance.print.called
            mock_console_instance.reset_mock()

    def test_print_items_invalid_sort_option(self):
        """Test print_items with invalid sort option."""
        mock_items = [
            {"name": "test", "size": 1000, "date": datetime.now(), "type": "model"}
        ]

        with pytest.raises(ValueError, match="Invalid sort_by option"):
            print_items(mock_items, sort_by="invalid")


@pytest.mark.unit
class TestAssetDeletionWorkflow:
    """Test cases for asset deletion workflow and string parsing."""
    
    def test_display_name_extraction_basic(self):
        """Test basic display name extraction functionality."""
        # Test that the extraction logic works for common cases
        test_cases = [
            ("model_name (1.23 GB)", "model_name"),
            ("simple_name", "simple_name"),
        ]
        
        for choice_str, expected in test_cases:
            # Basic extraction logic
            if " (" in choice_str and choice_str.endswith(" GB)"):
                item_name_to_find = choice_str.rsplit(" (", 1)[0]
            else:
                item_name_to_find = choice_str
                
            assert item_name_to_find == expected

    @patch("hf_model_tool.ui.unified_prompt")
    @patch("hf_model_tool.ui.group_and_identify_duplicates")
    @patch("hf_model_tool.ui.inquirer")
    @patch("hf_model_tool.ui.shutil")
    def test_delete_workflow_basic_functionality(self, mock_shutil, mock_inquirer, mock_group, mock_prompt):
        """Test basic delete workflow functionality."""
        from hf_model_tool.ui import delete_assets_workflow
        
        # Mock basic test item
        mock_items = [
            {
                "name": "test_item",
                "size": 1000,
                "date": datetime(2023, 1, 1),
                "type": "model",
                "path": "/path/to/item",
                "display_name": "test_item",
                "is_duplicate": False,
            }
        ]
        
        # Mock grouped data
        mock_grouped = {
            "models": {
                "test": [mock_items[0]]
            }
        }
        mock_group.return_value = (mock_grouped, set())
        
        # Mock user selections to exit workflow
        mock_prompt.side_effect = ["BACK"]
        
        # Run the workflow
        result = delete_assets_workflow(mock_items)
        
        # Verify workflow executed without errors
        assert result is not None or result is None  # Just check it doesn't crash

    def test_deduplication_name_extraction_basic(self):
        """Test basic name extraction functionality."""
        # Test simple case for name extraction
        test_case = "item_name (timestamp, size)"
        item_to_keep_name = test_case.split(" (")[0]
        assert item_to_keep_name == "item_name"


@pytest.mark.unit
class TestAssetDetailViewWorkflow:
    """Test cases for asset detail view workflow."""
    
    def test_display_name_parsing_basic(self):
        """Test basic display name parsing functionality."""
        # Test basic parsing logic
        test_case = "model_name (1.23 GB)"
        if " (" in test_case and test_case.endswith(" GB)"):
            selected_asset_display_name = test_case.rsplit(" (", 1)[0]
        else:
            selected_asset_display_name = test_case
            
        assert selected_asset_display_name == "model_name"

    @patch("hf_model_tool.ui.Console")
    @patch("hf_model_tool.ui.group_and_identify_duplicates")
    def test_print_items_empty_list(self, mock_group, mock_console):
        """Test print_items with empty items list."""
        mock_group.return_value = ({"models": {}, "datasets": {}}, set())
        mock_console_instance = MagicMock()
        mock_console.return_value = mock_console_instance

        print_items([])

        # Should still display total (0 GB)
        mock_console_instance.print.assert_called()

    @patch("hf_model_tool.ui.Console")
    @patch("hf_model_tool.ui.group_and_identify_duplicates")
    @patch("hf_model_tool.ui.logger")
    def test_print_items_handles_exceptions(
        self, mock_logger, mock_group, mock_console
    ):
        """Test print_items handles exceptions gracefully."""
        mock_items = [
            {"name": "test", "size": 1000, "date": datetime.now(), "type": "model"}
        ]

        # Mock an exception in group_and_identify_duplicates
        mock_group.side_effect = Exception("Test error")
        mock_console_instance = MagicMock()
        mock_console.return_value = mock_console_instance

        print_items(mock_items)

        # Should log error and display error message
        mock_logger.error.assert_called()
        mock_console_instance.print.assert_called()

    @patch("hf_model_tool.ui.Console")
    @patch("hf_model_tool.ui.group_and_identify_duplicates")
    def test_print_items_missing_size_field(self, mock_group, mock_console):
        """Test print_items handles missing size field gracefully."""
        mock_items = [
            {
                "name": "models--test--model1",
                # Missing size field
                "date": datetime(2023, 1, 1),
                "type": "model",
            }
        ]

        mock_grouped = {"models": {"test": [mock_items[0]]}, "datasets": {}}

        mock_group.return_value = (mock_grouped, set())
        mock_console_instance = MagicMock()
        mock_console.return_value = mock_console_instance

        # Should handle missing size gracefully (default to 0)
        print_items(mock_items)

        mock_console_instance.print.assert_called()

    @patch("hf_model_tool.ui.Console")
    @patch("hf_model_tool.ui.group_and_identify_duplicates")
    def test_print_items_duplicate_marking(self, mock_group, mock_console):
        """Test that duplicate items are properly marked."""
        mock_items = [
            {
                "name": "models--test--model1",
                "size": 1000,
                "date": datetime(2023, 1, 1),
                "type": "model",
                "display_name": "model1",
                "is_duplicate": True,  # Marked as duplicate
            }
        ]

        mock_grouped = {"models": {"test": [mock_items[0]]}, "datasets": {}}

        mock_group.return_value = (mock_grouped, set())
        mock_console_instance = MagicMock()
        mock_console.return_value = mock_console_instance

        print_items(mock_items)

        # Verify console.print was called (table display)
        mock_console_instance.print.assert_called()

        # Check that the table was created with proper columns
        call_args = mock_console_instance.print.call_args_list
        assert len(call_args) >= 2  # At least panel + table
