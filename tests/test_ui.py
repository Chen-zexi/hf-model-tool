"""
Unit tests for ui.py module.

Tests the user interface functionality.
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

from hf_model_tool.ui import print_items


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
