"""
Unit tests for navigation.py module.

Tests the unified navigation system functionality.
"""

import pytest
from unittest.mock import patch

from hf_model_tool.navigation import show_help, show_config, unified_prompt


@pytest.mark.unit
class TestShowHelp:
    """Test cases for show_help function."""

    @patch("builtins.input")
    @patch("builtins.print")
    def test_show_help_basic_functionality(self, mock_print, mock_input):
        """Test that help display works without errors."""
        mock_input.return_value = ""  # User presses Enter

        # Should not raise any exceptions
        show_help()

        # Should print something
        mock_print.assert_called()


@pytest.mark.unit
class TestShowConfig:
    """Test cases for show_config function."""

    @patch("hf_model_tool.navigation.unified_prompt")
    def test_show_config_basic_functionality(self, mock_unified_prompt):
        """Test that config menu handles basic selections."""
        mock_unified_prompt.return_value = "Sort Assets By Size"

        result = show_config()

        assert result is not None
        mock_unified_prompt.assert_called_once()

    @patch("hf_model_tool.navigation.unified_prompt")
    def test_show_config_back_selection(self, mock_unified_prompt):
        """Test config menu with back selection."""
        mock_unified_prompt.return_value = "BACK"

        result = show_config()

        assert result is None


@pytest.mark.unit
class TestUnifiedPrompt:
    """Test cases for unified_prompt function."""

    @patch("hf_model_tool.navigation.inquirer.prompt")
    @patch("hf_model_tool.navigation.Console")
    def test_unified_prompt_basic_functionality(self, mock_console, mock_prompt):
        """Test basic unified prompt functionality."""
        mock_prompt.return_value = {"test": "Test Option"}

        result = unified_prompt("test", "Test Menu", ["Test Option"])

        assert result == "Test Option"
        mock_console.return_value.print.assert_called()

    @patch("hf_model_tool.navigation.inquirer.prompt")
    @patch("hf_model_tool.navigation.Console")
    def test_unified_prompt_navigation_options(self, mock_console, mock_prompt):
        """Test navigation options return correct values."""
        # Test back navigation
        mock_prompt.return_value = {"test": "← Back"}
        result = unified_prompt("test", "Test Menu", ["Option 1"])
        assert result == "BACK"

        # Test main menu navigation
        mock_prompt.return_value = {"test": "Main Menu"}
        result = unified_prompt("test", "Test Menu", ["Option 1"])
        assert result == "MAIN_MENU"

    @patch("hf_model_tool.navigation.inquirer.prompt")
    @patch("hf_model_tool.navigation.Console")
    def test_unified_prompt_exit_selection(self, mock_console, mock_prompt):
        """Test exit selection."""
        mock_prompt.return_value = {"test": "Exit"}

        with pytest.raises(SystemExit):
            unified_prompt("test", "Test Menu", ["Option 1"])

    @patch("hf_model_tool.navigation.inquirer.prompt")
    @patch("hf_model_tool.navigation.Console")
    def test_unified_prompt_keyboard_interrupt(self, mock_console, mock_prompt):
        """Test keyboard interrupt handling."""
        mock_prompt.side_effect = KeyboardInterrupt()
        with pytest.raises(SystemExit):
            unified_prompt("test", "Test Menu", ["Option 1"])

    @patch("hf_model_tool.navigation.inquirer.prompt")
    @patch("hf_model_tool.navigation.Console")
    def test_unified_prompt_no_answer(self, mock_console, mock_prompt):
        """Test handling when user cancels prompt."""
        mock_prompt.return_value = None
        result = unified_prompt("test", "Test Menu", ["Option 1"])
        assert result is None

    def test_unified_prompt_invalid_choices(self):
        """Test unified_prompt with invalid choices parameter."""
        with pytest.raises(ValueError, match="Choices must be a list"):
            unified_prompt("test", "Test Menu", "not a list")
