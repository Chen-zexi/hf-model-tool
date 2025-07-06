"""
Unit tests for navigation.py module.

Tests the unified navigation system functionality.
"""

import pytest
from unittest.mock import patch

from hf_model_tool.navigation import show_help, show_config, unified_prompt


class TestShowHelp:
    """Test cases for show_help function."""

    @patch("builtins.input")
    @patch("builtins.print")
    def test_show_help_normal_flow(self, mock_print, mock_input):
        """Test normal help display flow."""
        mock_input.return_value = ""  # User presses Enter

        show_help()

        # Verify help content was printed
        mock_print.assert_called()
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        help_content = "\n".join(print_calls)

        assert "NAVIGATION HELP:" in help_content
        assert "↑/↓ arrows: Navigate menu options" in help_content
        assert "Enter: Select current option" in help_content

    @patch("builtins.input", side_effect=KeyboardInterrupt())
    @patch("hf_model_tool.navigation.logger")
    def test_show_help_keyboard_interrupt(self, mock_logger, mock_input):
        """Test help display with keyboard interrupt."""
        show_help()

        # Should log the interruption
        mock_logger.info.assert_called_with("Help display interrupted by user")


class TestShowConfig:
    """Test cases for show_config function."""

    @patch("hf_model_tool.navigation.unified_prompt")
    def test_show_config_sort_selection(self, mock_unified_prompt):
        """Test config menu with sort option selection."""
        mock_unified_prompt.return_value = "Sort Assets By Size"

        result = show_config()

        assert result == "Sort Assets By Size"
        mock_unified_prompt.assert_called_once()

    @patch("hf_model_tool.navigation.unified_prompt")
    def test_show_config_back_selection(self, mock_unified_prompt):
        """Test config menu with back selection."""
        mock_unified_prompt.return_value = "BACK"

        result = show_config()

        assert result is None

    @patch("hf_model_tool.navigation.unified_prompt")
    @patch("builtins.input")
    @patch("builtins.print")
    def test_show_config_future_features(
        self, mock_print, mock_input, mock_unified_prompt
    ):
        """Test config menu future feature selections."""
        mock_input.return_value = ""  # User presses Enter

        # Test cache directory feature
        mock_unified_prompt.side_effect = ["Set Default Cache Directory", "BACK"]
        result = show_config()

        assert result is None
        mock_print.assert_called()
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        output = "\n".join(print_calls)
        assert "[Future Feature] Cache directory configuration" in output


class TestUnifiedPrompt:
    """Test cases for unified_prompt function."""

    @patch("hf_model_tool.navigation.inquirer.prompt")
    @patch("hf_model_tool.navigation.Console")
    def test_unified_prompt_normal_selection(self, mock_console, mock_prompt):
        """Test normal menu selection."""
        mock_prompt.return_value = {"test": "List Assets"}

        result = unified_prompt("test", "Test Menu", ["List Assets", "Exit"])

        assert result == "List Assets"
        mock_console.return_value.print.assert_called()  # Panel should be printed

    @patch("hf_model_tool.navigation.inquirer.prompt")
    @patch("hf_model_tool.navigation.Console")
    def test_unified_prompt_back_selection(self, mock_console, mock_prompt):
        """Test back navigation selection."""
        mock_prompt.return_value = {"test": "← Back"}

        result = unified_prompt("test", "Test Menu", ["Option 1"])

        assert result == "BACK"

    @patch("hf_model_tool.navigation.inquirer.prompt")
    @patch("hf_model_tool.navigation.Console")
    def test_unified_prompt_main_menu_selection(self, mock_console, mock_prompt):
        """Test main menu navigation selection."""
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
    def test_unified_prompt_config_selection(self, mock_console, mock_prompt):
        """Test config selection with sort return."""
        mock_prompt.side_effect = [
            {"test": "→ Config"},
            {"config": "Sort Assets By Date"},
            {"config": "BACK"},
        ]

        with patch(
            "hf_model_tool.navigation.show_config", return_value="Sort Assets By Date"
        ):
            result = unified_prompt("test", "Test Menu", ["Option 1"])
            assert result == "Sort Assets By Date"

    @patch("hf_model_tool.navigation.inquirer.prompt")
    @patch("hf_model_tool.navigation.Console")
    def test_unified_prompt_separator_ignored(self, mock_console, mock_prompt):
        """Test that separator selections are ignored."""
        mock_prompt.side_effect = [
            {"test": "─────"},  # Separator - should be ignored
            {"test": "Option 1"},  # Valid selection
        ]

        result = unified_prompt("test", "Test Menu", ["Option 1"])

        assert result == "Option 1"

    @patch("hf_model_tool.navigation.inquirer.prompt")
    @patch("hf_model_tool.navigation.Console")
    def test_unified_prompt_no_answer(self, mock_console, mock_prompt):
        """Test handling when user cancels prompt."""
        mock_prompt.return_value = None

        result = unified_prompt("test", "Test Menu", ["Option 1"])

        assert result is None

    @patch("hf_model_tool.navigation.inquirer.prompt")
    @patch("hf_model_tool.navigation.Console")
    def test_unified_prompt_keyboard_interrupt(self, mock_console, mock_prompt):
        """Test handling of keyboard interrupt."""
        mock_prompt.side_effect = KeyboardInterrupt()

        with pytest.raises(SystemExit):
            unified_prompt("test", "Test Menu", ["Option 1"])

    @patch("hf_model_tool.navigation.inquirer.prompt")
    @patch("hf_model_tool.navigation.Console")
    @patch("hf_model_tool.navigation.logger")
    def test_unified_prompt_ioctl_error(self, mock_logger, mock_console, mock_prompt):
        """Test handling of ioctl error (non-interactive environment)."""
        mock_prompt.side_effect = OSError("Inappropriate ioctl for device")

        result = unified_prompt("test", "Test Menu", ["Option 1"])

        assert result is None
        mock_logger.warning.assert_called_with("Running in non-interactive environment")

    def test_unified_prompt_invalid_choices(self):
        """Test unified_prompt with invalid choices parameter."""
        with pytest.raises(ValueError, match="Choices must be a list"):
            unified_prompt("test", "Test Menu", "not a list")

    @patch("hf_model_tool.navigation.inquirer.prompt")
    @patch("hf_model_tool.navigation.Console")
    def test_unified_prompt_choice_enhancement(self, mock_console, mock_prompt):
        """Test that choices are properly enhanced with navigation options."""
        mock_prompt.return_value = {"test": "Test Option"}

        result = unified_prompt("test", "Test Menu", ["Test Option"], allow_back=True)

        # Verify the return value is correct
        assert result == "Test Option"

        # Verify inquirer was called with enhanced choices
        call_args = mock_prompt.call_args[0][0][0]  # First question object
        choices = call_args.choices

        # Should include original choice plus navigation options
        assert "Test Option" in choices
        assert "← Back" in choices
        assert "→ Config" in choices
        assert "Main Menu" in choices
        assert "Exit" in choices
        assert "─────" in choices  # Separator

    @patch("hf_model_tool.navigation.inquirer.prompt")
    @patch("hf_model_tool.navigation.Console")
    def test_unified_prompt_no_back_option(self, mock_console, mock_prompt):
        """Test unified_prompt with allow_back=False."""
        mock_prompt.return_value = {"test": "Test Option"}

        result = unified_prompt("test", "Test Menu", ["Test Option"], allow_back=False)

        # Verify the return value is correct
        assert result == "Test Option"

        # Verify inquirer was called without back option
        call_args = mock_prompt.call_args[0][0][0]  # First question object
        choices = call_args.choices

        assert "Test Option" in choices
        assert "← Back" not in choices
        assert "→ Config" in choices
        assert "Main Menu" in choices
        assert "Exit" in choices
