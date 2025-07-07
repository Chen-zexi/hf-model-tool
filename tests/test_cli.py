"""
Unit tests for CLI functionality.

Tests the command line argument parsing and handling.
"""

import pytest
import argparse
from unittest.mock import patch, MagicMock
from pathlib import Path

from hf_model_tool.__main__ import (
    create_parser,
    handle_cli_list,
    handle_cli_manage,
    handle_cli_details,
    handle_cli_add_path,
)


@pytest.mark.unit
class TestCreateParser:
    """Test cases for CLI argument parser creation."""

    def test_parser_creation(self):
        """Test that parser is created with correct arguments."""
        parser = create_parser()
        
        assert isinstance(parser, argparse.ArgumentParser)
        assert parser.prog == "hf-model-tool"
        
        # Test that help doesn't raise errors
        with pytest.raises(SystemExit):
            parser.parse_args(["--help"])

    def test_version_argument(self):
        """Test version argument."""
        parser = create_parser()
        
        with pytest.raises(SystemExit):
            parser.parse_args(["--version"])

    def test_list_argument(self):
        """Test list argument parsing."""
        parser = create_parser()
        
        args = parser.parse_args(["-l"])
        assert args.list is True
        
        args = parser.parse_args(["--list"])
        assert args.list is True

    def test_manage_argument(self):
        """Test manage argument parsing."""
        parser = create_parser()
        
        args = parser.parse_args(["-m"])
        assert args.manage is True
        
        args = parser.parse_args(["--manage"])
        assert args.manage is True

    def test_details_argument(self):
        """Test details argument parsing."""
        parser = create_parser()
        
        args = parser.parse_args(["-v"])
        assert args.details is True
        
        args = parser.parse_args(["--view"])
        assert args.details is True
        
        args = parser.parse_args(["--details"])
        assert args.details is True

    def test_add_path_argument(self):
        """Test add path argument parsing."""
        parser = create_parser()
        
        args = parser.parse_args(["-path", "/test/path"])
        assert args.add_path == "/test/path"
        
        args = parser.parse_args(["--add-path", "/another/path"])
        assert args.add_path == "/another/path"

    def test_sort_argument(self):
        """Test sort argument parsing."""
        parser = create_parser()
        
        # Default sort
        args = parser.parse_args([])
        assert args.sort == "size"
        
        # Custom sort options
        for sort_option in ["size", "name", "date"]:
            args = parser.parse_args(["--sort", sort_option])
            assert args.sort == sort_option

    def test_combined_arguments(self):
        """Test combining arguments."""
        parser = create_parser()
        
        args = parser.parse_args(["-l", "--sort", "name"])
        assert args.list is True
        assert args.sort == "name"


@pytest.mark.integration
class TestCliHandlers:
    """Test cases for CLI command handlers."""

    @patch("hf_model_tool.__main__.scan_all_directories")
    @patch("hf_model_tool.__main__.print_items")
    @patch("hf_model_tool.__main__.Console")
    def test_handle_cli_list_with_items(self, mock_console, mock_print_items, mock_scan):
        """Test CLI list handler with items found."""
        mock_items = [{"name": "test", "size": 1000}]
        mock_scan.return_value = mock_items
        
        handle_cli_list("size")
        
        mock_scan.assert_called_once()
        mock_print_items.assert_called_once_with(mock_items, sort_by="size")

    @patch("hf_model_tool.__main__.scan_all_directories")
    @patch("hf_model_tool.__main__.Console")
    def test_handle_cli_list_no_items(self, mock_console, mock_scan):
        """Test CLI list handler with no items found."""
        mock_scan.return_value = []
        mock_console_instance = MagicMock()
        mock_console.return_value = mock_console_instance
        
        handle_cli_list("size")
        
        mock_scan.assert_called_once()
        mock_console_instance.print.assert_called()

    @patch("hf_model_tool.__main__.scan_all_directories")
    @patch("hf_model_tool.__main__.unified_prompt")
    @patch("hf_model_tool.__main__.Console")
    def test_handle_cli_manage_with_items(self, mock_console, mock_prompt, mock_scan):
        """Test CLI manage handler with items found."""
        mock_items = [{"name": "test", "size": 1000}]
        mock_scan.return_value = mock_items
        mock_prompt.return_value = "BACK"  # Exit immediately
        
        handle_cli_manage()
        
        mock_scan.assert_called_once()
        mock_prompt.assert_called()

    @patch("hf_model_tool.__main__.scan_all_directories")
    @patch("hf_model_tool.__main__.view_asset_details_workflow")
    @patch("hf_model_tool.__main__.Console")
    def test_handle_cli_details_with_items(self, mock_console, mock_workflow, mock_scan):
        """Test CLI details handler with items found."""
        mock_items = [{"name": "test", "size": 1000}]
        mock_scan.return_value = mock_items
        
        handle_cli_details()
        
        mock_scan.assert_called_once()
        mock_workflow.assert_called_once_with(mock_items)

    @patch("hf_model_tool.__main__.ConfigManager")
    @patch("hf_model_tool.__main__.Console")
    @patch("builtins.input", side_effect=["1"])  # Choose HuggingFace Cache
    def test_handle_cli_add_path_success(self, mock_input, mock_console, mock_config_class):
        """Test CLI add path handler with successful addition."""
        mock_console_instance = MagicMock()
        mock_console.return_value = mock_console_instance
        
        mock_config = MagicMock()
        mock_config.add_directory.return_value = True
        mock_config_class.return_value = mock_config
        
        with patch("hf_model_tool.__main__.Path") as mock_path:
            mock_path_instance = MagicMock()
            mock_path_instance.expanduser.return_value.resolve.return_value = mock_path_instance
            mock_path_instance.exists.return_value = True
            mock_path_instance.is_dir.return_value = True
            mock_path.return_value = mock_path_instance
            
            handle_cli_add_path("/test/path")
            
            mock_config.add_directory.assert_called_once()

    @patch("hf_model_tool.__main__.Console")
    def test_handle_cli_add_path_nonexistent(self, mock_console):
        """Test CLI add path handler with non-existent path."""
        mock_console_instance = MagicMock()
        mock_console.return_value = mock_console_instance
        
        with patch("hf_model_tool.__main__.Path") as mock_path:
            mock_path_instance = MagicMock()
            mock_path_instance.expanduser.return_value.resolve.return_value = mock_path_instance
            mock_path_instance.exists.return_value = False
            mock_path.return_value = mock_path_instance
            
            handle_cli_add_path("/nonexistent/path")
            
            # Should print error message
            mock_console_instance.print.assert_called()


@pytest.mark.unit
class TestAssetDeletionStringParsing:
    """Test cases for asset deletion string parsing logic."""

    def test_lora_adapter_display_name_extraction(self):
        """Test extracting display names from LoRA adapter choices with timestamps."""
        test_cases = [
            # LoRA adapter with timestamp
            ("lora_unsloth_Qwen3_4B_unsloth_bnb_4bit (2025-07-04 12:16) (0.56 GB)", 
             "lora_unsloth_Qwen3_4B_unsloth_bnb_4bit (2025-07-04 12:16)"),
            # Regular model without timestamp
            ("regular_model (1.23 GB)", 
             "regular_model"),
            # Complex model name
            ("microsoft--DialoGPT-medium (2.45 GB)", 
             "microsoft--DialoGPT-medium"),
        ]
        
        for choice_str, expected in test_cases:
            # Simulate the extraction logic from ui.py
            if " (" in choice_str and choice_str.endswith(" GB)"):
                extracted = choice_str.rsplit(" (", 1)[0]
            else:
                extracted = choice_str
                
            assert extracted == expected, f"Failed for {choice_str}: got {extracted}, expected {expected}"

    def test_deduplication_name_extraction(self):
        """Test extracting names from deduplication choices."""
        test_cases = [
            # Deduplication format: "name (date, size GB)"
            ("models--huggingface--bert (2023-01-01, 1.23 GB)", 
             "models--huggingface--bert"),
            ("lora_adapter (2025-07-04, 0.56 GB)", 
             "lora_adapter"),
        ]
        
        for choice_str, expected in test_cases:
            # Simulate the extraction logic from ui.py
            extracted = choice_str.split(" (")[0]
            assert extracted == expected, f"Failed for {choice_str}: got {extracted}, expected {expected}"