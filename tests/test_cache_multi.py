#!/usr/bin/env python3
"""
Test suite for multi-directory cache scanning functionality.
"""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from hf_model_tool.cache import scan_all_directories


@pytest.mark.integration
class TestMultiDirectoryScanning:
    """Test scanning across multiple directories."""

    def create_mock_hf_directory(
        self, base_path: Path, asset_name: str, size: int = 1000
    ):
        """Create a mock HuggingFace asset directory structure."""
        asset_dir = base_path / asset_name
        asset_dir.mkdir()

        blobs_dir = asset_dir / "blobs"
        blobs_dir.mkdir()

        # Create mock blob files
        blob_file = blobs_dir / "blob1"
        blob_file.write_bytes(b"x" * size)

        return asset_dir

    @patch("hf_model_tool.cache.ConfigManager")
    def test_scan_single_directory(self, mock_config_class):
        """Test scanning with a single configured directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup mock config - return list of dictionaries as expected
            mock_config = Mock()
            mock_config.get_all_directories_with_types.return_value = [
                {"path": temp_dir, "type": "huggingface", "source": "test"}
            ]
            mock_config_class.return_value = mock_config

            # Create test assets
            self.create_mock_hf_directory(Path(temp_dir), "models--test-model", 1000)

            # Scan
            items = scan_all_directories()

            assert len(items) == 1
            assert items[0]["name"] == "models--test-model"
            assert items[0]["size"] == 1000
            assert items[0]["source_dir"] == temp_dir

    @patch("hf_model_tool.cache.ConfigManager")
    def test_scan_multiple_directories(self, mock_config_class):
        """Test scanning across multiple directories."""
        with tempfile.TemporaryDirectory() as temp_base:
            dir1 = Path(temp_base) / "dir1"
            dir2 = Path(temp_base) / "dir2"
            dir1.mkdir()
            dir2.mkdir()

            # Setup mock config
            mock_config = Mock()
            mock_config.get_all_directories_with_types.return_value = [
                {"path": str(dir1), "type": "huggingface", "source": "test1"},
                {"path": str(dir2), "type": "huggingface", "source": "test2"}
            ]
            mock_config_class.return_value = mock_config

            # Create assets in different directories
            self.create_mock_hf_directory(dir1, "models--model1", 1000)
            self.create_mock_hf_directory(dir2, "models--model2", 2000)

            # Scan
            items = scan_all_directories()

            assert len(items) == 2

            # Find items by name
            item1 = next(i for i in items if i["name"] == "models--model1")
            item2 = next(i for i in items if i["name"] == "models--model2")

            assert item1["source_dir"] == str(dir1)
            assert item1["size"] == 1000

            assert item2["source_dir"] == str(dir2)
            assert item2["size"] == 2000

    @patch("hf_model_tool.cache.ConfigManager")
    def test_duplicate_removal(self, mock_config_class):
        """Test that duplicates across directories are removed."""
        with tempfile.TemporaryDirectory() as temp_base:
            dir1 = Path(temp_base) / "dir1"
            dir2 = Path(temp_base) / "dir2"
            dir1.mkdir()
            dir2.mkdir()

            # Setup mock config
            mock_config = Mock()
            mock_config.get_all_directories_with_types.return_value = [
                {"path": str(dir1), "type": "huggingface", "source": "test1"},
                {"path": str(dir2), "type": "huggingface", "source": "test2"}
            ]
            mock_config_class.return_value = mock_config

            # Create same asset in both directories
            self.create_mock_hf_directory(dir1, "models--duplicate", 1000)
            self.create_mock_hf_directory(dir2, "models--duplicate", 2000)

            # Scan
            items = scan_all_directories()

            # Should only have one item (first found)
            assert len(items) == 1
            assert items[0]["name"] == "models--duplicate"
            assert items[0]["source_dir"] == str(dir1)  # First directory wins

    @patch("hf_model_tool.cache.ConfigManager")
    def test_scan_with_missing_directory(self, mock_config_class):
        """Test scanning handles missing directories gracefully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            existing_dir = Path(temp_dir) / "exists"
            existing_dir.mkdir()

            # Setup mock config with one existing and one missing directory
            mock_config = Mock()
            mock_config.get_all_directories_with_types.return_value = [
                {"path": str(existing_dir), "type": "huggingface", "source": "test1"},
                {"path": "/nonexistent/directory", "type": "huggingface", "source": "test2"}
            ]
            mock_config_class.return_value = mock_config

            # Create asset in existing directory
            self.create_mock_hf_directory(existing_dir, "models--test", 1000)

            # Scan should not fail
            items = scan_all_directories()

            assert len(items) == 1
            assert items[0]["name"] == "models--test"

    @patch("hf_model_tool.cache.ConfigManager")
    def test_scan_empty_directories(self, mock_config_class):
        """Test scanning empty directories returns empty list."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup mock config
            mock_config = Mock()
            mock_config.get_all_directories_with_types.return_value = [
                {"path": temp_dir, "type": "huggingface", "source": "test"}
            ]
            mock_config_class.return_value = mock_config

            # Scan empty directory
            items = scan_all_directories()

            assert items == []

    @patch("hf_model_tool.cache.ConfigManager")
    def test_scan_no_configured_directories(self, mock_config_class):
        """Test scanning with no configured directories."""
        # Setup mock config with no directories
        mock_config = Mock()
        mock_config.get_all_directories_with_types.return_value = []
        mock_config_class.return_value = mock_config

        # Scan
        items = scan_all_directories()

        assert items == []

    @patch("hf_model_tool.cache.ConfigManager")
    @patch("hf_model_tool.cache.logger")
    def test_logging_output(self, mock_logger, mock_config_class):
        """Test that appropriate logging occurs during scanning."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup mock config
            mock_config = Mock()
            mock_config.get_all_directories_with_types.return_value = [
                {"path": temp_dir, "type": "huggingface", "source": "test"}
            ]
            mock_config_class.return_value = mock_config

            # Create test asset
            self.create_mock_hf_directory(Path(temp_dir), "models--test", 1000)

            # Scan
            scan_all_directories()

            # Check logging calls
            mock_logger.info.assert_any_call("Scanning 1 directories for assets")
            mock_logger.info.assert_any_call(
                "Found 1 assets across all directories"
            )
