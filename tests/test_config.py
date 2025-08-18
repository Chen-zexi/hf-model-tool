#!/usr/bin/env python3
"""
Test suite for configuration management functionality.
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from hf_model_tool.config import ConfigManager


@pytest.mark.integration
class TestConfigManager:
    """Test ConfigManager functionality."""

    def test_init_creates_config_dir(self):
        """Test that ConfigManager creates config directory on init."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / "config"
            manager = ConfigManager(config_dir)

            assert config_dir.exists()
            assert config_dir.is_dir()

    def test_load_default_config(self):
        """Test loading default configuration when no file exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConfigManager(Path(temp_dir))
            config = manager.load_config()

            assert "custom_directories" in config
            assert config["custom_directories"] == []
            assert config["include_default_cache"] is True
            assert "last_updated" in config

    def test_save_and_load_config(self):
        """Test saving and loading configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConfigManager(Path(temp_dir))

            # Save custom config
            test_config = {
                "custom_directories": ["/test/path1", "/test/path2"],
                "include_default_cache": False,
            }
            manager.save_config(test_config)

            # Load and verify
            loaded_config = manager.load_config()
            assert loaded_config["custom_directories"] == ["/test/path1", "/test/path2"]
            assert loaded_config["include_default_cache"] is False
            assert "last_updated" in loaded_config

    def test_add_directory(self):
        """Test adding a directory to configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConfigManager(Path(temp_dir))

            # Create a test directory
            test_dir = Path(temp_dir) / "test_assets"
            test_dir.mkdir()

            # Add directory with default path type
            result = manager.add_directory(str(test_dir))
            assert result is True

            # Verify it was added with new dict format
            config = manager.load_config()
            custom_dirs = config["custom_directories"]
            assert len(custom_dirs) == 1

            dir_entry = custom_dirs[0]
            assert isinstance(dir_entry, dict)
            assert "path" in dir_entry
            assert "type" in dir_entry
            assert "added_date" in dir_entry
            assert Path(dir_entry["path"]).resolve() == test_dir.resolve()
            assert dir_entry["type"] == "auto"  # default type

            # Try adding again - should return False
            result = manager.add_directory(str(test_dir))
            assert result is False

    def test_add_directory_with_type(self):
        """Test adding a directory with specific path type."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConfigManager(Path(temp_dir))

            # Create a test directory
            test_dir = Path(temp_dir) / "test_assets"
            test_dir.mkdir()

            # Add directory with custom path type
            result = manager.add_directory(str(test_dir), "custom_directory")
            assert result is True

            # Verify it was added with correct type
            config = manager.load_config()
            custom_dirs = config["custom_directories"]
            assert len(custom_dirs) == 1

            dir_entry = custom_dirs[0]
            assert dir_entry["type"] == "custom_directory"

    def test_add_nonexistent_directory(self):
        """Test adding a non-existent directory raises error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConfigManager(Path(temp_dir))

            with pytest.raises(ValueError, match="Directory does not exist"):
                manager.add_directory("/nonexistent/path")

    def test_add_file_as_directory(self):
        """Test adding a file instead of directory raises error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConfigManager(Path(temp_dir))

            # Create a test file
            test_file = Path(temp_dir) / "test.txt"
            test_file.write_text("test")

            with pytest.raises(ValueError, match="Path is not a directory"):
                manager.add_directory(str(test_file))

    def test_remove_directory(self):
        """Test removing a directory from configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConfigManager(Path(temp_dir))

            # Add directories
            test_dir1 = Path(temp_dir) / "test1"
            test_dir2 = Path(temp_dir) / "test2"
            test_dir1.mkdir()
            test_dir2.mkdir()

            manager.add_directory(str(test_dir1))
            manager.add_directory(str(test_dir2))

            # Remove one
            result = manager.remove_directory(str(test_dir1))
            assert result is True

            # Verify removal with new dict format
            config = manager.load_config()
            custom_dirs = config["custom_directories"]
            assert len(custom_dirs) == 1

            # Check that only test_dir2 remains
            remaining_dir = custom_dirs[0]
            assert Path(remaining_dir["path"]).resolve() == test_dir2.resolve()

            # Try removing non-existent - should return False
            result = manager.remove_directory("/nonexistent/path")
            assert result is False

    def test_legacy_string_format_removal(self):
        """Test removing directories from legacy string format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConfigManager(Path(temp_dir))

            # Create a config with legacy string format
            test_dir = Path(temp_dir) / "legacy_test"
            test_dir.mkdir()

            config = {
                "custom_directories": [str(test_dir)],  # Legacy string format
                "include_default_cache": True,
            }
            manager.save_config(config)

            # Should be able to remove legacy format directory
            result = manager.remove_directory(str(test_dir))
            assert result is True

            # Verify removal
            config = manager.load_config()
            assert len(config["custom_directories"]) == 0

    def test_get_all_directories(self):
        """Test getting all directories including defaults."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConfigManager(Path(temp_dir))

            # Add custom directory
            test_dir = Path(temp_dir) / "custom"
            test_dir.mkdir()
            manager.add_directory(str(test_dir))

            # Mock home directory to control default paths
            with patch.object(Path, "home", return_value=Path(temp_dir)):
                # Create fake default directories
                default_hub = Path(temp_dir) / ".cache" / "huggingface" / "hub"
                default_hub.mkdir(parents=True)

                dirs = manager.get_all_directories()

                # Should include both default and custom
                # Compare resolved paths
                resolved_dirs = [Path(d).resolve() for d in dirs]
                assert default_hub.resolve() in resolved_dirs
                assert test_dir.resolve() in resolved_dirs

    def test_toggle_default_cache(self):
        """Test toggling default cache inclusion."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConfigManager(Path(temp_dir))

            # Should start as True
            config = manager.load_config()
            assert config["include_default_cache"] is True

            # Toggle to False
            new_state = manager.toggle_default_cache()
            assert new_state is False

            # Verify saved
            config = manager.load_config()
            assert config["include_default_cache"] is False

            # Toggle back to True
            new_state = manager.toggle_default_cache()
            assert new_state is True

    def test_validate_directory_with_hf_assets(self):
        """Test validating directory with HuggingFace assets."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConfigManager(Path(temp_dir))

            # Create directory with HF-like structure
            hf_dir = Path(temp_dir) / "hf_cache"
            hf_dir.mkdir()

            # Create model directory
            model_dir = hf_dir / "models--bert-base-uncased"
            model_dir.mkdir()

            # Should be valid
            assert manager.validate_directory(str(hf_dir)) is True

    def test_validate_directory_with_blobs(self):
        """Test validating directory with blobs subdirectory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConfigManager(Path(temp_dir))

            # Create directory with blobs
            asset_dir = Path(temp_dir) / "asset"
            asset_dir.mkdir()
            (asset_dir / "blobs").mkdir()

            # Should be valid
            assert manager.validate_directory(str(asset_dir)) is True

    def test_validate_empty_directory(self):
        """Test validating empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConfigManager(Path(temp_dir))

            # Empty directory should be invalid
            empty_dir = Path(temp_dir) / "empty"
            empty_dir.mkdir()

            assert manager.validate_directory(str(empty_dir)) is False

    def test_config_cache(self):
        """Test that config is cached after first load."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConfigManager(Path(temp_dir))

            # First load
            config1 = manager.load_config()

            # Modify file directly
            config_file = manager.config_file
            with open(config_file, "w") as f:
                json.dump(
                    {
                        "custom_directories": ["modified"],
                        "include_default_cache": False,
                    },
                    f,
                )

            # Second load should return cached value
            config2 = manager.load_config()
            assert config2 == config1

            # After save, cache should be updated
            manager.save_config(
                {"custom_directories": ["new"], "include_default_cache": True}
            )
            config3 = manager.load_config()
            assert config3["custom_directories"] == ["new"]

    def test_corrupted_config_file(self):
        """Test handling of corrupted config file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConfigManager(Path(temp_dir))

            # Create corrupted config file
            config_file = manager.config_file
            config_file.write_text("{ invalid json }")

            # Should return default config
            config = manager.load_config()
            assert config["custom_directories"] == []
            assert config["include_default_cache"] is True

    def test_toggle_ollama_scanning(self):
        """Test toggling Ollama scanning feature."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConfigManager(Path(temp_dir))

            # Should start as False by default
            config = manager.load_config()
            assert config.get("scan_ollama", False) is False

            # Toggle to True
            new_state = manager.toggle_ollama_scanning()
            assert new_state is True

            # Verify saved
            config = manager.load_config()
            assert config["scan_ollama"] is True

            # Toggle back to False
            new_state = manager.toggle_ollama_scanning()
            assert new_state is False

    def test_add_ollama_directory(self):
        """Test adding an Ollama directory to configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConfigManager(Path(temp_dir))

            # Create a test Ollama directory structure
            ollama_dir = Path(temp_dir) / ".ollama" / "models"
            ollama_dir.mkdir(parents=True)
            (ollama_dir / "manifests").mkdir()
            (ollama_dir / "blobs").mkdir()

            # Add Ollama directory
            result = manager.add_ollama_directory(str(ollama_dir))
            assert result is True

            # Verify it was added
            config = manager.load_config()
            ollama_dirs = config.get("ollama_directories", [])
            assert len(ollama_dirs) == 1
            assert Path(ollama_dirs[0]).resolve() == ollama_dir.resolve()

            # Try adding again - should return False
            result = manager.add_ollama_directory(str(ollama_dir))
            assert result is False

    def test_add_ollama_directory_without_structure(self):
        """Test adding directory without Ollama structure (should warn but proceed)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConfigManager(Path(temp_dir))

            # Create directory without Ollama structure
            not_ollama_dir = Path(temp_dir) / "not_ollama"
            not_ollama_dir.mkdir()

            # Should still add but with warning in logs
            result = manager.add_ollama_directory(str(not_ollama_dir))
            assert result is True

            # Verify it was added
            ollama_dirs = manager.get_ollama_directories()
            assert len(ollama_dirs) == 1

    def test_add_nonexistent_ollama_directory(self):
        """Test adding a non-existent Ollama directory raises error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConfigManager(Path(temp_dir))

            with pytest.raises(ValueError, match="Directory does not exist"):
                manager.add_ollama_directory("/nonexistent/ollama/path")

    def test_remove_ollama_directory(self):
        """Test removing an Ollama directory from configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConfigManager(Path(temp_dir))

            # Add Ollama directories
            ollama_dir1 = Path(temp_dir) / "ollama1"
            ollama_dir2 = Path(temp_dir) / "ollama2"
            ollama_dir1.mkdir()
            ollama_dir2.mkdir()
            (ollama_dir1 / "manifests").mkdir()
            (ollama_dir1 / "blobs").mkdir()
            (ollama_dir2 / "manifests").mkdir()
            (ollama_dir2 / "blobs").mkdir()

            manager.add_ollama_directory(str(ollama_dir1))
            manager.add_ollama_directory(str(ollama_dir2))

            # Remove one
            result = manager.remove_ollama_directory(str(ollama_dir1))
            assert result is True

            # Verify removal
            ollama_dirs = manager.get_ollama_directories()
            assert len(ollama_dirs) == 1
            assert Path(ollama_dirs[0]).resolve() == ollama_dir2.resolve()

            # Try removing non-existent - should return False
            result = manager.remove_ollama_directory("/nonexistent/path")
            assert result is False

    def test_get_ollama_directories(self):
        """Test getting list of configured Ollama directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConfigManager(Path(temp_dir))

            # Initially should be empty
            ollama_dirs = manager.get_ollama_directories()
            assert ollama_dirs == []

            # Add some directories
            ollama_dir1 = Path(temp_dir) / "ollama1"
            ollama_dir2 = Path(temp_dir) / "ollama2"
            ollama_dir1.mkdir()
            ollama_dir2.mkdir()

            manager.add_ollama_directory(str(ollama_dir1))
            manager.add_ollama_directory(str(ollama_dir2))

            # Should return both
            ollama_dirs = manager.get_ollama_directories()
            assert len(ollama_dirs) == 2
            resolved_dirs = [Path(d).resolve() for d in ollama_dirs]
            assert ollama_dir1.resolve() in resolved_dirs
            assert ollama_dir2.resolve() in resolved_dirs

    def test_get_all_directories_with_ollama(self):
        """Test that get_all_directories includes Ollama dirs when scanning is enabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConfigManager(Path(temp_dir))

            # Create custom Ollama directory
            custom_ollama = Path(temp_dir) / "custom_ollama"
            custom_ollama.mkdir()
            (custom_ollama / "manifests").mkdir()
            (custom_ollama / "blobs").mkdir()

            # Add custom Ollama directory
            manager.add_ollama_directory(str(custom_ollama))

            # Enable Ollama scanning
            manager.toggle_ollama_scanning()

            # Mock home directory to control default paths
            with patch.object(Path, "home", return_value=Path(temp_dir)):
                # Create fake default Ollama directory
                default_ollama = Path(temp_dir) / ".ollama" / "models"
                default_ollama.mkdir(parents=True)

                dirs = manager.get_all_directories_with_types()

                # Should include both default and custom Ollama directories
                dir_info = [d for d in dirs if d.get("type") == "ollama"]
                assert len(dir_info) >= 1  # At least custom should be there

                # Check that custom Ollama dir is included
                custom_found = any(
                    Path(d["path"]).resolve() == custom_ollama.resolve()
                    for d in dir_info
                )
                assert custom_found
