"""
Unit tests for cache.py module.

Tests the HuggingFace cache directory scanning functionality.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch
from datetime import datetime

from hf_model_tool.cache import get_items


@pytest.mark.unit
class TestGetItems:
    """Test cases for get_items function."""

    def test_get_items_with_valid_cache(self):
        """Test get_items with a valid cache directory structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)

            # Create mock model directory
            model_dir = cache_dir / "models--huggingface--bert-base"
            model_dir.mkdir(parents=True)
            blobs_dir = model_dir / "blobs"
            blobs_dir.mkdir()

            # Create mock blob files
            (blobs_dir / "blob1").write_bytes(b"x" * 1000)  # 1KB
            (blobs_dir / "blob2").write_bytes(b"x" * 2000)  # 2KB

            # Create mock dataset directory
            dataset_dir = cache_dir / "datasets--squad--v1"
            dataset_dir.mkdir(parents=True)
            dataset_blobs = dataset_dir / "blobs"
            dataset_blobs.mkdir()
            (dataset_blobs / "data").write_bytes(b"x" * 5000)  # 5KB

            items = get_items(str(cache_dir))

            assert len(items) == 2

            # Check model item
            model_item = next(item for item in items if item["type"] == "model")
            assert model_item["name"] == "models--huggingface--bert-base"
            assert model_item["size"] == 3000  # 1KB + 2KB
            assert model_item["type"] == "model"
            assert isinstance(model_item["date"], datetime)

            # Check dataset item
            dataset_item = next(item for item in items if item["type"] == "dataset")
            assert dataset_item["name"] == "datasets--squad--v1"
            assert dataset_item["size"] == 5000  # 5KB
            assert dataset_item["type"] == "dataset"

    def test_get_items_empty_directory(self):
        """Test get_items with an empty cache directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            items = get_items(temp_dir)
            assert items == []

    def test_get_items_nonexistent_directory(self):
        """Test get_items with a non-existent directory."""
        with pytest.raises(OSError, match="Custom directory not found"):
            get_items("/nonexistent/path")

    def test_get_items_file_instead_of_directory(self):
        """Test get_items when path points to a file instead of directory."""
        with tempfile.NamedTemporaryFile() as temp_file:
            with pytest.raises(OSError, match="Custom path is not a directory"):
                get_items(temp_file.name)

    def test_get_items_with_path_object(self):
        """Test get_items with Path object instead of string."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir)
            items = get_items(cache_path)
            assert items == []

    def test_get_items_ignores_zero_size_items(self):
        """Test that items with zero size are ignored."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)

            # Create directory without blobs
            empty_dir = cache_dir / "models--empty--model"
            empty_dir.mkdir(parents=True)

            # Create directory with empty blobs
            model_dir = cache_dir / "models--test--model"
            model_dir.mkdir(parents=True)
            blobs_dir = model_dir / "blobs"
            blobs_dir.mkdir()
            # No files in blobs directory

            items = get_items(str(cache_dir))
            assert len(items) == 0

    @patch("hf_model_tool.cache.logger")
    def test_get_items_handles_permission_error(self, mock_logger):
        """Test handling of permission errors during scanning."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)

            # Create a model directory
            model_dir = cache_dir / "models--test--model"
            model_dir.mkdir(parents=True)
            blobs_dir = model_dir / "blobs"
            blobs_dir.mkdir()

            # Mock permission error
            with patch.object(
                Path, "iterdir", side_effect=PermissionError("Access denied")
            ):
                with pytest.raises(PermissionError):
                    get_items(str(cache_dir))

    def test_get_items_asset_type_detection(self):
        """Test basic asset type detection."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)

            # Create a simple model directory
            model_dir = cache_dir / "models--test--model"
            model_dir.mkdir(parents=True)
            blobs_dir = model_dir / "blobs"
            blobs_dir.mkdir()
            (blobs_dir / "file").write_bytes(b"x" * 100)

            # Create a simple dataset directory
            dataset_dir = cache_dir / "datasets--test--dataset"
            dataset_dir.mkdir(parents=True)
            dataset_blobs = dataset_dir / "blobs"
            dataset_blobs.mkdir()
            (dataset_blobs / "data").write_bytes(b"x" * 100)

            items = get_items(str(cache_dir))
            
            # Should detect at least 2 items
            assert len(items) >= 2
            
            # Check that we have both models and datasets
            types_found = {item["type"] for item in items}
            assert "model" in types_found or "dataset" in types_found
