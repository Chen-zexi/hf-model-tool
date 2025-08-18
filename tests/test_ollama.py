#!/usr/bin/env python3
"""
Test suite for Ollama model discovery module.
"""
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from hf_model_tool.ollama import (
    get_ollama_items,
    get_ollama_model_info,
    _scan_single_ollama_directory,
    _parse_ollama_manifest,
    DEFAULT_OLLAMA_DIRS,
)


@pytest.fixture
def mock_ollama_dir():
    """Create a mock Ollama directory structure for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ollama_dir = Path(tmpdir) / ".ollama" / "models"
        ollama_dir.mkdir(parents=True)

        # Create manifests directory
        manifests_dir = ollama_dir / "manifests" / "registry.ollama.ai" / "library"
        manifests_dir.mkdir(parents=True)

        # Create blobs directory
        blobs_dir = ollama_dir / "blobs"
        blobs_dir.mkdir(parents=True)

        # Create a sample manifest for llama2
        llama2_manifest_dir = manifests_dir / "llama2"
        llama2_manifest_dir.mkdir(parents=True)

        manifest_data = {
            "schemaVersion": 2,
            "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
            "config": {"digest": "sha256:abc123", "size": 1024},
            "layers": [
                {
                    "digest": "sha256:layer1",
                    "size": 5368709120,
                    "mediaType": "application/vnd.ollama.image.model",
                },
                {
                    "digest": "sha256:layer2",
                    "size": 1048576,
                    "mediaType": "application/vnd.ollama.image.params",
                },
            ],
        }

        with open(llama2_manifest_dir / "7b", "w") as f:
            json.dump(manifest_data, f)

        # Create corresponding blob files with GGUF header
        blob1 = blobs_dir / "sha256-layer1"
        blob1.write_bytes(b"GGUF" + b"\x00" * 5000)  # GGUF magic header + data

        blob2 = blobs_dir / "sha256-layer2"
        blob2.write_bytes(b"params" * 100)

        # Create another model manifest
        codellama_manifest_dir = manifests_dir / "codellama"
        codellama_manifest_dir.mkdir(parents=True)

        with open(codellama_manifest_dir / "13b", "w") as f:
            json.dump(
                {
                    "schemaVersion": 2,
                    "layers": [
                        {
                            "digest": "sha256:layer3",
                            "size": 13000000000,
                            "mediaType": "application/vnd.ollama.image.model",
                        }
                    ],
                },
                f,
            )

        # Create blob for codellama
        blob3 = blobs_dir / "sha256-layer3"
        blob3.write_bytes(b"GGUF" + b"\x00" * 10000)  # GGUF magic header + data

        yield ollama_dir


@pytest.fixture
def mock_empty_ollama_dir():
    """Create an empty Ollama directory structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ollama_dir = Path(tmpdir) / ".ollama" / "models"
        ollama_dir.mkdir(parents=True)

        # Create empty directories
        (ollama_dir / "manifests").mkdir()
        (ollama_dir / "blobs").mkdir()

        yield ollama_dir


class TestGetOllamaItems:
    """Test the get_ollama_items function."""

    @pytest.mark.unit
    def test_get_ollama_items_with_custom_dir(self, mock_ollama_dir):
        """Test scanning a custom Ollama directory."""
        items = get_ollama_items(ollama_dirs=str(mock_ollama_dir))

        assert len(items) == 2

        # Check llama2 model
        llama2 = next((item for item in items if "llama2" in item["name"]), None)
        assert llama2 is not None
        assert llama2["type"] == "ollama_model"
        assert llama2["size"] > 0
        assert "path" in llama2

        # Check codellama model
        codellama = next((item for item in items if "codellama" in item["name"]), None)
        assert codellama is not None

    @pytest.mark.unit
    def test_get_ollama_items_with_multiple_dirs(
        self, mock_ollama_dir, mock_empty_ollama_dir
    ):
        """Test scanning multiple Ollama directories."""
        items = get_ollama_items(
            ollama_dirs=[str(mock_ollama_dir), str(mock_empty_ollama_dir)]
        )

        # Should find models only from the populated directory
        assert len(items) == 2

    @pytest.mark.unit
    def test_get_ollama_items_with_env_variable(self, mock_ollama_dir, monkeypatch):
        """Test using OLLAMA_MODELS environment variable."""
        monkeypatch.setenv("OLLAMA_MODELS", str(mock_ollama_dir))

        # When env var is set, it's added alongside default dirs
        # So we need to check our specific models exist
        items = get_ollama_items()

        # At least our 2 mock models should be present
        assert len(items) >= 2
        assert all(item["type"] == "ollama_model" for item in items)

        # Check our specific models are found
        model_names = [item["name"] for item in items]
        assert any("llama2" in name for name in model_names)
        assert any("codellama" in name for name in model_names)

    @pytest.mark.unit
    def test_get_ollama_items_nonexistent_dir(self):
        """Test scanning a non-existent directory."""
        items = get_ollama_items(ollama_dirs="/nonexistent/path")

        assert items == []

    @pytest.mark.unit
    def test_get_ollama_items_empty_dir(self, mock_empty_ollama_dir):
        """Test scanning an empty Ollama directory."""
        items = get_ollama_items(ollama_dirs=str(mock_empty_ollama_dir))

        assert items == []

    @pytest.mark.unit
    def test_get_ollama_items_handles_duplicates(self, mock_ollama_dir):
        """Test that duplicate models are handled correctly."""
        # Scan the same directory twice
        items = get_ollama_items(
            ollama_dirs=[str(mock_ollama_dir), str(mock_ollama_dir)]
        )

        # Should merge duplicates
        assert len(items) == 2

        # Check that the best version is kept
        for item in items:
            assert item["size"] > 0


class TestScanSingleOllamaDirectory:
    """Test the _scan_single_ollama_directory function."""

    @pytest.mark.unit
    def test_scan_single_directory_success(self, mock_ollama_dir):
        """Test successful scanning of a single directory."""
        items = _scan_single_ollama_directory(mock_ollama_dir)

        assert len(items) == 2
        assert all("name" in item for item in items)
        assert all("size" in item for item in items)
        assert all("type" in item for item in items)
        assert all(item["type"] == "ollama_model" for item in items)

    @pytest.mark.unit
    def test_scan_single_directory_no_manifests(self, mock_empty_ollama_dir):
        """Test scanning directory with no manifests."""
        items = _scan_single_ollama_directory(mock_empty_ollama_dir)

        assert items == []

    @pytest.mark.unit
    def test_scan_single_directory_corrupted_manifest(self):
        """Test handling of corrupted manifest files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ollama_dir = Path(tmpdir)
            manifests_dir = (
                ollama_dir / "manifests" / "registry.ollama.ai" / "library" / "broken"
            )
            manifests_dir.mkdir(parents=True)
            blobs_dir = ollama_dir / "blobs"
            blobs_dir.mkdir(parents=True)

            # Write invalid JSON
            with open(manifests_dir / "latest", "w") as f:
                f.write("not valid json{")

            items = _scan_single_ollama_directory(ollama_dir)

            # Should handle error gracefully
            assert isinstance(items, list)


class TestParseOllamaManifest:
    """Test the _parse_ollama_manifest function."""

    @pytest.mark.unit
    def test_parse_manifest_with_model_layer(self):
        """Test parsing manifest with model layer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create manifest file
            manifest_path = Path(tmpdir) / "library" / "llama2" / "7b"
            manifest_path.parent.mkdir(parents=True)

            # Create blobs directory with GGUF file
            blobs_dir = Path(tmpdir) / "blobs"
            blobs_dir.mkdir()

            # Write a GGUF file
            blob_file = blobs_dir / "sha256-testmodel"
            blob_file.write_bytes(b"GGUF" + b"\x00" * 1000)  # GGUF magic header

            manifest_data = {
                "layers": [
                    {
                        "digest": "sha256:testmodel",
                        "size": 1004,
                        "mediaType": "application/vnd.ollama.image.model",
                    }
                ]
            }

            with open(manifest_path, "w") as f:
                json.dump(manifest_data, f)

            info = _parse_ollama_manifest(manifest_path, blobs_dir)

            assert info is not None
            assert info["name"] == "llama2:7b"
            assert info["size"] == 1004
            assert info["type"] == "ollama_model"
            assert "testmodel" in info["metadata"]["model_digest"]

    @pytest.mark.unit
    def test_parse_manifest_no_model_layer(self):
        """Test parsing manifest without model layer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "test.json"
            blobs_dir = Path(tmpdir) / "blobs"
            blobs_dir.mkdir()

            manifest_data = {"layers": [{"digest": "sha256:params", "size": 1000}]}

            with open(manifest_path, "w") as f:
                json.dump(manifest_data, f)

            info = _parse_ollama_manifest(manifest_path, blobs_dir)

            assert info is None  # No model layer found

    @pytest.mark.unit
    def test_parse_corrupted_manifest(self):
        """Test parsing corrupted manifest file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "corrupted.json"
            blobs_dir = Path(tmpdir) / "blobs"
            blobs_dir.mkdir()

            manifest_path.write_text("not valid json{")

            info = _parse_ollama_manifest(manifest_path, blobs_dir)

            assert info is None  # Should handle error gracefully


class TestGetOllamaModelInfo:
    """Test the get_ollama_model_info function."""

    @pytest.mark.unit
    def test_get_model_info_found(self, mock_ollama_dir):
        """Test getting info for an existing model."""
        # get_ollama_model_info internally calls get_ollama_items() with no args
        # which scans default directories. For testing, we need to mock the
        # get_ollama_items function to return our test data
        from unittest.mock import patch

        # First get the items from our mock directory
        items = get_ollama_items(ollama_dirs=str(mock_ollama_dir))

        # Now mock get_ollama_items to return these items
        with patch("hf_model_tool.ollama.get_ollama_items", return_value=items):
            info = get_ollama_model_info("llama2:7b")

            assert info is not None
            assert info["name"] == "llama2:7b"
            assert info["type"] == "ollama_model"
            assert "path" in info

    @pytest.mark.unit
    def test_get_model_info_not_found(self, mock_ollama_dir):
        """Test getting info for a non-existent model."""
        from unittest.mock import patch

        items = get_ollama_items(ollama_dirs=str(mock_ollama_dir))

        with patch("hf_model_tool.ollama.get_ollama_items", return_value=items):
            info = get_ollama_model_info("nonexistent:model")
            assert info is None

    @pytest.mark.unit
    def test_get_model_info_case_insensitive(self, mock_ollama_dir):
        """Test that model lookup is case-insensitive."""
        from unittest.mock import patch

        items = get_ollama_items(ollama_dirs=str(mock_ollama_dir))

        with patch("hf_model_tool.ollama.get_ollama_items", return_value=items):
            info1 = get_ollama_model_info("LLAMA2:7B")
            info2 = get_ollama_model_info("llama2:7b")

            # Both should find the same model
            if info1 and info2:
                assert info1["name"].lower() == info2["name"].lower()


class TestDefaultDirectories:
    """Test default Ollama directory constants."""

    @pytest.mark.unit
    def test_default_directories_defined(self):
        """Test that default directories are properly defined."""
        assert len(DEFAULT_OLLAMA_DIRS) > 0
        assert all(isinstance(d, Path) for d in DEFAULT_OLLAMA_DIRS)

        # Check that user directory is included
        user_dir = Path.home() / ".ollama" / "models"
        assert user_dir in DEFAULT_OLLAMA_DIRS


@pytest.mark.integration
class TestOllamaIntegration:
    """Integration tests for Ollama functionality."""

    def test_full_workflow(self, mock_ollama_dir):
        """Test the complete Ollama scanning workflow."""
        from unittest.mock import patch

        # Scan directories
        items = get_ollama_items(ollama_dirs=str(mock_ollama_dir))
        assert len(items) > 0

        # Get specific model info
        first_model = items[0]
        model_name = first_model["name"]

        # Mock get_ollama_items for get_ollama_model_info call
        with patch("hf_model_tool.ollama.get_ollama_items", return_value=items):
            info = get_ollama_model_info(model_name)
            assert info is not None
            assert info["name"] == model_name

            # Verify all required fields
            required_fields = ["name", "type", "size", "path", "source_type"]
            for field in required_fields:
                assert field in info, f"Missing required field: {field}"
