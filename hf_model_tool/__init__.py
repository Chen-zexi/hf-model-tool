"""
HF-MODEL-TOOL: HuggingFace Model Management Tool

A CLI tool for managing locally downloaded HuggingFace models and datasets
"""

import sys

try:
    # Python 3.8+
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    # Python 3.7 fallback
    try:
        from importlib_metadata import version, PackageNotFoundError
    except ImportError:
        # If importlib_metadata is not available, use a simple fallback
        def version(package_name):
            return "0.2.1"
        
        class PackageNotFoundError(Exception):
            pass

def get_version():
    """Get the package version with multiple fallback strategies."""
    try:
        return version("hf-model-tool")
    except PackageNotFoundError:
        # Development mode fallback - try to read from pyproject.toml
        from pathlib import Path
        import re
        
        try:
            pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
            if pyproject_path.exists():
                with open(pyproject_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    # Simple regex to extract version
                    match = re.search(r'version\s*=\s*"([^"]+)"', content)
                    if match:
                        return match.group(1)
        except Exception:
            pass
        
        # Ultimate fallback
        return "0.2.1"

__version__ = get_version()

# Make version available at package level
__all__ = ["__version__"]
