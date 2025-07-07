.PHONY: help install test test-simple test-cov test-unit test-integration lint format type-check clean build publish dev-install qa qa-relaxed qa-simple

help:  ## Show this help message
	@echo "HF-MODEL-TOOL Development Commands"
	@echo "=================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install package in development mode
	pip install -e .

dev-install:  ## Install with development dependencies
	pip install -e ".[dev]"

test-simple:  ## Run tests without coverage
	pytest tests/ -v

test:  ## Run tests with coverage (requires pytest-cov)
	@pytest tests/ -v --cov=hf_model_tool --cov-report=term-missing --cov-report=xml || \
	(echo "Coverage failed, running simple tests..." && pytest tests/ -v)

test-cov:  ## Run tests with coverage reporting
	pytest tests/ -v --cov=hf_model_tool --cov-report=term-missing --cov-report=xml

test-unit:  ## Run only unit tests (fast)
	pytest -m unit --tb=short

test-integration:  ## Run only integration tests
	pytest -m integration --tb=short

test-ci:  ## Run tests in CI order (unit first, then integration)
	@echo "Running unit tests..."
	@pytest -m unit --tb=short
	@echo "Running integration tests..."
	@pytest -m integration --tb=short

lint:  ## Run linting (flake8)
	flake8 hf_model_tool/ tests/

format:  ## Format code with black
	black hf_model_tool/ tests/

format-check:  ## Check if code is formatted
	black --check hf_model_tool/ tests/

format-fix:  ## Auto-fix formatting issues
	@echo "Auto-fixing formatting..."
	@black hf_model_tool/ tests/ || echo "Formatting completed"

type-check:  ## Run type checking with mypy
	mypy hf_model_tool/

qa: format-check lint type-check  ## Run all quality checks (strict)

qa-simple: format-fix test-simple  ## Run basic quality checks with auto-fix
	@echo "Running simple quality checks..."
	@flake8 hf_model_tool/ tests/ || echo "⚠ Linting issues found (continuing)"
	@mypy hf_model_tool/ || echo "⚠ Type checking issues found (continuing)"
	@echo "✓ Simple quality checks completed"

qa-relaxed:  ## Run quality checks with auto-fixes and allow failures
	@echo "Running relaxed quality checks..."
	@black hf_model_tool/ tests/ || echo "✓ Code formatted"
	@flake8 hf_model_tool/ tests/ || echo "⚠ Linting issues found (continuing)"
	@mypy hf_model_tool/ || echo "⚠ Type checking issues found (continuing)"
	@echo "✓ Quality checks completed"

clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf .pytest_cache/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build:  ## Build package
	python -m build

publish-test:  ## Publish to TestPyPI
	twine upload --repository testpypi dist/*

publish:  ## Publish to PyPI
	twine upload dist/*

check-deps:  ## Check for outdated dependencies
	pip list --outdated

run:  ## Run the application
	python -m hf_model_tool