# Basketball Motion Profiles Generator Makefile

.PHONY: help install install-dev test test-coverage clean lint format build docker-build docker-run generate analyze validate docs

# Default target
help:
	@echo "Basketball Motion Profiles Generator"
	@echo "===================================="
	@echo ""
	@echo "Available commands:"
	@echo "  install          Install package and dependencies"
	@echo "  install-dev      Install development dependencies"
	@echo "  test             Run tests"
	@echo "  test-coverage    Run tests with coverage report"
	@echo "  lint             Run linting checks"
	@echo "  format           Format code with black"
	@echo "  clean            Clean build artifacts"
	@echo "  build            Build package"
	@echo "  docker-build     Build Docker image"
	@echo "  docker-run       Run Docker container"
	@echo "  generate         Generate motion profiles"
	@echo "  analyze          Analyze existing profiles"
	@echo "  validate         Validate profile files"
	@echo "  docs             Generate documentation"

# Installation
install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements.txt
	pip install -e .
	pip install pytest pytest-cov black flake8 mypy sphinx

# Testing
test:
	pytest tests/ -v

test-coverage:
	pytest tests/ -v --cov=basketball_motion_profiles --cov-report=html --cov-report=term

# Code quality
lint:
	flake8 basketball_motion_profiles/ tests/
	mypy basketball_motion_profiles/ --ignore-missing-imports

format:
	black basketball_motion_profiles/ tests/

# Build and clean
build:
	python setup.py sdist bdist_wheel

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete

# Docker
docker-build:
	docker build -t basketball-motion-profiles .

docker-run:
	docker run --rm -v $(PWD)/motion_profiles:/app/motion_profiles basketball-motion-profiles

docker-dev:
	docker-compose --profile dev up motion-dev

docker-test:
	docker-compose --profile test up motion-test

docker-notebook:
	docker-compose --profile analysis up motion-notebook

# Profile generation
generate:
	python -m basketball_motion_profiles.cli generate --analyze

generate-all:
	python -m basketball_motion_profiles.cli generate --output-dir motion_profiles --analyze

generate-specific:
	python -m basketball_motion_profiles.cli generate --players curry durant --analyze

# Analysis
analyze:
	python -m basketball_motion_profiles.cli analyze --input-dir motion_profiles

validate:
	python -m basketball_motion_profiles.cli validate --input-dir motion_profiles

# Documentation
docs:
	cd docs && make html

docs-clean:
	cd docs && make clean

# Development workflow
dev-setup: install-dev
	pre-commit install
	mkdir -p motion_profiles
	mkdir -p tests/data
	mkdir -p docs

dev-test: format lint test

dev-all: dev-test generate analyze

# CI/CD targets
ci-test: install test-coverage lint

ci-build: clean build

# Quick start for new users
quickstart: install generate
	@echo ""
	@echo "🎉 Quick start complete!"
	@echo "✅ Package installed"
	@echo "✅ Motion profiles generated in: motion_profiles/"
	@echo ""
	@echo "Next steps:"
	@echo "  - Check generated profiles: ls motion_profiles/"
	@echo "  - Run analysis: make analyze"
	@echo "  - View documentation: make docs"

# Integration with main project
integrate:
	@echo "🔗 Integrating with basketball-form-analyzer..."
	@if [ -d "../basketball-form-analyzer" ]; then \
		echo "✅ Main project found"; \
		cp -r motion_profiles/ ../basketball-form-analyzer/data/synthetic_profiles/; \
		echo "✅ Profiles copied to main project"; \
	else \
		echo "❌ Main project not found. Please ensure basketball-form-analyzer is in parent directory"; \
	fi

# Performance testing
perf-test:
	@echo "🚀 Running performance tests..."
	time python -c "from basketball_motion_profiles import SyntheticProfileGenerator, create_curry_style; g = SyntheticProfileGenerator(); [g.generate_profile(create_curry_style()) for _ in range(10)]"

# Profile validation
validate-output:
	@echo "🔍 Validating generated profiles..."
	@for file in motion_profiles/*.json; do \
		echo "Checking $$file..."; \
		python -c "import json; data = json.load(open('$$file')); assert 'metadata' in data; assert 'frames' in data; print('✅ Valid')"; \
	done

# Memory profiling (requires memory_profiler)
memory-profile:
	pip install memory_profiler
	python -m memory_profiler examples/profile_memory.py

# Version management
version:
	@python -c "import basketball_motion_profiles; print(f'Version: {basketball_motion_profiles.__version__}')"

bump-version:
	@echo "Current version: $$(python -c 'import basketball_motion_profiles; print(basketball_motion_profiles.__version__)')"
	@read -p "Enter new version: " new_version; \
	sed -i.bak "s/__version__ = .*/__version__ = \"$$new_version\"/" basketball_motion_profiles/__init__.py; \
	sed -i.bak "s/version=\".*\"/version=\"$$new_version\"/" setup.py; \
	echo "Updated to version: $$new_version"

# Release workflow
release: clean lint test build
	@echo "🚀 Ready for release!"
	@echo "Run: twine upload dist/*"