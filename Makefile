.PHONY: help install dev test run-collector run-ui docker-build docker-up docker-down clean

PYTHON := python3
PIP := pip3

help:
	@echo "AgentOps OSS - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install      Install SDK in development mode"
	@echo "  make dev          Install all development dependencies"
	@echo ""
	@echo "Run locally:"
	@echo "  make clickhouse      Start ClickHouse only"
	@echo "  make run-collector   Start collector on port 8000"
	@echo "  make run-ui          Start UI on port 8501"
	@echo "  make run-demo        Run example agent"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build    Build Docker images"
	@echo "  make docker-up       Start all services"
	@echo "  make docker-down     Stop all services"
	@echo ""
	@echo "Testing:"
	@echo "  make test            Run all tests"

# ============================================================================
# Setup
# ============================================================================

install:
	$(PIP) install -e ./sdk/python

dev: install
	$(PIP) install -r ./collector/requirements.txt
	$(PIP) install -r ./ui/requirements.txt
	$(PIP) install pytest pytest-asyncio httpx

# ============================================================================
# Local Development
# ============================================================================

clickhouse:
	docker-compose up -d clickhouse
	@echo "✓ ClickHouse started on localhost:8123"

run-collector:
	@echo "Note: ClickHouse must be running (make clickhouse)"
	cd collector && CLICKHOUSE_HOST=localhost $(PYTHON) -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

run-ui:
	cd ui && streamlit run app.py --server.port 8501

run-demo: install
	$(PYTHON) examples/basic_agent.py

# ============================================================================
# Docker
# ============================================================================

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d
	@echo ""
	@echo "Services started:"
	@echo "  ClickHouse: http://localhost:8123"
	@echo "  Collector:  http://localhost:8000"
	@echo "  UI:         http://localhost:8501"

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

# ============================================================================
# Testing
# ============================================================================

test: test-sdk

test-sdk:
	cd sdk/python && $(PYTHON) -m pytest tests/ -v

# ============================================================================
# Cleanup
# ============================================================================

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache build dist 2>/dev/null || true

clean-docker:
	docker-compose down -v --rmi local
