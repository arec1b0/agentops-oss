.PHONY: help install dev test lint format clean docker-build docker-up docker-down run-collector run-ui run-demo clickhouse generate-keys

# Default target
help:
	@echo "AgentOps Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install all dependencies"
	@echo "  make dev            Install with dev dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make run-collector  Run collector locally (requires ClickHouse)"
	@echo "  make run-ui         Run UI locally"
	@echo "  make run-demo       Run demo agent"
	@echo "  make clickhouse     Start ClickHouse only (for local dev)"
	@echo ""
	@echo "Security:"
	@echo "  make generate-keys  Generate secure API keys"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build   Build all Docker images"
	@echo "  make docker-up      Start full stack with Docker Compose"
	@echo "  make docker-down    Stop all containers"
	@echo ""
	@echo "Testing:"
	@echo "  make test           Run all tests"
	@echo "  make lint           Run linters"
	@echo "  make format         Format code"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          Remove build artifacts"

# =============================================================================
# Setup
# =============================================================================

install:
	cd sdk && pip install -e .
	cd collector && pip install -r requirements.txt
	cd ui && pip install -r requirements.txt

dev: install
	pip install pytest pytest-cov black ruff mypy httpx

# =============================================================================
# Security
# =============================================================================

generate-keys:
	@echo "=== AgentOps API Key Generator ==="
	@echo ""
	@echo "Admin Key (full access):"
	@echo "  AGENTOPS_ADMIN_KEY=sk-admin-$$(openssl rand -hex 24)"
	@echo ""
	@echo "Ingest Key (agents only):"
	@echo "  AGENTOPS_INGEST_KEY=sk-ingest-$$(openssl rand -hex 24)"
	@echo ""
	@echo "Copy these to your .env file or export them."
	@echo ""
	@echo "Quick start:"
	@echo "  cp .env.example .env"
	@echo "  # Edit .env with generated keys"
	@echo "  docker-compose up -d"

# =============================================================================
# Local Development
# =============================================================================

clickhouse:
	docker-compose up -d clickhouse
	@echo ""
	@echo "ClickHouse started:"
	@echo "  HTTP:   http://localhost:8123"
	@echo "  Native: localhost:9000"
	@echo ""
	@echo "Wait for healthy status, then run: make run-collector"

run-collector:
	@echo "Starting collector (connecting to localhost ClickHouse)..."
	@echo ""
	@echo "=== Security Mode ==="
	@if [ -z "$$AGENTOPS_ADMIN_KEY" ] && [ -z "$$AGENTOPS_INGEST_KEY" ]; then \
		echo "WARNING: No API keys set. Authentication DISABLED."; \
		echo "For production, set AGENTOPS_ADMIN_KEY or AGENTOPS_INGEST_KEY"; \
	else \
		echo "API key authentication ENABLED"; \
	fi
	@echo ""
	cd collector && \
		CLICKHOUSE_HOST=localhost \
		CLICKHOUSE_PORT=8123 \
		uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

run-ui:
	@echo "Starting UI..."
	@echo "Set AGENTOPS_API_KEY if collector requires authentication"
	cd ui && streamlit run app.py --server.port 8501

run-demo:
	@echo "Running demo agent..."
	@echo "Set AGENTOPS_API_KEY if collector requires authentication"
	cd examples && python basic_agent.py

# =============================================================================
# Docker
# =============================================================================

docker-build:
	docker-compose build

docker-up:
	@echo "=== Starting AgentOps Stack ==="
	@echo ""
	@if [ ! -f .env ]; then \
		echo "WARNING: No .env file found. Using defaults (no authentication)."; \
		echo "For production, run: make generate-keys"; \
		echo ""; \
	fi
	docker-compose up -d
	@echo ""
	@echo "Services:"
	@echo "  Collector:   http://localhost:8000"
	@echo "  UI:          http://localhost:8501"
	@echo "  ClickHouse:  http://localhost:8123"
	@echo ""
	@echo "API Docs:      http://localhost:8000/docs"
	@echo ""
	@echo "Logs: docker-compose logs -f"

docker-down:
	docker-compose down

docker-clean:
	docker-compose down -v --remove-orphans

# =============================================================================
# Testing
# =============================================================================

test:
	cd sdk && pytest tests/ -v --cov=agentops_sdk --cov-report=term-missing

test-integration:
	@echo "Running integration tests (requires running stack)..."
	pytest tests/integration/ -v

lint:
	ruff check sdk/ collector/ ui/
	mypy sdk/ --ignore-missing-imports

format:
	black sdk/ collector/ ui/
	ruff check --fix sdk/ collector/ ui/

# =============================================================================
# Cleanup
# =============================================================================

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true