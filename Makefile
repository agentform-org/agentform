.PHONY: install test lint typecheck \
        install-cli install-compiler install-mcp install-runtime install-schema \
        test-cli test-compiler test-mcp test-runtime test-schema \
        lint-cli lint-compiler lint-mcp lint-runtime lint-schema \
        typecheck-cli typecheck-compiler typecheck-mcp typecheck-runtime typecheck-schema

LIBS = agentform-cli agentform-compiler agentform-mcp agentform-runtime agentform-schema

# Install all libraries
install: install-cli install-compiler install-mcp install-runtime install-schema

install-cli:
	cd agentform-cli && poetry install

install-compiler:
	cd agentform-compiler && poetry install

install-mcp:
	cd agentform-mcp && poetry install

install-runtime:
	cd agentform-runtime && poetry install

install-schema:
	cd agentform-schema && poetry install

# Test all libraries
test: test-cli test-compiler test-mcp test-runtime test-schema

test-cli:
	cd agentform-cli && poetry run pytest

test-compiler:
	cd agentform-compiler && poetry run pytest

test-mcp:
	cd agentform-mcp && poetry run pytest

test-runtime:
	cd agentform-runtime && poetry run pytest

test-schema:
	cd agentform-schema && poetry run pytest

# Lint all libraries
lint: lint-cli lint-compiler lint-mcp lint-runtime lint-schema

lint-cli:
	cd agentform-cli && poetry run ruff check .

lint-compiler:
	cd agentform-compiler && poetry run ruff check .

lint-mcp:
	cd agentform-mcp && poetry run ruff check .

lint-runtime:
	cd agentform-runtime && poetry run ruff check .

lint-schema:
	cd agentform-schema && poetry run ruff check .

# Type check all libraries
typecheck: typecheck-cli typecheck-compiler typecheck-mcp typecheck-runtime typecheck-schema

typecheck-cli:
	cd agentform-cli && poetry run mypy .

typecheck-compiler:
	cd agentform-compiler && poetry run mypy .

typecheck-mcp:
	cd agentform-mcp && poetry run mypy .

typecheck-runtime:
	cd agentform-runtime && poetry run mypy .

typecheck-schema:
	cd agentform-schema && poetry run mypy .

