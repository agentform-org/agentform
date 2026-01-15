.PHONY: install test lint typecheck \
        install-cli install-compiler install-mcp install-runtime install-schema \
        test-cli test-compiler test-mcp test-runtime test-schema \
        lint-cli lint-compiler lint-mcp lint-runtime lint-schema \
        typecheck-cli typecheck-compiler typecheck-mcp typecheck-runtime typecheck-schema

LIBS = acp-cli acp-compiler acp-mcp acp-runtime acp-schema

# Install all libraries
install: install-cli install-compiler install-mcp install-runtime install-schema

install-cli:
	cd acp-cli && poetry install

install-compiler:
	cd acp-compiler && poetry install

install-mcp:
	cd acp-mcp && poetry install

install-runtime:
	cd acp-runtime && poetry install

install-schema:
	cd acp-schema && poetry install

# Test all libraries
test: test-cli test-compiler test-mcp test-runtime test-schema

test-cli:
	cd acp-cli && poetry run pytest

test-compiler:
	cd acp-compiler && poetry run pytest

test-mcp:
	cd acp-mcp && poetry run pytest

test-runtime:
	cd acp-runtime && poetry run pytest

test-schema:
	cd acp-schema && poetry run pytest

# Lint all libraries
lint: lint-cli lint-compiler lint-mcp lint-runtime lint-schema

lint-cli:
	cd acp-cli && poetry run ruff check .

lint-compiler:
	cd acp-compiler && poetry run ruff check .

lint-mcp:
	cd acp-mcp && poetry run ruff check .

lint-runtime:
	cd acp-runtime && poetry run ruff check .

lint-schema:
	cd acp-schema && poetry run ruff check .

# Type check all libraries
typecheck: typecheck-cli typecheck-compiler typecheck-mcp typecheck-runtime typecheck-schema

typecheck-cli:
	cd acp-cli && poetry run mypy .

typecheck-compiler:
	cd acp-compiler && poetry run mypy .

typecheck-mcp:
	cd acp-mcp && poetry run mypy .

typecheck-runtime:
	cd acp-runtime && poetry run mypy .

typecheck-schema:
	cd acp-schema && poetry run mypy .

