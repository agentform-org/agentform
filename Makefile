.PHONY: install test install-cli install-compiler install-mcp install-runtime install-schema \
        test-cli test-compiler test-mcp test-runtime test-schema

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

