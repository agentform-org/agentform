#!/bin/bash
# Version bump script for ACP monorepo
# Usage: ./bump-version.sh [major|minor|patch]
# Defaults to 'patch' if no argument provided

set -e

BUMP_TYPE="${1:-patch}"

# Validate bump type
if [[ ! "$BUMP_TYPE" =~ ^(major|minor|patch)$ ]]; then
    echo "Error: Invalid bump type '$BUMP_TYPE'. Must be 'major', 'minor', or 'patch'"
    exit 1
fi

# Get the repository root (script is in .github/scripts/)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Read current version from acp-cli/pyproject.toml
CURRENT_VERSION=$(grep -Po '(?<=^version = ")[^"]*' "$REPO_ROOT/acp-cli/pyproject.toml")

if [[ -z "$CURRENT_VERSION" ]]; then
    echo "Error: Could not read current version from acp-cli/pyproject.toml"
    exit 1
fi

echo "Current version: $CURRENT_VERSION"
echo "Bump type: $BUMP_TYPE"

# Parse version components
IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT_VERSION"

# Calculate new version based on bump type
case "$BUMP_TYPE" in
    major)
        NEW_MAJOR=$((MAJOR + 1))
        NEW_VERSION="${NEW_MAJOR}.0.0"
        ;;
    minor)
        NEW_MINOR=$((MINOR + 1))
        NEW_VERSION="${MAJOR}.${NEW_MINOR}.0"
        ;;
    patch)
        NEW_PATCH=$((PATCH + 1))
        NEW_VERSION="${MAJOR}.${MINOR}.${NEW_PATCH}"
        ;;
esac

echo "New version: $NEW_VERSION"

# List of pyproject.toml files to update
PYPROJECT_FILES=(
    "acp-cli/pyproject.toml"
    "acp-compiler/pyproject.toml"
    "acp-mcp/pyproject.toml"
    "acp-runtime/pyproject.toml"
    "acp-schema/pyproject.toml"
)

# Update all pyproject.toml files
for file in "${PYPROJECT_FILES[@]}"; do
    filepath="$REPO_ROOT/$file"
    if [[ -f "$filepath" ]]; then
        sed -i "s/^version = \".*\"/version = \"$NEW_VERSION\"/" "$filepath"
        echo "Updated $file"
    else
        echo "Warning: $file not found"
    fi
done

# Update version.py
VERSION_PY="$REPO_ROOT/acp-schema/acp_schema/version.py"
if [[ -f "$VERSION_PY" ]]; then
    sed -i "s/^VERSION = \".*\"/VERSION = \"$NEW_VERSION\"/" "$VERSION_PY"
    echo "Updated acp-schema/acp_schema/version.py"
else
    echo "Warning: version.py not found"
fi

# Output the new version (for use in GitHub Actions)
echo "new_version=$NEW_VERSION" >> "${GITHUB_OUTPUT:-/dev/stdout}"

echo "Version bump complete: $CURRENT_VERSION -> $NEW_VERSION"
