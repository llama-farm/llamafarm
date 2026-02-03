#!/usr/bin/env bash
# Manual script to generate and update changelog documentation
#
# This script can be run manually if the automated workflow fails,
# or for backfilling historical releases.
#
# Usage:
#   ./scripts/update-changelog-docs.sh                    # Latest version
#   ./scripts/update-changelog-docs.sh 0.0.26             # Specific version
#   ./scripts/update-changelog-docs.sh --all              # All versions

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CHANGELOG_FILE="$REPO_ROOT/CHANGELOG.md"
DOCS_CHANGELOG_DIR="$REPO_ROOT/docs/website/docs/changelog"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to generate changelog for a specific version
generate_changelog_for_version() {
    local version="$1"

    log_info "Generating changelog for version $version..."

    # Check if version exists in CHANGELOG.md
    if ! grep -q "## \[${version}\]" "$CHANGELOG_FILE"; then
        log_error "Version ${version} not found in CHANGELOG.md"
        return 1
    fi

    # Extract release date
    local date
    date=$(grep -m1 "## \[${version}\]" "$CHANGELOG_FILE" | sed 's/.*(\([0-9-]*\)).*/\1/')
    if [[ -z "$date" ]]; then
        date=$(date +%Y-%m-%d)
        log_warn "Could not extract date from CHANGELOG.md, using today: $date"
    fi

    # Generate prose changelog using the existing action script
    log_info "Running prose changelog generator..."
    cd "$REPO_ROOT"

    INPUT_VERSION="$version" \
    INPUT_CHANGELOG_FILE="$CHANGELOG_FILE" \
    .github/actions/prose-changelog/generate.sh > /tmp/prose-changelog-${version}.txt

    local prose_content
    prose_content=$(cat /tmp/prose-changelog-${version}.txt)

    if [[ -z "$prose_content" ]]; then
        log_error "Failed to generate prose changelog for version ${version}"
        return 1
    fi

    # Update index with expandable section instead of separate file
    if grep -q "v${version}" "$DOCS_CHANGELOG_DIR/index.md"; then
        log_info "v${version} already in index.md, skipping"
        return 0
    fi

    log_info "Adding v${version} as expandable section to index.md..."

    # Create temporary file with new release section
    cat > /tmp/new-release-section-${version}.md <<RELEASE_EOF
<details open>
<summary><strong>v${version} (${date})</strong> - Latest</summary>

<br/>

*Released on ${date}*

${prose_content}

---

**Full Changelog**: [v${version}](https://github.com/llama-farm/llamafarm/blob/main/CHANGELOG.md#${version//./})

</details>

RELEASE_EOF

    # Remove 'open' attribute from existing details tags
    sed -i.bak 's/<details open>/<details>/g' "$DOCS_CHANGELOG_DIR/index.md"

    # Remove " - Latest" from existing releases
    sed -i.bak 's/ - Latest<\/strong>/<\/strong>/g' "$DOCS_CHANGELOG_DIR/index.md"

    # Insert new section after "## Recent Releases"
    awk '
        /^## Recent Releases$/ {
            print
            print ""
            system("cat /tmp/new-release-section-'"$version"'.md")
            next
        }
        { print }
    ' "$DOCS_CHANGELOG_DIR/index.md" > "$DOCS_CHANGELOG_DIR/index.md.tmp"

    mv "$DOCS_CHANGELOG_DIR/index.md.tmp" "$DOCS_CHANGELOG_DIR/index.md"
    rm -f "$DOCS_CHANGELOG_DIR/index.md.bak" /tmp/new-release-section-${version}.md

    log_info "✅ Updated index.md with expandable section for v${version}"

    rm -f /tmp/prose-changelog-${version}.txt

    log_info "✅ Successfully generated changelog docs for v${version}"
}

# Main script logic
main() {
    cd "$REPO_ROOT"

    # Ensure changelog directory exists
    mkdir -p "$DOCS_CHANGELOG_DIR"

    if [[ $# -eq 0 ]]; then
        # No arguments - use latest version from CHANGELOG.md
        log_info "No version specified, using latest from CHANGELOG.md..."
        version=$(grep -m1 '## \[' "$CHANGELOG_FILE" | sed 's/.*\[\([^]]*\)\].*/\1/')

        if [[ -z "$version" ]]; then
            log_error "Could not determine latest version from CHANGELOG.md"
            exit 1
        fi

        log_info "Detected latest version: $version"
        generate_changelog_for_version "$version"

    elif [[ "$1" == "--all" ]]; then
        # Generate for all versions in CHANGELOG.md
        log_info "Generating changelogs for all versions..."

        versions=$(grep '## \[' "$CHANGELOG_FILE" | sed 's/.*\[\([^]]*\)\].*/\1/')

        for version in $versions; do
            log_info "Processing version $version..."
            generate_changelog_for_version "$version" || log_warn "Failed to generate changelog for $version, continuing..."
        done

    else
        # Specific version provided
        version="$1"
        generate_changelog_for_version "$version"
    fi

    log_info ""
    log_info "🎉 Changelog documentation updated successfully!"
    log_info ""
    log_info "Next steps:"
    log_info "  1. Review the generated files in docs/website/docs/changelog/"
    log_info "  2. Commit the changes: git add docs/website/docs/changelog/ && git commit -m 'docs: update changelog'"
    log_info "  3. Push to your branch"
}

# Run main function
main "$@"
