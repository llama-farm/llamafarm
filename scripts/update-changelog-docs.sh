#!/usr/bin/env bash
# Manual script to generate and update changelog documentation
#
# This script generates prose release notes and adds them as accordion
# sections in the changelog index. Use this if the automated workflow
# fails or for backfilling historical releases.
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

    # Check if version already exists in index before expensive AI generation
    local version_regex
    version_regex=$(echo "$version" | sed 's/\./\\./g')
    if grep -qE "<strong>v${version_regex}</strong>" "$DOCS_CHANGELOG_DIR/index.md"; then
        log_warn "v${version} already in index.md, skipping"
        return 0
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

    local prose_content
    prose_content=$(INPUT_VERSION="$version" \
        INPUT_CHANGELOG_FILE="$CHANGELOG_FILE" \
        .github/actions/prose-changelog/generate.sh 2>/dev/null || echo "")

    if [[ -z "$prose_content" ]]; then
        log_error "Failed to generate prose changelog for version ${version}"
        return 1
    fi

    log_info "Adding v${version} as accordion to index.md..."

    # Create new accordion section
    local tmp_section="/tmp/new-release-section-${version}.md"
    cat > "$tmp_section" <<EOF
<details open>
<summary><strong>v${version}</strong> — ${date}</summary>

${prose_content}

**[Full Changelog →](https://github.com/llama-farm/llamafarm/releases/tag/v${version})**

</details>
EOF

    # Close the currently open accordion (remove 'open' attribute)
    # Use portable sed syntax (works on both BSD/macOS and GNU/Linux)
    sed 's/<details open>/<details>/g' "$DOCS_CHANGELOG_DIR/index.md" > "$DOCS_CHANGELOG_DIR/index.md.tmp"
    mv "$DOCS_CHANGELOG_DIR/index.md.tmp" "$DOCS_CHANGELOG_DIR/index.md"

    # Insert new section after "## Latest Release" heading
    awk -v section_file="$tmp_section" '
        /^## Latest Release$/ {
            print
            print ""
            while ((getline line < section_file) > 0) {
                print line
            }
            close(section_file)
            next
        }
        { print }
    ' "$DOCS_CHANGELOG_DIR/index.md" > "$DOCS_CHANGELOG_DIR/index.md.tmp"

    mv "$DOCS_CHANGELOG_DIR/index.md.tmp" "$DOCS_CHANGELOG_DIR/index.md"
    rm -f "$tmp_section"
    log_info "Updated index.md with v${version} accordion"

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

        # Extract versions and store in array (portable approach for Bash 3.2+)
        versions=()
        while IFS= read -r version; do
            versions+=("$version")
        done < <(grep '## \[' "$CHANGELOG_FILE" | sed 's/.*\[\([^]]*\)\].*/\1/')

        # Reverse the array so oldest is processed first
        # This ensures newest ends up at the top when inserting at "## Latest Release"
        for ((i=${#versions[@]}-1; i>=0; i--)); do
            version="${versions[$i]}"
            log_info "Processing version $version..."
            generate_changelog_for_version "$version" || log_warn "Failed for $version, continuing..."
        done

    else
        # Specific version provided
        version="$1"
        generate_changelog_for_version "$version"
    fi

    log_info ""
    log_info "🎉 Changelog documentation updated!"
    log_info ""
    log_info "Next steps:"
    log_info "  1. Review files in docs/website/docs/changelog/"
    log_info "  2. Commit: git add docs/website/docs/changelog/ && git commit -m 'docs: update changelog'"
    log_info "  3. Push to your branch"
}

# Run main function
main "$@"
