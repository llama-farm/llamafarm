# Changelog Documentation System

This directory contains human-readable release notes for LlamaFarm, automatically generated from conventional commits.

## How It Works

### Automated Workflow (Primary Method)

When a release PR is created by release-please:

1. **Workflow triggers** - [.github/workflows/update-changelog-docs.yml](../../../../.github/workflows/update-changelog-docs.yml) detects the release PR (filtered by `CHANGELOG.md` changes)
2. **Extracts version** - Reads the latest version from CHANGELOG.md
3. **Generates prose** - Uses LlamaFarm AI to transform conventional commits into narrative release notes
4. **Updates index** - Adds a new `<details>` accordion section to `index.md`
5. **Commits to PR** - Pushes the changelog docs back to the release PR
6. **Posts comment** - Adds a comment with status (success/failure)

### Manual Fallback (Backup Method)

If the automated workflow fails or you need to backfill historical releases:

```bash
# Generate changelog for latest release
./scripts/update-changelog-docs.sh

# Generate for specific version
./scripts/update-changelog-docs.sh 0.0.26

# Generate for all versions in CHANGELOG.md
./scripts/update-changelog-docs.sh --all
```

**Requirements for manual script:**
- LlamaFarm CLI (`lf`) must be installed
- LlamaFarm services must be running (`lf start`)
- Or `OPENAI_API_KEY` environment variable set for OpenAI provider

## File Structure

```
docs/website/docs/changelog/
├── README.md       # This file (developer documentation)
└── index.md        # Single changelog page with <details> accordion sections
```

All releases live in a single `index.md` file as `<details>` accordion sections. The latest release is expanded by default (`<details open>`), while older releases are collapsed. No individual version files are created.

## Testing the System

### Test with Workflow Dispatch (Recommended)

You can test the workflow without creating a real release PR:

1. Go to GitHub Actions → "Update Changelog Docs" workflow
2. Click "Run workflow"
3. Leave inputs empty (auto-detects latest version) or specify:
   - `version`: e.g., "0.0.26"
   - `pr_number`: If testing on an existing PR
4. Check the workflow output and generated files

### Test Locally with Manual Script

```bash
# 1. Start LlamaFarm services
lf start

# 2. Run the script
./scripts/update-changelog-docs.sh 0.0.26

# 3. Check the generated files
ls -la docs/website/docs/changelog/

# 4. Preview in Docusaurus
cd docs/website
npm run start
# Visit http://localhost:3000/docs/changelog
```

## Prose Changelog Generation

The human-readable release notes are generated using:

**Action:** [.github/actions/prose-changelog/](../../../../.github/actions/prose-changelog/)

**Process:**
1. Extracts conventional commits from CHANGELOG.md
2. Feeds them to `lf chat` with a specialized prompt
3. AI transforms commits into user-friendly narrative
4. Focuses on **user impact** and **value**, not technical details

**Prompt configuration:** [.github/actions/prose-changelog/llamafarm.yaml](../../../../.github/actions/prose-changelog/llamafarm.yaml)

## Docusaurus Integration

Release notes appear under the "Changelog" section in the docs sidebar:
- Changelog page: https://docs.llamafarm.dev/docs/changelog

### Sidebar Configuration

The sidebar in [sidebars.ts](../../sidebars.ts) links to the single changelog page:

```typescript
{
  type: 'doc',
  id: 'changelog/index',
  label: 'Changelog',
}
```

New releases are automatically added as accordion sections — no sidebar changes needed.

## Troubleshooting

### Workflow Failed

Check the workflow logs for errors. Common issues:

1. **Prose generation failed**
   - Ensure GPU runner has LlamaFarm CLI installed
   - Check if `lf start` is working on the runner
   - Verify the model is downloaded

2. **Git push failed**
   - Check GH_RELEASE_TOKEN permissions
   - Verify bot has write access to the repo

3. **Version already exists**
   - The workflow skips if the version accordion already exists in `index.md`
   - Remove the `<details>` section from `index.md` if you want to regenerate

### Manual Script Failed

1. **`lf: command not found`**
   ```bash
   curl -fsSL https://raw.githubusercontent.com/llama-farm/llamafarm/main/install.sh | bash
   ```

2. **Generation timeout or error**
   ```bash
   lf start
   lf services status
   ```

3. **Version not found**
   - Check that the version exists in CHANGELOG.md
   - Use exact version format (e.g., "0.0.26", not "v0.0.26")

## Adding a New Release Manually

If you need to add a release without the AI generation:

1. Add a `<details>` accordion section to `index.md` under `## Latest Release`
2. Close the previous latest by removing the `open` attribute from its `<details>` tag

## Questions or Issues?

- **Workflow issues**: Check [.github/workflows/update-changelog-docs.yml](../../../../.github/workflows/update-changelog-docs.yml)
- **Generation issues**: Check [.github/actions/prose-changelog/](../../../../.github/actions/prose-changelog/)
- **Docs issues**: Check `index.md` accordion sections
