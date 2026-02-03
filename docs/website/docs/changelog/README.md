# Changelog Documentation System

This directory contains human-readable release notes for LlamaFarm, automatically generated from conventional commits.

## How It Works

### Automated Workflow (Primary Method)

When a release PR is created by release-please:

1. **Workflow triggers** - [.github/workflows/update-changelog-docs.yml](../../../../.github/workflows/update-changelog-docs.yml) detects the release PR
2. **Extracts version** - Reads the latest version from CHANGELOG.md
3. **Generates prose** - Uses LlamaFarm AI to transform conventional commits into narrative release notes
4. **Creates docs** - Writes Docusaurus-formatted markdown to this directory
5. **Updates index** - Adds the new release to the changelog index
6. **Commits to PR** - Pushes the changelog docs back to the release PR
7. **Posts comment** - Adds a comment with status (success/failure)

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
├── README.md              # This file (developer documentation)
└── index.md               # Changelog page with all releases as expandable sections
```

All release notes are embedded in `index.md` as expandable `<details>` sections. The most recent release is expanded by default.

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

### Test on a Real Release PR

1. Wait for release-please to create a release PR (or create one manually)
2. The workflow will automatically trigger
3. Check for the comment on the PR
4. Review the committed changelog files
5. Merge when ready

## Prose Changelog Generation

The human-readable release notes are generated using:

**Action:** [.github/actions/prose-changelog/](../../../../.github/actions/prose-changelog/)

**Process:**
1. Extracts conventional commits from CHANGELOG.md
2. Feeds them to `lf chat` with a specialized prompt
3. AI transforms commits into user-friendly narrative
4. Focuses on **user impact** and **value**, not technical details

**Prompt configuration:** [.github/actions/prose-changelog/llamafarm.yaml](../../../../.github/actions/prose-changelog/llamafarm.yaml)

The prompt instructs the AI to:
- Write in a professional yet approachable tone
- Group changes into logical sections
- Explain WHAT users can do and WHY it matters
- Avoid technical jargon and commit references
- Combine related commits into cohesive narratives

## Docusaurus Integration

All release notes appear on a single page at `/docs/changelog`:
- https://docs.llamafarm.dev/docs/changelog

Each release is an expandable section. The most recent release is expanded by default for immediate visibility.

### Sidebar Configuration

Changelog section added to [sidebars.ts](../../sidebars.ts):

```typescript
{
  type: 'doc',
  id: 'changelog/index',
  label: 'Changelog',
}
```

All releases are on a single page with expandable sections.

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

3. **No changes detected**
   - Changelog might already exist for this version
   - Check if the version was already processed

### Manual Script Failed

1. **`lf: command not found`**
   ```bash
   # Install LlamaFarm CLI
   curl -fsSL https://raw.githubusercontent.com/llama-farm/llamafarm/main/install.sh | bash
   ```

2. **Generation timeout or error**
   ```bash
   # Start services
   lf start

   # Verify services are running
   lf services status
   ```

3. **Version not found**
   - Check that the version exists in CHANGELOG.md
   - Use exact version format (e.g., "0.0.26", not "v0.0.26")

## Future Improvements

Potential enhancements:

- [ ] Add version-specific sidebar items (might get too long)
- [ ] Generate changelog on release tag (not just PR)
- [ ] Support for pre-releases and release candidates
- [ ] Automated Reddit posting using generated prose
- [ ] RSS feed for changelog updates
- [ ] Email notifications for new releases

## Questions or Issues?

- **Workflow issues**: Check [.github/workflows/update-changelog-docs.yml](../../../../.github/workflows/update-changelog-docs.yml)
- **Generation issues**: Check [.github/actions/prose-changelog/](../../../../.github/actions/prose-changelog/)
- **Docs issues**: Check [index.md](./index.md) and individual version files
