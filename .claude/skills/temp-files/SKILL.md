---
name: temp-files
description: Guidelines for creating temporary files in system temp directory. Use when agents need to create reports, logs, or progress files without cluttering the repository.
allowed-tools: Bash, Write, Read
---

# Temporary Files Skill

When you need to create files to track progress, generate reports, or store temporary data, use the system's temporary directory instead of the repository root.

## Directory Structure

Use this base path for all temporary files (aligns with Claude Code's existing task output convention):

```
/tmp/claude/{sanitized-cwd}/
```

Where `{sanitized-cwd}` is the current working directory path with `/` replaced by `-` (leading slash stripped first to avoid a leading dash).

Example: Working in `/Users/bobby/workspace/pivot/llamafarm` → `/tmp/claude/Users-bobby-workspace-pivot-llamafarm/`

## Creating a Temp File

### Step 1: Create the directory

```bash
SANITIZED_PATH=$(echo "$PWD" | sed 's|^/||' | tr '/' '-')
REPORT_DIR="/tmp/claude/${SANITIZED_PATH}/reviews"
mkdir -p "$REPORT_DIR"
```

### Step 2: Generate a unique filename

Use this pattern: `{descriptor}-{YYYYMMDD-HHMMSS}.{ext}`

```bash
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
FILENAME="code-review-${TIMESTAMP}.md"
FILEPATH="${REPORT_DIR}/${FILENAME}"
```

### Step 3: Write the file

Use the Write tool with the full temp path.

### Step 4: Inform the user

Always tell the user where the file was created:

> Report saved to: `/tmp/claude/Users-bobby-workspace-pivot-llamafarm/reviews/code-review-20260108-143052.md`

## When to Use This Pattern

- Code review reports
- Analysis outputs
- Progress tracking files
- Test result summaries
- Any generated documentation not explicitly requested in a specific location

## When NOT to Use This Pattern

- User explicitly specifies a file path
- Creating files that should be committed (e.g., README, config files)
- Editing existing files

## Cleanup Note

Files in `/tmp/` are cleared on system restart. If the user needs to preserve a file, suggest they copy it to a permanent location.
