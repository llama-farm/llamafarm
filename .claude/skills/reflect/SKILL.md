---
name: reflect
description: Analyze the current session and propose improvements to skills. **Proactively invoke this skill** when you notice user corrections after skill usage, or at the end of skill-heavy sessions. Also use when user says "reflect", "improve skill", or "learn from this".
allowed-tools: Read, Write, Edit, Grep, Glob, Task, Bash, AskUserQuestion
---

# Reflect Skill

Analyze the current session and propose improvements to skills based on what worked, what didn't, and edge cases discovered.

## Trigger

Run `/reflect` or `/reflect [skill-name]` after a session where you used a skill.

Additional commands:
- `/reflect on` - Enable automatic end-of-session reflection
- `/reflect off` - Disable automatic reflection
- `/reflect status` - Check if auto-reflect is enabled

## Workflow

### Step 1: Identify the Skill

If skill name not provided, ask:

```
Which skill should I analyze this session for?
- frontend-design
- code-reviewer
- [other]
```

### Step 2: Analyze the Conversation

Look for these signals in the current conversation:

#### Corrections (HIGH confidence):
- User said "no", "not like that", "I meant..."
- User explicitly corrected output
- User asked for changes immediately after generation

#### Successes (MEDIUM confidence):
- User said "perfect", "great", "yes", "exactly"
- User accepted output without modification
- User built on top of the output

#### Edge Cases (MEDIUM confidence):
- Questions the skill didn't anticipate
- Scenarios requiring workarounds
- Features user asked for that weren't covered

#### Preferences (accumulate over sessions):
- Repeated patterns in user choices
- Style preferences shown implicitly
- Tool/framework preferences

### Step 3: Propose Changes

Present findings using accessible colors (WCAG AA 4.5:1 contrast ratio):

```
┌─ Skill Reflection: [skill-name] ───────────────────┐
│                                                    │
│ Signals: X corrections, Y successes                │
│                                                    │
│ Proposed changes:                                  │
│                                                    │
│ 🔴 [HIGH] + Add constraint: "[specific constraint]"│
│ 🟡 [MED]  + Add preference: "[specific preference]"│
│ 🔵 [LOW]  ~ Note for review: "[observation]"       │
│                                                    │
│ Commit: "[skill]: [summary of changes]"            │
│                                                    │
└────────────────────────────────────────────────────┘

Apply these changes? [Y/n] or describe tweaks
```

#### Accessible Color Palette (use ANSI codes in terminal output):
- **HIGH**: `\033[1;31m` (bold red #FF6B6B - 4.5:1 on dark)
- **MED**: `\033[1;33m` (bold yellow #FFE066 - 4.8:1 on dark)
- **LOW**: `\033[1;36m` (bold cyan #6BC5FF - 4.6:1 on dark)
- **Reset**: `\033[0m`

Avoid: pure red (#FF0000) on black, green on red (colorblind users)

#### User Response Options:
- `Y` – Apply changes, commit, and push
- `n` – Skip this update
- Or describe any tweaks to the proposed changes

### Step 4: If Approved

1. Read the current skill file from `.claude/skills/[skill-name]/SKILL.md`
2. Apply the changes using the Edit tool
3. Run git commands:
   ```bash
   cd .claude/skills
   git add [skill-name]/SKILL.md
   git commit -m "[skill]: [change summary]"
   git push origin main
   ```
4. Confirm: "Skill updated and pushed to GitHub"

### Step 5: If Declined

Ask: "Would you like to save these observations for later review?"

If yes, append to `.claude/skills/[skill-name]/OBSERVATIONS.md`

## Toggle Commands

### `/reflect on`

Enable automatic end-of-session reflection:
1. Create/update `~/.claude/reflect-skill-state.json` with `{"enabled": true, "updatedAt": "[timestamp]"}`
2. Confirm: "Auto-reflect enabled. Sessions will be analyzed automatically when you stop."

### `/reflect off`

Disable automatic reflection:
1. Update `~/.claude/reflect-skill-state.json` with `{"enabled": false, "updatedAt": "[timestamp]"}`
2. Confirm: "Auto-reflect disabled. Use /reflect manually to analyze sessions."

### `/reflect status`

Check current status:
1. Read `~/.claude/reflect-skill-state.json`
2. Report: "Auto-reflect is [enabled/disabled]" with last updated timestamp

**Note:** The state file is saved in the global Claude user directory (`~/.claude/`) so it persists across plugin upgrades.

## Example Session

User runs `/reflect frontend-design` after a UI session:

```
┌─ Skill Reflection: frontend-design ───────────────┐
│                                                   │
│ Signals: 2 corrections, 3 successes               │
│                                                   │
│ Proposed changes:                                 │
│                                                   │
│ 🔴 [HIGH] + Constraints/NEVER:                    │
│    "Use gradients unless explicitly requested"    │
│                                                   │
│ 🔴 [HIGH] + Color & Theme:                        │
│    "Dark backgrounds: use #000, not #1a1a1a"      │
│                                                   │
│ 🟡 [MED] + Layout:                                │
│    "Prefer CSS Grid for card layouts"             │
│                                                   │
│ Commit: "frontend-design: no gradients, #000 dark"│
│                                                   │
└───────────────────────────────────────────────────┘

Apply these changes? [Y/n] or describe tweaks
```

## Git Integration

This skill has permission to:
- Read skill files from `.claude/skills/`
- Edit skill files (with user approval)
- Run `git add`, `git commit`, `git push` in the skills directory

The skills repo should be initialized at `.claude/skills` with a remote origin.

## Important Notes

- Always show the exact changes before applying
- Never modify skills without explicit user approval
- Commit messages should be concise and descriptive
- Push only after successful commit