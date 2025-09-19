# GitHub - Issue Management

You are tasked with managing GitHub issues, including creating issues from thoughts documents, updating existing issues, and following the team's specific workflow patterns.

## Initial Setup

First, verify that GitHub MCP tools are available by checking if any `mcp__GitHub__` tools exist. If not, attempt to use the GitHub API directly or VSCode extension instead.

If tools are available, respond based on the user's request:

### For general requests:
```
I can help you with GitHub issues. What would you like to do?
1. Create a new issue from a thoughts document
2. Create a new issue ad-hoc
3. Add a comment to an issue (I'll use our conversation context)
4. Search for issues
5. Update issue status or details
```

### For specific create requests:
```
I'll help you create a GitHub issue from your thoughts document. Please provide:
1. The path to the thoughts document (or topic to search for) -or- a good description of the issue to create.
2. Any specific focus or angle for the issue (optional)
```

Then wait for the user's input.

## Team Workflow & Status Progression

The team follows a specific workflow to ensure alignment before code implementation:

1. **New** → All new issues start here for initial review
4. **Researching** → Active research/investigation underway
7. **Planning** → Actively writing the implementation plan
10. **Implementing** → Active development and review
12. **Done** → Completed

**Key principle**: Review and alignment happen at the planning stage (not PR stage) to move faster and avoid rework.

## Important Conventions

### URL Mapping for Thoughts Documents
When referencing thoughts documents, always provide GitHub links using the `links` parameter:
- `thoughts/shared/...` → `https://github.com/llama-farm/thoughts/blob/main/repos/llamafarm/shared/...`
- `thoughts/alice/...` → `https://github.com/llama-farm/thoughts/blob/main/repos/llamafarm/alice/...`
- `thoughts/global/...` → `https://github.com/llama-farm/thoughts/blob/main/global/...`

### Default Values
- **Status**: Always create new issues in "New" status
- **Project**: For new issues, default to "LlamaFarm development pipeline" unless told otherwise
- **Priority**: Default to Medium (3) for most tasks, use best judgment or ask user
  - Urgent (1): Critical blockers, security issues
  - High (2): Important features with deadlines, major bugs
  - Medium (3): Standard implementation tasks (default)
  - Low (4): Nice-to-haves, minor improvements
- **Links**: Use the `links` parameter to attach URLs (not just markdown links in description)

### Automatic Label Assignment
Automatically apply labels based on the issue content:
- **component::cli**: For issues about the `cli/` directory (the daemon)
- **component::server**: For issues about `server/`
- **component::designer**: For issues about `designer/`
- **component::docs**: For issues about `docs/`
- **component::rag**: For issues about `rag/`
- **component::models**: For issues about `models/`

## Action-Specific Instructions

### 1. Creating issues from Thoughts

#### Steps to follow after receiving the request:

1. **Locate and read the thoughts document:**
   - If given a path, read the document directly
   - If given a topic/keyword, search thoughts/ directory using Grep to find relevant documents
   - If multiple matches found, show list and ask user to select
   - Create a TodoWrite list to track: Read document → Analyze content → Draft issue → Get user input → Create issue

2. **Analyze the document content:**
   - Identify the core problem or feature being discussed
   - Extract key implementation details or technical decisions
   - Note any specific code files or areas mentioned
   - Look for action items or next steps
   - Identify what stage the idea is at (early ideation vs ready to implement)
   - Take time to ultrathink about distilling the essence of this document into a clear problem statement and solution approach

3. **Check for related context (if mentioned in doc):**
   - If the document references specific code files, read relevant sections
   - If it mentions other thoughts documents, quickly check them
   - Look for any existing GitHub issues mentioned

4. **Get GitHub workspace context:**
   - List teams: `mcp__GitHub__list_teams`
   - If multiple teams, ask user to select one
   - List projects for selected team: `mcp__GitHub__list_projects`

5. **Draft the issue summary:**
   Present a draft to the user:
   ```
   ## Draft GitHub issue

   **Title**: [Clear, action-oriented title]

   **Description**:
   [2-3 sentence summary of the problem/goal]

   ## Key Details
   - [Bullet points of important details from thoughts]
   - [Technical decisions or constraints]
   - [Any specific requirements]

   ## Implementation Notes (if applicable)
   [Any specific technical approach or steps outlined]

   ## References
   - Source: `thoughts/[path/to/document.md]` ([View on GitHub](converted GitHub URL))
   - Related code: [any file:line references]
   - Parent issue: [if applicable]

   ---
   Based on the document, this seems to be at the stage of: [ideation/planning/ready to implement]
   ```

6. **Interactive refinement:**
   Ask the user:
   - Does this summary capture the issue accurately?
   - Which project should this go in? [show list]
   - What priority? (Default: Medium/3)
   - Any additional context to add?
   - Should we include more/less implementation detail?
   - Do you want to assign it to yourself?

   Note: issue will be created in "New" status by default.

7. **Create the GitHub issue:**
   ```
   mcp__GitHub__create_issue with:
   - title: [refined title]
   - description: [final description in markdown]
   - teamId: [selected team]
   - projectId: [use default project from above unless user specifies]
   - priority: [selected priority number, default 3]
   - stateId: [New status ID]
   - assigneeId: [if requested]
   - labelIds: [apply automatic label assignment from above]
   - links: [{url: "GitHub URL", title: "Document Title"}]
   ```

8. **Post-creation actions:**
   - Show the created issue URL
   - Ask if user wants to:
     - Add a comment with additional implementation details
     - Create sub-tasks for specific action items
     - Update the original thoughts document with the issue reference
   - If yes to updating thoughts doc:
     ```
     Add at the top of the document:
     ---
     GitHub_issue: [URL]
     created: [date]
     ---
     ```

## Example transformations:

### From verbose thoughts:
```
"I've been thinking about how our resumed sessions don't remember history properly.
This is causing issues where users have to re-specify everything. We should probably
store all the history in the user's home dir and then pull it when resuming. Maybe we need
a new file/spec for storing this"
```

### To concise issue:
```
Title: Fix resumed sessions to inherit all history from previous

Description:

## Problem to solve
Currently, resumed sessions only inherit Model and WorkingDir from parent sessions,
causing all other configuration to be lost. Users must re-specify permissions and
settings when resuming.

## Solution
Store all session configuration in the database and automatically inherit it when
resuming sessions, with support for explicit overrides.
```

### 2. Adding Comments and Links to Existing issues

When user wants to add a comment to an issue:

1. **Determine which issue:**
   - Use context from the current conversation to identify the relevant issue
   - If uncertain, use `mcp__GitHub__get_issue` to show issue details and confirm with user
   - Look for issue references in recent work discussed

2. **Format comments for clarity:**
   - Attempt to keep comments concise (~10 lines) unless more detail is needed
   - Focus on the key insight or most useful information for a human reader
   - Not just what was done, but what matters about it
   - Include relevant file references with backticks and GitHub links

3. **File reference formatting:**
   - Wrap paths in backticks: `thoughts/alice/example.md`
   - Add GitHub link after: `([View](url))`
   - Do this for both thoughts/ and code files mentioned

4. **Comment structure example:**
   ```markdown
   Implemented retry logic in webhook handler to address rate limit issues.

   Key insight: The 429 responses were clustered during batch operations,
   so exponential backoff alone wasn't sufficient - added request queuing.

   Files updated:
   - `cli/cmd/main.go` ([GitHub](link))
   - `thoughts/shared/rate_limit_analysis.md` ([GitHub](link))
   ```

5. **Handle links properly:**
   - If adding a link with a comment: Update the issue with the link AND mention it in the comment
   - If only adding a link: Still create a comment noting what link was added for posterity
   - Always add links to the issue itself using the `links` parameter

6. **For comments with links:**
   ```
   # First, update the issue with the link
   mcp__GitHub__update_issue with:
   - id: [issue ID]
   - links: [existing links + new link with proper title]

   # Then, create the comment mentioning the link
   mcp__GitHub__create_comment with:
   - issueId: [issue ID]
   - body: [formatted comment with key insights and file references]
   ```

7. **For links only:**
   ```
   # Update the issue with the link
   mcp__GitHub__update_issue with:
   - id: [issue ID]
   - links: [existing links + new link with proper title]

   # Add a brief comment for posterity
   mcp__GitHub__create_comment with:
   - issueId: [issue ID]
   - body: "Added link: `path/to/document.md` ([View](url))"
   ```

### 3. Searching for issues

When user wants to find issues:

1. **Gather search criteria:**
   - Query text
   - Team/Project filters
   - Status filters
   - Date ranges (createdAt, updatedAt)

2. **Execute search:**
   ```
   mcp__GitHub__list_issues with:
   - query: [search text]
   - teamId: [if specified]
   - projectId: [if specified]
   - stateId: [if filtering by status]
   - limit: 20
   ```

3. **Present results:**
   - Show issue ID, title, status, assignee
   - Group by project if multiple projects
   - Include direct links to GitHub

### 4. Updating issue Status

When moving issues through the workflow:

1. **Get current status:**
   - Fetch issue details
   - Show current status in workflow

2. **Suggest next status:**
   - New → Researching (starting research/investigation)
   - Researching → Planning (research complete, starting implementation plan)
   - Planning → Implementing (plan approved, starting development)
   - Implementing → Done (work completed)

3. **Update with context:**
   ```
   mcp__GitHub__update_issue with:
   - id: [issue ID]
   - stateId: [new status ID]
   ```

   Consider adding a comment explaining the status change.

## Important Notes

- Tag users in descriptions and comments using @username format, e.g., @alice
- Keep issues concise but complete - aim for scannable content
- All issues should include a clear "problem to solve" - if the user asks for an issue and only gives implementation details, you MUST ask "To write a good issue, please explain the problem you're trying to solve from a user perspective"
- Focus on the "what" and "why", include "how" only if well-defined
- Always preserve links to source material using the `links` parameter
- Don't create issues from early-stage brainstorming unless requested
- Use proper GitHub markdown formatting
- Include code references as: `path/to/file.ext:linenum`
- Ask for clarification rather than guessing project/status
- Remember that GitHub descriptions support full markdown including code blocks
- Always use the `links` parameter for external URLs (not just markdown links)
- remember - you must get a "Problem to solve"!

## Comment Quality Guidelines

When creating comments, focus on extracting the **most valuable information** for a human reader:

- **Key insights over summaries**: What's the "aha" moment or critical understanding?
- **Decisions and tradeoffs**: What approach was chosen and what it enables/prevents
- **Blockers resolved**: What was preventing progress and how it was addressed
- **State changes**: What's different now and what it means for next steps
- **Surprises or discoveries**: Unexpected findings that affect the work

Avoid:
- Mechanical lists of changes without context
- Restating what's obvious from code diffs
- Generic summaries that don't add value

Remember: The goal is to help a future reader (including yourself) quickly understand what matters about this update.
