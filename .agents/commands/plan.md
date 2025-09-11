## Implementation Planning Playbook

Turn your completed research into a minimal, verifiable change plan. This document helps you define scope, sequence work, and specify how you’ll prove success before writing code.

### Inputs
- The research note in `thoughts/shared/research/<issue_id>_<slug>.md`
- Clarifications from the user for any uncertainties identified in research
- Awareness of repo guidelines in `AGENTS.md` and project conventions

### Outputs
- A plan document at `thoughts/shared/plans/<issue_id>_<slug>-plan.md` that includes:
  - Clear scope and acceptance criteria
  - File-level change list and impact analysis
  - Test and verification plan (commands you’ll run)
  - Risk/rollout/rollback strategy
  - Docs and config update plan
  - Estimated effort and sequencing

## Planning Workflow
1. Confirm problem and success criteria from the research note.
2. Identify impacted projects and targets (prefer workspace graph/tools when available).
3. Map entry points and contracts to be changed; list interfaces/CLIs/APIs affected.
4. Draft the smallest viable change list; postpone refactors.
5. Define the test strategy and exact commands you will run to verify.
6. Specify risk controls: flags, canaries, and rollback steps.
7. List required updates to docs and configuration.
8. Sequence tasks and add rough estimates; call out external dependencies.
9. Share the plan; adjust based on feedback before coding.

## Change List: plan by files and boundaries
For each change, capture:
- **File(s)/Module(s)**: paths to create/edit/remove
- **Change summary**: what will be added/modified and why
- **Contracts**: CLI flags, API schemas, function signatures, data shapes
- **Dependencies**: services, tasks/targets, feature flags, migrations
- **Blast radius**: who calls this code; risks and mitigations

Example structure:
```markdown
- server/api/items.py: add GET /items/{id} handler returning X schema
- server/services/items.py: implement fetch_item with validation Y
- cli/cmd/items.go: add `lf items get` to call server endpoint
- docs/website/...: add usage docs and examples
```

## Test and Verification Plan
Define how you will prove the change works before and after coding.
- **Unit tests**: files to add/modify; what behaviors are asserted
- **Integration/e2e**: endpoints or CLI flows exercised
- **Commands to run**:
```bash
# Server
cd server && uv run pytest -q
cd server && uv sync && uv run uvicorn server.main:app --reload

# CLI
cd cli && go test ./...
cd cli && go build -o lf && ./lf --help

# Docs
nx build docs
```
- **Success checks**: specific outputs/logs/HTTP codes expected
- **Coverage or instrumentation goals**: if applicable

## Risk, Rollout, and Rollback
- **Risk assessment**: user-facing, data, performance, or compatibility risks
- **Rollout**: behind a flag, staged release, or immediate
- **Rollback**: how to revert safely; data migration backout if applicable

## Observability, Logging, and Ops
- Note any logs/metrics/traces to add
- Ensure logging and operational conventions from `AGENTS.md` are followed

## Docs and Configuration Updates
- Enumerate docs pages or READMEs to update
- Configuration changes, defaults, and `.env.example` updates

## Estimation and Sequencing
- Break the plan into 2–6 tasks with rough estimates
- Order tasks to deliver value early and reduce risk

## Anti-patterns to Avoid
- Planning large refactors alongside the feature/fix
- Vague acceptance criteria (“works as expected”)
- No test plan or unverifiable steps
- No rollback or ops considerations
- Ignoring cross-project impacts

## Plan Deliverable Template
Create `thoughts/shared/plans/<issue_id>_<slug>-plan.md` using this template:
```markdown
# <Issue Title>

## Context and Goals
- Summary of the problem
- Acceptance criteria (bullet list)

## Scope
- In scope
- Out of scope

## Change List
- <path>: <concise change summary>
- <path>: <concise change summary>

## Impact and Contracts
- Affected APIs/CLIs/schemas and compatibility notes

## Test and Verification Plan
```bash
<commands to run>
```
- Unit tests to add/modify
- Integration/e2e flows
- Expected outputs/HTTP codes

## Risk, Rollout, Rollback
- Risks and mitigations
- Rollout strategy
- Rollback steps

## Observability and Ops
- Logs/metrics/traces to add

## Docs and Config Updates
- Files/pages to update

## Task Breakdown and Estimates
- Task 1 — <description> (~Xh)
- Task 2 — <description> (~Xh)

## Open Questions
- List unknowns and how to resolve them
```

---
Keep plans minimal and measurable. If anything is uncertain after research, ask the user before finalizing the plan. Then implement according to the plan and verify using the specified commands.
