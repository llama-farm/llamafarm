---
description: Create a detailed implementation plan and save to PLAN.md
---

Create a detailed, step-by-step implementation plan for the following task:

$ARGUMENTS

## Instructions

1. **Read the relevant agent files first** (see `.claude/CLAUDE.md` for which files)
2. **Analyze the codebase** to understand current architecture and patterns
3. **Create a plan** with clear phases, each with its own tests and demos
4. **Save the plan** to `PLAN.md` in the project root (NOT in .claude/)
5. **Wait for approval** before executing

## CRITICAL: Test-First Phase Structure

**EVERY phase MUST follow this order:**
1. **Tests FIRST** - Define what success looks like before coding
2. **Demo FIRST** - Define the demo script that will prove it works
3. **Implementation** - Build to make the tests pass
4. **Verification** - Run tests and demo, confirm everything passes
5. **Checkpoint** - Mark phase complete only after verification

**This is Test-Driven Development (TDD):**
- Write the tests and demo FIRST (they will fail initially)
- Implement until tests pass
- Tests CAN be modified if requirements change, but this is the EXCEPTION, not the rule
- If you must modify tests, document WHY in the plan

**DO NOT move to the next phase until:**
- All tests for the current phase pass
- The demo runs successfully
- The checkpoint is explicitly marked complete

## Plan Format

Use this EXACT format for the plan:

```markdown
# Plan: [Task Title]

## Overview
[Brief description of what will be built]

## Agents to Use
[List which specialized agents will be invoked automatically during execution]
- **llamafarm** - For classifier training, anomaly detection, RAG setup
- **database-architect** - For DuckDB schema design, vector storage
- **backend-architect** - For FastAPI/Fastify API implementation
- **ui-architect** - For React/Next.js frontend components
- **test-runner** - After each phase to run and verify tests
- **debugger** - If any tests fail, to fix issues
- **security-auditor** - Before final checkpoint for security review
- **demo-builder** - To create phase demos
- **code-reviewer** - After significant implementations

## LlamaFarm API Usage (if applicable)
[List EVERY LlamaFarm API that will be called, with endpoint and purpose]
- `POST /v1/classifier/fit` - Train text classifier
- `POST /v1/classifier/predict` - Classify texts
- `POST /v1/classifier/save` - Persist classifier
- `POST /v1/anomaly/fit` - Train anomaly detector
- `POST /v1/anomaly/detect` - Detect anomalies
- `POST /v1/anomaly/save` - Persist model
- etc.

## Phase 1: [Phase Name]

### Phase 1 Tests (Define FIRST)
- [ ] Test: [specific test - what should pass when phase is complete]
- [ ] Test: [specific test - edge cases, error handling]
- [ ] Test file: `tests/test_phase1_feature.py` or similar

### Phase 1 Demo (Define FIRST)
- [ ] Demo script: `demos/demo-phase1.sh` or similar
- [ ] Demo shows: [what the demo will demonstrate]
- [ ] Expected output: [what success looks like]

### Phase 1 Implementation
- [ ] Step 1 description
- [ ] Step 2 description

### Phase 1 Verification
- [ ] Run tests: `pytest tests/test_phase1_*.py -v`
- [ ] All tests pass
- [ ] Run demo: `bash demos/demo-phase1.sh`
- [ ] Demo runs successfully

### Phase 1 Checkpoint
- [ ] Tests verified passing
- [ ] Demo verified working
- [ ] Ready for Phase 2

## Phase 2: [Phase Name]

### Phase 2 Tests (Define FIRST)
- [ ] Test: [specific test description]
- [ ] Test file: `tests/test_phase2_feature.py`

### Phase 2 Demo (Define FIRST)
- [ ] Demo script: `demos/demo-phase2.sh`
- [ ] Demo shows: [what the demo will demonstrate]

### Phase 2 Implementation
- [ ] Step 3 description

### Phase 2 Verification
- [ ] Run tests: all Phase 2 tests pass
- [ ] Run demo: demo runs successfully

### Phase 2 Checkpoint
- [ ] Tests verified passing
- [ ] Demo verified working
- [ ] Ready for Phase 3

[Continue pattern for all phases...]

## Final Success Criteria
- [ ] All phase checkpoints complete
- [ ] Full integration test passes
- [ ] End-to-end demo runs for full duration
```

## ML Requirements

When the plan involves machine learning:
- **Anomaly Detection**: Use One-Class SVM (preferred over Isolation Forest)
- **Training Data**: Minimum 200 examples for any classifier or anomaly model
- **Classification**: Document all labels and example counts per label

## Python Backend Requirements

When the plan involves Python:
- **Always use UV** for package management (`uv init`, `uv add`, `uv run`)
- **Never use pip directly**
- Reference `.claude/agents/backend-architect.md` for patterns

## Execution Rules

After creating the plan:
1. Present it to the user for review
2. Discuss and refine as needed
3. Wait for explicit approval before executing
4. Once approved, run: `python3 .claude/hooks/plan-tracker.py approve`

**During execution (Test-First Workflow):**

For EACH phase:
1. **Write tests first** - Create the test file(s) defined in the plan
2. **Write demo first** - Create the demo script defined in the plan
3. **Run tests** - They should FAIL initially (this confirms they're testing something real)
4. **Implement** - Build the feature to make tests pass
5. **Run tests again** - They should now PASS
6. **Run demo** - Verify it works end-to-end
7. **Mark checkpoint complete** - Only after verification

**If tests need modification:**
- This should be RARE - tests define the contract
- Document WHY the test changed (requirements changed, test was wrong, etc.)
- Get user approval if the change is significant

Use `python3 .claude/hooks/plan-tracker.py complete "step description"` after each step.

Do NOT start implementing until the user says "approved", "execute", "go ahead", or similar.
