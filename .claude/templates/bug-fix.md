# Bug Fix Template

Follow these steps to fix a bug:

## Phase 1: Reproduce
1. Understand the bug report
2. Reproduce the issue locally
3. Capture the exact error message/behavior
4. Identify the conditions that trigger it

## Phase 2: Diagnose
1. Trace the error to its source
2. Understand the root cause (not just symptoms)
3. Check if it affects other areas
4. Document the diagnosis

## Phase 3: Write Failing Test
1. Write a test that reproduces the bug
2. The test should FAIL before the fix
3. The test should PASS after the fix
4. Include edge cases if applicable

## Phase 4: Fix
1. Implement the MINIMAL fix
2. Don't refactor unrelated code
3. Don't add features while fixing
4. Keep the change focused

## Phase 5: Verify
1. Run the new test - should PASS
2. Run ALL tests - ensure no regressions
3. Manually verify the fix works
4. Check related functionality

## Phase 6: Cleanup
1. Run linters
2. Remove any debug code
3. Update comments if needed

## Phase 7: Commit
1. Create descriptive commit message
2. Reference issue number if applicable
3. NO Claude attribution
4. Message format: `fix(scope): description`

## Remember
- Reproduce FIRST, fix SECOND
- Write a test that catches the bug
- Minimal, focused changes only
- Verify no regressions
