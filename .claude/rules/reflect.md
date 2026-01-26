# Automatic Reflection

## When to Invoke /reflect Proactively

Invoke the `/reflect` skill when you observe:

1. **User corrections after skill usage** - If the user says "no", "not like that", or asks for changes immediately after you used a skill (like /commit, /code-review, /frontend-design), suggest or invoke reflection

2. **Skill-heavy sessions** - If you've invoked 2+ skills in a session and the session is winding down, offer to reflect

3. **Clear preference patterns** - If the user has shown repeated preferences that should be captured in a skill

## How to Check Auto-Reflect Status

Before invoking, check `~/.claude/reflect-skill-state.json`:
- If `{"enabled": true}` - Invoke `/reflect` automatically
- If `{"enabled": false}` or missing - Ask user: "Would you like me to capture these learnings with /reflect?"

## Don't Over-Reflect

- Don't suggest reflection for every minor correction
- Focus on patterns that would improve future skill outputs
- One reflection per session is usually enough
