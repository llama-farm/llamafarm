# Pull Request Workflow

## Always verify CI after pushing

After pushing commits to a PR, verify the state before declaring work done:

- `gh pr view <num> --json statusCheckRollup` — check for `FAILURE` / `CANCELLED`
- `gh api /repos/{owner}/{repo}/code-scanning/alerts?ref=refs/pull/<num>/head&state=open` — authoritative CodeQL state
- If checks are long-running, wait and re-check rather than stopping at "pushed"

`mergeStateStatus: BLOCKED` with `mergeable: MERGEABLE` and `reviewDecision: REVIEW_REQUIRED` means branch protection is waiting on a human reviewer, not a technical blocker.

## Triaging bot reviewer comments

cubic, qodo, github-advanced-security, and github-code-quality re-post the same findings on every new commit with updated line numbers. The comment list is full of duplicates; don't read it as the source of truth.

- **CodeQL**: query `code-scanning/alerts?state=open` via API. Alerts reflect current head; inline comments reflect historical state.
- **cubic / qodo**: filter `pulls/{num}/comments` by `commit_id` and only trust the latest commit's entries.
- **Verify "stale" empirically**: grep the current file for the fix before dismissing a comment — cubic re-flags already-fixed items.

When an alert is a known false positive with justification, dismiss via API rather than fighting the analyzer further:

```
gh api --method PATCH /repos/{owner}/{repo}/code-scanning/alerts/<num> \
  -f state=dismissed \
  -f dismissed_reason="false positive" \
  -f dismissed_comment="<detailed justification referencing the sanitization>"
```

## CodeQL py/path-injection: use the recognized pattern

CodeQL's `py/path-injection` dataflow requires the sanitizer and sink in the same basic block. It recognizes this shape:

```python
root_abs = os.path.abspath(env_var_or_param)
candidate = os.path.realpath(os.path.join(root_abs, safe_name))
if (os.path.commonpath([root_abs, candidate]) == root_abs
        and os.path.isfile(candidate)):
    # use `candidate` here — CodeQL sees the guard as protecting the use
    ...
```

Patterns that **don't** satisfy the analyzer despite being semantically safe:

- Sanitization in a helper function — parameters become fresh taint sources on return
- `candidate.startswith(root + os.sep)` — functionally equivalent but not in the recognized set
- A single `commonpath` check at the top of a function, then multiple downstream uses — the sanitizer doesn't flow across basic blocks
- `pathlib.Path` wrapping — `Path` methods re-introduce taint after a string-level check

If the above in-place pattern isn't achievable, either (a) restructure so every filesystem operation is in a compound conditional with a fresh `commonpath` check, or (b) dismiss via API with justification. See `common/llamafarm_common/model_utils.py::get_gguf_file_path`'s `GGUF_MODELS_DIR` lookup for a working example.

## Pre-existing bugs can surface on merge

When merging `main` into a feature branch, CI failures that look new may actually be pre-existing bugs that were dormant:

- Concurrent-write races (e.g. `os.replace` on Windows) only fire under specific timing
- Unicode encoding crashes only fire when log output happens to contain non-latin1 bytes (llama.cpp emits `Ġ` / `Ċ` byte-BPE markers)
- Flaky tests pass on lucky runs and fail on others

Before assuming a failure is caused by your PR, check:

1. Did the same test pass on the last green `main` CI run?
2. Does your PR touch any file in the failing code path (`git diff --stat main..HEAD <path>`)?

If the failure is pre-existing but your merge triggers it, fix it in your PR anyway — it's blocking the merge — and note the root cause in the commit message so reviewers know it's not net-new from your work.

## Windows runtime considerations

Python runtimes that shell out to llama.cpp or process user-generated text on Windows should force UTF-8 on `stdout`/`stderr` at startup:

```python
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        # Pre-3.7 or stream replaced by pytest capture / PyApp wrapper.
        pass
```

Windows' default `cp1252` console codec crashes on characters like `\u0120` (Ġ) and `\u010a` (Ċ) that llama.cpp's native logger emits during tokenizer loading. This must run **before** any logger is configured.
