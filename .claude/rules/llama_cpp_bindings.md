# llama.cpp Bindings

LlamaFarm vendors its own Python bindings to llama.cpp via `packages/llamafarm-llama`. This file captures why, where the version is pinned, and how to bump it without surprises.

## Why we own the bindings

We previously used `llama-cpp-python`. We moved away from it because:

1. **Maintenance gaps.** The project had periods of slow maintenance that delayed bug fixes we needed.
2. **Missing wheels.** It did not publish wheels for all platforms LlamaFarm supports, **Linux ARM64 in particular**. Building from source on user machines is not acceptable for our install story.
3. **Release cadence.** Even when wheels existed, they shipped behind upstream by enough that we could not chase fast-moving model support.

Owning `llamafarm-llama` gives us:

- Direct control over which llama.cpp version we ship
- A LlamaFarm-published Linux ARM64 binary via our own release pipeline (`build-llama.yml`)
- Freedom from upstream wrapper-package release cadence

The trade-off: every llama.cpp version bump is our responsibility. We have to track upstream API changes, struct layout changes, and removed/deprecated functions ourselves. This file documents the discipline that makes that sustainable.

## Where the version is pinned

The single source of truth is **`llama-cpp-version.txt`** at the repo root. It contains a single line: the upstream tag name (e.g. `b8708`).

The version propagates to four other locations as fallback constants. These MUST be kept in sync with `llama-cpp-version.txt` in the same commit as any bump:

| File | What to update |
|---|---|
| `packages/llamafarm-llama/src/llamafarm_llama/_binary.py` | Hardcoded fallback in `_read_llama_cpp_version()` (used when the version file is missing, e.g. installed package) |
| `cli/internal/llamabinary/llamabinary.go` | `var Version = "..."` package variable |
| `cli/internal/llamabinary/llamabinary_test.go` | Test fixture strings (multiple occurrences — use `replace_all`) |
| `cli/internal/llamabinary/download_test.go` | Test fixture string |

To find any drift, grep:

```
rg 'b\d{4,5}' --type go --type py llama-cpp-version.txt
```

The CI workflow `.github/workflows/build-llama.yml` reads from `llama-cpp-version.txt` directly and does not need a manual update.

## Bundled binaries

Release packaging now stages the pinned `llama.cpp` binary into two deterministic bundle layouts:

- CLI archives: `llama-cpp/<os>-<arch>/<binary-name>`
- Python/PyApp bundles: `packages/llamafarm-llama/src/llamafarm_llama/_bundled/<os>-<arch>/<binary-name>`

When the pin changes, verify the staging workflows still fetch the matching upstream artifacts for:

- `darwin-arm64`
- `darwin-x86_64`
- `linux-x86_64`
- `linux-arm64` via LlamaFarm's own `build-llama.yml` release asset
- `windows-x86_64`

CPU-only bundles ship first. GPU-specific bundled variants remain a follow-up.

## How to bump

### 1. Pick a target tag

```
git ls-remote --tags https://github.com/ggml-org/llama.cpp.git \
  | grep -oE 'refs/tags/b[0-9]{4,5}$' | sort -V | tail -10
```

Pick the latest tag, OR — if you are bumping for a specific bug fix or model — confirm the relevant commit is reachable from the tag you pick.

### 2. Run the header-diff procedure

This is the **most important** step. Clone upstream into a temp directory and diff the public headers between the current pin and the target:

```
git clone --filter=blob:none --no-checkout https://github.com/ggml-org/llama.cpp.git /tmp/llama-diff
cd /tmp/llama-diff
git fetch --depth 1 origin tag <old>
git fetch --depth 1 origin tag <new>
git diff <old>..<new> -- include/llama.h ggml/include/ggml.h ggml/include/ggml-backend.h ggml/include/ggml-cpu.h ggml/include/gguf.h
```

Walk through the diff looking for **struct layout changes** that affect any type declared in `packages/llamafarm-llama/src/llamafarm_llama/_bindings.py`:

- `llama_model_params`
- `llama_context_params`
- `llama_batch`
- (and any others currently in the cffi cdef block)

Field additions, removals, reorderings, or type changes in these structs are ABI-breaking and require corresponding edits in `_bindings.py`. Field additions in structs we **don't** bind are safe to ignore.

Also look for:

- **Removed functions** we call (rare; check `llama_*` and `ggml_backend_*` symbols referenced in `_bindings.py` and `llama.py`)
- **Deprecated functions** we call (usually safe to leave for one bump; replace at next opportunity, see below)

If the diff is large, you can scope the search by extracting just the deprecation markers from the new header and intersecting with our usage:

```
git show <new>:include/llama.h \
  | awk '/DEPRECATED\(/{flag=1} flag{print; if(/instead/){flag=0; print "---"}}'
```

### 3. Update version constants

Update `llama-cpp-version.txt` and the four propagation locations listed above.

### 4. Replace deprecated APIs (if any)

For any deprecated function we call where the replacement has an **identical signature** (a pure rename), update the cffi declaration in `_bindings.py` and the call sites in `llama.py` in the same commit. Pure renames are cheap.

For deprecations with **non-trivial migration** (signature changes, semantic differences), leave the deprecated call in place and open a follow-up issue. Don't block a version bump on a refactor.

### 5. Smoke-test the ARM64 build

Upstream does **not** ship Linux ARM64 binaries. We build them ourselves via `build-llama.yml`. Trigger it manually against the new tag as a **build smoke test**:

```
gh workflow run build-llama.yml -f llama_version=<tag>
gh run watch
```

This validates that llama.cpp at the new tag actually compiles for ARM64 with our cmake flags. If the build fails, the bump is blocked until either upstream is fixed or we patch the workflow.

**Important:** the `Release` step in `build-llama.yml` is gated on `if: startsWith(github.ref, 'refs/tags/')`, so a `workflow_dispatch` run does **not** attach the artifact to any GitHub release — it only uploads to the workflow run as a downloadable artifact. The actual release attachment happens automatically when the next LlamaFarm release tag (`v*`) is pushed: the workflow re-runs against the version pin in `llama-cpp-version.txt` at that tag and attaches the built artifact to the release. This means:

- The pre-merge `workflow_dispatch` run is a **smoke test only**, not a release publication step
- Linux ARM64 users will not be able to download the new binary until the next LlamaFarm release is cut
- That's normally fine because the LlamaFarm release tag and the pin bump ship together — users on the new LlamaFarm version automatically get the matching binary

If you ever need to make a binary at a new pin available to existing-release users (e.g. backporting a fix to a release that already shipped), you have two options:
1. Cut a new LlamaFarm release at the bumped version (cleanest)
2. Manually upload the workflow artifact to an existing release with `gh release upload <tag> <artifact>` (polluting; only do this for hotfixes)

### 6. Smoke-test inference

The unit tests in `packages/llamafarm-llama/tests/` mock the cffi layer — they catch wiring errors but **not** runtime correctness regressions. You must additionally:

1. Load at least one GGUF model end-to-end through the universal runtime
2. Generate output and verify it is coherent (not garbage tokens, not a crash, not a tokenizer error)
3. **If the bump was motivated by a specific model architecture**, smoke-test that architecture specifically — passing tests on previously-supported models does not prove the new architecture works

### 7. Cross-platform CI

After pushing, wait for the full Linux/macOS/Windows matrix to pass before declaring done. Per `.claude/rules/pr_workflow.md`, "pushed" is not "done."

## Bumping during periods of upstream churn

When upstream is actively patching support for a hot model architecture (Gemma 4 in April 2026 is the canonical example), you may find that **no soaked tag exists with all current fixes**. The fixes are landing live; any tag old enough to be soaked is missing fixes; any tag with all current fixes has zero soak time.

In this situation:

- **Take the latest tag.** Soaking is impossible.
- **Plan for follow-up bumps every 1–2 weeks** until upstream churn settles. Schedule them in advance — don't wait for someone to notice the next round of fixes.
- **Front-load the binding-layer work** (deprecated API renames, this rules doc) into the first bump so subsequent bumps are pure version-string updates with no code changes.
- **Note in the PR** that this is part of an expected bump sequence so reviewers understand the cadence.

## What NOT to bundle into a version bump

Resist the temptation to:

- Refactor the binding layer "while you're here"
- Adopt new upstream APIs (e.g. `llama_init_from_user`, `llama_set_adapters_lora`) that we don't currently need
- Fix unrelated bugs in `llamafarm-llama`
- Replace deprecated APIs that require non-trivial migration (do it as a separate change)
- Update other dependencies

A version bump should be reviewable in five minutes. Anything bigger is its own change.

## Reference: file inventory

- `llama-cpp-version.txt` — single source of truth
- `packages/llamafarm-llama/src/llamafarm_llama/_binary.py` — binary download, cache management, fallback version constant
- `packages/llamafarm-llama/src/llamafarm_llama/_bindings.py` — cffi cdef block mirroring upstream `llama.h`
- `packages/llamafarm-llama/src/llamafarm_llama/llama.py` — high-level Python `Llama` class that calls the cffi layer
- `packages/llamafarm-llama/tests/test_llama.py` — mocks the cffi layer, must reference the current API names
- `cli/internal/llamabinary/llamabinary.go` — Go-side binary download/cache for cross-platform deployment, holds `Version` constant
- `cli/internal/llamabinary/llamabinary_test.go` and `download_test.go` — Go test fixtures that hardcode the version
- `.github/workflows/build-llama.yml` — builds the LlamaFarm Linux ARM64 binary, reads `llama-cpp-version.txt` automatically
