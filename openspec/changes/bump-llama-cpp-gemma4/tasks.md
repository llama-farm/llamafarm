## 1. Resolve target tag

- [x] 1.1 Listed recent llama.cpp tags via `git ls-remote --tags`. Latest tag is `b8708`.
- [x] 1.2 Verified `d9a12c82` (Gemma 4 EOG fix) is on master but not yet in any tag — `b8708` was cut ~4.5h before `d9a12c82` landed. Per user direction ("let's go with the latest"), proceed with `b8708` and capture the missing EOG fix in the planned follow-up bump.
- [x] 1.3 Recorded `b8708` and the d9a12c82-deferred rationale in PR #809 description

## 2. Bump version constants

- [x] 2.1 Updated `llama-cpp-version.txt` to `b8708`
- [x] 2.2 Updated the hardcoded fallback in `packages/llamafarm-llama/src/llamafarm_llama/_binary.py` to `b8708`
- [x] 2.3 Updated `cli/internal/llamabinary/llamabinary.go` `var Version` to `b8708`
- [x] 2.4 Updated 16 fixture strings in `llamabinary_test.go` (replace_all) and 1 in `download_test.go`
- [x] 2.5 Grepped remaining references: only historical comments in `_binary.py` (about CUDA 11 dropping at b7694+) and example fixture strings in `common/tests/test_offline_mode.py` (not tied to the pin); both intentionally left

## 3. Replace deprecated llama.cpp APIs

- [x] 3.1 Renamed cdef `llama_load_model_from_file` → `llama_model_load_from_file` in `_bindings.py` (and a docstring reference)
- [x] 3.2 Renamed cdef `llama_new_context_with_model` → `llama_init_from_model` in `_bindings.py`
- [x] 3.3 Renamed cdef `llama_free_model` → `llama_model_free` in `_bindings.py`
- [x] 3.4 Updated call site at `llama.py:177` to `llama_model_load_from_file`
- [x] 3.5 Updated call site at `llama.py:241` to `llama_init_from_model`
- [x] 3.6 Updated 2 call sites for `llama_free_model` in `llama.py` (replace_all)
- [x] 3.7 Also updated `tests/test_llama.py` mocks (which referenced the old names and would have broken). Final grep confirms no source-code references remain — only openspec change artifacts mention the old names

## 4. Update documentation

- [x] 4.1 Created `.claude/rules/llama_cpp_bindings.md` covering rationale, version pinning, propagation locations, header-diff procedure, smoke-test procedure, ARM64 build dependency, and expected bump cadence during upstream churn
- [x] 4.2 CLAUDE.md already has umbrella `See .claude/rules/` reference (no per-file links needed). Also fixed a stale tech-stack line in CLAUDE.md that listed the Universal Runtime as using `llama-cpp-python` — corrected to `llamafarm-llama`

## 5. Smoke-test ARM64 binary build (release attachment is automatic on next v* tag)

- [x] 5.1 Triggered `gh workflow run build-llama.yml --ref chore-bump-llama-cpp-b8708-gemma4 -f llama_version=b8708` — run id 24148418292 (https://github.com/llama-farm/llamafarm/actions/runs/24148418292)
- [x] 5.2 Run completed successfully in ~4 minutes. Artifact `llama-b8708-bin-linux-arm64.zip` (6.4 MB) uploaded to the workflow run as artifact ID 6333000647.
- [x] 5.3 N/A — `build-llama.yml`'s `Release` step is gated on `if: startsWith(github.ref, 'refs/tags/')`, so a `workflow_dispatch` run does not attach to a release. The artifact will be attached to the next LlamaFarm release (v0.0.30+) automatically when the v* tag is pushed and the workflow re-runs at that tag against the bumped version pin. Documented this behavior in `.claude/rules/llama_cpp_bindings.md`.
- [x] 5.4 Build succeeded — no investigation needed

## 6. Run tests

- [x] 6.1 Ran `uv run pytest tests/` in `packages/llamafarm-llama` — 54/54 pass on macOS arm64. Also fixed a pre-existing broken test (`test_download_binary_uses_prebuilt_on_linux_arm64`) that was deselected from CI since before b7694 — stale `bin/libllama.so` expectation in the manifest assertion. One-line fix.
- [x] 6.2 Ran `go test ./internal/llamabinary/...` in `cli/` — pass
- [ ] 6.3 Confirm cross-platform CI (Linux, macOS, Windows) passes after pushing — wait for the full matrix, do not declare done at "pushed"

## 7. Smoke-test inference

- [ ] 7.1 Download a small Gemma 4 GGUF model (smallest variant that exercises the tokenizer and EOG fixes is sufficient)
- [ ] 7.2 Start the universal runtime locally and load the Gemma 4 model
- [ ] 7.3 Issue a chat completion or text generation request and verify the output is coherent (not garbage tokens, not a tokenizer error, not a crash)
- [ ] 7.4 Repeat 7.1–7.3 with a non-Gemma model already known to work (e.g. a small Llama 3 or Qwen GGUF) as a regression check
- [ ] 7.5 Document the exact model variants tested in the PR description

## 8. Verify and ship

- [x] 8.1 `openspec validate bump-llama-cpp-gemma4` — passes
- [x] 8.2 Opened PR #809 https://github.com/llama-farm/llamafarm/pull/809 with chosen tag, justification, ARM64 build run URL, and follow-up bump cadence note. Smoke-test results to be added in a follow-up comment after manual testing.
- [ ] 8.3 After merge, verify CI on `main` is green (per `.claude/rules/pr_workflow.md` — do not stop at "pushed")
- [ ] 8.4 Schedule (or note in a tracking issue) the next bump for ~1–2 weeks out, as a Gemma 4 stabilization sweep
