## Context

LlamaFarm vendors its own Python bindings to llama.cpp via the `packages/llamafarm-llama` package. The pinned version (`b7694`, tagged 2026-01-10) predates upstream Gemma 4 support, which began landing 2026-04-02 and is still being patched as of 2026-04-08 (most recent: commit `d9a12c82`, which lands ~4.5 hours after the latest tag `b8708`).

Three constraints shape this work:

1. **No "stable" Gemma 4 tag exists.** The fixes are landing live. Any tag old enough to be "soaked" is missing fixes; any tag with all current fixes has zero soak time. We have no choice but to track upstream HEAD until the churn settles.
2. **We own the bindings ourselves.** A previous attempt to use `llama-cpp-python` was abandoned because the project had maintenance issues and did not publish wheels for all platforms LlamaFarm supports — Linux ARM64 in particular. Owning the bindings means each upstream version bump is our responsibility, but it also means we can ship Linux ARM64 binaries via our own release pipeline (`build-llama.yml`).
3. **The binding diff for this jump is unusually small.** Header analysis between `b7694` and `b8708` shows zero struct layout changes affecting our cffi declarations. The risk surface is concentrated in runtime behavior (model loading, inference correctness), not the binding layer.

The project has no OpenSpec capability covering the llama.cpp binary subsystem today. This change introduces one (`llama-cpp-binary`) so that the maintenance discipline we are committing to is captured in testable form.

## Goals / Non-Goals

**Goals:**

- Bump the pinned llama.cpp version to `b8708`, the latest upstream tag at implementation time. This includes 7 of the 8 Gemma 4 commits that had landed upstream as of 2026-04-08; the trailing EOG token fix (`d9a12c82`) is deferred to the planned follow-up bump.
- Replace three deprecated llama.cpp APIs with their renamed equivalents (pure rename, no behavior change)
- Publish a Linux ARM64 binary at the new tag via `build-llama.yml` before merge
- Document the binding-ownership rationale and upgrade procedure in `.claude/rules/llama_cpp_bindings.md` so this conversation does not happen again on the next bump
- Establish the `llama-cpp-binary` capability spec with testable invariants

**Non-Goals:**

- Adopting any of the new upstream APIs introduced between `b7694` and target (`init_from_user`, `load_from_file_ptr`, `sampler_init_adaptive_p`, `set_adapters_lora`)
- Replacing the 32 other deprecated llama.cpp APIs upstream marked but that we do not call (token/vocab/session helpers)
- Switching away from `llamafarm-llama` to `llama-cpp-python` or building a native inference engine in Python or Go
- Refactoring the binding layer "while we're here"
- Updating any non-llama.cpp dependencies
- Making the upgrade process automatic (e.g. a Renovate rule). The next 2–3 bumps will be manual on purpose, until churn settles.

## Decisions

### Decision 1: Take the latest upstream tag, not a soaked one

**Choice:** `b8708` — the latest upstream tag at implementation time. It contains 7 of the 8 Gemma 4 commits as of 2026-04-08; the trailing EOG token fix (`d9a12c82`) was cut ~4.5 hours after `b8708` and is deferred to the next bump.

The original framing of this decision required the chosen tag to include `d9a12c82`. That floor was relaxed at implementation time because (a) no tag yet contains that commit, (b) the EOG fix is an end-of-generation edge case, not a "Gemma 4 doesn't work" blocker, and (c) Decision 2 plans for follow-up bumps that will pick it up. The principle — "take the latest, not a soaked one" — is unchanged.

**Alternatives considered:**
- *Take an older "stable" tag.* Rejected because no such tag exists with full Gemma 4 support — the fixes are landing live.
- *Pin to a specific commit hash instead of a tag.* Rejected because `build-llama.yml` and `_binary.py` both expect upstream-tagged release artifacts (the URL format embeds the tag). Custom commit pins would require building from source on every platform.
- *Wait until churn settles.* Rejected because the user needs Gemma 4 working now, and "settled" is undefined for an actively maintained model architecture.

**Rationale:** The header diff confirms low binding risk for any tag in the b76xx–b87xx range. The real risk is runtime correctness for Gemma 4 specifically, and the only way to mitigate that is to test against a real Gemma 4 model. Older tags do not reduce that risk; they only delay it.

### Decision 2: Plan for a 2–3 bump sequence over ~2 weeks

**Choice:** Treat this as the first of an expected 2–3 bumps. Land the deprecated-API rename and the rules documentation in this change so subsequent bumps are pure version-string updates.

**Alternatives considered:**
- *One bump and done.* Rejected because Gemma 4 fixes are still landing daily upstream. Any single tag we pick will be missing fixes within days.
- *Wait until churn settles, then do one bump.* Same rejection as Decision 1.

**Rationale:** Front-loading the binding-layer work (rename, docs) makes follow-up bumps cheap. The marginal cost of bump #2 should be: edit `llama-cpp-version.txt`, edit four propagation locations, run tests. No design work required.

### Decision 3: Replace deprecated APIs with renamed equivalents now

**Choice:** Update `_bindings.py` and `llama.py` in this change to use:
- `llama_model_load_from_file` (was `llama_load_model_from_file`)
- `llama_init_from_model` (was `llama_new_context_with_model`)
- `llama_model_free` (was `llama_free_model`)

**Alternatives considered:**
- *Leave the deprecated calls in place.* Upstream still supports them in `b8708`, so technically we could defer. Rejected because they will eventually be removed, and the cost of fixing them now is ~6 line changes with identical signatures.
- *Replace all 35 deprecated APIs upstream marks, not just the 3 we use.* Rejected because we do not bind the other 32 — they are token/vocab/session helpers we never wrapped. Deleting unbound declarations is not in scope.

**Rationale:** Pure renames are the cheapest possible fix. Bundling them into a version bump (rather than a separate change) avoids two CI cycles and two reviews for what is conceptually one piece of housekeeping.

### Decision 4: Introduce `llama-cpp-binary` capability spec

**Choice:** Create a new OpenSpec capability spec that captures testable invariants about llama.cpp binary management — version pinning, cache key isolation, supported architectures, and upgrade procedure — even though this change does not modify those behaviors except to add Gemma 4 to the supported list and bump the version.

**Alternatives considered:**
- *Create no spec, mark this as a pure dependency bump.* Rejected because the OpenSpec workflow expects a capability artifact, and more importantly: we are committing to a maintenance discipline (track upstream, validate via header diff, run ARM64 build before merge) that has no codified contract today. Future bumps will benefit from having something to point at.
- *Stuff the rules into `.claude/rules/llama_cpp_bindings.md` only and skip the capability spec.* Rejected because rules are guidance for AI assistants, not testable system behavior. The capability spec captures invariants that can be linted (e.g. "the version constants in 4 files match `llama-cpp-version.txt`").

**Rationale:** The spec is small but real. It covers behavior that exists today, has just never been written down. This change is the natural moment to capture it because we are extending the supported-architecture list anyway.

### Decision 5: Smoke-test ARM64 build before merge (release attachment is automatic)

**Choice:** Manually invoke `build-llama.yml` via `workflow_dispatch` against the new tag as part of the change implementation, **as a build smoke test only**.

**Important context discovered during implementation:** The `Release` step in `build-llama.yml` is gated on `if: startsWith(github.ref, 'refs/tags/')`. A `workflow_dispatch` run does NOT attach the artifact to any release — only a `v*` tag push does. So the manual run validates that the build works at the new pin; the actual release-asset publication happens automatically when the next LlamaFarm release tag is cut.

This means:
- Linux ARM64 users will not see the new binary until the next LlamaFarm release ships
- That's fine in the normal flow because the LlamaFarm release and the pin bump ship together — users on the new version automatically get the matching binary
- The pre-merge smoke test catches build failures (e.g. upstream introduced an incompatible cmake flag) before they become a release blocker

**Alternatives considered:**
- *Let the next release tag trigger it without a pre-merge smoke test.* Rejected because if the build fails at the new tag, we want to know before merging — fixing it post-merge is more painful, and "ship a release that breaks ARM64" would be embarrassing.
- *Make the ARM64 build a hard CI gate on the PR.* Rejected because the workflow is `workflow_dispatch` only and runs on a non-PR runner. Adding it to PR CI is a separate, larger change.
- *Modify `build-llama.yml` to also attach to a release on `workflow_dispatch`.* Rejected because there is no clear release to attach it to before the LlamaFarm version bump that includes the new pin actually ships. The current decoupling is correct.

**Rationale:** The smoke test is the value of the pre-merge run. The release publication is decoupled and handled automatically by the existing tag-push trigger.

## Risks / Trade-offs

**[Risk] Runtime correctness regression on a non-Gemma model** — A binding layer that compiles and passes unit tests can still produce broken inference if upstream changed an op kernel. → **Mitigation**: smoke-test inference against at least one non-Gemma model (e.g. Llama 3 or Qwen) end-to-end as part of validation. The existing `packages/llamafarm-llama` test suite is necessary but not sufficient.

**[Risk] Gemma 4 support is itself buggy in the chosen tag** — We are picking a tag that is hours old. There may be Gemma 4 issues that have not been reported yet. → **Mitigation**: this is acknowledged and accepted as the price of needing Gemma 4 now. Decision 2's "plan for follow-up bumps" is the explicit mitigation. Document any observed Gemma 4 issues in the next bump's design doc.

**[Risk] ARM64 build fails at the new tag** — The Linux ARM64 binary is built from source via `build-llama.yml`, and upstream may have introduced a build dependency or flag we do not handle. → **Mitigation**: Decision 5 (smoke-test build before merge) catches this. If the build fails, the change is blocked until either upstream is fixed or we patch our workflow. Note: the smoke-test run does NOT publish to a release — see Decision 5.

**[Risk] Cache eviction does not happen** — Users with cached `b7694` binaries should automatically download the new version because the cache key includes the version string. If that assumption is wrong, they will silently keep running the old binary. → **Mitigation**: verify the cache key behavior in `_binary.py` (`_get_cache_dir() / LLAMA_CPP_VERSION / _get_lib_name()`) is correct, and add a scenario to the capability spec asserting it.

**[Trade-off] We are tracking upstream more closely than ollama or other reference projects** — Ollama's vendored llama.cpp is from December 2025; they have their own Go inference engine for new models and do not need to chase upstream. We do need to chase upstream because `llamafarm-llama` is our only inference path. This is a genuine architectural tax of the "own the bindings" choice. The trade-off was made deliberately when llama-cpp-python was abandoned, but it should be revisited if upstream-chasing becomes a recurring source of pain. (Not in scope for this change.)

**[Trade-off] Bundling the deprecated-API rename into a version bump** — In strict terms, the rename is unrelated to the bump. They could be separate changes. We are bundling them because they will both touch the same files, both need the same validation, and the cost of two changes vs. one is meaningfully higher with no offsetting benefit. If review feedback objects, the rename can be split out cheaply.

## Migration Plan

**Pre-merge steps (in order):**

1. Resolve the target tag: latest upstream release. For this change, `b8708`. (The original plan required `b8709+` to include `d9a12c82`; relaxed at implementation time — see Decision 1.)
2. Update `llama-cpp-version.txt` and the four propagation locations.
3. Update `_bindings.py` and `llama.py` to use the renamed APIs.
4. Add `.claude/rules/llama_cpp_bindings.md`.
5. Add the `llama-cpp-binary` spec under `openspec/changes/bump-llama-cpp-gemma4/specs/`.
6. Run `packages/llamafarm-llama` test suite locally on the host platform.
7. Run `cli/internal/llamabinary` test suite locally.
8. Manually trigger `build-llama.yml` against the new tag via `gh workflow run` as a build smoke test and confirm the ARM64 artifact uploads to the workflow run. (Release attachment happens automatically on the next LlamaFarm `v*` tag push — see Decision 5.)
9. Smoke-test a Gemma 4 model end-to-end through the universal runtime (load → generate → check output is coherent).
10. Smoke-test a non-Gemma 4 model end-to-end as a regression check.
11. Push and let CI run on Linux/macOS/Windows.

**Rollback strategy:**

- The change is gated by a single version string in `llama-cpp-version.txt`. Reverting that string and the four propagation locations is sufficient to restore previous behavior.
- Cached binaries on user machines remain isolated by version, so a rollback does not require cache cleanup.
- The deprecated-API rename is signature-compatible, so a rollback of the rename is also a pure source revert.
- The `.claude/rules/llama_cpp_bindings.md` and capability spec can stay regardless of whether the version bump rolls back; they document intent that survives version churn.

## Open Questions

- **Exactly which tag?** Resolved: `b8708`. The original constraint to include commit `d9a12c82` was relaxed at implementation time (see Decision 1) — no tag yet contains that commit, and the EOG fix is captured in the planned follow-up bump.
- **Which Gemma 4 model variant for the smoke test?** Implementer's choice — the smallest Gemma 4 model that exercises the tokenizer fixes is sufficient. Document which variant was tested in the PR description.
- **Should the rules doc reference the next-bump cadence?** Yes — it should explicitly state that during periods of upstream churn around a hot model (like Gemma 4 today), expect to do follow-up bumps every 1–2 weeks until the churn settles. This sets expectations for whoever does bump #2.
