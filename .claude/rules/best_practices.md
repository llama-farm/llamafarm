# Software Development Best Practices

This document defines **non‑negotiable software engineering standards** that AI agents must follow when generating, modifying, or refactoring code in this monorepo. These rules exist to preserve long‑term maintainability, security, testability, and team velocity across frontend, API, and AI runtime services.

AI agents are expected to **default to these practices unless explicitly instructed otherwise**. Deviations must be justified in comments or commit messages.

---

## 1. Architectural Principles

### 1.1 Separation of Concerns

Each module, service, or package must have **one primary responsibility**.

* UI logic must not contain business logic
* Business logic must not contain infrastructure concerns
* Infrastructure code must not encode domain rules

**Anti‑patterns:**

* Database queries inside UI components
* HTTP request handling mixed with core business logic
* Model inference logic tightly coupled to transport layers

**Preferred pattern:**

```
UI / Controller → Application Service → Domain Logic → Infrastructure Adapters
```

---

### 1.2 Loose Coupling, High Cohesion

* Dependencies must flow **inward** toward stable abstractions
* Modules should depend on **interfaces**, not concrete implementations
* Internal details must be hidden behind well‑defined boundaries

**Guidelines:**

* Prefer dependency injection over imports of concrete classes
* Avoid shared mutable state across modules
* Avoid global configuration access inside business logic

---

### 1.3 Composition Over Inheritance

* Prefer composition, delegation, and small units of behavior
* Inheritance is allowed only for:

  * Framework integration
  * Clearly modeled is‑a relationships

**Reasoning:**
Inheritance increases coupling, reduces flexibility, and complicates refactoring—especially in AI‑assisted codebases.

---

## 2. Code Quality Standards

### 2.1 DRY (Don’t Repeat Yourself)

Duplication of **logic**, not syntax, is the primary concern.

* Extract shared logic into reusable modules
* Avoid copy‑pasted code with minor variations
* Centralize business rules

**Allowed duplication:**

* Trivial glue code
* Readability‑motivated duplication (with justification)

---

### 2.2 Explicitness Over Cleverness

Code must optimize for **readability and predictability**, not brevity.

* Prefer clear variable and function names
* Avoid implicit side effects
* Avoid overuse of metaprogramming, reflection, or magic behavior

AI agents must assume that humans will maintain this code long‑term.

---

### 2.3 Small, Focused Functions

* Functions should do **one thing**
* Ideal length: 5–30 lines
* Avoid deeply nested conditionals

If a function requires extensive explanation, it is too complex.

---

### 2.4 Deterministic Behavior

* Given the same inputs, functions should produce the same outputs
* Avoid hidden dependencies on:

  * Global state
  * System time (unless injected)
  * Randomness (unless seeded or abstracted)

This is especially critical for AI pipelines and inference systems.

---

## 3. Error Handling & Reliability

### 3.1 Fail Loudly and Early

* Validate inputs at boundaries
* Throw or return explicit errors
* Avoid silent failures

Errors should:

* Be descriptive
* Contain actionable context
* Preserve original stack traces

---

### 3.2 No Exception Swallowing

* Never catch errors without handling or rethrowing
* Logging an error is **not** handling it

---

### 3.3 Graceful Degradation

For long‑running services and AI systems:

* Handle partial failures
* Use timeouts and circuit breakers
* Avoid cascading failures

---

## 4. Testing Discipline

### 4.1 Test Pyramid

AI agents must follow this hierarchy:

1. **Unit tests** – fast, deterministic, isolated
2. **Integration tests** – real dependencies, limited scope
3. **End‑to‑end tests** – minimal and targeted

---

### 4.2 Test What Matters

* Test business logic, not framework internals
* Avoid brittle snapshot tests unless justified
* Tests must be deterministic and reproducible

---

### 4.3 Tests as Specification

Well‑written tests should:

* Document expected behavior
* Clarify edge cases
* Serve as living documentation

AI agents should read existing tests before modifying behavior.

---

## 5. Dependency Management

### 5.1 Minimize Dependencies

* Prefer standard libraries
* Avoid adding dependencies for trivial functionality
* Evaluate long‑term maintenance risk

---

### 5.2 Stable Interfaces

* Internal APIs must be versioned or backward‑compatible
* Breaking changes require migration paths

---

## 6. Performance & Scalability

### 6.1 Measure Before Optimizing

* Do not prematurely optimize
* Use profiling and metrics

---

### 6.2 Predictable Resource Usage

* Avoid unbounded memory growth
* Avoid unbounded concurrency
* Clean up resources deterministically

Critical for AI runtimes and long‑lived workers.

---

## 7. Documentation Expectations

### 7.1 Code Should Be Self‑Documenting

Comments are for:

* Why something exists
* Why a decision was made
* Non‑obvious constraints

Not for narrating obvious code behavior.

---

### 7.2 Architecture‑Level Documentation

* Major components must have a README
* Public APIs must be documented
* Assumptions and invariants must be explicit

---

## 8. Change Safety

### 8.1 Backward Compatibility First

AI agents must assume:

* Existing users depend on current behavior
* Changes may affect multiple services

---

### 8.2 Small, Reviewable Changes

* Prefer incremental commits
* Avoid large, unfocused refactors

---

## 9. AI‑Specific Expectations

### 9.1 Transparency

* AI behavior must be inspectable
* Prompts, configs, and decision logic must be explicit

---

### 9.2 Reproducibility

* Model versions must be pinned
* Configuration must be deterministic
* Outputs should be traceable to inputs

---

## 10. Common Pitfalls to Avoid

### 10.1 Platform & Environment Assumptions

* **Never hardcode `/tmp`** – Use `tempfile.gettempdir()` in Python or equivalent. Respects `TMPDIR`, `TEMP`, and platform conventions.
* **Never hardcode branch names** – Detect the default branch dynamically: `git symbolic-ref refs/remotes/origin/HEAD | sed 's@^refs/remotes/origin/@@'`
* **Consider case-insensitive filesystems** – macOS and Windows treat `.env` and `.ENV` as the same file. Use case-insensitive comparisons for file matching.
* **Avoid leading dashes in generated paths** – Paths like `-Users-foo` can be interpreted as command flags. Strip leading dashes or use `./` prefix.

### 10.2 API & URL Conventions

* **Ollama URLs should NOT include `/v1`** – Client libraries auto-append this. Use `http://localhost:11434`, not `http://localhost:11434/v1`.
* **Verify API paths before documenting** – Run the actual command to confirm it exists.

### 10.3 Security Pattern Completeness

* **Negative security patterns must be precise** – When checking "yaml.load without SafeLoader", don't just exclude `Loader=`; exclude only `Loader=SafeLoader` or `Loader=yaml.SafeLoader`.
* **Detection patterns must be actionable** – If documenting a check, provide the complete command. "Check for X" without a working grep/search is useless.

### 10.4 Exception Handling

* **Never swallow exceptions silently** – At minimum, log a warning. Silent `except: pass` makes debugging impossible.
* **Log before ignoring** – If you intentionally ignore an exception, log it first so failures are visible.

### 10.5 Documentation Accuracy

* **Verify commands exist** – Before documenting a CLI command, confirm it exists in the codebase.
* **Proofread for typos and grammar** – Check possessives (`user's` not `users`), verb tenses, and spelling.

---

## 11. Enforcement

AI agents must:

* Treat this document as authoritative
* Call out conflicts with existing code
* Ask for clarification when requirements conflict

Violations are considered defects.

