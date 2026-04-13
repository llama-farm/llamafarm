## 1. Reorder priority in context calculator

- [x] 1.1 In `runtimes/edge/utils/context_calculator.py`, swap the `elif n_ctx_train` and `elif pattern_n_ctx` branches in `get_default_context_size()` so pattern match is checked before `n_ctx_train`
- [x] 1.2 Update the function docstring priority list to reflect the new order (pattern match at tier 2, `n_ctx_train` at tier 3)
- [x] 1.3 Update inline comments on each `elif` branch to match the new priority numbering

## 2. Add override logging

- [x] 2.1 Add an info-level log when a pattern match is used and `n_ctx_train` is also available, showing both values (e.g., "Pattern default overrides n_ctx_train (32768 → 4096) for model navlink-v2-Q8_0.gguf")

## 3. Update compose template in llamadrone

- [x] 3.1 Add `GGUF_MODELS_DIR=/models` to the edge-runtime environment in `infra/ansible/roles/deploy/templates/docker-compose.yml.j2`
