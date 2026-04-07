## ADDED Requirements

### Requirement: Project config supports deployment section

The `llamafarm.yaml` schema SHALL support an optional top-level `deployment` key containing deployment-target defaults that complement the existing `environments` section.

#### Scenario: Config with deployment.model_dir defined

- **WHEN** `llamafarm.yaml` contains `deployment: { model_dir: /srv/lf/models }`
- **THEN** the config is valid and `deployment.model_dir` is accessible to CLI commands

#### Scenario: Config without deployment section

- **WHEN** `llamafarm.yaml` has no `deployment` section
- **THEN** the config is valid and all existing behavior is unchanged

### Requirement: `deployment.model_dir` provides default target root for model paths

The `deployment.model_dir` field SHALL serve as the default value for the `target_root` used by `lf models path`. When this field is absent, CLI commands SHALL fall back to the hardcoded path `/opt/llamafarm/models`. The `--target-root` flag on `lf models path` SHALL override this field when both are present.

#### Scenario: `lf models path` uses deployment.model_dir

- **WHEN** `llamafarm.yaml` specifies `deployment.model_dir: /srv/lf/models` and the user runs `lf models path --format json` without `--target-root`
- **THEN** the emitted `target_root` is `/srv/lf/models`

#### Scenario: `lf models path` falls back to hardcoded default

- **WHEN** `llamafarm.yaml` has no `deployment` section and the user runs `lf models path --format json` without `--target-root`
- **THEN** the emitted `target_root` is `/opt/llamafarm/models`

#### Scenario: `--target-root` flag wins over deployment.model_dir

- **WHEN** `llamafarm.yaml` specifies `deployment.model_dir: /srv/lf/models` and the user runs `lf models path --format json --target-root /custom/path`
- **THEN** the emitted `target_root` is `/custom/path`

### Requirement: Deployment section is stripped before server push

When deploying a project config to a remote server (e.g., via `lf deploy`), the `deployment` section SHALL be removed from the config payload, consistent with how the `environments` section is handled today. Deployment defaults are local-only metadata about how to prepare artifacts for transport.

#### Scenario: Deploy strips deployment section

- **WHEN** `lf deploy <env>` pushes config to a remote server and the local `llamafarm.yaml` contains a `deployment` section
- **THEN** the `PUT /{ns}/{proj}` request body does not contain the `deployment` key
