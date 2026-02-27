## ADDED Requirements

### Requirement: Project config supports environments section
The `llamafarm.yaml` schema SHALL support an optional top-level `environments` key containing named deploy targets.

#### Scenario: Config with environments defined
- **WHEN** `llamafarm.yaml` contains an `environments` section with entries like `staging` and `production`
- **THEN** each entry SHALL have a required `server_url` field and optional `deploy_models` (default: true) and `deploy_data` (default: false) boolean fields

#### Scenario: Config without environments
- **WHEN** `llamafarm.yaml` has no `environments` section
- **THEN** the config is valid and all existing behavior is unchanged

### Requirement: Environments section is stripped before server push
When deploying a project config to a server, the `environments` section SHALL be removed from the config payload.

#### Scenario: Deploy strips environments
- **WHEN** `lf deploy staging` pushes config to a remote server
- **THEN** the `PUT /{ns}/{proj}` request body does not contain the `environments` key

### Requirement: Environment config provides deploy settings
Each named environment SHALL configure how deploy behaves for that target.

#### Scenario: Environment with deploy_models enabled
- **WHEN** an environment has `deploy_models: true`
- **THEN** `lf deploy <env>` triggers model downloads on the target server after pushing config

#### Scenario: Environment with deploy_data enabled
- **WHEN** an environment has `deploy_data: true`
- **THEN** `lf deploy <env>` uploads dataset documents and triggers ingestion on the target server

#### Scenario: Environment with deploy_data disabled
- **WHEN** an environment has `deploy_data: false` (or omitted, as false is the default)
- **THEN** `lf deploy <env>` does not upload or process datasets
