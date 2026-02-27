## ADDED Requirements

### Requirement: Deploy supports uploading dataset documents
The `lf deploy --with-data` flag SHALL upload local dataset document files to the remote server and trigger RAG ingestion.

#### Scenario: Upload and ingest dataset docs
- **WHEN** user runs `lf deploy production --with-data`
- **THEN** for each dataset in the project config, the CLI uploads local document files via the bulk upload endpoint and triggers ingestion on the server

#### Scenario: Dataset has no local files
- **WHEN** a dataset is configured but has no local document files in the project directory
- **THEN** the CLI skips that dataset and reports it as having no files to upload

### Requirement: Data push is configurable per environment
The `deploy_data` setting in an environment config SHALL control whether dataset upload and ingestion happens automatically on deploy.

#### Scenario: deploy_data enabled in environment
- **WHEN** an environment has `deploy_data: true` and user runs `lf deploy <env>`
- **THEN** dataset upload and ingestion is performed automatically as part of the deploy

#### Scenario: deploy_data overridden by --with-data flag
- **WHEN** an environment has `deploy_data: false` but user runs `lf deploy <env> --with-data`
- **THEN** dataset upload and ingestion is performed for this run

### Requirement: Data push shows ingestion progress
The CLI SHALL display progress for dataset file uploads and ingestion status.

#### Scenario: Multiple datasets uploaded
- **WHEN** 2 datasets with documents are uploaded
- **THEN** the CLI shows per-dataset file upload progress and ingestion task status
