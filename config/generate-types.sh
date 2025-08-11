#! /bin/bash
set -e
uv run datamodel-codegen \
    --input schema.yaml \
    --output datamodel.py \
    --input-file-type=jsonschema \
    --use-standard-collections \
    --formatters=ruff-format \
    --class-name=LlamaFarmConfig
