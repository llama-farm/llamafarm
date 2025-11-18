#! /bin/bash
set -e
echo "Compiling schema..."
uv run python compile_schema.py
echo ""
echo "Generating Python types..."
uv run datamodel-codegen \
    --input schema.deref.yaml \
    --output datamodel.py \
    --input-file-type=jsonschema \
    --output-model-type=pydantic_v2.BaseModel \
    --target-python-version=3.12 \
    --use-standard-collections \
    --use-title-as-name \
    --formatters=ruff-format \
    --set-default-enum-member \
    --class-name=LlamaFarmConfig
echo "✅ Python types generated"
echo ""
echo "✅ Done!"
