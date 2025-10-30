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
    --formatters=ruff-format \
    --class-name=LlamaFarmConfig
echo "✅ Python types generated"
echo ""
echo "Generating Go reference types (for validation)..."
./generate-go-types.sh
echo ""
echo "Validating manual Go types against schema..."
./validate-go-types.sh
echo ""
echo "✅ Done!"
echo ""
echo "If validation shows mismatches, review cli/cmd/config/types.go and ensure"
echo "it matches the generated reference at config/config_types_generated.go"
