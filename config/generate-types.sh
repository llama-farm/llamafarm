#! /bin/bash
set -e
echo "Compiling schema..."
uv run python compile_schema.py
echo "Generating types..."
uv run datamodel-codegen \
    --input schema.deref.yaml \
    --output datamodel.py \
    --input-file-type=jsonschema \
    --output-model-type=pydantic_v2.BaseModel \
    --target-python-version=3.12 \
    --use-standard-collections \
    --formatters=ruff-format \
    --class-name=LlamaFarmConfig

echo "Post-processing generated types..."
# Add use_enum_values=True to all ConfigDict instances (cross-platform)
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS requires empty string after -i
    sed -i '' 's/model_config = ConfigDict(/model_config = ConfigDict(\
        use_enum_values=True,/g' datamodel.py
else
    # Linux doesn't require empty string
    sed -i 's/model_config = ConfigDict(/model_config = ConfigDict(\
        use_enum_values=True,/g' datamodel.py
fi

echo "Formatting with ruff..."
uv run ruff format datamodel.py

echo "Done!"