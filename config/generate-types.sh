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
# Add use_enum_values=True to all ConfigDict instances
sed -i '' 's/model_config = ConfigDict(/model_config = ConfigDict(\
        use_enum_values=True,/g' datamodel.py

echo "Formatting with ruff..."
uv run ruff format datamodel.py

echo "Done!"