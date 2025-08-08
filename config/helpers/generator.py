from config.datamodel import LlamaFarmConfig, Version


def generate_base_config():
    return LlamaFarmConfig(
        version=Version.v1,
        name="llamafarm",
        rag={},
        prompts=[],
        datasets=[],
        models=[],
    ).model_dump(mode="json")

# Example usage:
# generate_base_config_from_schema("schema.yaml", "llamafarm.yaml")
