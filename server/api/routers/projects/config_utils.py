from typing import Any, Dict
from config.datamodel import LlamaFarmConfig

def sanitize_config_for_client(config: LlamaFarmConfig) -> Dict[str, Any]:
    config_dict = config.model_dump(mode="json", exclude_none=True)

    if "runtime" in config_dict and isinstance(config_dict["runtime"], dict):
        runtime = config_dict["runtime"]

        if "api_key" in runtime:
            api_key_value = runtime["api_key"]

            if isinstance(api_key_value, dict) and _is_encrypted_dict(api_key_value):
                runtime["api_key"] = "[ENCRYPTED]"
            elif isinstance(api_key_value, str):
                if api_key_value and api_key_value not in ["[ENCRYPTED]", "[ENCRYPTION_FAILED]"]:
                    if len(api_key_value) > 8:
                        runtime["api_key"] = f"{api_key_value[:4]}****{api_key_value[-4:]}"
                    else:
                        runtime["api_key"] = "****"

    return config_dict

def _is_encrypted_dict(value: Dict[str, Any]) -> bool:
    required_fields = {"ciphertext", "salt", "iv", "version", "algorithm"}
    return all(field in value for field in required_fields)

def get_plaintext_api_key(config: LlamaFarmConfig) -> str | None:
    try:
        if hasattr(config.runtime, 'api_key') and config.runtime.api_key:
            if hasattr(config.runtime.api_key, 'get_plaintext'):
                return config.runtime.api_key.get_plaintext()
    except Exception:
        pass
    return None
