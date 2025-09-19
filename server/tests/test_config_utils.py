import pytest
from unittest.mock import patch

from config.datamodel import LlamaFarmConfig, Version, Provider

class TestConfigSanitization:
    def setup_method(self):
        from core.encryption import generate_encryption_key
        self.test_key = generate_encryption_key()

    def test_sanitize_plaintext_api_key(self):
        from api.routers.projects.config_utils import sanitize_config_for_client
        config = LlamaFarmConfig(
            version=Version.v1,
            name="test-project",
            namespace="test-namespace",
            runtime={
                "provider": "openai",
                "model": "gpt-4",
                "api_key": "sk-test-key-123456789"
            }
        )
        sanitized = sanitize_config_for_client(config)
        runtime = sanitized["runtime"]
        assert runtime["api_key"] == "sk-t****6789"

    def test_sanitize_encrypted_api_key(self):
        with patch.dict(os.environ, {"LF_ENCRYPTION_KEY": self.test_key}):
            from api.routers.projects.config_utils import sanitize_config_for_client
            config = LlamaFarmConfig(
                version=Version.v1,
                name="test-project",
                namespace="test-namespace",
                runtime={
                    "provider": "openai",
                    "model": "gpt-4",
                    "api_key": "sk-test-key-123456789"
                }
            )
            config_dict = config.model_dump()
            config_from_dict = LlamaFarmConfig(**config_dict)
            sanitized = sanitize_config_for_client(config_from_dict)
            runtime = sanitized["runtime"]
            assert runtime["api_key"] == "[ENCRYPTED]"

    def test_sanitize_short_api_key(self):
        from api.routers.projects.config_utils import sanitize_config_for_client
        config = LlamaFarmConfig(
            version=Version.v1,
            name="test-project",
            namespace="test-namespace",
            runtime={
                "provider": "openai",
                "model": "gpt-4",
                "api_key": "short"
            }
        )
        sanitized = sanitize_config_for_client(config)
        runtime = sanitized["runtime"]
        assert runtime["api_key"] == "****"

    def test_sanitize_empty_api_key(self):
        from api.routers.projects.config_utils import sanitize_config_for_client
        config = LlamaFarmConfig(
            version=Version.v1,
            name="test-project",
            namespace="test-namespace",
            runtime={
                "provider": "openai",
                "model": "gpt-4",
                "api_key": ""
            }
        )
        sanitized = sanitize_config_for_client(config)
        runtime = sanitized["runtime"]
        assert runtime["api_key"] == ""

    def test_sanitize_config_without_runtime(self):
        from api.routers.projects.config_utils import sanitize_config_for_client
        config = LlamaFarmConfig(
            version=Version.v1,
            name="test-project",
            namespace="test-namespace"
        )
        sanitized = sanitize_config_for_client(config)
        assert "runtime" not in sanitized

    def test_get_plaintext_api_key(self):
        from api.routers.projects.config_utils import get_plaintext_api_key
        config = LlamaFarmConfig(
            version=Version.v1,
            name="test-project",
            namespace="test-namespace",
            runtime={
                "provider": "openai",
                "model": "gpt-4",
                "api_key": "sk-test-key-123456789"
            }
        )
        plaintext = get_plaintext_api_key(config)
        assert plaintext == "sk-test-key-123456789"

    def test_get_plaintext_api_key_none(self):
        from api.routers.projects.config_utils import get_plaintext_api_key
        config = LlamaFarmConfig(
            version=Version.v1,
            name="test-project",
            namespace="test-namespace",
            runtime={
                "provider": "openai",
                "model": "gpt-4"
            }
        )
        plaintext = get_plaintext_api_key(config)
        assert plaintext is None


class TestEncryptedValueDetection:
    def test_is_encrypted_dict_valid(self):
        from api.routers.projects.config_utils import _is_encrypted_dict
        encrypted_dict = {
            "ciphertext": "encrypted-data",
            "salt": "salt",
            "iv": "iv",
            "version": "1.0",
            "algorithm": "AES-256-GCM"
        }
        assert _is_encrypted_dict(encrypted_dict)

    def test_is_encrypted_dict_invalid(self):
        from api.routers.projects.config_utils import _is_encrypted_dict
        incomplete_dict = {
            "ciphertext": "encrypted-data",
            "salt": "salt"
        }
        assert not _is_encrypted_dict(incomplete_dict)
        assert not _is_encrypted_dict("string")
        assert not _is_encrypted_dict(None)
        assert not _is_encrypted_dict([])

class TestProjectResponseModel:
    def setup_method(self):
        from core.encryption import generate_encryption_key
        self.test_key = generate_encryption_key()

    def test_project_from_project_service(self):
        from api.routers.projects.projects import Project
        from services.project_service import Project as ProjectServiceProject
        config = LlamaFarmConfig(
            version=Version.v1,
            name="test-project",
            namespace="test-namespace",
            runtime={
                "provider": "openai",
                "model": "gpt-4",
                "api_key": "sk-test-key-123456789"
            }
        )
        service_project = ProjectServiceProject(
            namespace="test-namespace",
            name="test-project",
            config=config
        )
        api_project = Project.from_project_service(
            service_project.namespace,
            service_project.name,
            service_project.config
        )
        assert api_project.namespace == "test-namespace"
        assert api_project.name == "test-project"
        assert isinstance(api_project.config, dict)
        runtime = api_project.config["runtime"]
        assert runtime["api_key"] == "sk-t****6789"
