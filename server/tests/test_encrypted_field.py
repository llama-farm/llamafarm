import pytest
from unittest.mock import patch
from pydantic import BaseModel

from core.encryption import generate_encryption_key

class TestEncryptedField:
    def setup_method(self):
        self.test_key = generate_encryption_key()

    def test_encrypted_string_from_plaintext(self):
        with patch.dict(os.environ, {"LF_ENCRYPTION_KEY": self.test_key}):
            from core.encrypted_field import EncryptedString
            secret = EncryptedString("my-api-key")
            assert secret.get_plaintext() == "my-api-key"
            assert not secret.is_encrypted()
            assert str(secret) == "[ENCRYPTED]"

    def test_encrypted_string_from_encrypted_dict(self):
        with patch.dict(os.environ, {"LF_ENCRYPTION_KEY": self.test_key}):
            from core.encryption import encrypt_secret_value
            from core.encrypted_field import EncryptedString
            encrypted_data = encrypt_secret_value("original-secret")
            secret = EncryptedString(encrypted_data)
            assert secret.get_plaintext() == "original-secret"
            assert secret.is_encrypted()
            assert str(secret) == "[ENCRYPTED]"

    def test_encrypted_string_decryption_failure(self):
        wrong_key = generate_encryption_key()
        original_key = generate_encryption_key()

        with patch.dict(os.environ, {"LF_ENCRYPTION_KEY": original_key}):
            from core.encryption import encrypt_secret_value
            encrypted_data = encrypt_secret_value("original-secret")

        with patch.dict(os.environ, {"LF_ENCRYPTION_KEY": wrong_key}):
            from core.encrypted_field import EncryptedString
            secret = EncryptedString(encrypted_data)
            assert secret.get_plaintext() is None
            assert secret.is_encrypted()
            assert str(secret) == "[ENCRYPTED_FAILED]"


class TestEncryptedFieldInModel:
    def setup_method(self):
        self.test_key = generate_encryption_key()

    def test_model_with_encrypted_field(self):
        with patch.dict(os.environ, {"LF_ENCRYPTION_KEY": self.test_key}):
            from core.encrypted_field import EncryptedString, encrypted_field

            class TestConfig(BaseModel):
                name: str
                api_key: EncryptedString = encrypted_field(None, description="API key")

            config = TestConfig(name="test", api_key="secret-key-123")
            assert config.api_key.get_plaintext() == "secret-key-123"
            assert not config.api_key.is_encrypted()

    def test_model_serialization_with_encryption(self):
        with patch.dict(os.environ, {"LF_ENCRYPTION_KEY": self.test_key}):
            from core.encrypted_field import EncryptedString, encrypted_field

            class TestConfig(BaseModel):
                name: str
                api_key: EncryptedString = encrypted_field(None, description="API key")

            config = TestConfig(name="test", api_key="secret-key-123")
            config_dict = config.model_dump()
            assert isinstance(config_dict["api_key"], dict)
            assert "ciphertext" in config_dict["api_key"]

    def test_model_deserialization_with_encryption(self):
        with patch.dict(os.environ, {"LF_ENCRYPTION_KEY": self.test_key}):
            from core.encrypted_field import EncryptedString, encrypted_field
            from core.encryption import encrypt_secret_value

            class TestConfig(BaseModel):
                name: str
                api_key: EncryptedString = encrypted_field(None, description="API key")

            encrypted_data = encrypt_secret_value("original-key")
            config_dict = {"name": "test", "api_key": encrypted_data}
            config = TestConfig(**config_dict)
            assert config.api_key.get_plaintext() == "original-key"
            assert config.api_key.is_encrypted()


class TestConfigIntegration:
    def setup_method(self):
        self.test_key = generate_encryption_key()

    def test_runtime_with_encrypted_api_key(self):
        with patch.dict(os.environ, {"LF_ENCRYPTION_KEY": self.test_key}):
            from config.datamodel import Runtime, Provider
            runtime = Runtime(
                provider=Provider.openai,
                model="gpt-4",
                api_key="sk-test-key-123456789"
            )
            assert hasattr(runtime.api_key, 'get_plaintext')
            assert runtime.api_key.get_plaintext() == "sk-test-key-123456789"
            assert not runtime.api_key.is_encrypted()

    def test_runtime_serialization_encryption(self):
        with patch.dict(os.environ, {"LF_ENCRYPTION_KEY": self.test_key}):
            from config.datamodel import Runtime, Provider
            runtime = Runtime(
                provider=Provider.openai,
                model="gpt-4",
                api_key="sk-test-key-123456789"
            )
            runtime_dict = runtime.model_dump()
            assert isinstance(runtime_dict["api_key"], dict)
            assert "ciphertext" in runtime_dict["api_key"]

    def test_full_config_with_encrypted_api_key(self):
        with patch.dict(os.environ, {"LF_ENCRYPTION_KEY": self.test_key}):
            from config.datamodel import LlamaFarmConfig, Version, Provider
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
            runtime = config_dict["runtime"]
            assert isinstance(runtime["api_key"], dict)
            assert "ciphertext" in runtime["api_key"]

class TestConfigLoaderEncryption:
    def setup_method(self):
        self.test_key = generate_encryption_key()

    def test_loader_processes_encrypted_fields(self):
        with patch.dict(os.environ, {"LF_ENCRYPTION_KEY": self.test_key}):
            from config.helpers.loader import _process_encrypted_fields_on_load, _process_encrypted_fields_for_save
            from core.encryption import encrypt_secret_value

            config_dict = {
                "version": "v1",
                "name": "test",
                "namespace": "test",
                "runtime": {
                    "provider": "openai",
                    "model": "gpt-4",
                    "api_key": encrypt_secret_value("test-key")
                }
            }

            _process_encrypted_fields_on_load(config_dict)
            assert isinstance(config_dict["runtime"]["api_key"], dict)
            assert "ciphertext" in config_dict["runtime"]["api_key"]

            config_dict_plaintext = {
                "version": "v1",
                "name": "test",
                "namespace": "test",
                "runtime": {
                    "provider": "openai",
                    "model": "gpt-4",
                    "api_key": "plaintext-key"
                }
            }

            result_dict = _process_encrypted_fields_for_save(config_dict_plaintext)
            assert isinstance(result_dict["runtime"]["api_key"], dict)
            assert "ciphertext" in result_dict["runtime"]["api_key"]
