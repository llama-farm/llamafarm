import os
import pytest
from unittest.mock import patch

from core.encryption import (
    SecretEncryption,
    EncryptedValue,
    EncryptionError,
    generate_encryption_key,
    validate_encryption_key,
    encrypt_secret_value,
    decrypt_secret_value,
    is_encrypted,
)

class TestSecretEncryption:
    def setup_method(self):
        self.test_key = generate_encryption_key()
        self.encryption = SecretEncryption(self.test_key)

    def test_generate_encryption_key(self):
        key = generate_encryption_key()
        assert validate_encryption_key(key)

    def test_validate_encryption_key(self):
        valid_key = generate_encryption_key()
        assert validate_encryption_key(valid_key)
        assert not validate_encryption_key("invalid-base64!")
        assert not validate_encryption_key("")
        assert not validate_encryption_key("short")

    def test_encrypt_decrypt_value(self):
        plaintext = "my-secret-api-key-12345"
        encrypted = self.encryption.encrypt_value(plaintext)
        decrypted = self.encryption.decrypt_value(encrypted)
        assert decrypted == plaintext
        assert isinstance(encrypted, EncryptedValue)

    def test_encrypt_different_values_produce_different_ciphertext(self):
        plaintext1 = "secret1"
        plaintext2 = "secret2"
        encrypted1 = self.encryption.encrypt_value(plaintext1)
        encrypted2 = self.encryption.encrypt_value(plaintext2)
        assert encrypted1.ciphertext != encrypted2.ciphertext

    def test_encrypt_same_value_produces_different_ciphertext(self):
        plaintext = "same-secret"
        encrypted1 = self.encryption.encrypt_value(plaintext)
        encrypted2 = self.encryption.encrypt_value(plaintext)
        assert encrypted1.ciphertext != encrypted2.ciphertext

    def test_decrypt_wrong_key_fails(self):
        plaintext = "secret"
        wrong_key = generate_encryption_key()
        encrypted = self.encryption.encrypt_value(plaintext)
        wrong_encryption = SecretEncryption(wrong_key)
        with pytest.raises(EncryptionError):
            wrong_encryption.decrypt_value(encrypted)

    def test_is_encrypted_value(self):
        encrypted = self.encryption.encrypt_value("secret")
        encrypted_dict = encrypted.model_dump()
        assert self.encryption.is_encrypted_value(encrypted_dict)
        assert not self.encryption.is_encrypted_value("plaintext")
        assert not self.encryption.is_encrypted_value({})

    def test_encrypt_empty_string(self):
        plaintext = ""
        encrypted = self.encryption.encrypt_value(plaintext)
        decrypted = self.encryption.decrypt_value(encrypted)
        assert decrypted == plaintext

    def test_encrypt_unicode_string(self):
        plaintext = "héllo wörld 🚀"
        encrypted = self.encryption.encrypt_value(plaintext)
        decrypted = self.encryption.decrypt_value(encrypted)
        assert decrypted == plaintext


class TestEncryptionWithEnvironment:
    def test_encryption_from_env_var(self):
        test_key = generate_encryption_key()
        with patch.dict(os.environ, {"LF_ENCRYPTION_KEY": test_key}):
            encryption = SecretEncryption()
            assert encryption is not None
            plaintext = "test-secret"
            encrypted = encryption.encrypt_value(plaintext)
            decrypted = encryption.decrypt_value(encrypted)
            assert decrypted == plaintext

    def test_encryption_missing_key_raises_error(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch('core.encryption.settings', None):
                with pytest.raises(EncryptionError, match="No encryption key provided"):
                    SecretEncryption()

class TestEncryptedValueModel:
    def test_encrypted_value_model_creation(self):
        data = {
            "ciphertext": "encrypted-data",
            "salt": "salt",
            "iv": "iv",
            "version": "1.0",
            "algorithm": "AES-256-GCM"
        }
        encrypted = EncryptedValue(**data)
        assert encrypted.ciphertext == "encrypted-data"

    def test_encrypted_value_missing_fields(self):
        incomplete_data = {
            "ciphertext": "encrypted-data",
            "salt": "salt",
        }
        with pytest.raises(ValueError):
            EncryptedValue(**incomplete_data)


class TestUtilityFunctions:
    def setup_method(self):
        self.test_key = generate_encryption_key()

    def test_encrypt_secret_value(self):
        with patch.dict(os.environ, {"LF_ENCRYPTION_KEY": self.test_key}):
            plaintext = "secret-value"
            encrypted = encrypt_secret_value(plaintext)
            assert isinstance(encrypted, dict)
            assert "ciphertext" in encrypted

    def test_decrypt_secret_value(self):
        with patch.dict(os.environ, {"LF_ENCRYPTION_KEY": self.test_key}):
            plaintext = "secret-value"
            encrypted = encrypt_secret_value(plaintext)
            decrypted = decrypt_secret_value(encrypted)
            assert decrypted == plaintext

    def test_is_encrypted(self):
        with patch.dict(os.environ, {"LF_ENCRYPTION_KEY": self.test_key}):
            encrypted = encrypt_secret_value("secret")
            assert is_encrypted(encrypted)
            assert not is_encrypted("plaintext")

class TestEncryptionIntegration:
    def test_full_encrypt_decrypt_cycle(self):
        test_key = generate_encryption_key()
        with patch.dict(os.environ, {"LF_ENCRYPTION_KEY": test_key}):
            original = "my-api-key-123456789"
            encrypted_data = encrypt_secret_value(original)
            assert isinstance(encrypted_data, dict)
            decrypted = decrypt_secret_value(encrypted_data)
            assert decrypted == original

    def test_multiple_keys_isolation(self):
        key1 = generate_encryption_key()
        key2 = generate_encryption_key()
        plaintext = "shared-secret"

        with patch.dict(os.environ, {"LF_ENCRYPTION_KEY": key1}):
            encrypted1 = encrypt_secret_value(plaintext)

        with patch.dict(os.environ, {"LF_ENCRYPTION_KEY": key2}):
            encrypted2 = encrypt_secret_value(plaintext)

        assert encrypted1["ciphertext"] != encrypted2["ciphertext"]

        with patch.dict(os.environ, {"LF_ENCRYPTION_KEY": key2}):
            with pytest.raises(EncryptionError):
                decrypt_secret_value(encrypted1)

        with patch.dict(os.environ, {"LF_ENCRYPTION_KEY": key1}):
            decrypted = decrypt_secret_value(encrypted1)
            assert decrypted == plaintext
