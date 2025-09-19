import base64
import os
from typing import Any, Dict, Optional

try:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.backends import default_backend
    from pydantic import BaseModel
    ENCRYPTION_AVAILABLE = True
except ImportError:
    ENCRYPTION_AVAILABLE = False
    class BaseModel:
        pass

from core.logging import FastAPIStructLogger
logger = FastAPIStructLogger()


class EncryptedValue(BaseModel if ENCRYPTION_AVAILABLE else object):
    if ENCRYPTION_AVAILABLE:
        ciphertext: str
        salt: str
        iv: str
        version: str = "1.0"
        algorithm: str = "AES-256-GCM"
    else:
        def __init__(self, ciphertext="", salt="", iv="", version="1.0", algorithm="AES-256-GCM"):
            self.ciphertext = ciphertext
            self.salt = salt
            self.iv = iv
            self.version = version
            self.algorithm = algorithm


class EncryptionError(Exception):
    pass


class SecretEncryption:
    ITERATIONS = 100000
    KEY_LENGTH = 32
    SALT_LENGTH = 16
    IV_LENGTH = 12

    def __init__(self, master_key: Optional[str] = None):
        if not ENCRYPTION_AVAILABLE:
            raise EncryptionError("Encryption dependencies not available.")

        if master_key is None:
            master_key = os.getenv("LF_ENCRYPTION_KEY")
            if not master_key:
                try:
                    from .settings import settings
                    master_key = settings.lf_encryption_key
                except ImportError:
                    pass

        if not master_key:
            raise EncryptionError("No encryption key provided.")

        try:
            self._master_key = base64.b64decode(master_key)
            if len(self._master_key) != 32:
                raise EncryptionError("Encryption key must be 256 bits (32 bytes)")
        except Exception as e:
            raise EncryptionError(f"Invalid encryption key: {e}") from e

    def _derive_key(self, salt: bytes) -> bytes:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=self.KEY_LENGTH,
            salt=salt,
            iterations=self.ITERATIONS,
            backend=default_backend()
        )
        return kdf.derive(self._master_key)

    def encrypt_value(self, plaintext: str) -> EncryptedValue:
        try:
            salt = os.urandom(self.SALT_LENGTH)
            iv = os.urandom(self.IV_LENGTH)
            key = self._derive_key(salt)

            cipher = Cipher(algorithms.AES(key), modes.GCM(iv), backend=default_backend())
            encryptor = cipher.encryptor()

            ciphertext = encryptor.update(plaintext.encode('utf-8')) + encryptor.finalize()
            ciphertext_with_tag = ciphertext + encryptor.tag

            return EncryptedValue(
                ciphertext=base64.b64encode(ciphertext_with_tag).decode('utf-8'),
                salt=base64.b64encode(salt).decode('utf-8'),
                iv=base64.b64encode(iv).decode('utf-8'),
            )
        except Exception as e:
            logger.error("Encryption failed", error=str(e))
            raise EncryptionError(f"Encryption failed: {e}") from e

    def decrypt_value(self, encrypted: EncryptedValue) -> str:
        try:
            ciphertext_with_tag = base64.b64decode(encrypted.ciphertext)
            salt = base64.b64decode(encrypted.salt)
            iv = base64.b64decode(encrypted.iv)

            ciphertext = ciphertext_with_tag[:-16]
            tag = ciphertext_with_tag[-16:]
            key = self._derive_key(salt)

            cipher = Cipher(algorithms.AES(key), modes.GCM(iv, tag), backend=default_backend())
            decryptor = cipher.decryptor()

            plaintext = decryptor.update(ciphertext) + decryptor.finalize()
            return plaintext.decode('utf-8')
        except Exception as e:
            logger.error("Decryption failed", error=str(e))
            raise EncryptionError(f"Decryption failed: {e}") from e

    def is_encrypted_value(self, value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        required_fields = {'ciphertext', 'salt', 'iv', 'version', 'algorithm'}
        return all(field in value for field in required_fields)


_encryption_service: Optional[SecretEncryption] = None

def get_encryption_service() -> SecretEncryption:
    global _encryption_service
    if _encryption_service is None:
        try:
            _encryption_service = SecretEncryption()
        except EncryptionError as e:
            logger.warning("Encryption service not available", error=str(e))
            raise
    return _encryption_service

def encrypt_secret_value(value: str) -> Dict[str, Any]:
    service = get_encryption_service()
    encrypted = service.encrypt_value(value)
    return encrypted.model_dump()

def decrypt_secret_value(encrypted_dict: Dict[str, Any]) -> str:
    service = get_encryption_service()
    encrypted = EncryptedValue(**encrypted_dict)
    return service.decrypt_value(encrypted)

def is_encrypted(value: Any) -> bool:
    if _encryption_service is None:
        return False
    return _encryption_service.is_encrypted_value(value)

def generate_encryption_key() -> str:
    if not ENCRYPTION_AVAILABLE:
        raise EncryptionError("Encryption dependencies not available.")
    key = os.urandom(32)
    return base64.b64encode(key).decode('utf-8')

def validate_encryption_key(key: str) -> bool:
    if not ENCRYPTION_AVAILABLE:
        return False
    try:
        decoded = base64.b64decode(key)
        return len(decoded) == 32
    except Exception:
        return False
