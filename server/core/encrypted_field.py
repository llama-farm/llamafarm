from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, Field, field_serializer, field_validator
from pydantic_core import PydanticUndefined

try:
    from .encryption import (
        encrypt_secret_value,
        decrypt_secret_value,
        is_encrypted,
        EncryptionError,
        ENCRYPTION_AVAILABLE,
    )
except ImportError:
    ENCRYPTION_AVAILABLE = False
    def encrypt_secret_value(value: str) -> Dict[str, Any]:
        raise EncryptionError("Encryption service not available")
    def decrypt_secret_value(encrypted_dict: Dict[str, Any]) -> str:
        raise EncryptionError("Encryption service not available")
    def is_encrypted(value: Any) -> bool:
        return False
    class EncryptionError(Exception):
        pass


class EncryptedString(str):
    _plaintext_value: Optional[str] = None
    _is_encrypted: bool = False

    if not ENCRYPTION_AVAILABLE:
        def __new__(cls, value: Union[str, Dict[str, Any], None] = None):
            if value is None:
                return str.__new__(cls, "")
            return str.__new__(cls, str(value))
        def get_plaintext(self) -> Optional[str]:
            return str(self)
        def is_encrypted(self) -> bool:
            return False

    def __new__(cls, value: Union[str, Dict[str, Any], None] = None):
        if value is None:
            instance = str.__new__(cls, "")
            instance._plaintext_value = None
            instance._is_encrypted = False
            return instance

        if isinstance(value, dict) and is_encrypted(value):
            try:
                plaintext = decrypt_secret_value(value)
                instance = str.__new__(cls, "[ENCRYPTED]")
                instance._plaintext_value = plaintext
                instance._is_encrypted = True
                return instance
            except EncryptionError:
                instance = str.__new__(cls, "[ENCRYPTED_FAILED]")
                instance._plaintext_value = None
                instance._is_encrypted = True
                return instance
        else:
            plaintext = str(value)
            instance = str.__new__(cls, "[ENCRYPTED]")
            instance._plaintext_value = plaintext
            instance._is_encrypted = False
            return instance

    def get_plaintext(self) -> Optional[str]:
        return self._plaintext_value

    def is_encrypted(self) -> bool:
        return self._is_encrypted

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Any) -> Any:
        from pydantic_core import core_schema

        def validate_encrypted_string(value: Any) -> 'EncryptedString':
            if value is None:
                return cls(None)
            if isinstance(value, dict) and is_encrypted(value):
                return cls(value)
            if isinstance(value, str):
                return cls(value)
            return cls(str(value))

        return core_schema.no_info_plain_validator_function(validate_encrypted_string)


def EncryptedField(default: Any = PydanticUndefined, **kwargs) -> Field:
    if default is not PydanticUndefined:
        default = EncryptedString(default)
    return Field(default=default, **kwargs)

class EncryptedFieldMixin:
    def __init__(self, **data):
        for field_name, field_info in self.model_fields.items():
            if field_name in data:
                field_type = field_info.annotation
                if hasattr(field_type, '__origin__') and field_type.__origin__ is Union:
                    for arg in field_type.__args__:
                        if hasattr(arg, '__name__') and arg.__name__ == 'EncryptedString':
                            data[field_name] = EncryptedString(data[field_name])
                            break
                elif hasattr(field_type, '__name__') and field_type.__name__ == 'EncryptedString':
                    data[field_name] = EncryptedString(data[field_name])
        super().__init__(**data)

    @field_serializer('*', when_used='json')
    def serialize_encrypted_fields(self, value: Any, info) -> Any:
        if isinstance(value, EncryptedString):
            if value.is_encrypted():
                return {
                    'ciphertext': getattr(value, '_encrypted_data', {}).get('ciphertext', ''),
                    'salt': getattr(value, '_encrypted_data', {}).get('salt', ''),
                    'iv': getattr(value, '_encrypted_data', {}).get('iv', ''),
                    'version': getattr(value, '_encrypted_data', {}).get('version', '1.0'),
                    'algorithm': getattr(value, '_encrypted_data', {}).get('algorithm', 'AES-256-GCM'),
                }
            else:
                plaintext = value.get_plaintext()
                if plaintext is not None:
                    try:
                        encrypted_data = encrypt_secret_value(plaintext)
                        value._encrypted_data = encrypted_data
                        value._is_encrypted = True
                        return encrypted_data
                    except EncryptionError:
                        return "[ENCRYPTION_FAILED]"
                return None
        return value


encrypted_field = EncryptedField
