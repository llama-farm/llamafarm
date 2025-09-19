#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'server'))

try:
    from core.encryption import generate_encryption_key, validate_encryption_key
except ImportError as e:
    print(f"Error: {e}")
    sys.exit(1)

def main():
    key = generate_encryption_key()
    if validate_encryption_key(key):
        print(f"LF_ENCRYPTION_KEY={key}")
    else:
        print("Error: Generated key is invalid!")
        sys.exit(1)


if __name__ == "__main__":
    main()
