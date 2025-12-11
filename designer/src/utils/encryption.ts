/**
 * Encrypts an API key using AES-GCM encryption with PBKDF2 key derivation.
 *
 * @param apiKey - The API key to encrypt
 * @param secret - The secret key used for encryption
 * @returns A JSON string containing the encrypted data, salt, and IV
 */
export async function encryptAPIKey(
  apiKey: string,
  secret: string
): Promise<string> {
  const enc = new TextEncoder()
  const keyMaterial = await window.crypto.subtle.importKey(
    'raw',
    enc.encode(secret),
    { name: 'PBKDF2' },
    false,
    ['deriveKey']
  )
  const salt = window.crypto.getRandomValues(new Uint8Array(16))
  const key = await window.crypto.subtle.deriveKey(
    {
      name: 'PBKDF2',
      salt: salt,
      iterations: 100000,
      hash: 'SHA-256',
    },
    keyMaterial,
    { name: 'AES-GCM', length: 256 },
    false,
    ['encrypt']
  )
  const iv = window.crypto.getRandomValues(new Uint8Array(12))
  const ciphertext = await window.crypto.subtle.encrypt(
    { name: 'AES-GCM', iv },
    key,
    enc.encode(apiKey)
  )
  const base64 = (ab: ArrayBuffer) =>
    window.btoa(String.fromCharCode(...new Uint8Array(ab)))
  return JSON.stringify({
    salt: base64(salt.buffer),
    iv: base64(iv.buffer),
    data: base64(ciphertext),
  })
}
