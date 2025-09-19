// Minimal Web Crypto helpers for encrypting/decrypting JSON payloads for localStorage
// NOTE: Client-side encryption only obscures data; do not rely on it for true secrecy.

export async function encryptJson(
  payload: unknown,
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
    enc.encode(JSON.stringify(payload))
  )
  return JSON.stringify({
    salt: toBase64(salt),
    iv: toBase64(iv),
    data: toBase64(ciphertext),
  })
}

export async function decryptJson<T = unknown>(
  ciphertextJson: string,
  secret: string
): Promise<T | null> {
  try {
    const parsed = JSON.parse(ciphertextJson)
    if (!parsed || typeof parsed !== 'object') return null
    const { salt, iv, data } = parsed as any
    if (!salt || !iv || !data) return null

    const enc = new TextEncoder()
    const keyMaterial = await window.crypto.subtle.importKey(
      'raw',
      enc.encode(secret),
      { name: 'PBKDF2' },
      false,
      ['deriveKey']
    )
    const key = await window.crypto.subtle.deriveKey(
      {
        name: 'PBKDF2',
        salt: fromBase64(salt),
        iterations: 100000,
        hash: 'SHA-256',
      },
      keyMaterial,
      { name: 'AES-GCM', length: 256 },
      false,
      ['decrypt']
    )
    const plaintext = await window.crypto.subtle.decrypt(
      { name: 'AES-GCM', iv: fromBase64(iv) },
      key,
      fromBase64(data)
    )
    const decoded = new TextDecoder().decode(new Uint8Array(plaintext))
    return JSON.parse(decoded) as T
  } catch {
    return null
  }
}

function toBase64(buf: ArrayBuffer | Uint8Array): string {
  const u8 = buf instanceof Uint8Array ? buf : new Uint8Array(buf)
  return window.btoa(String.fromCharCode(...u8))
}

function fromBase64(b64: string): ArrayBuffer {
  const binary = window.atob(b64)
  const out = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i)
  return out.buffer
}

// Derive a client-side secret for local obfuscation purposes.
// Prefer an environment-provided value; fallback to a random per-session secret.
let cachedClientSecret: string | null = null
export function getClientSideSecret(): string {
  if (cachedClientSecret) return cachedClientSecret
  const env = (import.meta as any)?.env?.VITE_LF_ENCRYPTION_SECRET
  if (typeof env === 'string' && env.trim().length >= 16) {
    cachedClientSecret = env
    return cachedClientSecret
  }
  // per-session random secret to avoid guessable constants
  const rand = crypto.getRandomValues(new Uint8Array(24))
  cachedClientSecret = toBase64(rand)
  return cachedClientSecret
}
