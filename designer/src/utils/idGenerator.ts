/**
 * Secure ID generation utilities using cryptographically secure random numbers
 * Replaces insecure Math.random() with crypto.getRandomValues()
 */

/**
 * Generates a cryptographically secure random string
 * @param length - Length of the random string to generate
 * @returns Secure random string using base36 encoding
 */
function generateSecureRandomString(length: number = 9): string {
  try {
    // Use crypto.getRandomValues for secure random generation
    const array = new Uint32Array(Math.ceil(length / 6)); // Each Uint32 gives ~6 base36 chars
    crypto.getRandomValues(array);
    
    // Convert to base36 string and concatenate
    let result = '';
    for (const num of array) {
      result += num.toString(36);
    }
    
    // Trim to exact length and ensure we have enough characters
    return result.substring(0, length).padEnd(length, '0');
  } catch (error) {
    // Fallback for environments without crypto support (shouldn't happen in modern browsers)
    console.warn('crypto.getRandomValues not available, falling back to less secure method');
    return Array.from({ length }, () => Math.floor(Math.random() * 36).toString(36)).join('');
  }
}

/**
 * Generate a unique session ID with cryptographically secure random numbers
 * Format: chat-{timestamp}-{secureRandom}
 * @returns Secure session ID string
 */
export function generateSessionId(): string {
  const timestamp = Date.now();
  const randomPart = generateSecureRandomString(9);
  return `chat-${timestamp}-${randomPart}`;
}

/**
 * Generate a unique message ID with cryptographically secure random numbers
 * Format: msg-{timestamp}-{secureRandom}
 * @returns Secure message ID string
 */
export function generateMessageId(): string {
  const timestamp = Date.now();
  const randomPart = generateSecureRandomString(9);
  return `msg-${timestamp}-${randomPart}`;
}

/**
 * Generate a generic secure ID with custom prefix
 * @param prefix - Prefix for the ID (e.g., 'user', 'session', 'message')
 * @param includeTimestamp - Whether to include timestamp in the ID
 * @returns Secure ID string
 */
export function generateSecureId(prefix: string, includeTimestamp: boolean = true): string {
  const randomPart = generateSecureRandomString(9);
  
  if (includeTimestamp) {
    const timestamp = Date.now();
    return `${prefix}-${timestamp}-${randomPart}`;
  }
  
  return `${prefix}-${randomPart}`;
}

export default {
  generateSessionId,
  generateMessageId,
  generateSecureId,
};
