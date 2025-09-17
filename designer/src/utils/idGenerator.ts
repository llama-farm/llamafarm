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
     // Never fall back to insecure randomness for session IDs; fail explicitly
     throw new Error('Secure random generation for session ID failed: crypto.getRandomValues is not available in this environment.');
  }
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

export default {
  generateMessageId,
};
