/**
 * Home Page Utilities for URL encoding/decoding
 *
 * This module provides utilities for encoding and decoding messages in URL parameters.
 */

/**
 * Generate a URL-safe encoded message for navigation
 *
 * @param message - The message to encode
 * @returns URL-encoded message
 */
export function encodeMessageForUrl(message: string): string {
  return encodeURIComponent(message.trim())
}

/**
 * Decode a message from URL parameters
 *
 * @param encodedMessage - The URL-encoded message
 * @returns Decoded message
 */
export function decodeMessageFromUrl(encodedMessage: string): string {
  try {
    return decodeURIComponent(encodedMessage)
  } catch (error) {
    console.warn('Failed to decode message from URL:', error)
    return ''
  }
}
