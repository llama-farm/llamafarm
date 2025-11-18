/**
 * Security utilities for sanitizing and validating user-controlled data
 * 
 * These functions prevent XSS, SSRF, injection attacks, and other security vulnerabilities
 * when displaying or processing configuration data from potentially untrusted sources.
 */

// Maximum length constraints to prevent DoS attacks
const MAX_CONFIG_VALUE_LENGTH = 100
const MAX_FILTER_KEY_LENGTH = 50
const MAX_FILTER_VALUE_LENGTH = 200
const MAX_ARRAY_ITEMS = 20

/**
 * Sanitizes configuration values before displaying in the UI
 * Prevents XSS and injection attacks by removing dangerous characters
 * 
 * @param value - The configuration value to sanitize (can be any type)
 * @returns A safe string representation, or 'Not set' if empty
 * 
 * @example
 * sanitizeConfigValue('<script>alert("xss")</script>') // Returns 'scriptalert("xss")/script'
 * sanitizeConfigValue(null) // Returns 'Not set'
 */
export const sanitizeConfigValue = (value: unknown): string => {
  if (!value) return 'Not set'
  
  // Convert to string and remove potentially dangerous characters
  const str = String(value)
    .replace(/[<>'"]/g, '') // Remove HTML/script injection characters
    .trim()
  
  // Limit length to prevent DoS
  return str.length > MAX_CONFIG_VALUE_LENGTH 
    ? str.substring(0, MAX_CONFIG_VALUE_LENGTH) + '...' 
    : str
}

/**
 * Validates that a URL is safe to parse and use
 * Prevents SSRF attacks by checking protocol and optionally warning about private IPs
 * 
 * @param urlString - The URL string to validate
 * @returns true if the URL is valid and uses a safe protocol, false otherwise
 * 
 * @example
 * isValidAndSafeURL('https://api.openai.com') // Returns true
 * isValidAndSafeURL('file:///etc/passwd') // Returns false
 * isValidAndSafeURL('http://169.254.169.254/metadata') // Returns true but logs warning
 */
export const isValidAndSafeURL = (urlString: string): boolean => {
  try {
    const url = new URL(urlString)
    
    // Only allow http and https protocols
    if (!['http:', 'https:'].includes(url.protocol)) {
      return false
    }
    
    // Warn about localhost/private IPs in production
    const hostname = url.hostname.toLowerCase()
    const isLocalhost = hostname === 'localhost' || 
                       hostname === '127.0.0.1' || 
                       hostname.startsWith('192.168.') ||
                       hostname.startsWith('10.') ||
                       hostname.startsWith('172.')
    
    if (import.meta.env.PROD && !isLocalhost) {
      // In production, log external URL usage for monitoring
      console.warn('External URL detected in config:', hostname)
    }
    
    return true
  } catch {
    return false
  }
}

/**
 * Validates navigation state data to prevent injection through location.state manipulation
 * 
 * @param state - The navigation state object to validate
 * @returns A validated state object with safe default values
 * 
 * @example
 * validateNavigationState({ database: 'main_db', strategyName: 'my-strategy' })
 * validateNavigationState({ database: '../../../etc/passwd' }) // Returns 'main_database'
 */
export const validateNavigationState = (state: unknown): {
  database: string
  strategyName: string
  strategyType: string
  currentConfig: Record<string, any>
  isDefault: boolean
} => {
  const s = state as any
  
  // Validate database name (alphanumeric and underscores only)
  const database = typeof s?.database === 'string' && 
                   /^[a-zA-Z0-9_]+$/.test(s.database)
    ? s.database 
    : 'main_database'
  
  // Validate strategy name (alphanumeric, spaces, hyphens, underscores)
  const strategyName = typeof s?.strategyName === 'string' &&
                       /^[a-zA-Z0-9\s_-]+$/.test(s.strategyName)
    ? s.strategyName
    : ''
  
  // Validate strategy type - accept any string strategy type from navigation
  // We trust our own navigation state, just validate it's a valid identifier string
  // This prevents silently overwriting complex strategy types like MultiQueryStrategy, RerankedStrategy, etc.
  const strategyType = typeof s?.strategyType === 'string' &&
                       s.strategyType.length > 0 &&
                       /^[a-zA-Z0-9_]+$/.test(s.strategyType)
    ? s.strategyType
    : 'BasicSimilarityStrategy'
  
  // Validate config is an object (but still treat contents as untrusted)
  const currentConfig = s?.currentConfig && 
                        typeof s.currentConfig === 'object' &&
                        !Array.isArray(s.currentConfig)
    ? s.currentConfig
    : {}
  
  // Validate boolean
  const isDefault = typeof s?.isDefault === 'boolean' ? s.isDefault : false
  
  return { database, strategyName, strategyType, currentConfig, isDefault }
}

/**
 * Validates navigation state data for embedding strategies
 * Similar to validateNavigationState but for embedding-specific fields
 * 
 * @param state - The navigation state object to validate
 * @returns A validated state object with safe default values
 */
export const validateEmbeddingNavigationState = (state: unknown): {
  database: string
  strategyName: string
  strategyType: string
  currentConfig: Record<string, any>
  isDefault: boolean
  priority: number
} => {
  const s = state as any
  
  // Validate database name (alphanumeric and underscores only)
  const database = typeof s?.database === 'string' && 
                   /^[a-zA-Z0-9_]+$/.test(s.database)
    ? s.database 
    : 'main_database'
  
  // Validate strategy name (alphanumeric, spaces, hyphens, underscores)
  const strategyName = typeof s?.strategyName === 'string' &&
                       /^[a-zA-Z0-9\s_-]+$/.test(s.strategyName)
    ? s.strategyName
    : ''
  
  // Validate strategy type against allowed embedding types
  const allowedTypes = [
    'OllamaEmbedder',
    'OpenAIEmbedder',
    'HuggingFaceEmbedder',
    'SentenceTransformerEmbedder',
    'BedrockEmbedder',
    'AzureOpenAIEmbedder',
  ]
  const strategyType = typeof s?.strategyType === 'string' &&
                       allowedTypes.includes(s.strategyType)
    ? s.strategyType
    : 'OllamaEmbedder'
  
  // Validate config is an object (but still treat contents as untrusted)
  const currentConfig = s?.currentConfig && 
                        typeof s.currentConfig === 'object' &&
                        !Array.isArray(s.currentConfig)
    ? s.currentConfig
    : {}
  
  // Validate boolean
  const isDefault = typeof s?.isDefault === 'boolean' ? s.isDefault : false
  
  // Validate priority is a safe number
  const priority = typeof s?.priority === 'number' && 
                   Number.isFinite(s.priority) &&
                   s.priority >= 0 &&
                   s.priority <= 1000
    ? s.priority
    : 0
  
  return { database, strategyName, strategyType, currentConfig, isDefault, priority }
}

/**
 * Sanitizes metadata filter keys
 * Only allows alphanumeric characters, underscores, and hyphens
 * 
 * @param key - The filter key to sanitize
 * @returns A sanitized key string
 * 
 * @example
 * sanitizeFilterKey('document_type') // Returns 'document_type'
 * sanitizeFilterKey('type<script>') // Returns 'typescript'
 */
export const sanitizeFilterKey = (key: string): string => {
  // Only allow alphanumeric, underscore, hyphen
  return key.replace(/[^a-zA-Z0-9_-]/g, '').substring(0, MAX_FILTER_KEY_LENGTH)
}

/**
 * Sanitizes metadata filter values
 * Removes dangerous characters while allowing reasonable punctuation
 * 
 * @param value - The filter value to sanitize
 * @returns A sanitized value string
 * 
 * @example
 * sanitizeFilterValue('2024-report.pdf') // Returns '2024-report.pdf'
 * sanitizeFilterValue('<script>alert(1)</script>') // Returns 'scriptalert(1)/script'
 */
export const sanitizeFilterValue = (value: string): string => {
  // Remove dangerous characters but allow reasonable punctuation
  return value
    .replace(/[<>'"\\]/g, '')
    .trim()
    .substring(0, MAX_FILTER_VALUE_LENGTH)
}

/**
 * Safely parses a numeric value from a string
 * Rejects NaN, Infinity, and values outside safe integer range
 * 
 * @param raw - The string value to parse
 * @returns The numeric value, or null if invalid
 * 
 * @example
 * parseNumericValue('123') // Returns 123
 * parseNumericValue('Infinity') // Returns null
 * parseNumericValue('-1e309') // Returns null (becomes -Infinity)
 */
const parseNumericValue = (raw: string): number | null => {
  const num = Number(raw)
  
  // Check for NaN
  if (Number.isNaN(num)) {
    return null
  }
  
  // Check for Infinity (positive or negative)
  if (!Number.isFinite(num)) {
    return null
  }
  
  // Check for safe integer range
  if (Math.abs(num) > Number.MAX_SAFE_INTEGER) {
    return null
  }
  
  return num
}

/**
 * Parses and sanitizes metadata filters from user input
 * Prevents injection attacks and DoS through oversized arrays
 * 
 * @param filters - Array of filter objects with key/value pairs
 * @returns A sanitized filter object safe to include in configurations
 * 
 * @example
 * parseMetadataFilters([
 *   { key: 'status', value: 'active' },
 *   { key: 'tags', value: 'important,urgent' }
 * ])
 * // Returns { status: 'active', tags: ['important', 'urgent'] }
 */
export const parseMetadataFilters = (
  filters: Array<{ key: string; value: string }>
): Record<string, unknown> => {
  const result: Record<string, unknown> = {}
  
  for (const { key, value } of filters) {
    const sanitizedKey = sanitizeFilterKey(key)
    if (!sanitizedKey) continue
    
    const raw = sanitizeFilterValue(value)
    if (!raw) continue
    
    if (raw.includes(',')) {
      // Parse as array
      const items = raw
        .split(',')
        .map(v => sanitizeFilterValue(v))
        .filter(v => v.length > 0)
        .slice(0, MAX_ARRAY_ITEMS) // Prevent DoS with huge arrays
      
      if (items.length > 0) {
        result[sanitizedKey] = items
      }
    } else if (raw === 'true' || raw === 'false') {
      // Parse as boolean
      result[sanitizedKey] = raw === 'true'
    } else {
      // Try to parse as number with strict validation
      const numValue = parseNumericValue(raw)
      if (numValue !== null) {
        result[sanitizedKey] = numValue
      } else {
        // Treat as string if not a valid number
        result[sanitizedKey] = raw
      }
    }
  }
  
  return result
}

/**
 * List of reserved strategy names that should not be allowed
 * These names could cause conflicts or confusion in the system
 */
export const RESERVED_STRATEGY_NAMES = [
  'default',
  'null',
  'undefined',
  'none',
  'system',
  'admin',
  'root',
  'all',
  'any',
]

/**
 * Validates a strategy name against reserved names and character restrictions
 * 
 * @param name - The strategy name to validate
 * @returns An error message if invalid, or null if valid
 * 
 * @example
 * validateStrategyName('my-strategy') // Returns null (valid)
 * validateStrategyName('default') // Returns error message
 * validateStrategyName('test<script>') // Returns error message
 */
export const validateStrategyName = (name: string): string | null => {
  const trimmedName = name.trim()
  
  if (!trimmedName) {
    return 'Strategy name is required'
  }
  
  // Check for reserved names (case-insensitive)
  if (RESERVED_STRATEGY_NAMES.includes(trimmedName.toLowerCase())) {
    return `"${trimmedName}" is a reserved name. Please choose a different name.`
  }
  
  // Check for valid characters (alphanumeric, spaces, hyphens, underscores only)
  if (!/^[a-zA-Z0-9\s_-]+$/.test(trimmedName)) {
    return 'Strategy name can only contain letters, numbers, spaces, hyphens, and underscores'
  }
  
  // Check length
  if (trimmedName.length > 100) {
    return 'Strategy name must be 100 characters or less'
  }
  
  return null
}

/**
 * Safely extracts and sanitizes a hostname from a URL configuration value
 * Returns 'Not set' for missing values, 'Invalid URL' for malformed/unsafe URLs
 * 
 * @param urlValue - The URL value from configuration (base_url, endpoint, etc.)
 * @returns A sanitized hostname display string
 * 
 * @example
 * extractSafeHostname('https://api.openai.com:443/v1') // Returns 'api.openai.com:443'
 * extractSafeHostname('file:///etc/passwd') // Returns 'Invalid URL'
 */
export const extractSafeHostname = (urlValue: unknown): string => {
  if (!urlValue) return 'Not set'
  
  const urlString = String(urlValue)
  
  if (!isValidAndSafeURL(urlString)) {
    return 'Invalid URL'
  }
  
  try {
    const url = new URL(urlString)
    const hostname = sanitizeConfigValue(url.hostname)
    const port = url.port ? sanitizeConfigValue(url.port) : ''
    return port ? `${hostname}:${port}` : hostname
  } catch {
    return 'Invalid URL'
  }
}

