/**
 * File validation utilities for secure file upload handling
 * 
 * Provides comprehensive validation including:
 * - Extension validation
 * - MIME type verification
 * - Magic number/file signature checking
 * 
 * This prevents malicious files from being uploaded by validating:
 * 1. File extensions match allowed types
 * 2. MIME types match extensions
 * 3. File content signatures match declared types (where applicable)
 */

// Valid file extensions and their corresponding MIME types
export const VALID_FILE_TYPES = {
  '.csv': ['text/csv', 'application/csv', 'text/plain'],
  '.xlsx': ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'],
  '.xls': ['application/vnd.ms-excel'],
  '.json': ['application/json', 'text/json', 'text/plain'],
  '.txt': ['text/plain'],
  '.tsv': ['text/tab-separated-values', 'text/plain'],
  '.parquet': ['application/octet-stream', 'application/vnd.apache.parquet'],
  '.pdf': ['application/pdf'],
  '.docx': ['application/vnd.openxmlformats-officedocument.wordprocessingml.document'],
  '.md': ['text/markdown', 'text/plain'],
  '.markdown': ['text/markdown', 'text/plain'],
} as const

// Magic number signatures for file type verification
const FILE_SIGNATURES = {
  pdf: { bytes: [0x25, 0x50, 0x44, 0x46], offset: 0 }, // %PDF
  zip: { bytes: [0x50, 0x4B, 0x03, 0x04], offset: 0 }, // PK.. (used by docx, xlsx)
  // Note: docx and xlsx are ZIP files, we verify MIME type separately
} as const

/**
 * Extracts the file extension from a filename (including the dot)
 * Returns only the final extension after the last dot to prevent
 * double-extension attacks (e.g., malicious.exe.txt)
 */
const getFileExtension = (filename: string): string | null => {
  const lower = filename.toLowerCase()
  const lastDotIndex = lower.lastIndexOf('.')
  
  // No extension found or dot is at the beginning
  if (lastDotIndex === -1 || lastDotIndex === 0) {
    return null
  }
  
  // Return extension including the dot (e.g., '.txt')
  return lower.substring(lastDotIndex)
}

/**
 * Validates file extension against allowed list
 * Only validates the final extension to prevent double-extension attacks
 */
export const hasValidExtension = (filename: string): boolean => {
  const extension = getFileExtension(filename)
  if (!extension) return false
  
  return Object.keys(VALID_FILE_TYPES).includes(extension)
}

/**
 * Validates MIME type matches the file extension
 */
export const hasValidMimeType = (file: File): boolean => {
  const extension = getFileExtension(file.name)
  
  if (!extension) return false
  
  const allowedMimes = VALID_FILE_TYPES[extension as keyof typeof VALID_FILE_TYPES]
  
  // Only allow empty MIME type for known safe extensions (e.g., .txt)
  const safeEmptyMimeExtensions = ['.txt', '.csv', '.md', '.markdown']
  if (file.type === '') {
    return safeEmptyMimeExtensions.includes(extension)
  }
  
  return (allowedMimes as readonly string[]).includes(file.type)
}

/**
 * Reads the first N bytes of a file
 */
const readFileHeader = (file: File, bytes: number): Promise<Uint8Array> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => {
      const arrayBuffer = reader.result as ArrayBuffer
      resolve(new Uint8Array(arrayBuffer))
    }
    reader.onerror = () => reject(reader.error)
    reader.readAsArrayBuffer(file.slice(0, bytes))
  })
}

/**
 * Checks if file header matches expected signature
 */
const matchesSignature = (
  header: Uint8Array,
  signature: { readonly bytes: readonly number[]; readonly offset: number }
): boolean => {
  if (header.length < signature.offset + signature.bytes.length) {
    return false
  }
  
  return signature.bytes.every(
    (byte, index) => header[signature.offset + index] === byte
  )
}

/**
 * Reads file magic number/signature to verify actual file type
 * Returns a promise that resolves to true if valid, false otherwise
 */
export const verifyFileSignature = async (file: File): Promise<boolean> => {
  // Only verify files where we have magic number signatures
  const extension = getFileExtension(file.name)
  
  if (!extension) return false
  
  // PDF verification
  if (extension === '.pdf') {
    try {
      const header = await readFileHeader(file, 4)
      return matchesSignature(header, FILE_SIGNATURES.pdf)
    } catch {
      return false
    }
  }
  
  // DOCX/XLSX verification (both are ZIP files)
  if (extension === '.docx' || extension === '.xlsx') {
    try {
      const header = await readFileHeader(file, 4)
      return matchesSignature(header, FILE_SIGNATURES.zip)
    } catch {
      return false
    }
  }
  
  // For other file types, we rely on extension + MIME validation
  // (text files, CSV, JSON, etc. don't have reliable magic numbers)
  return true
}

/**
 * Comprehensive file validation with extension, MIME type, and content verification
 * 
 * Performs 4-layer validation:
 * 1. Non-zero file size
 * 2. Valid extension from allowed list
 * 3. MIME type matches extension
 * 4. File signature/magic number verification (for supported types)
 * 
 * @param file - The File object to validate
 * @returns Promise resolving to validation result with optional reason for rejection
 */
export const isValidFile = async (file: File): Promise<{ valid: boolean; reason?: string }> => {
  // Check 1: Non-zero file size
  if (file.size === 0) {
    return { valid: false, reason: 'File is empty' }
  }
  
  // Check 2: Valid extension
  if (!hasValidExtension(file.name)) {
    return { valid: false, reason: 'Invalid file extension' }
  }
  
  // Check 3: MIME type matches extension
  if (!hasValidMimeType(file)) {
    return { valid: false, reason: 'File type does not match extension' }
  }
  
  // Check 4: File signature/magic number verification
  const signatureValid = await verifyFileSignature(file)
  if (!signatureValid) {
    return { valid: false, reason: 'File content does not match declared type' }
  }
  
  return { valid: true }
}

/**
 * Validates multiple files and returns categorized results
 * 
 * @param files - Array of File objects to validate
 * @returns Promise resolving to object with valid and invalid file arrays
 */
export const validateFiles = async (files: File[]) => {
  const validationResults = await Promise.all(
    files.map(async (file) => ({
      file,
      validation: await isValidFile(file),
    }))
  )

  const validFiles = validationResults
    .filter(result => result.validation.valid)
    .map(result => result.file)

  const invalidFiles = validationResults.filter(result => !result.validation.valid)

  return { validFiles, invalidFiles, validationResults }
}

