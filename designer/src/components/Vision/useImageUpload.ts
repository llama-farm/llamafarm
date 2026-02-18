import { useState, useCallback } from 'react'

/**
 * Shared hook for image upload and base64 conversion
 */
export function useImageUpload() {
  const [imageBase64, setImageBase64] = useState<string | null>(null)
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const [fileName, setFileName] = useState<string | null>(null)

  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    setFileName(file.name)
    const reader = new FileReader()
    reader.onload = () => {
      const result = reader.result as string
      setImagePreview(result)
      // Strip data URL prefix for API
      const base64 = result.split(',')[1]
      setImageBase64(base64)
    }
    reader.readAsDataURL(file)
  }, [])

  const clear = useCallback(() => {
    setImageBase64(null)
    setImagePreview(null)
    setFileName(null)
  }, [])

  return { imageBase64, imagePreview, fileName, handleFileChange, clear }
}
