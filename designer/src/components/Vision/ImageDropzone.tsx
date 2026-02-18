import { useState, useRef, useCallback } from 'react'
import { cn } from '@/lib/utils'
import FontIcon from '../../common/FontIcon'

interface ImageDropzoneProps {
  onFileSelected: (e: React.ChangeEvent<HTMLInputElement>) => void
  fileName?: string | null
  imagePreview?: string | null
  className?: string
  accept?: string
}

/**
 * Unified drag-and-drop + click-to-browse upload zone.
 * Matches the pattern used in Data and RAG panels.
 */
export function ImageDropzone({
  onFileSelected,
  fileName,
  imagePreview,
  className,
  accept = 'image/*',
}: ImageDropzoneProps) {
  const [isDragging, setIsDragging] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
  }, [])

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      e.stopPropagation()
      setIsDragging(false)

      const file = e.dataTransfer.files?.[0]
      if (!file) return

      // Create a synthetic event for the existing handler
      const dt = new DataTransfer()
      dt.items.add(file)
      const input = fileInputRef.current
      if (input) {
        input.files = dt.files
        input.dispatchEvent(new Event('change', { bubbles: true }))
      }
    },
    []
  )

  const handleClick = () => {
    fileInputRef.current?.click()
  }

  return (
    <div className={cn('flex flex-col gap-2', className)}>
      <input
        ref={fileInputRef}
        type="file"
        accept={accept}
        onChange={onFileSelected}
        className="hidden"
      />
      <button
        type="button"
        onClick={handleClick}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={cn(
          'w-full rounded-lg border-2 border-dashed transition-colors cursor-pointer text-left',
          'flex flex-col items-center justify-center gap-2 p-6',
          isDragging
            ? 'border-primary bg-primary/5'
            : 'border-border hover:border-primary/50 hover:bg-accent/30'
        )}
      >
        {imagePreview ? (
          <div className="flex flex-col items-center gap-2 w-full">
            <img
              src={imagePreview}
              alt="Preview"
              className="max-h-32 rounded-md object-contain"
            />
            <p className="text-xs text-muted-foreground">
              {fileName ?? 'Image loaded'} — click or drag to replace
            </p>
          </div>
        ) : (
          <>
            <FontIcon type="upload" className="w-8 h-8 text-muted-foreground" />
            <p className="text-sm text-muted-foreground">
              Drag and drop an image here, or click to browse
            </p>
          </>
        )}
      </button>
    </div>
  )
}
