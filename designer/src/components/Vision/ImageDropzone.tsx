import { useState, useRef, useCallback, useEffect } from 'react'
import { cn } from '@/lib/utils'
import { X } from 'lucide-react'
import FontIcon from '../../common/FontIcon'

interface ImageDropzoneProps {
  onFileSelected: (e: React.ChangeEvent<HTMLInputElement>) => void
  onFileDirect?: (file: File) => void
  fileName?: string | null
  fileSize?: number | null
  imagePreview?: string | null
  onClear?: () => void
  className?: string
  accept?: string
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

/**
 * Unified drag-and-drop + click-to-browse + paste upload zone.
 * Matches the pattern used in Data and RAG panels.
 */
export function ImageDropzone({
  onFileSelected,
  onFileDirect,
  fileName,
  fileSize,
  imagePreview,
  onClear,
  className,
  accept = 'image/jpeg,image/png,image/webp,image/gif',
}: ImageDropzoneProps) {
  const [isDragging, setIsDragging] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  const triggerFileFromFile = useCallback(
    (file: File) => {
      if (onFileDirect) {
        onFileDirect(file)
        return
      }
      const dt = new DataTransfer()
      dt.items.add(file)
      const input = fileInputRef.current
      if (input) {
        input.files = dt.files
        input.dispatchEvent(new Event('change', { bubbles: true }))
      }
    },
    [onFileDirect]
  )

  // Clipboard paste support
  useEffect(() => {
    const handlePaste = (e: ClipboardEvent) => {
      const items = e.clipboardData?.items
      if (!items) return
      for (const item of Array.from(items)) {
        if (item.type.startsWith('image/')) {
          e.preventDefault()
          const file = item.getAsFile()
          if (file) triggerFileFromFile(file)
          return
        }
      }
    }
    document.addEventListener('paste', handlePaste)
    return () => document.removeEventListener('paste', handlePaste)
  }, [triggerFileFromFile])

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
      triggerFileFromFile(file)
    },
    [triggerFileFromFile]
  )

  const handleClick = () => {
    fileInputRef.current?.click()
  }

  return (
    <div ref={containerRef} className={cn('flex flex-col gap-2', className)}>
      <input
        ref={fileInputRef}
        type="file"
        accept={accept}
        onChange={onFileSelected}
        className="hidden"
      />

      {imagePreview ? (
        <div className="relative rounded-lg border border-border p-3">
          <div className="flex items-center gap-3">
            <img
              src={imagePreview}
              alt="Preview"
              className="h-20 w-20 rounded-md object-cover flex-shrink-0"
            />
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium truncate">{fileName ?? 'Image loaded'}</p>
              {fileSize != null && (
                <p className="text-xs text-muted-foreground">{formatFileSize(fileSize)}</p>
              )}
              <button
                type="button"
                onClick={handleClick}
                className="text-xs text-primary hover:underline mt-1"
              >
                Replace image
              </button>
            </div>
            {onClear && (
              <button
                type="button"
                onClick={onClear}
                className="p-1 rounded-md hover:bg-destructive/10 text-muted-foreground hover:text-destructive transition-colors"
                title="Remove image"
              >
                <X className="w-4 h-4" />
              </button>
            )}
          </div>
        </div>
      ) : (
        <button
          type="button"
          onClick={handleClick}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className={cn(
            'w-full rounded-lg border-2 border-dashed transition-colors cursor-pointer text-left',
            'flex flex-col items-center justify-center gap-2 p-8',
            isDragging
              ? 'border-primary bg-primary/5'
              : 'border-border hover:border-primary/50 hover:bg-accent/30'
          )}
        >
          <FontIcon type="upload" className="w-8 h-8 text-muted-foreground" />
          <p className="text-sm text-muted-foreground text-center">
            Drag and drop an image here, or click to browse
          </p>
          <p className="text-xs text-muted-foreground/60">
            JPG, PNG, WEBP, GIF • Or paste from clipboard
          </p>
        </button>
      )}
    </div>
  )
}
