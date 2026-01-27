/**
 * Data status selection screen (Step 2 of wizard)
 * Shows sample dataset picker when "sample-data" is selected
 */

import { useState, useCallback, useRef } from 'react'
import { cn } from '@/lib/utils'
import { Check, Gamepad2, FolderOpen, Search, Upload, X, FileText, ExternalLink } from 'lucide-react'
import { Input } from '@/components/ui/input'
import type { DataStatus, ProjectType, OnboardingUploadedFile } from '../../types/onboarding'
import { getDemosByProjectType } from '../../config/demos'
import { validateDatasetName } from '../../utils/datasetValidation'

interface DataStatusSelectorProps {
  selected: DataStatus | null
  onSelect: (status: DataStatus) => void
  selectedSampleDataset: string | null
  onSelectSampleDataset: (demoId: string | null) => void
  projectType: ProjectType | null
  uploadedFiles: OnboardingUploadedFile[]
  onUploadedFilesChange: (files: OnboardingUploadedFile[]) => void
  datasetName: string | null
  onDatasetNameChange: (name: string | null) => void
  // File storage actions (actual File objects stored in context ref)
  onAddActualFiles: (files: File[]) => void
  onRemoveActualFile: (index: number) => void
  className?: string
}

// Format file size for display
function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

const dataStatusOptions = [
  {
    id: 'has-data' as DataStatus,
    icon: <FolderOpen className="w-5 h-5" />,
    title: 'Yep, ready to go!',
    description: 'I have PDFs, docs, or text files ready to upload',
    iconBg: 'bg-emerald-500/20 text-emerald-600 dark:bg-emerald-500/25 dark:text-emerald-300',
  },
  {
    id: 'sample-data' as DataStatus,
    icon: <Gamepad2 className="w-5 h-5" />,
    title: 'Let me kick the tires first',
    description: 'Use sample data so I can see how it works',
    iconBg: 'bg-violet-500/20 text-violet-600 dark:bg-violet-500/25 dark:text-violet-300',
  },
  {
    id: 'need-data' as DataStatus,
    icon: <Search className="w-5 h-5" />,
    title: 'Still gathering my data',
    description: 'Point me to some resources to get started',
    iconBg: 'bg-amber-500/20 text-amber-600 dark:bg-amber-500/25 dark:text-amber-300',
  },
]

// HuggingFace task filter URLs for different project types
const HF_TASK_URLS: Record<ProjectType, { url: string; label: string }> = {
  'doc-qa': {
    url: 'https://huggingface.co/datasets?task_categories=task_categories%3Atext-generation&sort=downloads',
    label: 'text generation datasets',
  },
  'exploring': {
    url: 'https://huggingface.co/datasets?task_categories=task_categories%3Atext-generation&sort=downloads',
    label: 'text datasets',
  },
  'classifier': {
    url: 'https://huggingface.co/datasets?task_categories=task_categories%3Atext-classification&sort=downloads',
    label: 'text classification datasets',
  },
  'anomaly': {
    url: 'https://huggingface.co/datasets?search=anomaly+detection&sort=downloads',
    label: 'anomaly detection datasets',
  },
  'doc-scan': {
    url: 'https://huggingface.co/datasets?task_categories=task_categories%3Atoken-classification&sort=downloads',
    label: 'entity extraction datasets',
  },
}

export function DataStatusSelector({
  selected,
  onSelect,
  selectedSampleDataset,
  onSelectSampleDataset,
  projectType,
  uploadedFiles,
  onUploadedFilesChange,
  datasetName,
  onDatasetNameChange,
  onAddActualFiles,
  onRemoveActualFile,
  className,
}: DataStatusSelectorProps) {
  // Drag-and-drop state
  const [isDragging, setIsDragging] = useState(false)
  const [datasetNameError, setDatasetNameError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const dropZoneRef = useRef<HTMLDivElement>(null)

  const hfTaskInfo = projectType ? HF_TASK_URLS[projectType] : null

  // Drag-and-drop handlers
  const handleDragEnter = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(true)
  }, [])

  const handleDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    e.dataTransfer.dropEffect = 'copy'
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    const rect = dropZoneRef.current?.getBoundingClientRect()
    const isLeavingZone =
      rect &&
      (e.clientX <= rect.left ||
        e.clientX >= rect.right ||
        e.clientY <= rect.top ||
        e.clientY >= rect.bottom)
    if (isLeavingZone) {
      setIsDragging(false)
    }
  }, [])

  const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
    const files = Array.from(e.dataTransfer.files)
    if (files.length > 0) {
      const fileInfos: OnboardingUploadedFile[] = files.map(f => ({
        name: f.name,
        size: f.size,
        type: f.type,
      }))
      onUploadedFilesChange([...uploadedFiles, ...fileInfos])
      // Store actual files in the onboarding context ref (not window)
      onAddActualFiles(files)
    }
  }, [uploadedFiles, onUploadedFilesChange, onAddActualFiles])

  const handleFileInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || [])
    if (files.length > 0) {
      const fileInfos: OnboardingUploadedFile[] = files.map(f => ({
        name: f.name,
        size: f.size,
        type: f.type,
      }))
      onUploadedFilesChange([...uploadedFiles, ...fileInfos])
      // Store actual files in the onboarding context ref (not window)
      onAddActualFiles(files)
    }
    // Reset input so same file can be selected again
    e.target.value = ''
  }, [uploadedFiles, onUploadedFilesChange, onAddActualFiles])

  const removeFile = useCallback((index: number) => {
    const newFiles = uploadedFiles.filter((_, i) => i !== index)
    onUploadedFilesChange(newFiles)
    // Also remove from the onboarding context ref
    onRemoveActualFile(index)
  }, [uploadedFiles, onUploadedFilesChange, onRemoveActualFile])

  return (
    <div className={cn('space-y-6', className)}>
      <div className="text-center">
        <h2 className="text-2xl font-semibold text-foreground">
          Do you have data to work with?
        </h2>
        <p className="mt-2 text-muted-foreground">
          Our local models are small—they need your data to shine.
        </p>
      </div>

      <div className="space-y-3">
        {dataStatusOptions.map(option => {
          const isSelected = selected === option.id
          const showSamplePicker = option.id === 'sample-data' && isSelected

          // For sample-data option, wrap everything in an expandable card
          if (option.id === 'sample-data') {
            return (
              <div
                key={option.id}
                className={cn(
                  'rounded-xl border-2 transition-all duration-200',
                  isSelected
                    ? 'border-primary bg-card shadow-md ring-1 ring-primary/20'
                    : 'border-border bg-card hover:bg-white dark:hover:bg-card hover:border-primary/40'
                )}
              >
                <button
                  type="button"
                  onClick={() => {
                    onSelect(option.id)
                  }}
                  className={cn(
                    'group w-full flex items-center gap-4 p-4 text-left transition-all duration-200',
                    'focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 rounded-xl',
                    !isSelected && 'hover:scale-[1.01]'
                  )}
                  role="radio"
                  aria-checked={isSelected}
                >
                  {/* Icon */}
                  <div
                    className={cn(
                      'flex-shrink-0 w-12 h-12 rounded-xl flex items-center justify-center transition-all duration-200',
                      option.iconBg,
                      isSelected ? 'scale-105' : 'group-hover:scale-105'
                    )}
                  >
                    {option.icon}
                  </div>

                  <div className="flex-1 min-w-0">
                    <div className="font-medium text-foreground">{option.title}</div>
                    <div className="text-sm text-muted-foreground mt-0.5">
                      {option.description}
                    </div>
                  </div>

                  {/* Selection indicator */}
                  <div
                    className={cn(
                      'flex-shrink-0 w-6 h-6 rounded-full border-2 flex items-center justify-center transition-all duration-200',
                      isSelected
                        ? 'border-primary bg-primary'
                        : 'border-muted-foreground/30 group-hover:border-primary/50'
                    )}
                  >
                    {isSelected && <Check className="w-4 h-4 text-primary-foreground" />}
                  </div>
                </button>

                {/* Sample dataset picker - expands inside the card */}
                {showSamplePicker && (
                  <div className="px-4 pb-4 animate-in fade-in slide-in-from-top-2 duration-300">
                    <div className="pt-3 border-t border-primary/20">
                      {getDemosByProjectType(projectType).length > 0 ? (
                        <>
                          <p className="text-sm text-muted-foreground mb-3">
                            Pick a sample dataset to start with:
                          </p>
                          <div className="space-y-2">
                            {getDemosByProjectType(projectType).map(demo => {
                              const isDemoSelected = selectedSampleDataset === demo.id
                              return (
                                <button
                                  type="button"
                                  key={demo.id}
                                  onClick={e => {
                                    e.stopPropagation()
                                    onSelectSampleDataset(demo.id)
                                  }}
                                  className={cn(
                                    'w-full flex items-center gap-4 p-3 rounded-lg border text-left transition-all duration-200',
                                    'hover:shadow-sm',
                                    isDemoSelected
                                      ? 'border-primary bg-primary/5 dark:bg-primary/10'
                                      : 'border-border bg-card/50 hover:bg-white dark:hover:bg-card hover:border-primary/40'
                                  )}
                                >
                                  <div className="flex-shrink-0 w-10 h-10 rounded-lg bg-violet-500/20 dark:bg-violet-500/25 flex items-center justify-center text-xl">
                                    {demo.icon}
                                  </div>
                                  <div className="flex-1 min-w-0">
                                    <div className="font-medium text-sm text-foreground">
                                      {demo.displayName}
                                    </div>
                                    <div className="text-xs text-muted-foreground mt-0.5">
                                      {demo.description}
                                    </div>
                                  </div>
                                  {/* Selection indicator */}
                                  <div
                                    className={cn(
                                      'flex-shrink-0 w-5 h-5 rounded-full border-2 flex items-center justify-center transition-all duration-200',
                                      isDemoSelected
                                        ? 'border-primary bg-primary'
                                        : 'border-muted-foreground/30'
                                    )}
                                  >
                                    {isDemoSelected && (
                                      <Check className="w-3 h-3 text-primary-foreground" />
                                    )}
                                  </div>
                                </button>
                              )
                            })}
                          </div>
                        </>
                      ) : (
                        <p className="text-sm text-muted-foreground py-2">
                          Sample datasets for this project type coming soon! Select "Yep, ready to go!" to upload your own data.
                        </p>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )
          }

          // For need-data option, show helpful links to browse datasets
          if (option.id === 'need-data') {
            return (
              <div
                key={option.id}
                className={cn(
                  'rounded-xl border-2 transition-all duration-200',
                  isSelected
                    ? 'border-primary bg-card shadow-md ring-1 ring-primary/20'
                    : 'border-border bg-card hover:bg-white dark:hover:bg-card hover:border-primary/40'
                )}
              >
                <button
                  type="button"
                  onClick={() => {
                    onSelect(option.id)
                    onSelectSampleDataset(null)
                  }}
                  className={cn(
                    'group w-full flex items-center gap-4 p-4 text-left transition-all duration-200',
                    'focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 rounded-xl',
                    !isSelected && 'hover:scale-[1.01]'
                  )}
                  role="radio"
                  aria-checked={isSelected}
                >
                  {/* Icon */}
                  <div
                    className={cn(
                      'flex-shrink-0 w-12 h-12 rounded-xl flex items-center justify-center transition-all duration-200',
                      option.iconBg,
                      isSelected ? 'scale-105' : 'group-hover:scale-105'
                    )}
                  >
                    {option.icon}
                  </div>

                  <div className="flex-1 min-w-0">
                    <div className="font-medium text-foreground">{option.title}</div>
                    <div className="text-sm text-muted-foreground mt-0.5">
                      {option.description}
                    </div>
                  </div>

                  {/* Selection indicator */}
                  <div
                    className={cn(
                      'flex-shrink-0 w-6 h-6 rounded-full border-2 flex items-center justify-center transition-all duration-200',
                      isSelected
                        ? 'border-primary bg-primary'
                        : 'border-muted-foreground/30 group-hover:border-primary/50'
                    )}
                  >
                    {isSelected && <Check className="w-4 h-4 text-primary-foreground" />}
                  </div>
                </button>

                {/* Resource links - expands inside the card */}
                {isSelected && hfTaskInfo && (
                  <div className="px-4 pb-4 pt-1 animate-in fade-in slide-in-from-top-2 duration-300">
                    <div className="space-y-3">
                      <p className="text-sm text-muted-foreground">
                        Browse Hugging Face for {hfTaskInfo.label}:
                      </p>
                      <a
                        href={hfTaskInfo.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex items-center gap-2 p-3 rounded-lg border border-border bg-card/50 hover:bg-white dark:hover:bg-card hover:border-primary/40 transition-all"
                      >
                        <span className="text-lg">🤗</span>
                        <div className="flex-1">
                          <div className="text-sm font-medium text-foreground">
                            Browse {hfTaskInfo.label}
                          </div>
                          <div className="text-xs text-muted-foreground">
                            huggingface.co/datasets
                          </div>
                        </div>
                        <ExternalLink className="w-4 h-4 text-muted-foreground" />
                      </a>
                      <p className="text-xs text-muted-foreground">
                        Download a dataset and upload it on the Data page after setup.
                      </p>
                    </div>
                  </div>
                )}
              </div>
            )
          }

          // has-data option with expandable drop zone
          if (option.id === 'has-data') {
            return (
              <div
                key={option.id}
                className={cn(
                  'rounded-xl border-2 transition-all duration-200',
                  isSelected
                    ? 'border-primary bg-card shadow-md ring-1 ring-primary/20'
                    : 'border-border bg-card hover:bg-white dark:hover:bg-card hover:border-primary/40'
                )}
              >
                <button
                  type="button"
                  onClick={() => {
                    onSelect(option.id)
                    onSelectSampleDataset(null)
                  }}
                  className={cn(
                    'group w-full flex items-center gap-4 p-4 text-left transition-all duration-200',
                    'focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 rounded-xl',
                    !isSelected && 'hover:scale-[1.01]'
                  )}
                  role="radio"
                  aria-checked={isSelected}
                >
                  {/* Icon */}
                  <div
                    className={cn(
                      'flex-shrink-0 w-12 h-12 rounded-xl flex items-center justify-center transition-all duration-200',
                      option.iconBg,
                      isSelected ? 'scale-105' : 'group-hover:scale-105'
                    )}
                  >
                    {option.icon}
                  </div>

                  <div className="flex-1 min-w-0">
                    <div className="font-medium text-foreground">{option.title}</div>
                    <div className="text-sm text-muted-foreground mt-0.5">
                      {option.description}
                    </div>
                  </div>

                  {/* Selection indicator */}
                  <div
                    className={cn(
                      'flex-shrink-0 w-6 h-6 rounded-full border-2 flex items-center justify-center transition-all duration-200',
                      isSelected
                        ? 'border-primary bg-primary'
                        : 'border-muted-foreground/30 group-hover:border-primary/50'
                    )}
                  >
                    {isSelected && <Check className="w-4 h-4 text-primary-foreground" />}
                  </div>
                </button>

                {/* Drop zone - expands inside the card */}
                {isSelected && (
                  <div className="px-4 pb-4 animate-in fade-in slide-in-from-top-2 duration-300">
                    <div className="pt-3 border-t border-primary/20">
                      {/* For classifier/anomaly, show guidance message instead of file upload */}
                      {(projectType === 'classifier' || projectType === 'anomaly') ? (
                        <div className="flex items-start gap-3 p-3 rounded-lg bg-primary/5 border border-primary/20">
                          <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center">
                            <Check className="w-4 h-4 text-primary" />
                          </div>
                          <div>
                            <p className="text-sm font-medium text-foreground">
                              Great! We'll show you where to add your data.
                            </p>
                            <p className="text-xs text-muted-foreground mt-1">
                              {projectType === 'classifier'
                                ? "You'll paste or upload your labeled training examples directly in the classifier training page."
                                : "You'll paste or upload your baseline data directly in the anomaly detector training page."}
                            </p>
                          </div>
                        </div>
                      ) : (
                        <>
                          <p className="text-sm text-muted-foreground mb-3">
                            Drop files here to get started (optional):
                          </p>

                          {/* Hidden file input */}
                          <input
                            ref={fileInputRef}
                            type="file"
                            multiple
                            onChange={handleFileInputChange}
                            className="hidden"
                            accept=".pdf,.doc,.docx,.txt,.md,.json,.jsonl,.csv,.html,.htm,.xml"
                          />

                          {/* Drop zone */}
                          <div
                            ref={dropZoneRef}
                            onDragEnter={handleDragEnter}
                            onDragOver={handleDragOver}
                            onDragLeave={handleDragLeave}
                            onDrop={handleDrop}
                            onClick={() => fileInputRef.current?.click()}
                            className={cn(
                              'border-2 border-dashed rounded-lg p-4 text-center cursor-pointer transition-all duration-200',
                              isDragging
                                ? 'border-primary bg-primary/10'
                                : 'border-border hover:border-primary/50 hover:bg-primary/5'
                            )}
                          >
                            <Upload className="w-6 h-6 mx-auto mb-2 text-muted-foreground" />
                            <p className="text-sm text-muted-foreground">
                              {isDragging ? 'Drop files here' : 'Drag & drop or click to browse'}
                            </p>
                            <p className="text-xs text-muted-foreground/70 mt-1">
                              PDF, Word, text, markdown, JSON, CSV
                            </p>
                          </div>

                          {/* File list */}
                          {uploadedFiles.length > 0 && (
                            <div className="mt-3 space-y-1.5">
                              {uploadedFiles.map((file, index) => (
                                <div
                                  key={`${file.name}-${index}`}
                                  className="flex items-center gap-2 p-2 rounded-lg bg-secondary/50 text-sm"
                                >
                                  <FileText className="w-4 h-4 text-muted-foreground flex-shrink-0" />
                                  <span className="flex-1 truncate text-foreground">{file.name}</span>
                                  <span className="text-xs text-muted-foreground flex-shrink-0">
                                    {formatFileSize(file.size)}
                                  </span>
                                  <button
                                    type="button"
                                    onClick={(e) => {
                                      e.stopPropagation()
                                      removeFile(index)
                                    }}
                                    className="p-0.5 rounded hover:bg-destructive/20 text-muted-foreground hover:text-destructive transition-colors"
                                  >
                                    <X className="w-3.5 h-3.5" />
                                  </button>
                                </div>
                              ))}
                            </div>
                          )}

                          {/* Dataset name input */}
                          {uploadedFiles.length > 0 && (
                            <div className="mt-3">
                              <Input
                                type="text"
                                placeholder="Dataset name (optional)"
                                value={datasetName || ''}
                                onChange={(e) => {
                                  const newName = e.target.value || null
                                  onDatasetNameChange(newName)
                                  // Validate on change (only if there's a value)
                                  if (newName) {
                                    const validation = validateDatasetName(newName)
                                    setDatasetNameError(validation.isValid ? null : validation.error || 'Invalid name')
                                  } else {
                                    setDatasetNameError(null)
                                  }
                                }}
                                className={cn(
                                  'h-9 text-sm',
                                  datasetNameError && 'border-destructive'
                                )}
                              />
                              {datasetNameError && (
                                <p className="text-xs text-destructive mt-1">
                                  {datasetNameError}
                                </p>
                              )}
                            </div>
                          )}
                        </>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )
          }

          // Standard option card (fallback, shouldn't be reached)
          return (
            <button
              type="button"
              key={option.id}
              onClick={() => {
                onSelect(option.id)
                onSelectSampleDataset(null)
              }}
              className={cn(
                'group w-full flex items-center gap-4 p-4 rounded-xl border-2 text-left transition-all duration-200',
                'hover:scale-[1.01] hover:shadow-md',
                'focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2',
                isSelected
                  ? 'border-primary bg-white dark:bg-primary/10 shadow-md ring-1 ring-primary/20'
                  : 'border-border bg-card hover:bg-white dark:hover:bg-card hover:border-primary/40'
              )}
              role="radio"
              aria-checked={isSelected}
            >
              {/* Icon */}
              <div
                className={cn(
                  'flex-shrink-0 w-12 h-12 rounded-xl flex items-center justify-center transition-all duration-200',
                  option.iconBg,
                  isSelected ? 'scale-105' : 'group-hover:scale-105'
                )}
              >
                {option.icon}
              </div>

              <div className="flex-1 min-w-0">
                <div className="font-medium text-foreground">{option.title}</div>
                <div className="text-sm text-muted-foreground mt-0.5">
                  {option.description}
                </div>
              </div>

              {/* Selection indicator */}
              <div
                className={cn(
                  'flex-shrink-0 w-6 h-6 rounded-full border-2 flex items-center justify-center transition-all duration-200',
                  isSelected
                    ? 'border-primary bg-primary'
                    : 'border-muted-foreground/30 group-hover:border-primary/50'
                )}
              >
                {isSelected && <Check className="w-4 h-4 text-primary-foreground" />}
              </div>
            </button>
          )
        })}
      </div>
    </div>
  )
}
