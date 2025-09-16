/**
 * Shared hook for project modal state and operations
 * Centralizes modal logic across Home, Dashboard, and Projects components
 */

import { useState } from 'react'
import { useToast } from '../components/ui/toast'
import { useNavigate } from 'react-router-dom'
import { useQueryClient } from '@tanstack/react-query'
import {
  useProject,
  useCreateProject,
  useUpdateProject,
  useDeleteProject,
} from './useProjects'
import { projectKeys } from './useProjects'
import type { Project } from '../types/project'
import { setActiveProject } from '../utils/projectUtils'
import {
  validateProjectConfig,
  mergeProjectConfig,
} from '../utils/projectConfigUtils'
import {
  validateProjectNameWithDuplicateCheck,
  sanitizeProjectName,
} from '../utils/projectValidation'

export type ProjectModalMode = 'create' | 'edit'

export interface UseProjectModalOptions {
  namespace: string
  existingProjects?: string[]
  onSuccess?: (projectName: string, mode: ProjectModalMode) => void
}

export interface UseProjectModalReturn {
  // Modal state
  isModalOpen: boolean
  modalMode: ProjectModalMode
  projectName: string
  currentProject: any

  // Validation state
  projectError: string | null

  // Loading states
  isLoading: boolean
  isProjectLoading: boolean

  // Actions
  openCreateModal: () => void
  openEditModal: (name: string) => void
  closeModal: () => void

  // CRUD operations
  saveProject: (name: string) => Promise<void>
  deleteProject: () => Promise<void>

  // Validation
  validateName: (name: string) => boolean
}

export const useProjectModal = ({
  namespace,
  existingProjects = [],
  onSuccess,
}: UseProjectModalOptions): UseProjectModalReturn => {
  const navigate = useNavigate()
  const { toast } = useToast()
  const queryClient = useQueryClient()

  // Modal state
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [modalMode, setModalMode] = useState<ProjectModalMode>('create')
  const [projectName, setProjectName] = useState('')
  const [projectError, setProjectError] = useState<string | null>(null)

  // API hooks
  const { data: currentProjectResponse, isLoading: isProjectLoading } =
    useProject(
      namespace,
      projectName,
      modalMode === 'edit' && !!projectName && isModalOpen
    )

  const createProjectMutation = useCreateProject()
  const updateProjectMutation = useUpdateProject()
  const deleteProjectMutation = useDeleteProject()

  // Current project data
  const currentProject = currentProjectResponse?.project

  // Combined loading state
  const isLoading =
    createProjectMutation.isPending ||
    updateProjectMutation.isPending ||
    deleteProjectMutation.isPending ||
    isProjectLoading

  // Actions
  const openCreateModal = () => {
    setModalMode('create')
    setProjectName('')
    setProjectError(null)
    setIsModalOpen(true)
  }

  const openEditModal = (name: string) => {
    setModalMode('edit')
    setProjectName(name)
    setProjectError(null)
    setIsModalOpen(true)
  }

  const closeModal = () => {
    setIsModalOpen(false)
    setProjectName('')
    setProjectError(null)
  }

  // Name validation with better error handling
  const validateName = (name: string): boolean => {
    const validation = validateProjectNameWithDuplicateCheck(
      name,
      existingProjects,
      modalMode === 'edit' ? projectName : null
    )

    if (!validation.isValid) {
      setProjectError(validation.error || 'Invalid project name')
      return false
    }

    setProjectError(null)
    return true
  }

  // Save project (create or update)
  const saveProject = async (name: string): Promise<void> => {
    const sanitizedName = sanitizeProjectName(name)

    // Validate name
    if (!validateName(sanitizedName)) {
      return
    }

    try {
      toast({ message: `Creating "${sanitizedName}"...` })
      if (modalMode === 'create') {
        await createProjectMutation.mutateAsync({
          namespace,
          request: { name: sanitizedName, config_template: 'default' },
        })

        setActiveProject(sanitizedName)
        // Update list cache optimistically to include the new project
        try {
          const prev = queryClient.getQueryData(projectKeys.list(namespace)) as
            | { total?: number; projects?: Project[] }
            | undefined
          const nextProject: Project = {
            namespace,
            name: sanitizedName,
            config: {},
          }
          const next = {
            total: (prev?.total ?? 0) + 1,
            projects: [...(prev?.projects ?? []), nextProject],
          }
          queryClient.setQueryData(projectKeys.list(namespace), next)
        } catch {}
        // Persist in local fallback list for offline support
        try {
          const raw = localStorage.getItem('lf_custom_projects')
          const arr: string[] = raw ? JSON.parse(raw) : []
          if (!arr.includes(sanitizedName)) {
            localStorage.setItem(
              'lf_custom_projects',
              JSON.stringify([...arr, sanitizedName])
            )
          }
        } catch {}
        closeModal()
        onSuccess?.(sanitizedName, 'create')
        toast({ message: `Project "${sanitizedName}" created` })

        // Navigate to new project dashboard
        navigate('/chat/dashboard')
      } else {
        // Edit mode - update existing project
        if (!currentProject?.config) {
          setProjectError('Cannot update project: configuration not loaded')
          return
        }

        // Update the config with the new name while preserving all other properties
        const updatedConfig = mergeProjectConfig(currentProject.config, {
          name: sanitizedName,
          namespace: namespace,
        })

        // Basic validation (backend will do detailed validation)
        if (!validateProjectConfig(updatedConfig)) {
          setProjectError('Invalid project configuration')
          return
        }

        await updateProjectMutation.mutateAsync({
          namespace,
          projectId: projectName,
          request: { config: updatedConfig },
        })

        setActiveProject(sanitizedName)
        closeModal()
        onSuccess?.(sanitizedName, 'edit')
      }
    } catch (error: any) {
      console.error(`Failed to ${modalMode} project:`, error)

      // Handle backend validation errors gracefully
      if (error?.response?.status === 409) {
        setProjectError('Project name already exists')
      } else if (error?.response?.status === 422) {
        setProjectError('Invalid project configuration')
      } else if (error?.response?.status === 400) {
        setProjectError('Invalid request. Please check your input.')
      } else {
        setProjectError(`Failed to ${modalMode} project. Please try again.`)
      }
    }
  }

  // Delete project
  const deleteProject = async (): Promise<void> => {
    if (modalMode !== 'edit') return

    try {
      await deleteProjectMutation.mutateAsync({
        namespace,
        projectId: projectName,
      })

      closeModal()
      onSuccess?.('', 'edit') // Empty name indicates deletion
    } catch (error: any) {
      console.error('Failed to delete project:', error)

      // Handle delete errors gracefully
      if (error?.response?.status === 404) {
        setProjectError('Project not found')
      } else if (error?.response?.status === 403) {
        setProjectError('Not authorized to delete this project')
      } else {
        setProjectError('Failed to delete project. Please try again.')
      }
    }
  }

  return {
    // State
    isModalOpen,
    modalMode,
    projectName,
    currentProject,
    projectError,

    // Loading
    isLoading,
    isProjectLoading,

    // Actions
    openCreateModal,
    openEditModal,
    closeModal,

    // Operations
    saveProject,
    deleteProject,
    validateName,
  }
}
