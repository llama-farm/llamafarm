/**
 * Hook for managing onboarding wizard and checklist state
 * Follows the pattern established by useProjectModal.ts
 */

import { useState, useCallback, useMemo, useEffect, useRef } from 'react'
import type {
  OnboardingState,
  WizardStep,
  ProjectType,
  DataStatus,
  DeployTarget,
  ExperienceLevel,
  UseOnboardingReturn,
  ChecklistStep,
  OnboardingUploadedFile,
} from '../types/onboarding'
import {
  DEFAULT_ONBOARDING_STATE,
  PROJECT_TYPE_LABELS,
  DEPLOY_TARGET_LABELS,
} from '../types/onboarding'
import {
  generateChecklist,
  generateDemoChecklist,
  getDescriptionForLevel,
} from '../utils/checklistGenerator'
import { isDemoProject, getDemoConfigForProject, removeDemoProject } from '../config/demos'
import { validateDatasetName } from '../utils/datasetValidation'

const STORAGE_KEY_PREFIX = 'lf_onboarding_'

/**
 * Get storage key for a specific project
 */
function getStorageKey(projectId: string | null): string {
  if (!projectId) {
    return `${STORAGE_KEY_PREFIX}default`
  }
  return `${STORAGE_KEY_PREFIX}${projectId}`
}

/**
 * Load state from localStorage for a specific project
 */
function loadState(projectId: string | null): OnboardingState {
  try {
    const storageKey = getStorageKey(projectId)
    const stored = localStorage.getItem(storageKey)
    if (stored) {
      const parsed = JSON.parse(stored)
      // Deep merge to handle missing fields from older versions
      return {
        ...DEFAULT_ONBOARDING_STATE,
        ...parsed,
        // Deep merge answers to ensure new fields have defaults
        answers: {
          ...DEFAULT_ONBOARDING_STATE.answers,
          ...(parsed.answers || {}),
        },
      }
    }
  } catch (e) {
    console.warn('Failed to load onboarding state from localStorage:', e)
  }
  return DEFAULT_ONBOARDING_STATE
}

/**
 * Save state to localStorage for a specific project
 */
function saveState(state: OnboardingState, projectId: string | null): void {
  try {
    const storageKey = getStorageKey(projectId)
    const toSave = { ...state, lastUpdated: new Date().toISOString() }
    localStorage.setItem(storageKey, JSON.stringify(toSave))
  } catch (e) {
    console.warn('Failed to save onboarding state to localStorage:', e)
  }
}

/**
 * Main onboarding hook - manages wizard and checklist state
 * @param projectId - Optional project identifier to make state project-specific
 */
export function useOnboarding(projectId: string | null = null): UseOnboardingReturn {
  const [state, setState] = useState<OnboardingState>(() => loadState(projectId))

  // Ref to store actual File objects (not persisted to localStorage)
  // This avoids storing files on the global window object
  const actualFilesRef = useRef<File[]>([])

  // Check if this is a demo project
  const demoConfig = useMemo(() => getDemoConfigForProject(projectId), [projectId])
  const isDemo = useMemo(() => isDemoProject(projectId), [projectId])

  // Reload state when project changes
  useEffect(() => {
    const loadedState = loadState(projectId)

    // For demo projects, auto-complete onboarding (skip wizard, show checklist)
    if (isDemo && !loadedState.onboardingCompleted) {
      setState({
        ...loadedState,
        wizardOpen: false,
        onboardingCompleted: true,
        checklistVisible: true,
        checklistDismissed: false,
        currentStep: 'complete',
        // Set project type to doc-qa for demo projects
        answers: {
          ...loadedState.answers,
          projectType: 'doc-qa',
          dataStatus: 'has-data',
        },
      })
    } else {
      setState(loadedState)
    }
  }, [projectId, isDemo])

  // Persist state changes to localStorage
  useEffect(() => {
    saveState(state, projectId)
  }, [state, projectId])

  // Generate checklist based on current answers (or demo config)
  const checklist = useMemo<ChecklistStep[]>(() => {
    // For demo projects, use the simplified demo checklist
    if (demoConfig) {
      const demoChecklist = generateDemoChecklist(demoConfig)
      if (demoChecklist) return demoChecklist
    }

    // Regular checklist based on user answers
    const { projectType, dataStatus, trainedModelName, trainedModelType, uploadedFiles, datasetName, selectedSampleDataset } = state.answers
    return generateChecklist(
      projectType,
      dataStatus,
      trainedModelName,
      trainedModelType,
      uploadedFiles?.length || 0,
      datasetName,
      selectedSampleDataset
    )
  }, [state.answers, demoConfig])

  // Check if current step has a valid selection to proceed
  const canProceed = useMemo(() => {
    const { currentStep, answers } = state
    switch (currentStep) {
      case 0:
        return true // Welcome screen always can proceed
      case 1:
        return answers.projectType !== null
      case 2:
        // If sample-data is selected, must also pick a sample dataset
        if (answers.dataStatus === 'sample-data') {
          return answers.selectedSampleDataset !== null
        }
        // If has-data is selected and dataset name is provided, validate it
        if (answers.dataStatus === 'has-data' && answers.datasetName) {
          const validation = validateDatasetName(answers.datasetName)
          if (!validation.isValid) {
            return false
          }
        }
        // If need-data is selected, HF dataset is optional - can proceed without it
        return answers.dataStatus !== null
      case 3:
        return answers.deployTarget !== null
      case 4:
        return answers.experienceLevel !== null
      default:
        return false
    }
  }, [state.currentStep, state.answers])

  // Wizard actions
  const openWizard = useCallback(() => {
    setState(prev => ({
      ...prev,
      wizardOpen: true,
      currentStep: 0,
    }))
    // Dispatch event to close chat panel during onboarding
    window.dispatchEvent(new CustomEvent('lf-onboarding-started'))
  }, [])

  const closeWizard = useCallback(() => {
    setState(prev => ({
      ...prev,
      wizardOpen: false,
    }))
  }, [])

  const setStep = useCallback((step: WizardStep) => {
    setState(prev => ({
      ...prev,
      currentStep: step,
    }))
  }, [])

  const nextStep = useCallback(() => {
    setState(prev => {
      const { currentStep } = prev
      let nextStepValue: WizardStep

      if (currentStep === 0) nextStepValue = 1
      else if (currentStep === 1) nextStepValue = 2
      else if (currentStep === 2) nextStepValue = 3
      else if (currentStep === 3) nextStepValue = 4
      else if (currentStep === 4) nextStepValue = 'transition'
      else nextStepValue = 'complete'

      return { ...prev, currentStep: nextStepValue }
    })
  }, [])

  const prevStep = useCallback(() => {
    setState(prev => {
      const { currentStep } = prev
      let prevStepValue: WizardStep

      if (currentStep === 1) prevStepValue = 0
      else if (currentStep === 2) prevStepValue = 1
      else if (currentStep === 3) prevStepValue = 2
      else if (currentStep === 4) prevStepValue = 3
      else prevStepValue = 0

      return { ...prev, currentStep: prevStepValue }
    })
  }, [])

  const skipWizard = useCallback(() => {
    setState(prev => ({
      ...prev,
      wizardOpen: false,
      onboardingCompleted: false,
      checklistDismissed: true,
    }))
  }, [])

  const completeWizard = useCallback(() => {
    setState(prev => {
      // If sample data was selected, dispatch event to trigger auto-import
      if (prev.answers.dataStatus === 'sample-data' && prev.answers.selectedSampleDataset) {
        // Dispatch event for Dashboard to handle navigation and import
        window.dispatchEvent(
          new CustomEvent('lf-onboarding-import-sample', {
            detail: { demoId: prev.answers.selectedSampleDataset },
          })
        )
      }

      // If user uploaded files during onboarding, dispatch event to trigger upload
      if (prev.answers.dataStatus === 'has-data' && prev.answers.uploadedFiles.length > 0) {
        // Get actual files from the ref (stored in React context, not window)
        const files = actualFilesRef.current
        if (files.length > 0) {
          window.dispatchEvent(
            new CustomEvent('lf-onboarding-upload-files', {
              detail: {
                files: [...files], // Copy the array to preserve files
                datasetName: prev.answers.datasetName || 'my-data',
              },
            })
          )
          // Clear the ref after dispatching
          actualFilesRef.current = []
        }
      }

      return {
        ...prev,
        wizardOpen: false,
        currentStep: 'complete',
        onboardingCompleted: true,
        checklistVisible: true,
        checklistDismissed: false,
      }
    })
  }, [])

  // Answer actions
  const setProjectType = useCallback((type: ProjectType) => {
    setState(prev => ({
      ...prev,
      answers: { ...prev.answers, projectType: type },
    }))
  }, [])

  const setDataStatus = useCallback((status: DataStatus) => {
    setState(prev => ({
      ...prev,
      answers: {
        ...prev.answers,
        dataStatus: status,
        // Clear sample dataset selection if not using sample-data
        selectedSampleDataset: status === 'sample-data' ? prev.answers.selectedSampleDataset : null,
      },
    }))
  }, [])

  const setSelectedSampleDataset = useCallback((demoId: string | null) => {
    setState(prev => ({
      ...prev,
      answers: { ...prev.answers, selectedSampleDataset: demoId },
    }))
  }, [])

  const setDeployTarget = useCallback((target: DeployTarget) => {
    setState(prev => ({
      ...prev,
      answers: { ...prev.answers, deployTarget: target },
    }))
  }, [])

  const setExperienceLevel = useCallback((level: ExperienceLevel) => {
    setState(prev => ({
      ...prev,
      answers: { ...prev.answers, experienceLevel: level },
    }))
  }, [])

  const setTrainedModel = useCallback((modelName: string, modelType: 'classifier' | 'anomaly') => {
    setState(prev => ({
      ...prev,
      answers: {
        ...prev.answers,
        trainedModelName: modelName,
        trainedModelType: modelType,
        isTrainingSampleModel: false, // Training complete
      },
    }))
  }, [])

  const setIsTrainingSampleModel = useCallback((isTraining: boolean) => {
    setState(prev => ({
      ...prev,
      answers: {
        ...prev.answers,
        isTrainingSampleModel: isTraining,
      },
    }))
  }, [])

  const setUploadedFiles = useCallback((files: OnboardingUploadedFile[]) => {
    setState(prev => ({
      ...prev,
      answers: { ...prev.answers, uploadedFiles: files },
    }))
  }, [])

  const setDatasetName = useCallback((name: string | null) => {
    setState(prev => ({
      ...prev,
      answers: { ...prev.answers, datasetName: name },
    }))
  }, [])

  // File storage actions (actual File objects stored in ref, not persisted)
  const addActualFiles = useCallback((files: File[]) => {
    actualFilesRef.current = [...actualFilesRef.current, ...files]
  }, [])

  const removeActualFile = useCallback((index: number) => {
    actualFilesRef.current = actualFilesRef.current.filter((_, i) => i !== index)
  }, [])

  const getActualFiles = useCallback(() => {
    return actualFilesRef.current
  }, [])

  const clearActualFiles = useCallback(() => {
    actualFilesRef.current = []
  }, [])

  // Checklist actions
  const completeChecklistStep = useCallback((stepId: string) => {
    setState(prev => {
      if (prev.completedSteps.includes(stepId)) {
        return prev
      }
      return {
        ...prev,
        completedSteps: [...prev.completedSteps, stepId],
      }
    })
  }, [])

  const uncompleteChecklistStep = useCallback((stepId: string) => {
    setState(prev => ({
      ...prev,
      completedSteps: prev.completedSteps.filter(id => id !== stepId),
    }))
  }, [])

  const dismissChecklist = useCallback(() => {
    setState(prev => ({
      ...prev,
      checklistDismissed: true,
      checklistVisible: false,
    }))
  }, [])

  const showChecklist = useCallback(() => {
    setState(prev => ({
      ...prev,
      checklistDismissed: false,
      checklistVisible: true,
    }))
  }, [])

  const toggleChecklistCollapsed = useCallback(() => {
    setState(prev => ({
      ...prev,
      checklistCollapsed: !prev.checklistCollapsed,
    }))
  }, [])

  const resetOnboarding = useCallback(() => {
    // If this was a demo project, remove it from the demo list so it becomes a regular project
    // This allows the user to "Build your own" from a demo and get a fresh onboarding experience
    if (isDemo) {
      removeDemoProject(projectId)
    }

    setState({
      ...DEFAULT_ONBOARDING_STATE,
      wizardOpen: true,
      currentStep: 1, // Skip welcome, go straight to "what are you building"
    })
    // Dispatch event to close chat panel during onboarding
    window.dispatchEvent(new CustomEvent('lf-onboarding-reset'))
  }, [isDemo, projectId])

  // Derived helpers
  const isStepCompleted = useCallback(
    (stepId: string) => {
      return state.completedSteps.includes(stepId)
    },
    [state.completedSteps]
  )

  const getDescription = useCallback(
    (step: ChecklistStep) => {
      const level = state.answers.experienceLevel || 'beginner'
      return getDescriptionForLevel(step, level)
    },
    [state.answers.experienceLevel]
  )

  const getProjectTypeLabel = useCallback(() => {
    const { projectType } = state.answers
    if (!projectType) return ''
    return PROJECT_TYPE_LABELS[projectType] || ''
  }, [state.answers])

  const getDeployTargetLabel = useCallback(() => {
    const { deployTarget } = state.answers
    if (!deployTarget) return ''
    return DEPLOY_TARGET_LABELS[deployTarget] || ''
  }, [state.answers])

  return {
    // State
    state,
    checklist,

    // Demo project info
    isDemo,
    demoConfig,

    // Wizard actions
    openWizard,
    closeWizard,
    setStep,
    nextStep,
    prevStep,
    skipWizard,
    completeWizard,

    // Answer actions
    setProjectType,
    setDataStatus,
    setSelectedSampleDataset,
    setDeployTarget,
    setExperienceLevel,
    setTrainedModel,
    setIsTrainingSampleModel,
    setUploadedFiles,
    setDatasetName,

    // File storage actions (actual File objects stored in ref, not persisted)
    addActualFiles,
    removeActualFile,
    getActualFiles,
    clearActualFiles,

    // Checklist actions
    completeChecklistStep,
    uncompleteChecklistStep,
    dismissChecklist,
    showChecklist,
    toggleChecklistCollapsed,
    resetOnboarding,

    // Derived helpers
    canProceed,
    isStepCompleted,
    getDescription,
    getProjectTypeLabel,
    getDeployTargetLabel,
  }
}
