/**
 * TypeScript types for the onboarding wizard and checklist
 */

import type { FileBasedDemo } from '../config/demos'

// Project type options (Screen 1)
export type ProjectType =
  | 'doc-qa' // Chat with my documents
  | 'classifier' // Sort & label content
  | 'anomaly' // Spot outliers
  | 'doc-scan' // Extract info from docs
  | 'exploring' // Just exploring

// Data availability options (Screen 2)
export type DataStatus =
  | 'has-data' // Yes, I have files ready
  | 'sample-data' // No, let me try sample data
  | 'need-data' // I need to find data first

// Deployment target options (Screen 3)
export type DeployTarget =
  | 'local' // My machine, on-prem, air-gapped
  | 'cloud' // AWS, GCP, Azure
  | 'tbd' // Not sure yet

// Experience level options (Screen 4)
export type ExperienceLevel =
  | 'beginner' // Lots of guidance
  | 'intermediate' // Some guidance
  | 'advanced' // Minimal guidance

// Wizard screen/step
export type WizardStep = 0 | 1 | 2 | 3 | 4 | 'transition' | 'complete'

// Checklist step definition
export interface ChecklistStep {
  id: string
  stepNumber: number
  title: string
  descriptionFull: string // For beginners
  descriptionShort: string // For intermediate
  descriptionMinimal: string // For advanced
  linkPath: string
  linkLabel: string
  // Optional secondary link (e.g., external HuggingFace link alongside primary action)
  secondaryLinkPath?: string
  secondaryLinkLabel?: string
}

// Uploaded file info for onboarding (stored without actual file data)
export interface OnboardingUploadedFile {
  name: string
  size: number
  type: string
}

// User's wizard answers
export interface OnboardingAnswers {
  projectType: ProjectType | null
  dataStatus: DataStatus | null
  selectedSampleDataset: string | null // Demo ID when sample-data is selected
  deployTarget: DeployTarget | null
  experienceLevel: ExperienceLevel | null
  // For classifier/anomaly sample data: the name of the trained model
  trainedModelName: string | null
  trainedModelType: 'classifier' | 'anomaly' | null
  // Whether sample model training is currently in progress
  isTrainingSampleModel: boolean
  // For has-data: uploaded files and dataset name
  uploadedFiles: OnboardingUploadedFile[]
  datasetName: string | null
}

// Complete onboarding state
export interface OnboardingState {
  // Wizard state
  wizardOpen: boolean
  currentStep: WizardStep
  answers: OnboardingAnswers

  // Checklist state
  checklistVisible: boolean
  checklistDismissed: boolean
  checklistCollapsed: boolean
  completedSteps: string[] // Array of step IDs

  // Meta
  onboardingCompleted: boolean
  lastUpdated: string | null
}

// Project type card data for wizard
export interface ProjectTypeOption {
  id: ProjectType
  icon: string
  title: string
  subtitle: string
}

// Radio option data for wizard screens 2-4
export interface RadioOption<T extends string> {
  id: T
  title: string
  description: string
}

// Hook return type
export interface UseOnboardingReturn {
  // State
  state: OnboardingState
  checklist: ChecklistStep[]

  // Demo project info
  isDemo: boolean
  demoConfig: FileBasedDemo | undefined

  // Wizard actions
  openWizard: () => void
  closeWizard: () => void
  setStep: (step: WizardStep) => void
  nextStep: () => void
  prevStep: () => void
  skipWizard: () => void
  completeWizard: () => void

  // Answer actions
  setProjectType: (type: ProjectType) => void
  setDataStatus: (status: DataStatus) => void
  setSelectedSampleDataset: (demoId: string | null) => void
  setDeployTarget: (target: DeployTarget) => void
  setExperienceLevel: (level: ExperienceLevel) => void
  setTrainedModel: (modelName: string, modelType: 'classifier' | 'anomaly') => void
  setIsTrainingSampleModel: (isTraining: boolean) => void
  setUploadedFiles: (files: OnboardingUploadedFile[]) => void
  setDatasetName: (name: string | null) => void

  // File storage actions (actual File objects stored in ref, not persisted)
  addActualFiles: (files: File[]) => void
  removeActualFile: (index: number) => void
  getActualFiles: () => File[]
  clearActualFiles: () => void

  // Checklist actions
  completeChecklistStep: (stepId: string) => void
  uncompleteChecklistStep: (stepId: string) => void
  dismissChecklist: () => void
  showChecklist: () => void
  toggleChecklistCollapsed: () => void
  resetOnboarding: () => void

  // Derived helpers
  canProceed: boolean
  isStepCompleted: (stepId: string) => boolean
  getDescription: (step: ChecklistStep) => string
  getProjectTypeLabel: () => string
  getDeployTargetLabel: () => string
}

// Default initial state
export const DEFAULT_ONBOARDING_STATE: OnboardingState = {
  wizardOpen: false,
  currentStep: 0,
  answers: {
    projectType: null,
    dataStatus: null,
    selectedSampleDataset: null,
    deployTarget: 'local', // Default to "On my own turf"
    experienceLevel: 'beginner', // Default to "Hold my hand"
    trainedModelName: null,
    trainedModelType: null,
    isTrainingSampleModel: false,
    uploadedFiles: [],
    datasetName: null,
  },
  checklistVisible: true,
  checklistDismissed: false,
  checklistCollapsed: false,
  completedSteps: [],
  onboardingCompleted: false,
  lastUpdated: null,
}

// Labels for display
export const PROJECT_TYPE_LABELS: Record<ProjectType, string> = {
  'doc-qa': 'Chat with documents',
  classifier: 'Sort & label content',
  anomaly: 'Spot outliers',
  'doc-scan': 'Extract info from docs',
  exploring: 'Just exploring',
}

export const DEPLOY_TARGET_LABELS: Record<DeployTarget, string> = {
  local: 'Staying local',
  cloud: 'Deploying to cloud',
  tbd: 'Target TBD',
}
