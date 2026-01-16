/**
 * Generates personalized checklists based on user's onboarding answers
 */

import type {
  ProjectType,
  DataStatus,
  ChecklistStep,
} from '../types/onboarding'
import { getDemoById, isFileBasedDemo, type FileBasedDemo } from '../config/demos'

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Create a "Ship it" step with consistent structure
 */
function createShipStep(
  id: string,
  stepNumber: number,
  modelLabel?: string
): ChecklistStep {
  const desc = modelLabel
    ? `Package your project for deployment. This includes your trained ${modelLabel}.`
    : 'Package your project for deployment.'
  return {
    id,
    stepNumber,
    title: 'Ship it',
    descriptionFull: desc,
    descriptionShort: 'Package your project for deployment.',
    descriptionMinimal: 'Package for deployment.',
    linkPath: '/chat/dashboard',
    linkLabel: 'Package',
  }
}

/**
 * Create a test step for classifier or anomaly models
 */
function createModelTestStep(
  type: 'classifier' | 'anomaly',
  stepNumber: number,
  id?: string
): ChecklistStep {
  const isClassifier = type === 'classifier'
  return {
    id: id || `${type}-test`,
    stepNumber,
    title: isClassifier ? 'Test your labels' : 'Test detection',
    descriptionFull: isClassifier
      ? 'Switch to Classifier mode in Test and see how it categorizes new content. Check if the labels make sense.'
      : 'Switch to Anomaly Detection mode in Test and see if it catches the weird stuff. Try both normal and unusual inputs.',
    descriptionShort: isClassifier
      ? 'Test classification in Classifier mode.'
      : 'Test detection in Anomaly Detection mode.',
    descriptionMinimal: isClassifier ? 'Test your classifier.' : 'Test your detector.',
    linkPath: `/chat/test?mode=${type}`,
    linkLabel: 'Go to Test',
  }
}

// ============================================================================
// Base Checklist Definitions
// ============================================================================

const DOC_QA_CHECKLIST: ChecklistStep[] = [
  {
    id: 'doc-qa-data',
    stepNumber: 1,
    title: 'Get your data in',
    descriptionFull:
      'Create a dataset, upload your files, and hit Process. You\'ll see "SUCCESS" when chunks are ready for your AI to use.',
    descriptionShort: 'Create a dataset, upload files, and process them.',
    descriptionMinimal: 'Upload and process your files.',
    linkPath: '/chat/data?modal=create',
    linkLabel: 'Create dataset',
  },
  {
    id: 'doc-qa-prompt',
    stepNumber: 2,
    title: 'Tweak your prompt',
    descriptionFull:
      "The default works, but you'll get better answers if you tell the AI what it's doing. Edit the system prompt to match your use case.",
    descriptionShort: 'Edit the system prompt to match your use case.',
    descriptionMinimal: 'Customize your system prompt.',
    linkPath: '/chat/prompt',
    linkLabel: 'Go to Prompts',
  },
  {
    id: 'doc-qa-test',
    stepNumber: 3,
    title: 'Take it for a spin',
    descriptionFull:
      'Open Test, make sure you\'re in Text Generation mode, and ask questions about your docs. See if the answers make sense.',
    descriptionShort: 'Test your setup in Text Generation mode.',
    descriptionMinimal: 'Test your RAG setup.',
    linkPath: '/chat/test',
    linkLabel: 'Go to Test',
  },
  createShipStep('doc-qa-ship', 4),
]

const CLASSIFIER_CHECKLIST: ChecklistStep[] = [
  {
    id: 'classifier-data',
    stepNumber: 1,
    title: 'Get training data ready',
    descriptionFull:
      "You'll need labeled examples for your classifier to learn from. Upload your own labeled data or use sample data within the training flow.",
    descriptionShort: 'Prepare labeled examples for training.',
    descriptionMinimal: 'Prepare labeled training data.',
    linkPath: '/chat/data?modal=create',
    linkLabel: 'Create dataset',
  },
  {
    id: 'classifier-create',
    stepNumber: 2,
    title: 'Create your classifier',
    descriptionFull: 'Configure your categories and train a new classifier model on your labeled data.',
    descriptionShort: 'Create and train a classifier model.',
    descriptionMinimal: 'Train your classifier model.',
    linkPath: '/chat/models/train/classifier/new',
    linkLabel: 'Create classifier',
  },
  createModelTestStep('classifier', 3),
  createShipStep('classifier-ship', 4, 'classifier model'),
]

const ANOMALY_CHECKLIST: ChecklistStep[] = [
  {
    id: 'anomaly-data',
    stepNumber: 1,
    title: 'Get baseline data ready',
    descriptionFull:
      "You'll need examples of \"normal\" so your detector can learn what's unusual. Upload representative samples of your typical data.",
    descriptionShort: 'Prepare examples of normal data for training.',
    descriptionMinimal: 'Prepare baseline training data.',
    linkPath: '/chat/data?modal=create',
    linkLabel: 'Create dataset',
  },
  {
    id: 'anomaly-create',
    stepNumber: 2,
    title: 'Create your detector',
    descriptionFull: 'Train a new anomaly detection model on your baseline data so it can spot unusual patterns.',
    descriptionShort: 'Create and train an anomaly detector.',
    descriptionMinimal: 'Train your anomaly detector.',
    linkPath: '/chat/models/train/anomaly/new',
    linkLabel: 'Create detector',
  },
  createModelTestStep('anomaly', 3),
  createShipStep('anomaly-ship', 4, 'anomaly detection model'),
]

// Sample data flows - model trains automatically, skip "Create" step
const CLASSIFIER_SAMPLE_CHECKLIST: ChecklistStep[] = [
  {
    id: 'classifier-data',
    stepNumber: 1,
    title: 'View your trained classifier',
    descriptionFull:
      "Your sample classifier is training! Head over to see the progress and try it out when it's ready.",
    descriptionShort: 'View your sample classifier training progress.',
    descriptionMinimal: 'View trained classifier.',
    linkPath: '/chat/models/train/classifier/new?autoTrain=true',
    linkLabel: 'View classifier',
  },
  createModelTestStep('classifier', 2),
  createShipStep('classifier-ship', 3, 'classifier model'),
]

const ANOMALY_SAMPLE_CHECKLIST: ChecklistStep[] = [
  {
    id: 'anomaly-data',
    stepNumber: 1,
    title: 'View your trained detector',
    descriptionFull:
      "Your sample anomaly detector is training! Head over to see the progress and try it out when it's ready.",
    descriptionShort: 'View your sample detector training progress.',
    descriptionMinimal: 'View trained detector.',
    linkPath: '/chat/models/train/anomaly/new?autoTrain=true',
    linkLabel: 'View detector',
  },
  createModelTestStep('anomaly', 2),
  createShipStep('anomaly-ship', 3, 'anomaly detection model'),
]

// Need data flows - step 1 goes to training page with sample modal
const CLASSIFIER_NEED_DATA_CHECKLIST: ChecklistStep[] = [
  {
    id: 'classifier-data',
    stepNumber: 1,
    title: 'Add training data',
    descriptionFull:
      "Browse [Hugging Face](https://huggingface.co/datasets?task_categories=task_categories%3Atext-classification&sort=downloads) for text classification datasets, or use our sample data to get started quickly.",
    descriptionShort:
      'Browse [HF datasets](https://huggingface.co/datasets?task_categories=task_categories%3Atext-classification&sort=downloads) or use sample data.',
    descriptionMinimal: 'Find training data.',
    linkPath: '/chat/models/train/classifier/new?showSampleModal=true',
    linkLabel: 'Use sample data',
  },
  {
    id: 'classifier-train',
    stepNumber: 2,
    title: 'Train & test',
    descriptionFull: 'Head to Test to try your trained classifier with some sample content.',
    descriptionShort: 'Test your trained model in the Test page.',
    descriptionMinimal: 'Test your model.',
    linkPath: '/chat/test?mode=classifier',
    linkLabel: 'Test model',
  },
  createShipStep('classifier-ship', 3, 'classifier model'),
]

const ANOMALY_NEED_DATA_CHECKLIST: ChecklistStep[] = [
  {
    id: 'anomaly-data',
    stepNumber: 1,
    title: 'Add baseline data',
    descriptionFull:
      "Browse [Hugging Face](https://huggingface.co/datasets?search=anomaly+detection&sort=downloads) for anomaly detection datasets, or use our sample data to get started quickly.",
    descriptionShort:
      'Browse [HF datasets](https://huggingface.co/datasets?search=anomaly+detection&sort=downloads) or use sample data.',
    descriptionMinimal: 'Find baseline data.',
    linkPath: '/chat/models/train/anomaly/new?showSampleModal=true',
    linkLabel: 'Use sample data',
  },
  {
    id: 'anomaly-train',
    stepNumber: 2,
    title: 'Train & test',
    descriptionFull: 'Head to Test to try your trained detector with normal and unusual inputs.',
    descriptionShort: 'Test your trained model in the Test page.',
    descriptionMinimal: 'Test your model.',
    linkPath: '/chat/test?mode=anomaly',
    linkLabel: 'Test model',
  },
  createShipStep('anomaly-ship', 3, 'anomaly detection model'),
]

// Has data flows - user already has labeled data, 4 steps with same-page transitions
const CLASSIFIER_HAS_DATA_CHECKLIST: ChecklistStep[] = [
  {
    id: 'classifier-train',
    stepNumber: 1,
    title: 'Add data & train',
    descriptionFull: 'Add your training data and hit Train to create your classifier.',
    descriptionShort: 'Add training data and train your model.',
    descriptionMinimal: 'Train your classifier.',
    linkPath: '/chat/models/train/classifier/new',
    linkLabel: 'Create classifier',
  },
  {
    id: 'classifier-quick-test',
    stepNumber: 2,
    title: 'Quick test',
    descriptionFull: 'Try a few examples in the test input field to see how your classifier categorizes content.',
    descriptionShort: 'Test with the input field on this page.',
    descriptionMinimal: 'Quick test here.',
    linkPath: '/chat/models/train/classifier/new', // Same page as step 1
    linkLabel: 'Test here',
  },
  {
    id: 'classifier-full-test',
    stepNumber: 3,
    title: 'Test at scale',
    descriptionFull: 'Head to the Test page to try your classifier with more examples and see detailed results.',
    descriptionShort: 'Test more thoroughly in the Test page.',
    descriptionMinimal: 'Full testing.',
    linkPath: '/chat/test?mode=classifier',
    linkLabel: 'Go to Test',
  },
  createShipStep('classifier-ship', 4, 'classifier model'),
]

const ANOMALY_HAS_DATA_CHECKLIST: ChecklistStep[] = [
  {
    id: 'anomaly-train',
    stepNumber: 1,
    title: 'Add data & train',
    descriptionFull: 'Add your baseline data and hit Train to create your anomaly detector.',
    descriptionShort: 'Add baseline data and train your model.',
    descriptionMinimal: 'Train your detector.',
    linkPath: '/chat/models/train/anomaly/new',
    linkLabel: 'Create detector',
  },
  {
    id: 'anomaly-quick-test',
    stepNumber: 2,
    title: 'Quick test',
    descriptionFull: 'Try a few examples in the test input field to see how your detector scores them.',
    descriptionShort: 'Test with the input field on this page.',
    descriptionMinimal: 'Quick test here.',
    linkPath: '/chat/models/train/anomaly/new', // Same page as step 1
    linkLabel: 'Test here',
  },
  {
    id: 'anomaly-full-test',
    stepNumber: 3,
    title: 'Test at scale',
    descriptionFull: 'Head to the Test page to try your detector with more examples and see detailed results.',
    descriptionShort: 'Test more thoroughly in the Test page.',
    descriptionMinimal: 'Full testing.',
    linkPath: '/chat/test?mode=anomaly',
    linkLabel: 'Go to Test',
  },
  createShipStep('anomaly-ship', 4, 'anomaly detection model'),
]

const DOC_SCAN_CHECKLIST: ChecklistStep[] = [
  {
    id: 'doc-scan-data',
    stepNumber: 1,
    title: 'Get your docs in',
    descriptionFull:
      'Create a dataset, upload the documents you want to extract information from, and process them.',
    descriptionShort: 'Upload and process your documents.',
    descriptionMinimal: 'Upload your documents.',
    linkPath: '/chat/data?modal=create',
    linkLabel: 'Create dataset',
  },
  {
    id: 'doc-scan-test',
    stepNumber: 2,
    title: 'Test extraction',
    descriptionFull:
      'Switch to Doc Scanning mode in Test and see what information it pulls out. Check if the extracted data looks right.',
    descriptionShort: 'Test extraction in Doc Scanning mode.',
    descriptionMinimal: 'Test document extraction.',
    linkPath: '/chat/test',
    linkLabel: 'Go to Test',
  },
  createShipStep('doc-scan-ship', 3),
]

const EXPLORING_CHECKLIST: ChecklistStep[] = [
  {
    id: 'exploring-sample',
    stepNumber: 1,
    title: 'Find & import data',
    descriptionFull: "Find data on Hugging Face or generate synthetic data, then import it.",
    descriptionShort: 'Import a sample dataset to experiment with.',
    descriptionMinimal: 'Import sample data.',
    linkPath: '/chat/data?modal=import',
    linkLabel: 'Import sample',
  },
  {
    id: 'exploring-prompt',
    stepNumber: 2,
    title: 'Tweak your prompt',
    descriptionFull:
      "The system prompt tells your AI what it knows and how to behave. Try editing it to see how it changes responses.",
    descriptionShort: 'Edit the system prompt to customize behavior.',
    descriptionMinimal: 'Edit system prompt.',
    linkPath: '/chat/prompt',
    linkLabel: 'Go to Prompts',
  },
  {
    id: 'exploring-test',
    stepNumber: 3,
    title: 'Try the Test page',
    descriptionFull:
      'Chat with the sample data, try different modes (Text Generation, Classifier, etc.). Get a feel for what LlamaFarm can do.',
    descriptionShort: 'Explore different modes in the Test page.',
    descriptionMinimal: 'Explore the Test page.',
    linkPath: '/chat/test',
    linkLabel: 'Go to Test',
  },
  {
    id: 'exploring-decide',
    stepNumber: 4,
    title: 'Pick a direction',
    descriptionFull:
      "Ready to build something real? Come back here and start over with a specific project type. We'll give you a focused checklist.",
    descriptionShort: 'Start over with a specific project type.',
    descriptionMinimal: 'Choose a project type.',
    linkPath: '',
    linkLabel: 'Start over',
  },
]

// ============================================================================
// Checklist Lookup Map
// ============================================================================

type ModelType = 'classifier' | 'anomaly'
type ChecklistMap = Partial<Record<DataStatus, Partial<Record<ModelType, ChecklistStep[]>>>>

const MODEL_CHECKLIST_MAP: ChecklistMap = {
  'sample-data': {
    classifier: CLASSIFIER_SAMPLE_CHECKLIST,
    anomaly: ANOMALY_SAMPLE_CHECKLIST,
  },
  'need-data': {
    classifier: CLASSIFIER_NEED_DATA_CHECKLIST,
    anomaly: ANOMALY_NEED_DATA_CHECKLIST,
  },
  'has-data': {
    classifier: CLASSIFIER_HAS_DATA_CHECKLIST,
    anomaly: ANOMALY_HAS_DATA_CHECKLIST,
  },
}

// ============================================================================
// Checklist Generation Functions
// ============================================================================

/**
 * Demo project checklist - simplified flow for pre-built demo projects
 */
function createDemoChecklist(demo: FileBasedDemo): ChecklistStep[] {
  return [
    {
      id: 'demo-prompt',
      stepNumber: 1,
      title: 'Check out the prompt',
      descriptionFull: `See how we set up ${demo.displayName}. The system prompt tells the AI what it knows and how to respond.`,
      descriptionShort: 'Review the system prompt configuration.',
      descriptionMinimal: 'View system prompt.',
      linkPath: '/chat/prompt',
      linkLabel: 'View prompt',
    },
    {
      id: 'demo-test',
      stepNumber: 2,
      title: 'Try it out',
      descriptionFull:
        demo.sampleQuestions && demo.sampleQuestions.length > 0
          ? `Ask questions and see how it responds! Try: "${demo.sampleQuestions[0]}"`
          : 'Chat with your demo project and see how it responds to your questions.',
      descriptionShort: 'Test the demo in the chat interface.',
      descriptionMinimal: 'Try the demo.',
      linkPath: '/chat/test',
      linkLabel: 'Go to Test',
    },
    {
      id: 'demo-build',
      stepNumber: 3,
      title: 'Build your own',
      descriptionFull: "Ready to create something custom? Start fresh with your own data, prompts, and configuration.",
      descriptionShort: 'Start over to build your own project.',
      descriptionMinimal: 'Build your own.',
      linkPath: '',
      linkLabel: 'Start over',
    },
  ]
}

/**
 * Get the base checklist for a project type
 */
function getBaseChecklist(projectType: ProjectType, dataStatus?: DataStatus | null): ChecklistStep[] {
  // Check if we have a specialized checklist for classifier/anomaly with specific data status
  if (dataStatus && (projectType === 'classifier' || projectType === 'anomaly')) {
    const modelChecklists = MODEL_CHECKLIST_MAP[dataStatus]
    const checklist = modelChecklists?.[projectType]
    if (checklist) {
      return checklist.map(step => ({ ...step }))
    }
  }

  // Default checklists by project type
  switch (projectType) {
    case 'doc-qa':
      return DOC_QA_CHECKLIST.map(step => ({ ...step }))
    case 'classifier':
      return CLASSIFIER_CHECKLIST.map(step => ({ ...step }))
    case 'anomaly':
      return ANOMALY_CHECKLIST.map(step => ({ ...step }))
    case 'doc-scan':
      return DOC_SCAN_CHECKLIST.map(step => ({ ...step }))
    case 'exploring':
      return EXPLORING_CHECKLIST.map(step => ({ ...step }))
    default:
      return DOC_QA_CHECKLIST.map(step => ({ ...step }))
  }
}

/**
 * Modify the first step based on data status
 */
function applyDataStatusModifications(
  checklist: ChecklistStep[],
  dataStatus: DataStatus,
  projectType: ProjectType
): ChecklistStep[] {
  if (checklist.length === 0) return checklist

  const modified = [...checklist]
  const firstStep = { ...modified[0] }

  if (dataStatus === 'sample-data') {
    // For classifier/anomaly, the first step should be "View your trained model"
    if (projectType === 'classifier') {
      firstStep.title = 'View your trained classifier'
      firstStep.descriptionFull =
        "Your sample classifier is training! Head over to see the progress and try it out when it's ready."
      firstStep.descriptionShort = 'View your sample classifier training progress.'
      firstStep.descriptionMinimal = 'View trained classifier.'
      firstStep.linkPath = '/chat/models/train/classifier/new?autoTrain=true'
      firstStep.linkLabel = 'View classifier'
    } else if (projectType === 'anomaly') {
      firstStep.title = 'View your trained detector'
      firstStep.descriptionFull =
        "Your sample anomaly detector is training! Head over to see the progress and try it out when it's ready."
      firstStep.descriptionShort = 'View your sample detector training progress.'
      firstStep.descriptionMinimal = 'View trained detector.'
      firstStep.linkPath = '/chat/models/train/anomaly/new?autoTrain=true'
      firstStep.linkLabel = 'View detector'
    } else {
      // For doc-qa and other types with sample data
      firstStep.title = 'Load sample data'
      firstStep.descriptionFull =
        "Start with our sample dataset to see how things work. You can always swap in your own data later."
      firstStep.descriptionShort = 'Import a sample dataset to get started.'
      firstStep.descriptionMinimal = 'Import sample data.'
      firstStep.linkPath = '/chat/data?modal=import'
      firstStep.linkLabel = 'Import sample'
    }
  } else if (dataStatus === 'need-data') {
    // For non-classifier/anomaly types, show generic "find data" step
    if (projectType !== 'classifier' && projectType !== 'anomaly') {
      firstStep.title = 'Find & import data'
      firstStep.descriptionFull =
        "Check out [Hugging Face datasets](https://huggingface.co/datasets) or synthetic data generators to find data for your project. Once you have files, come back and create a dataset."
      firstStep.descriptionShort = 'Find data on [Hugging Face](https://huggingface.co/datasets) or generate synthetic data, then import it.'
      firstStep.descriptionMinimal = 'Find and import data.'
      firstStep.linkPath = '/chat/data?modal=import'
      firstStep.linkLabel = 'Import sample'
    }
  }

  modified[0] = firstStep
  return modified
}

/**
 * Options for generating a checklist
 */
export interface GenerateChecklistOptions {
  projectType: ProjectType | null
  dataStatus: DataStatus | null
  trainedModelName?: string | null
  trainedModelType?: 'classifier' | 'anomaly' | null
}

/**
 * Generate a personalized checklist based on user's answers
 */
export function generateChecklist(
  projectType: ProjectType | null,
  dataStatus: DataStatus | null,
  trainedModelName?: string | null,
  trainedModelType?: 'classifier' | 'anomaly' | null,
  uploadedFilesCount?: number,
  datasetName?: string | null,
  selectedSampleDataset?: string | null
): ChecklistStep[] {
  if (!projectType) {
    return []
  }

  let checklist = getBaseChecklist(projectType, dataStatus)

  // Apply modifications for non-sample flows (sample flows already have correct first step)
  if (dataStatus && dataStatus !== 'sample-data') {
    checklist = applyDataStatusModifications(checklist, dataStatus, projectType)
  } else if (dataStatus === 'sample-data' && projectType !== 'classifier' && projectType !== 'anomaly') {
    checklist = applyDataStatusModifications(checklist, dataStatus, projectType)
  }

  // If user uploaded files during onboarding, update the first step
  if (dataStatus === 'has-data' && uploadedFilesCount && uploadedFilesCount > 0) {
    const fileWord = uploadedFilesCount === 1 ? 'file' : 'files'
    const dsName = datasetName || 'my-data'

    if (checklist.length > 0) {
      checklist[0] = {
        ...checklist[0],
        title: 'View your uploaded data',
        descriptionFull:
          `You added ${uploadedFilesCount} ${fileWord} during setup. View your "${dsName}" dataset and hit Process to prepare it for your AI.`,
        descriptionShort: `Check your ${uploadedFilesCount} uploaded ${fileWord} and process them.`,
        descriptionMinimal: 'View and process uploaded data.',
        linkPath: `/chat/data/${encodeURIComponent(dsName)}`,
        linkLabel: 'View dataset',
      }
    }
  }

  // If user selected a file-based sample dataset (doc-qa), update the first step
  if (dataStatus === 'sample-data' && selectedSampleDataset && projectType !== 'classifier' && projectType !== 'anomaly') {
    const demo = getDemoById(selectedSampleDataset)

    if (demo && isFileBasedDemo(demo)) {
      if (checklist.length > 0) {
        checklist[0] = {
          ...checklist[0],
          title: 'View sample data',
          descriptionFull:
            `Your "${demo.displayName}" sample data is importing! Head to the Data page to see the import progress.`,
          descriptionShort: `View your "${demo.displayName}" sample dataset.`,
          descriptionMinimal: 'View sample data.',
          linkPath: `/chat/data/${encodeURIComponent(demo.datasetName)}`,
          linkLabel: 'View data',
        }
      }
    }
  }

  // If a model was trained from sample data, update the first step's link to the trained model
  if (trainedModelName && trainedModelType && dataStatus === 'sample-data') {
    const modelPath = trainedModelType === 'classifier' ? 'classifier' : 'anomaly'
    const modelLabel = trainedModelType === 'classifier' ? 'classifier' : 'detector'
    const newLinkPath = `/chat/models/train/${modelPath}/${encodeURIComponent(trainedModelName)}`

    if (checklist.length > 0) {
      checklist[0] = {
        ...checklist[0],
        descriptionFull: `Your sample ${modelLabel} is ready! Check it out and try it with some test inputs.`,
        descriptionShort: `View and test your trained ${modelLabel}.`,
        descriptionMinimal: `View trained ${modelLabel}.`,
        linkPath: newLinkPath,
      }
    }
  }

  return checklist
}

/**
 * Get the appropriate description based on experience level
 */
export function getDescriptionForLevel(
  step: ChecklistStep,
  level: 'beginner' | 'intermediate' | 'advanced'
): string {
  switch (level) {
    case 'beginner':
      return step.descriptionFull
    case 'intermediate':
      return step.descriptionShort
    case 'advanced':
      return step.descriptionMinimal
    default:
      return step.descriptionFull
  }
}

/**
 * Generate a simplified checklist for demo projects
 * Returns null if not a demo project
 */
export function generateDemoChecklist(demoConfig: FileBasedDemo | undefined): ChecklistStep[] | null {
  if (!demoConfig) return null
  return createDemoChecklist(demoConfig)
}
