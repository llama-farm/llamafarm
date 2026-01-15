/**
 * Generates personalized checklists based on user's onboarding answers
 */

import type {
  ProjectType,
  DataStatus,
  ChecklistStep,
} from '../types/onboarding'
import type { SelectedHFDataset } from '../types/huggingface'
import type { FileBasedDemo } from '../config/demos'

// Base checklist definitions for each project type
const DOC_QA_CHECKLIST: ChecklistStep[] = [
  {
    id: 'doc-qa-data',
    stepNumber: 1,
    title: 'Get your data in',
    descriptionFull:
      'Create a dataset, upload your files, and hit Process. You\'ll see "SUCCESS" when chunks are ready for your AI to use.',
    descriptionShort:
      'Create a dataset, upload files, and process them.',
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
    descriptionShort:
      'Edit the system prompt to match your use case.',
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
    descriptionShort:
      'Test your setup in Text Generation mode.',
    descriptionMinimal: 'Test your RAG setup.',
    linkPath: '/chat/test',
    linkLabel: 'Go to Test',
  },
  {
    id: 'doc-qa-ship',
    stepNumber: 4,
    title: 'Ship it',
    descriptionFull:
      'Package your project for deployment. This creates everything you need to run your AI assistant in production.',
    descriptionShort: 'Package your project for deployment.',
    descriptionMinimal: 'Package for deployment.',
    linkPath: '/chat/dashboard',
    linkLabel: 'Package',
  },
]

const CLASSIFIER_CHECKLIST: ChecklistStep[] = [
  {
    id: 'classifier-data',
    stepNumber: 1,
    title: 'Get training data ready',
    descriptionFull:
      "You'll need labeled examples for your classifier to learn from. Upload your own labeled data or use sample data within the training flow.",
    descriptionShort:
      'Prepare labeled examples for training.',
    descriptionMinimal: 'Prepare labeled training data.',
    linkPath: '/chat/data?modal=create',
    linkLabel: 'Create dataset',
  },
  {
    id: 'classifier-create',
    stepNumber: 2,
    title: 'Create your classifier',
    descriptionFull:
      'Configure your categories and train a new classifier model on your labeled data.',
    descriptionShort:
      'Create and train a classifier model.',
    descriptionMinimal: 'Train your classifier model.',
    linkPath: '/chat/models/train/classifier/new',
    linkLabel: 'Create classifier',
  },
  {
    id: 'classifier-test',
    stepNumber: 3,
    title: 'Test your labels',
    descriptionFull:
      'Switch to Classifier mode in Test and see how it categorizes new content. Check if the labels make sense.',
    descriptionShort:
      'Test classification in Classifier mode.',
    descriptionMinimal: 'Test your classifier.',
    linkPath: '/chat/test',
    linkLabel: 'Go to Test',
  },
  {
    id: 'classifier-ship',
    stepNumber: 4,
    title: 'Ship it',
    descriptionFull:
      'Package your project for deployment. This includes your trained classifier model.',
    descriptionShort: 'Package your project for deployment.',
    descriptionMinimal: 'Package for deployment.',
    linkPath: '/chat/dashboard',
    linkLabel: 'Package',
  },
]

const ANOMALY_CHECKLIST: ChecklistStep[] = [
  {
    id: 'anomaly-data',
    stepNumber: 1,
    title: 'Get baseline data ready',
    descriptionFull:
      "You'll need examples of \"normal\" so your detector can learn what's unusual. Upload representative samples of your typical data.",
    descriptionShort:
      'Prepare examples of normal data for training.',
    descriptionMinimal: 'Prepare baseline training data.',
    linkPath: '/chat/data?modal=create',
    linkLabel: 'Create dataset',
  },
  {
    id: 'anomaly-create',
    stepNumber: 2,
    title: 'Create your detector',
    descriptionFull:
      'Train a new anomaly detection model on your baseline data so it can spot unusual patterns.',
    descriptionShort:
      'Create and train an anomaly detector.',
    descriptionMinimal: 'Train your anomaly detector.',
    linkPath: '/chat/models/train/anomaly/new',
    linkLabel: 'Create detector',
  },
  {
    id: 'anomaly-test',
    stepNumber: 3,
    title: 'Test detection',
    descriptionFull:
      'Switch to Anomaly Detection mode in Test and see if it catches the weird stuff. Try both normal and unusual inputs.',
    descriptionShort:
      'Test detection in Anomaly Detection mode.',
    descriptionMinimal: 'Test your detector.',
    linkPath: '/chat/test',
    linkLabel: 'Go to Test',
  },
  {
    id: 'anomaly-ship',
    stepNumber: 4,
    title: 'Ship it',
    descriptionFull:
      'Package your project for deployment. This includes your trained anomaly detection model.',
    descriptionShort: 'Package your project for deployment.',
    descriptionMinimal: 'Package for deployment.',
    linkPath: '/chat/dashboard',
    linkLabel: 'Package',
  },
]

// Shortened checklists for sample-data flows (model already trained, skip "Create" step)
const CLASSIFIER_SAMPLE_CHECKLIST: ChecklistStep[] = [
  {
    id: 'classifier-data',
    stepNumber: 1,
    title: 'View your trained classifier',
    descriptionFull:
      "Your sample classifier is training! Head over to see the progress and try it out when it's ready.",
    descriptionShort:
      'View your sample classifier training progress.',
    descriptionMinimal: 'View trained classifier.',
    linkPath: '/chat/models/train/classifier/new?autoTrain=true',
    linkLabel: 'View classifier',
  },
  {
    id: 'classifier-test',
    stepNumber: 2,
    title: 'Test your labels',
    descriptionFull:
      'Switch to Classifier mode in Test and see how it categorizes new content. Check if the labels make sense.',
    descriptionShort:
      'Test classification in Classifier mode.',
    descriptionMinimal: 'Test your classifier.',
    linkPath: '/chat/test',
    linkLabel: 'Go to Test',
  },
  {
    id: 'classifier-ship',
    stepNumber: 3,
    title: 'Ship it',
    descriptionFull:
      'Package your project for deployment. This includes your trained classifier model.',
    descriptionShort: 'Package your project for deployment.',
    descriptionMinimal: 'Package for deployment.',
    linkPath: '/chat/dashboard',
    linkLabel: 'Package',
  },
]

const ANOMALY_SAMPLE_CHECKLIST: ChecklistStep[] = [
  {
    id: 'anomaly-data',
    stepNumber: 1,
    title: 'View your trained detector',
    descriptionFull:
      "Your sample anomaly detector is training! Head over to see the progress and try it out when it's ready.",
    descriptionShort:
      'View your sample detector training progress.',
    descriptionMinimal: 'View trained detector.',
    linkPath: '/chat/models/train/anomaly/new?autoTrain=true',
    linkLabel: 'View detector',
  },
  {
    id: 'anomaly-test',
    stepNumber: 2,
    title: 'Test detection',
    descriptionFull:
      'Switch to Anomaly Detection mode in Test and see if it catches the weird stuff. Try both normal and unusual inputs.',
    descriptionShort:
      'Test detection in Anomaly Detection mode.',
    descriptionMinimal: 'Test your detector.',
    linkPath: '/chat/test',
    linkLabel: 'Go to Test',
  },
  {
    id: 'anomaly-ship',
    stepNumber: 3,
    title: 'Ship it',
    descriptionFull:
      'Package your project for deployment. This includes your trained anomaly detection model.',
    descriptionShort: 'Package your project for deployment.',
    descriptionMinimal: 'Package for deployment.',
    linkPath: '/chat/dashboard',
    linkLabel: 'Package',
  },
]

const DOC_SCAN_CHECKLIST: ChecklistStep[] = [
  {
    id: 'doc-scan-data',
    stepNumber: 1,
    title: 'Get your docs in',
    descriptionFull:
      'Create a dataset, upload the documents you want to extract information from, and process them.',
    descriptionShort:
      'Upload and process your documents.',
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
    descriptionShort:
      'Test extraction in Doc Scanning mode.',
    descriptionMinimal: 'Test document extraction.',
    linkPath: '/chat/test',
    linkLabel: 'Go to Test',
  },
  {
    id: 'doc-scan-ship',
    stepNumber: 3,
    title: 'Ship it',
    descriptionFull:
      'Package your project for deployment. This creates everything you need to run document scanning in production.',
    descriptionShort: 'Package your project for deployment.',
    descriptionMinimal: 'Package for deployment.',
    linkPath: '/chat/dashboard',
    linkLabel: 'Package',
  },
]

const EXPLORING_CHECKLIST: ChecklistStep[] = [
  {
    id: 'exploring-sample',
    stepNumber: 1,
    title: 'Find & import data',
    descriptionFull:
      "Find data on Hugging Face or generate synthetic data, then import it.",
    descriptionShort:
      'Import a sample dataset to experiment with.',
    descriptionMinimal: 'Import sample data.',
    linkPath: '/chat/data?modal=import',
    linkLabel: 'Import sample',
  },
  {
    id: 'exploring-test',
    stepNumber: 2,
    title: 'Try the Test page',
    descriptionFull:
      'Chat with the sample data, try different modes (Text Generation, Classifier, etc.). Get a feel for what LlamaFarm can do.',
    descriptionShort:
      'Explore different modes in the Test page.',
    descriptionMinimal: 'Explore the Test page.',
    linkPath: '/chat/test',
    linkLabel: 'Go to Test',
  },
  {
    id: 'exploring-decide',
    stepNumber: 3,
    title: 'Pick a direction',
    descriptionFull:
      "Ready to build something real? Come back here and start over with a specific project type. We'll give you a focused checklist.",
    descriptionShort:
      'Start over with a specific project type.',
    descriptionMinimal: 'Choose a project type.',
    linkPath: '',
    linkLabel: 'Start over',
  },
]

/**
 * Demo project checklist - simplified flow for pre-built demo projects
 * (LlamaEncyclopedia, Santa Helper, etc.)
 */
function createDemoChecklist(demo: FileBasedDemo): ChecklistStep[] {
  return [
    {
      id: 'demo-prompt',
      stepNumber: 1,
      title: 'Check out the prompt',
      descriptionFull:
        `See how we set up ${demo.displayName}. The system prompt tells the AI what it knows and how to respond.`,
      descriptionShort:
        'Review the system prompt configuration.',
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
      descriptionShort:
        'Test the demo in the chat interface.',
      descriptionMinimal: 'Try the demo.',
      linkPath: '/chat/test',
      linkLabel: 'Go to Test',
    },
    {
      id: 'demo-build',
      stepNumber: 3,
      title: 'Build your own',
      descriptionFull:
        "Ready to create something custom? Start fresh with your own data, prompts, and configuration.",
      descriptionShort:
        'Start over to build your own project.',
      descriptionMinimal: 'Build your own.',
      linkPath: '',
      linkLabel: 'Start over',
    },
  ]
}

/**
 * Get the base checklist for a project type
 * For sample-data flows with classifier/anomaly, use shortened checklists (no "Create" step)
 */
function getBaseChecklist(projectType: ProjectType, dataStatus?: DataStatus | null): ChecklistStep[] {
  // Use shortened sample checklists for classifier/anomaly with sample-data
  if (dataStatus === 'sample-data') {
    if (projectType === 'classifier') {
      return CLASSIFIER_SAMPLE_CHECKLIST.map(step => ({ ...step }))
    }
    if (projectType === 'anomaly') {
      return ANOMALY_SAMPLE_CHECKLIST.map(step => ({ ...step }))
    }
  }

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
 * Modify the first step based on data status and selected HF dataset
 */
function applyDataStatusModifications(
  checklist: ChecklistStep[],
  dataStatus: DataStatus,
  projectType: ProjectType,
  selectedHFDataset?: SelectedHFDataset | null
): ChecklistStep[] {
  if (checklist.length === 0) return checklist

  const modified = [...checklist]
  const firstStep = { ...modified[0] }

  if (dataStatus === 'sample-data') {
    // For classifier/anomaly, the first step should be "View your trained model"
    // (training happens automatically in background)
    if (projectType === 'classifier') {
      firstStep.title = 'View your trained classifier'
      firstStep.descriptionFull =
        "Your sample classifier is training! Head over to see the progress and try it out when it's ready."
      firstStep.descriptionShort =
        'View your sample classifier training progress.'
      firstStep.descriptionMinimal = 'View trained classifier.'
      firstStep.linkPath = '/chat/models/train/classifier/new?autoTrain=true'
      firstStep.linkLabel = 'View classifier'
    } else if (projectType === 'anomaly') {
      firstStep.title = 'View your trained detector'
      firstStep.descriptionFull =
        "Your sample anomaly detector is training! Head over to see the progress and try it out when it's ready."
      firstStep.descriptionShort =
        'View your sample detector training progress.'
      firstStep.descriptionMinimal = 'View trained detector.'
      firstStep.linkPath = '/chat/models/train/anomaly/new?autoTrain=true'
      firstStep.linkLabel = 'View detector'
    } else {
      // For doc-qa and other types, keep the original sample data flow
      firstStep.title = 'Load sample data'
      firstStep.descriptionFull =
        "Start with our sample dataset to see how things work. You can always swap in your own data later."
      firstStep.descriptionShort =
        'Import a sample dataset to get started.'
      firstStep.descriptionMinimal = 'Import sample data.'
      firstStep.linkPath = '/chat/data?modal=import'
      firstStep.linkLabel = 'Import sample'
    }
  } else if (dataStatus === 'need-data') {
    // If a HF dataset was selected, customize the first step to show they're importing it
    if (selectedHFDataset) {
      // Generate the same dataset name that Dashboard uses for the import
      const datasetName = `hf_${selectedHFDataset.id.replace(/\//g, '_')}`
      firstStep.title = 'Check out your imported dataset'
      firstStep.descriptionFull =
        `We're importing "${selectedHFDataset.name}" from Hugging Face in the background. Head to your dataset to see the import progress and explore your new data.`
      firstStep.descriptionShort =
        `Your "${selectedHFDataset.name}" dataset is being imported. Check its status.`
      firstStep.descriptionMinimal = 'View your imported HF dataset.'
      firstStep.linkPath = `/chat/data/${encodeURIComponent(datasetName)}`
      firstStep.linkLabel = 'View dataset'
    } else {
      firstStep.title = 'Find & import data'
      firstStep.descriptionFull =
        "Check out Hugging Face datasets or synthetic data generators to find data for your project. Once you have files, come back and create a dataset."
      firstStep.descriptionShort =
        'Find data on Hugging Face or generate synthetic data, then import it.'
      firstStep.descriptionMinimal = 'Find and import data.'
      firstStep.linkPath = '/chat/data?modal=import'
      firstStep.linkLabel = 'Import sample'
    }
  }

  // For classifier and anomaly, the first step points to the training page if they have data
  if (projectType === 'classifier' && dataStatus === 'has-data') {
    firstStep.linkPath = '/chat/models/train/classifier/new'
    firstStep.linkLabel = 'Create classifier'
  } else if (projectType === 'anomaly' && dataStatus === 'has-data') {
    firstStep.linkPath = '/chat/models/train/anomaly/new'
    firstStep.linkLabel = 'Create detector'
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
  selectedHFDataset?: SelectedHFDataset | null
  trainedModelName?: string | null
  trainedModelType?: 'classifier' | 'anomaly' | null
}

/**
 * Generate a personalized checklist based on user's answers
 */
export function generateChecklist(
  projectType: ProjectType | null,
  dataStatus: DataStatus | null,
  selectedHFDataset?: SelectedHFDataset | null,
  trainedModelName?: string | null,
  trainedModelType?: 'classifier' | 'anomaly' | null,
  uploadedFilesCount?: number,
  datasetName?: string | null
): ChecklistStep[] {
  if (!projectType) {
    return []
  }

  // Pass dataStatus so we get the right base checklist (sample flows have fewer steps)
  let checklist = getBaseChecklist(projectType, dataStatus)

  // Apply modifications for non-sample flows (sample flows already have correct first step)
  if (dataStatus && dataStatus !== 'sample-data') {
    checklist = applyDataStatusModifications(checklist, dataStatus, projectType, selectedHFDataset)
  } else if (dataStatus === 'sample-data' && projectType !== 'classifier' && projectType !== 'anomaly') {
    // For non-classifier/anomaly sample data flows, still apply modifications
    checklist = applyDataStatusModifications(checklist, dataStatus, projectType, selectedHFDataset)
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
        descriptionShort:
          `Check your ${uploadedFilesCount} uploaded ${fileWord} and process them.`,
        descriptionMinimal: 'View and process uploaded data.',
        linkPath: `/chat/data/${encodeURIComponent(dsName)}`,
        linkLabel: 'View dataset',
      }
    }
  }

  // If a model was trained from sample data, update the first step's link to the trained model
  if (trainedModelName && trainedModelType && dataStatus === 'sample-data') {
    const modelPath = trainedModelType === 'classifier' ? 'classifier' : 'anomaly'
    const modelLabel = trainedModelType === 'classifier' ? 'classifier' : 'detector'
    const newLinkPath = `/chat/models/train/${modelPath}/${encodeURIComponent(trainedModelName)}`

    console.log('[checklistGenerator] Updating link path to trained model:', newLinkPath)

    // Update the first step to point to the trained model (with updated description)
    if (checklist.length > 0) {
      checklist[0] = {
        ...checklist[0],
        descriptionFull:
          `Your sample ${modelLabel} is ready! Check it out and try it with some test inputs.`,
        descriptionShort:
          `View and test your trained ${modelLabel}.`,
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
 * Returns null if not a demo project (call generateChecklist instead)
 */
export function generateDemoChecklist(demoConfig: FileBasedDemo | undefined): ChecklistStep[] | null {
  if (!demoConfig) return null
  return createDemoChecklist(demoConfig)
}
