/**
 * Sample datasets for classifier and anomaly detection demo/testing
 * Used by both the model training pages and auto-training from onboarding
 */

// Re-export anomaly sample datasets
export { SAMPLE_DATASETS as ANOMALY_SAMPLE_DATASETS } from './sampleData'
export type { SampleDataset as AnomalySampleDataset } from './sampleData'

// Classifier sample datasets
export interface ClassifierSampleDataset {
  id: string
  name: string
  description: string
  classes: number
  examples: number
  data: Array<{ text: string; label: string }>
}

export const CLASSIFIER_SAMPLE_DATASETS: ClassifierSampleDataset[] = [
  {
    id: 'sentiment',
    name: 'Sentiment analysis',
    description: '3 classes, 72 examples',
    classes: 3,
    examples: 72,
    data: [
      { text: 'Love this product so much!', label: 'positive' },
      { text: 'Absolutely terrible experience', label: 'negative' },
      { text: 'The package arrived on Tuesday', label: 'neutral' },
      { text: 'Best purchase I\'ve ever made', label: 'positive' },
      { text: 'Complete waste of money', label: 'negative' },
      { text: 'It comes in three colors', label: 'neutral' },
      { text: 'Exceeded all my expectations', label: 'positive' },
      { text: 'Would not recommend to anyone', label: 'negative' },
      { text: 'The dimensions are listed on the box', label: 'neutral' },
      { text: 'Amazing quality for the price', label: 'positive' },
      { text: 'Broke after one day of use', label: 'negative' },
      { text: 'Shipping took five days', label: 'neutral' },
      { text: 'Can\'t imagine life without it now', label: 'positive' },
      { text: 'Customer service was unhelpful', label: 'negative' },
      { text: 'Made of plastic and metal', label: 'neutral' },
      { text: 'Five stars isn\'t enough', label: 'positive' },
      { text: 'Wish I could get a refund', label: 'negative' },
      { text: 'Available in stores and online', label: 'neutral' },
      { text: 'My whole family loves it', label: 'positive' },
      { text: 'Cheaply made garbage', label: 'negative' },
      { text: 'The manual is 20 pages', label: 'neutral' },
      { text: 'Worth every single penny', label: 'positive' },
      { text: 'Never buying from here again', label: 'negative' },
      { text: 'Comes with a one year warranty', label: 'neutral' },
      { text: 'Incredible value', label: 'positive' },
      { text: 'Disappointed with the quality', label: 'negative' },
      { text: 'Standard size fits most', label: 'neutral' },
      { text: 'Makes my life so much easier', label: 'positive' },
      { text: 'Arrived damaged and broken', label: 'negative' },
      { text: 'Ships from California', label: 'neutral' },
      { text: 'Highly recommend to everyone', label: 'positive' },
      { text: 'Not as described at all', label: 'negative' },
      { text: 'Battery life is 8 hours', label: 'neutral' },
      { text: 'Perfect gift for anyone', label: 'positive' },
      { text: 'Terrible build quality', label: 'negative' },
      { text: 'Weight is approximately 2 pounds', label: 'neutral' },
      { text: 'Obsessed with this thing', label: 'positive' },
      { text: 'Completely misleading photos', label: 'negative' },
      { text: 'Instructions are included', label: 'neutral' },
      { text: 'Game changer for real', label: 'positive' },
      { text: 'Falls apart immediately', label: 'negative' },
      { text: 'Compatible with most devices', label: 'neutral' },
      { text: 'So happy with this purchase', label: 'positive' },
      { text: 'Regret buying this', label: 'negative' },
      { text: 'The color is blue', label: 'neutral' },
      { text: 'Fantastic product overall', label: 'positive' },
      { text: 'Worst quality imaginable', label: 'negative' },
      { text: 'Measures 10 inches long', label: 'neutral' },
      { text: 'Absolutely love it', label: 'positive' },
      { text: 'Total disappointment', label: 'negative' },
      { text: 'Made in China', label: 'neutral' },
      { text: 'Best gift I\'ve given', label: 'positive' },
      { text: 'Does not work properly', label: 'negative' },
      { text: 'Requires two AA batteries', label: 'neutral' },
      { text: 'Exceeded expectations completely', label: 'positive' },
      { text: 'Poor customer support', label: 'negative' },
      { text: 'Available for pickup', label: 'neutral' },
      { text: 'Would buy again instantly', label: 'positive' },
      { text: 'Money down the drain', label: 'negative' },
      { text: 'Fits in standard compartments', label: 'neutral' },
      { text: 'Super impressed with quality', label: 'positive' },
      { text: 'Falling apart already', label: 'negative' },
      { text: 'Packaging is recyclable', label: 'neutral' },
      { text: 'Life changing purchase', label: 'positive' },
      { text: 'Horrible experience overall', label: 'negative' },
      { text: 'Delivery was contactless', label: 'neutral' },
      { text: 'Everyone should own this', label: 'positive' },
      { text: 'Very cheaply constructed', label: 'negative' },
      { text: 'Sold by authorized dealers', label: 'neutral' },
      { text: 'Remarkable quality', label: 'positive' },
      { text: 'Would give zero stars if possible', label: 'negative' },
      { text: 'Product code is on label', label: 'neutral' },
    ],
  },
  {
    id: 'expense',
    name: 'Expense reports',
    description: '5 classes, 40 examples',
    classes: 5,
    examples: 40,
    data: [
      { text: 'Uber to airport', label: 'travel' },
      { text: 'Team lunch at Chipotle', label: 'meals' },
      { text: 'Adobe Creative Cloud subscription', label: 'software' },
      { text: 'Staples office supplies', label: 'supplies' },
      { text: 'Hilton hotel 2 nights', label: 'lodging' },
      { text: 'Delta flight to Chicago', label: 'travel' },
      { text: 'Coffee with client', label: 'meals' },
      { text: 'Slack annual license', label: 'software' },
      { text: 'Printer paper 10 reams', label: 'supplies' },
      { text: 'Marriott conference stay', label: 'lodging' },
      { text: 'Lyft to client meeting', label: 'travel' },
      { text: 'Dinner with prospects', label: 'meals' },
      { text: 'Zoom pro subscription', label: 'software' },
      { text: 'Pens and notebooks', label: 'supplies' },
      { text: 'Airbnb for trade show', label: 'lodging' },
      { text: 'Train ticket to Boston', label: 'travel' },
      { text: 'Lunch during workshop', label: 'meals' },
      { text: 'Figma team license', label: 'software' },
      { text: 'Toner cartridges', label: 'supplies' },
      { text: 'Hotel & suites 3 nights', label: 'lodging' },
      { text: 'Taxi from airport', label: 'travel' },
      { text: 'Team dinner celebration', label: 'meals' },
      { text: 'GitHub enterprise', label: 'software' },
      { text: 'Sticky notes and markers', label: 'supplies' },
      { text: 'Hampton Inn business trip', label: 'lodging' },
      { text: 'Mileage reimbursement 45 miles', label: 'travel' },
      { text: 'Breakfast meeting', label: 'meals' },
      { text: 'Notion workspace', label: 'software' },
      { text: 'Filing folders', label: 'supplies' },
      { text: 'Extended stay hotel', label: 'lodging' },
      { text: 'Rental car 3 days', label: 'travel' },
      { text: 'Client entertainment dinner', label: 'meals' },
      { text: 'Microsoft 365 license', label: 'software' },
      { text: 'Desk organizer', label: 'supplies' },
      { text: 'Holiday Inn express', label: 'lodging' },
      { text: 'Parking at conference', label: 'travel' },
      { text: 'Working lunch catering', label: 'meals' },
      { text: 'Salesforce subscription', label: 'software' },
      { text: 'Whiteboard markers', label: 'supplies' },
      { text: 'Hyatt regency 2 nights', label: 'lodging' },
    ],
  },
]

/**
 * Get classifier sample dataset by ID
 */
export function getClassifierSampleDataset(id: string): ClassifierSampleDataset | undefined {
  return CLASSIFIER_SAMPLE_DATASETS.find(d => d.id === id)
}

/**
 * Get anomaly sample dataset by ID
 */
export function getAnomalySampleDataset(id: string) {
  // Import dynamically to avoid circular dependency
  const { SAMPLE_DATASETS } = require('./sampleData')
  return SAMPLE_DATASETS.find((d: { id: string }) => d.id === id)
}
