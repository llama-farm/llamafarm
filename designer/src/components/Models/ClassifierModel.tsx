import { useState, useEffect, useCallback, useMemo, useRef } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { Button } from '../ui/button'
import { Input } from '../ui/input'
import { Label } from '../ui/label'
import { Textarea } from '../ui/textarea'
import { Badge } from '../ui/badge'
import { Checkbox } from '../ui/checkbox'
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '../ui/dialog'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '../ui/dropdown-menu'
import { useToast } from '../ui/toast'
import FontIcon from '../../common/FontIcon'
import TrainingLoadingOverlay from './TrainingLoadingOverlay'
import type { ClassifierTestResult } from './types'
import {
  useListClassifierModels,
  useTrainAndSaveClassifier,
  usePredictClassifier,
  useLoadClassifier,
  useDeleteClassifierModel,
} from '../../hooks/useMLModels'
import {
  parseVersionedModelName,
  formatModelTimestamp,
  generateUniqueModelName,
  type ClassifierModelInfo,
  type ClassifierTrainingData,
} from '../../types/ml'

type TrainingState = 'idle' | 'training' | 'success' | 'error'

// Default SetFit base model
const DEFAULT_BASE_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

// Available embedder models for classifier training
const EMBEDDER_MODELS = [
  { id: 'sentence-transformers/all-MiniLM-L6-v2', label: 'all-MiniLM-L6-v2 (default)', dim: 384 },
  { id: 'BAAI/bge-small-en-v1.5', label: 'bge-small-en-v1.5', dim: 384 },
  { id: 'BAAI/bge-base-en-v1.5', label: 'bge-base-en-v1.5', dim: 768 },
  { id: 'BAAI/bge-large-en-v1.5', label: 'bge-large-en-v1.5', dim: 1024 },
  { id: 'BAAI/bge-m3', label: 'bge-m3 (multilingual)', dim: 1024 },
  { id: 'intfloat/e5-base-v2', label: 'e5-base-v2', dim: 768 },
  { id: 'intfloat/e5-large-v2', label: 'e5-large-v2', dim: 1024 },
]

// Sample datasets for classifier training
interface ClassifierSampleDataset {
  id: string
  name: string
  description: string
  classes: number
  examples: number
  data: Array<{ text: string; label: string }>
}

const CLASSIFIER_SAMPLE_DATASETS: ClassifierSampleDataset[] = [
  {
    id: 'sentiment',
    name: 'Sentiment analysis',
    description: '3 classes, 200 examples',
    classes: 3,
    examples: 200,
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
      { text: 'Better than expected', label: 'positive' },
      { text: 'Extremely frustrating product', label: 'negative' },
      { text: 'Return policy is 30 days', label: 'neutral' },
      { text: 'Such a great find', label: 'positive' },
      { text: 'Complete junk', label: 'negative' },
      { text: 'Assembled in Mexico', label: 'neutral' },
      { text: 'Pleasantly surprised', label: 'positive' },
      { text: 'Don\'t waste your time', label: 'negative' },
      { text: 'UPC code included', label: 'neutral' },
      { text: 'Top notch quality', label: 'positive' },
      { text: 'Misleading product description', label: 'negative' },
      { text: 'Dimensions as advertised', label: 'neutral' },
      { text: 'Really happy with this', label: 'positive' },
      { text: 'Feels like a scam', label: 'negative' },
      { text: 'Standard ground shipping', label: 'neutral' },
      { text: 'Incredible purchase', label: 'positive' },
      { text: 'Very poorly made', label: 'negative' },
      { text: 'Item weight as listed', label: 'neutral' },
      { text: 'So glad I bought this', label: 'positive' },
      { text: 'Unusable after a week', label: 'negative' },
      { text: 'Multiple colors available', label: 'neutral' },
      { text: 'Must have item', label: 'positive' },
      { text: 'Extremely disappointed', label: 'negative' },
      { text: 'Tracking number provided', label: 'neutral' },
      { text: 'Wonderful product', label: 'positive' },
      { text: 'Low quality materials', label: 'negative' },
      { text: 'Express shipping available', label: 'neutral' },
      { text: 'Thrilled with purchase', label: 'positive' },
      { text: 'Defective right out of box', label: 'negative' },
      { text: 'Gift wrapping offered', label: 'neutral' },
      { text: 'Amazing find', label: 'positive' },
      { text: 'Not worth the hassle', label: 'negative' },
      { text: 'Bulk pricing available', label: 'neutral' },
      { text: 'Perfect in every way', label: 'positive' },
      { text: 'Broke during first use', label: 'negative' },
      { text: 'Store hours vary', label: 'neutral' },
      { text: 'Fantastic quality', label: 'positive' },
      { text: 'Very misleading listing', label: 'negative' },
      { text: 'Free returns accepted', label: 'neutral' },
      { text: 'Love love love this', label: 'positive' },
      { text: 'Awful product overall', label: 'negative' },
      { text: 'Same day delivery option', label: 'neutral' },
      { text: 'Best decision ever', label: 'positive' },
      { text: 'Huge waste of money', label: 'negative' },
      { text: 'Price match guaranteed', label: 'neutral' },
      { text: 'Couldn\'t be happier', label: 'positive' },
      { text: 'Cheapest quality ever', label: 'negative' },
      { text: 'Curbside pickup ready', label: 'neutral' },
      { text: 'Impressive product', label: 'positive' },
      { text: 'Do not recommend', label: 'negative' },
      { text: 'Member discounts apply', label: 'neutral' },
      { text: 'Awesome purchase', label: 'positive' },
      { text: 'Terrible terrible terrible', label: 'negative' },
      { text: 'Loyalty points earned', label: 'neutral' },
      { text: 'So worth it', label: 'positive' },
      { text: 'Absolutely awful', label: 'negative' },
      { text: 'Gift cards accepted', label: 'neutral' },
      { text: 'Delightful product', label: 'positive' },
      { text: 'Frustrating experience', label: 'negative' },
      { text: 'Subscribe and save option', label: 'neutral' },
      { text: 'Brilliant quality', label: 'positive' },
      { text: 'Very unsatisfied customer', label: 'negative' },
      { text: 'Clearance items final sale', label: 'neutral' },
      { text: 'Outstanding product', label: 'positive' },
      { text: 'Would never buy again', label: 'negative' },
      { text: 'Size chart provided', label: 'neutral' },
      { text: 'Phenomenal purchase', label: 'positive' },
      { text: 'Poor poor quality', label: 'negative' },
      { text: 'Installation guide online', label: 'neutral' },
      { text: 'Superb quality', label: 'positive' },
      { text: 'Waste of packaging', label: 'negative' },
      { text: 'Customer reviews enabled', label: 'neutral' },
      { text: 'Genuinely impressed', label: 'positive' },
      { text: 'Pathetic product', label: 'negative' },
      { text: 'In stock notification', label: 'neutral' },
      { text: 'Remarkable find', label: 'positive' },
      { text: 'Horribly disappointing', label: 'negative' },
      { text: 'Wishlist feature available', label: 'neutral' },
      { text: 'Stellar product', label: 'positive' },
      { text: 'Inferior quality', label: 'negative' },
      { text: 'Compare prices online', label: 'neutral' },
      { text: 'Blown away by this', label: 'positive' },
      { text: 'Completely useless', label: 'negative' },
      { text: 'Shipping calculator provided', label: 'neutral' },
      { text: 'Top quality item', label: 'positive' },
      { text: 'Beyond disappointed', label: 'negative' },
      { text: 'Order history saved', label: 'neutral' },
      { text: 'Sensational product', label: 'positive' },
      { text: 'Dreadful experience', label: 'negative' },
      { text: 'Account required to purchase', label: 'neutral' },
      { text: 'Marvelous quality', label: 'positive' },
      { text: 'Subpar materials used', label: 'negative' },
      { text: 'Newsletter signup offered', label: 'neutral' },
      { text: 'Exceptional find', label: 'positive' },
      { text: 'Horrendous product', label: 'negative' },
      { text: 'Social media links below', label: 'neutral' },
      { text: 'First rate quality', label: 'positive' },
      { text: 'Abysmal customer service', label: 'negative' },
      { text: 'FAQ section available', label: 'neutral' },
      { text: 'Premium product', label: 'positive' },
      { text: 'Laughably bad quality', label: 'negative' },
      { text: 'Live chat support hours', label: 'neutral' },
      { text: 'Superior craftsmanship', label: 'positive' },
      { text: 'Shockingly poor product', label: 'negative' },
      { text: 'Email confirmation sent', label: 'neutral' },
      { text: 'Impeccable quality', label: 'positive' },
      { text: 'Unbelievably bad', label: 'negative' },
      { text: 'Mobile app available', label: 'neutral' },
      { text: 'Exquisite product', label: 'positive' },
      { text: 'Disastrous purchase', label: 'negative' },
      { text: 'Browser cookies used', label: 'neutral' },
      { text: 'Magnificent find', label: 'positive' },
      { text: 'Appalling quality', label: 'negative' },
      { text: 'Terms and conditions apply', label: 'neutral' },
      { text: 'Splendid product', label: 'positive' },
      { text: 'Atrocious experience', label: 'negative' },
      { text: 'Privacy policy updated', label: 'neutral' },
    ],
  },
  {
    id: 'expense',
    name: 'Expense reports',
    description: '5 classes, 200 examples',
    classes: 5,
    examples: 200,
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
      { text: 'Microsoft 365 subscription', label: 'software' },
      { text: 'Filing folders', label: 'supplies' },
      { text: 'Hyatt for customer visit', label: 'lodging' },
      { text: 'Parking at conference', label: 'travel' },
      { text: 'Catering for workshop', label: 'meals' },
      { text: 'Jira subscription', label: 'software' },
      { text: 'Desk organizers', label: 'supplies' },
      { text: 'Holiday Inn 1 night', label: 'lodging' },
      { text: 'Toll roads roundtrip', label: 'travel' },
      { text: 'Pizza for late night work', label: 'meals' },
      { text: 'Notion team plan', label: 'software' },
      { text: 'Whiteboard markers', label: 'supplies' },
      { text: 'Courtyard Marriott stay', label: 'lodging' },
      { text: 'Rental car 3 days', label: 'travel' },
      { text: 'Snacks for meeting', label: 'meals' },
      { text: 'Dropbox business', label: 'software' },
      { text: 'Binders and dividers', label: 'supplies' },
      { text: 'Best Western 2 nights', label: 'lodging' },
      { text: 'Airport parking 4 days', label: 'travel' },
      { text: 'Client appreciation dinner', label: 'meals' },
      { text: 'Asana premium', label: 'software' },
      { text: 'Envelopes bulk pack', label: 'supplies' },
      { text: 'Residence Inn extended stay', label: 'lodging' },
      { text: 'Uber to downtown office', label: 'travel' },
      { text: 'Lunch and learn catering', label: 'meals' },
      { text: 'Canva pro subscription', label: 'software' },
      { text: 'Paper clips and staples', label: 'supplies' },
      { text: 'Sheraton conference hotel', label: 'lodging' },
      { text: 'Lyft from hotel', label: 'travel' },
      { text: 'Team happy hour', label: 'meals' },
      { text: 'Trello business class', label: 'software' },
      { text: 'Scissors and tape', label: 'supplies' },
      { text: 'Doubletree by Hilton', label: 'lodging' },
      { text: 'Gas for rental car', label: 'travel' },
      { text: 'Donuts for team', label: 'meals' },
      { text: 'Mailchimp subscription', label: 'software' },
      { text: 'Printer ink black', label: 'supplies' },
      { text: 'Crowne Plaza 2 nights', label: 'lodging' },
      { text: 'Subway fare monthly', label: 'travel' },
      { text: 'Lunch with vendor', label: 'meals' },
      { text: 'HubSpot marketing', label: 'software' },
      { text: 'Label maker tape', label: 'supplies' },
      { text: 'Radisson business trip', label: 'lodging' },
      { text: 'Bridge toll', label: 'travel' },
      { text: 'Bagels for morning meeting', label: 'meals' },
      { text: 'Salesforce license', label: 'software' },
      { text: 'Stamps postage', label: 'supplies' },
      { text: 'Westin hotel 1 night', label: 'lodging' },
      { text: 'Amtrak business class', label: 'travel' },
      { text: 'Dinner during travel', label: 'meals' },
      { text: 'Atlassian suite', label: 'software' },
      { text: 'Copy paper white', label: 'supplies' },
      { text: 'Aloft hotel stay', label: 'lodging' },
      { text: 'Taxi to train station', label: 'travel' },
      { text: 'Coffee and pastries', label: 'meals' },
      { text: 'Monday.com subscription', label: 'software' },
      { text: 'Highlighters assorted', label: 'supplies' },
      { text: 'Four Points stay', label: 'lodging' },
      { text: 'Rideshare to conference', label: 'travel' },
      { text: 'Sandwich trays', label: 'meals' },
      { text: 'Miro team license', label: 'software' },
      { text: 'Rubber bands box', label: 'supplies' },
      { text: 'SpringHill Suites', label: 'lodging' },
      { text: 'Parking garage monthly', label: 'travel' },
      { text: 'Team building lunch', label: 'meals' },
      { text: 'Webflow subscription', label: 'software' },
      { text: 'Push pins', label: 'supplies' },
      { text: 'Fairfield Inn stay', label: 'lodging' },
      { text: 'Flight change fee', label: 'travel' },
      { text: 'Working lunch supplies', label: 'meals' },
      { text: 'Airtable pro plan', label: 'software' },
      { text: 'Correction tape', label: 'supplies' },
      { text: 'AC Hotel 2 nights', label: 'lodging' },
      { text: 'Car service to airport', label: 'travel' },
      { text: 'Interview candidate lunch', label: 'meals' },
      { text: 'Linear subscription', label: 'software' },
      { text: 'Pencils mechanical', label: 'supplies' },
      { text: 'Element hotel extended', label: 'lodging' },
      { text: 'Metro card reload', label: 'travel' },
      { text: 'Fruit platter meeting', label: 'meals' },
      { text: 'Loom business plan', label: 'software' },
      { text: 'Index cards', label: 'supplies' },
      { text: 'Homewood Suites stay', label: 'lodging' },
      { text: 'Valet parking', label: 'travel' },
      { text: 'Appetizers for event', label: 'meals' },
      { text: 'Calendly teams', label: 'software' },
      { text: 'Clipboard', label: 'supplies' },
      { text: 'Embassy Suites 1 night', label: 'lodging' },
      { text: 'Uber pool to office', label: 'travel' },
      { text: 'Drinks with team', label: 'meals' },
      { text: 'Grammarly business', label: 'software' },
      { text: 'Batteries AA pack', label: 'supplies' },
      { text: 'TownePlace Suites', label: 'lodging' },
      { text: 'Lyft to dinner', label: 'travel' },
      { text: 'Sushi team lunch', label: 'meals' },
      { text: '1Password teams', label: 'software' },
      { text: 'Calculator', label: 'supplies' },
      { text: 'Staybridge Suites', label: 'lodging' },
      { text: 'Airport shuttle', label: 'travel' },
      { text: 'Thai food delivery', label: 'meals' },
      { text: 'Zapier subscription', label: 'software' },
      { text: 'Desk lamp', label: 'supplies' },
      { text: 'Candlewood Suites', label: 'lodging' },
      { text: 'Bus fare', label: 'travel' },
      { text: 'Burgers for team', label: 'meals' },
      { text: 'Intercom subscription', label: 'software' },
      { text: 'Mouse pad', label: 'supplies' },
      { text: 'Hyatt Place stay', label: 'lodging' },
      { text: 'Toll tag reload', label: 'travel' },
      { text: 'Tacos for lunch', label: 'meals' },
      { text: 'Amplitude analytics', label: 'software' },
      { text: 'USB cables', label: 'supplies' },
      { text: 'Hyatt House extended', label: 'lodging' },
      { text: 'Parking meter', label: 'travel' },
      { text: 'Wings for game night', label: 'meals' },
      { text: 'Mixpanel subscription', label: 'software' },
      { text: 'Phone charger', label: 'supplies' },
      { text: 'Home2 Suites stay', label: 'lodging' },
      { text: 'Rental car gas refill', label: 'travel' },
      { text: 'BBQ team event', label: 'meals' },
      { text: 'Segment subscription', label: 'software' },
      { text: 'Extension cord', label: 'supplies' },
      { text: 'Tru by Hilton', label: 'lodging' },
      { text: 'Ferry ticket', label: 'travel' },
      { text: 'Mediterranean lunch', label: 'meals' },
      { text: 'Hotjar subscription', label: 'software' },
      { text: 'Surge protector', label: 'supplies' },
      { text: 'Moxy hotel stay', label: 'lodging' },
      { text: 'Congestion charge', label: 'travel' },
      { text: 'Poke bowls team', label: 'meals' },
      { text: 'FullStory license', label: 'software' },
      { text: 'Webcam', label: 'supplies' },
      { text: 'Aloft 2 nights', label: 'lodging' },
      { text: 'Bike share rental', label: 'travel' },
      { text: 'Indian food delivery', label: 'meals' },
      { text: 'Typeform subscription', label: 'software' },
      { text: 'Headphone splitter', label: 'supplies' },
      { text: 'Motto by Hilton', label: 'lodging' },
      { text: 'Scooter rental', label: 'travel' },
      { text: 'Ramen team lunch', label: 'meals' },
      { text: 'SurveyMonkey', label: 'software' },
      { text: 'Cable organizer', label: 'supplies' },
      { text: 'Tempo by Hilton', label: 'lodging' },
      { text: 'Train monthly pass', label: 'travel' },
      { text: 'Greek food catering', label: 'meals' },
      { text: 'Clearbit subscription', label: 'software' },
      { text: 'Monitor stand', label: 'supplies' },
      { text: 'Caption by Hyatt', label: 'lodging' },
      { text: 'Express toll lane', label: 'travel' },
      { text: 'Dim sum team', label: 'meals' },
      { text: 'Gong subscription', label: 'software' },
      { text: 'Keyboard wrist rest', label: 'supplies' },
      { text: 'Thompson hotel', label: 'lodging' },
      { text: 'Helicopter transfer', label: 'travel' },
      { text: 'Steakhouse dinner', label: 'meals' },
      { text: 'Outreach subscription', label: 'software' },
      { text: 'Laptop sleeve', label: 'supplies' },
      { text: 'W Hotel stay', label: 'lodging' },
      { text: 'Water taxi', label: 'travel' },
      { text: 'Seafood team dinner', label: 'meals' },
      { text: 'ZoomInfo license', label: 'software' },
      { text: 'Document scanner', label: 'supplies' },
      { text: 'Le Meridien stay', label: 'lodging' },
      { text: 'Cable car fare', label: 'travel' },
      { text: 'Vietnamese lunch', label: 'meals' },
      { text: 'Drift subscription', label: 'software' },
      { text: 'Wireless mouse', label: 'supplies' },
      { text: 'St Regis 1 night', label: 'lodging' },
      { text: 'Tram ticket', label: 'travel' },
      { text: 'Korean BBQ team', label: 'meals' },
      { text: 'Chorus subscription', label: 'software' },
      { text: 'USB hub', label: 'supplies' },
      { text: 'Ritz Carlton stay', label: 'lodging' },
      { text: 'Monorail fare', label: 'travel' },
      { text: 'Brunch meeting', label: 'meals' },
      { text: 'Clari subscription', label: 'software' },
      { text: 'HDMI cable', label: 'supplies' },
    ],
  },
]

interface ClassLabel {
  id: string
  name: string
  examples: string[]
}

interface ModelVersion {
  id: string
  versionNumber: number
  versionedName: string
  createdAt: string
  trainingSamples: number
  isActive: boolean
  labels: string[]
}

interface ClassifierTableRow {
  id: string
  className: string
  example: string
}

function ClassifierModel() {
  const navigate = useNavigate()
  const { id } = useParams<{ id: string }>()
  const isNewModel = !id || id === 'new'

  // Form state - modelName will be set after loading existing models
  const [modelName, setModelName] = useState('')
  const [description, setDescription] = useState('')
  const [nameExistsWarning, setNameExistsWarning] = useState(false)

  // Class labels state
  const [classLabels, setClassLabels] = useState<ClassLabel[]>([
    { id: '1', name: '', examples: [] },
    { id: '2', name: '', examples: [] },
  ])
  const [activeClassId, setActiveClassId] = useState<string>('1')
  const [trainingDataInput, setTrainingDataInput] = useState('')

  // Input mode toggle: 'text' for textarea per class, 'table' for all data in table
  const [inputMode, setInputMode] = useState<'text' | 'table'>('text')

  // Table view state - rows with class and example
  const [tableRows, setTableRows] = useState<ClassifierTableRow[]>([])

  // CSV import modal state
  const [showCsvModal, setShowCsvModal] = useState(false)
  const [csvFile, setCsvFile] = useState<File | null>(null)
  const [csvFirstRowIsHeader, setCsvFirstRowIsHeader] = useState(true)
  const [isDraggingCsv, setIsDraggingCsv] = useState(false)
  const [isDraggingTrainingArea, setIsDraggingTrainingArea] = useState(false)
  const csvFileInputRef = useRef<HTMLInputElement>(null)
  const trainingAreaRef = useRef<HTMLDivElement>(null)
  const { toast } = useToast()

  // Sample data modal state
  const [showSampleDataModal, setShowSampleDataModal] = useState(false)
  const [selectedSampleDataset, setSelectedSampleDataset] = useState<string | null>(null)
  const [isImportingSampleData, setIsImportingSampleData] = useState(false)

  // Track if user has interacted with training data (for showing low entry warning)
  const [hasBlurredTrainingData, setHasBlurredTrainingData] = useState(false)

  // Format pasted text (convert tabs/multiple spaces to structure)
  const formatPastedText = useCallback((text: string): string => {
    // Handle tab-separated or multi-space separated data
    const lines = text.split('\n')
    return lines
      .map(line => {
        // Replace tabs with newlines for single-column paste
        if (line.includes('\t')) {
          return line.split('\t').filter(Boolean).join('\n')
        }
        return line
      })
      .join('\n')
  }, [])

  // Settings state - base model selection
  const [baseModel, setBaseModel] = useState(DEFAULT_BASE_MODEL)

  // Training state
  const [trainingState, setTrainingState] = useState<TrainingState>('idle')
  const [trainingError, setTrainingError] = useState('')
  const [isTrainingExpanded, setIsTrainingExpanded] = useState(isNewModel)

  // Test state
  const [testInput, setTestInput] = useState('')
  const [testHistory, setTestHistory] = useState<ClassifierTestResult[]>([])

  // Versions - derived from API models with same base name
  const [versions, setVersions] = useState<ModelVersion[]>([])
  const [activeVersionName, setActiveVersionName] = useState<string | null>(null)

  // API hooks
  const { data: modelsData, isLoading: isLoadingModels } = useListClassifierModels()
  const trainAndSaveMutation = useTrainAndSaveClassifier()
  const predictMutation = usePredictClassifier()
  const loadMutation = useLoadClassifier()
  const deleteMutation = useDeleteClassifierModel()

  // Parse the model ID to get base name for filtering versions
  const baseModelName = useMemo(() => {
    if (isNewModel) return null
    if (!id) return null
    const parsed = parseVersionedModelName(id)
    return parsed.baseName
  }, [id, isNewModel])

  // Extract all existing base names from models for uniqueness check
  const existingBaseNames = useMemo(() => {
    const names = new Set<string>()
    if (modelsData?.models) {
      for (const model of modelsData.models) {
        const parsed = parseVersionedModelName(model.name)
        names.add(parsed.baseName)
      }
    }
    return names
  }, [modelsData])

  // Set unique default model name for new models once data is loaded
  useEffect(() => {
    if (isNewModel && !modelName && !isLoadingModels) {
      const uniqueName = generateUniqueModelName('new-classifier-model', existingBaseNames)
      setModelName(uniqueName)
    }
  }, [isNewModel, modelName, isLoadingModels, existingBaseNames])

  // Check if model name already exists (for warning display)
  useEffect(() => {
    if (isNewModel && modelName) {
      setNameExistsWarning(existingBaseNames.has(modelName))
    } else {
      setNameExistsWarning(false)
    }
  }, [isNewModel, modelName, existingBaseNames])

  // Build versions list from API models
  useEffect(() => {
    if (!modelsData?.models || !baseModelName) {
      setVersions([])
      return
    }

    // Filter models that match our base name
    const matchingModels = modelsData.models.filter((m: ClassifierModelInfo) => {
      const parsed = parseVersionedModelName(m.name)
      return parsed.baseName === baseModelName
    })

    // Sort by timestamp (newest first) and build version list
    const sortedModels = [...matchingModels].sort((a, b) => {
      const parsedA = parseVersionedModelName(a.name)
      const parsedB = parseVersionedModelName(b.name)
      return (parsedB.timestamp || '').localeCompare(parsedA.timestamp || '')
    })

    const versionList: ModelVersion[] = sortedModels.map((m, index) => ({
      id: m.name,
      versionNumber: sortedModels.length - index,
      versionedName: m.name,
      createdAt: m.created || new Date().toISOString(),
      trainingSamples: 0,
      isActive: m.name === activeVersionName,
      labels: m.labels || [],
    }))

    setVersions(versionList)

    // Set first model as active if none selected
    if (!activeVersionName && versionList.length > 0) {
      setActiveVersionName(versionList[0].versionedName)
    }
  }, [modelsData, baseModelName, activeVersionName])

  // Load model metadata when editing existing model
  useEffect(() => {
    if (isNewModel || !baseModelName) return
    setModelName(baseModelName)
  }, [isNewModel, baseModelName])

  // Track the previous active class ID to detect actual class switches
  const prevActiveClassIdRef = useRef<string | null>(null)

  // Update training data input ONLY when switching to a different class
  // (not when classLabels changes due to typing in the current class)
  useEffect(() => {
    // Only update input when we switch to a different class
    if (prevActiveClassIdRef.current !== activeClassId) {
      const activeClass = classLabels.find(c => c.id === activeClassId)
      if (activeClass) {
        setTrainingDataInput(activeClass.examples.join('\n'))
      }
      prevActiveClassIdRef.current = activeClassId
    }
  }, [activeClassId, classLabels])

  // Ensure activeClassId is always valid (points to an existing class)
  useEffect(() => {
    const activeExists = classLabels.some(c => c.id === activeClassId)
    if (!activeExists && classLabels.length > 0) {
      setActiveClassId(classLabels[0].id)
    }
  }, [classLabels, activeClassId])

  const hasVersions = versions.length > 0
  const canTest = hasVersions || trainingState === 'success'

  // Check if we have at least 2 classes with names and examples
  const validClasses = classLabels.filter(
    c => c.name.trim() && c.examples.length > 0
  )
  const canTrain = modelName.trim() && validClasses.length >= 2

  const handleAddClass = useCallback(() => {
    const newId = String(Date.now())
    setClassLabels(prev => [...prev, { id: newId, name: '', examples: [] }])
    setActiveClassId(newId)
    setTrainingDataInput('')
  }, [])

  const handleRemoveClass = useCallback(
    (classId: string) => {
      if (classLabels.length <= 2) return
      setClassLabels(prev => prev.filter(c => c.id !== classId))
      if (activeClassId === classId) {
        const remaining = classLabels.filter(c => c.id !== classId)
        setActiveClassId(remaining[0]?.id || '')
      }
    },
    [classLabels, activeClassId]
  )

  const handleClassNameChange = useCallback((classId: string, name: string) => {
    setClassLabels(prev =>
      prev.map(c => (c.id === classId ? { ...c, name } : c))
    )
  }, [])

  const handleTrainingDataChange = useCallback(
    (value: string) => {
      setTrainingDataInput(value)
      // Parse and update examples for active class
      const examples = value
        .split(/[\n]/)
        .map(s => s.trim())
        .filter(Boolean)
      setClassLabels(prev =>
        prev.map(c => (c.id === activeClassId ? { ...c, examples } : c))
      )
    },
    [activeClassId]
  )

  // Handle paste in textarea - format spreadsheet data
  const handleTextareaPaste = useCallback(
    (e: React.ClipboardEvent<HTMLTextAreaElement>) => {
      const pastedText = e.clipboardData.getData('text')
      // If pasted text has tabs, format it
      if (pastedText.includes('\t')) {
        e.preventDefault()
        const formatted = formatPastedText(pastedText)
        const textarea = e.currentTarget
        const start = textarea.selectionStart
        const end = textarea.selectionEnd
        const newValue =
          trainingDataInput.substring(0, start) +
          formatted +
          trainingDataInput.substring(end)
        handleTrainingDataChange(newValue)
      }
    },
    [trainingDataInput, formatPastedText, handleTrainingDataChange]
  )

  // Sync table rows to class labels
  const syncTableToClassLabels = useCallback((rows: ClassifierTableRow[]) => {
    // Group examples by class name
    const classMap = new Map<string, string[]>()
    for (const row of rows) {
      if (row.className.trim() && row.example.trim()) {
        const existing = classMap.get(row.className) || []
        existing.push(row.example)
        classMap.set(row.className, existing)
      }
    }

    // Update class labels based on table data
    setClassLabels(prev => {
      // Filter out empty default classes (no name or no examples) when we have imported data
      const hasImportedData = classMap.size > 0
      const filtered = hasImportedData
        ? prev.filter(cl => cl.name.trim() && cl.examples.length > 0)
        : prev

      const updated = [...filtered]

      // Update existing classes that match imported class names
      for (const cl of updated) {
        if (classMap.has(cl.name)) {
          cl.examples = classMap.get(cl.name) || []
          classMap.delete(cl.name)
        }
      }

      // Add new classes from table/import
      for (const [name, examples] of classMap) {
        updated.push({
          id: String(Date.now()) + Math.random(),
          name,
          examples,
        })
      }

      // Ensure we always have at least 2 class slots if no data
      if (updated.length === 0) {
        return [
          { id: '1', name: '', examples: [] },
          { id: '2', name: '', examples: [] },
        ]
      }

      return updated
    })
  }, [])

  // Handle table cell change
  const handleTableCellChange = useCallback(
    (rowId: string, field: 'className' | 'example', value: string) => {
      setTableRows(prev => {
        const updated = prev.map(row =>
          row.id === rowId ? { ...row, [field]: value } : row
        )
        syncTableToClassLabels(updated)
        return updated
      })
    },
    [syncTableToClassLabels]
  )

  // Add row to table
  const handleAddTableRow = useCallback(() => {
    const activeClass = classLabels.find(c => c.id === activeClassId)
    setTableRows(prev => [
      ...prev,
      {
        id: String(Date.now()),
        className: activeClass?.name || '',
        example: '',
      },
    ])
  }, [classLabels, activeClassId])

  // Remove row from table
  const handleRemoveTableRow = useCallback(
    (rowId: string) => {
      setTableRows(prev => {
        const updated = prev.filter(row => row.id !== rowId)
        syncTableToClassLabels(updated)
        return updated
      })
    },
    [syncTableToClassLabels]
  )

  // Handle paste in table - parse spreadsheet data into rows
  const handleTablePaste = useCallback(
    (e: React.ClipboardEvent) => {
      const pastedText = e.clipboardData.getData('text')
      if (!pastedText.includes('\t') && !pastedText.includes('\n')) return

      e.preventDefault()
      const lines = pastedText.split('\n').filter(line => line.trim())
      const newRows: ClassifierTableRow[] = []

      for (const line of lines) {
        const parts = line.split('\t')
        if (parts.length >= 2) {
          // Two columns: example (col 1), class (col 2)
          newRows.push({
            id: String(Date.now()) + Math.random(),
            example: parts[0].trim(),
            className: parts[1].trim(),
          })
        } else if (parts.length === 1 && parts[0].trim()) {
          // Single column: use active class
          const activeClass = classLabels.find(c => c.id === activeClassId)
          newRows.push({
            id: String(Date.now()) + Math.random(),
            className: activeClass?.name || '',
            example: parts[0].trim(),
          })
        }
      }

      if (newRows.length > 0) {
        setTableRows(prev => {
          const updated = [...prev, ...newRows]
          syncTableToClassLabels(updated)
          return updated
        })
      }
    },
    [classLabels, activeClassId, syncTableToClassLabels]
  )

  // Sync class labels to table when switching to table mode
  const syncClassLabelsToTable = useCallback(() => {
    const rows: ClassifierTableRow[] = []
    for (const cl of classLabels) {
      for (const example of cl.examples) {
        rows.push({
          id: String(Date.now()) + Math.random(),
          className: cl.name,
          example,
        })
      }
    }
    setTableRows(rows)
  }, [classLabels])

  // Handle mode switch
  const handleModeSwitch = useCallback(
    (mode: 'text' | 'table') => {
      if (mode === 'table' && inputMode === 'text') {
        syncClassLabelsToTable()
      }
      setInputMode(mode)
    },
    [inputMode, syncClassLabelsToTable]
  )

  const handleTrain = useCallback(async () => {
    if (!canTrain) return

    setTrainingState('training')
    setTrainingError('')

    // Use unique name if the current name already exists
    const finalModelName = isNewModel
      ? generateUniqueModelName(modelName, existingBaseNames)
      : modelName

    try {
      // Convert class labels to API training data format
      const trainingData: ClassifierTrainingData[] = []
      for (const classLabel of validClasses) {
        for (const example of classLabel.examples) {
          trainingData.push({
            text: example,
            label: classLabel.name,
          })
        }
      }

      const result = await trainAndSaveMutation.mutateAsync({
        model: finalModelName,
        base_model: baseModel,
        training_data: trainingData,
        overwrite: false,
        description: description.trim() || undefined,
      })

      // Update state with new version
      const newVersionName = result.fitResult.versioned_name
      setActiveVersionName(newVersionName)
      setTrainingState('success')
      setIsTrainingExpanded(false)

      // If new model, redirect to edit page with the base name
      if (isNewModel) {
        navigate(`/chat/models/train/classifier/${finalModelName}`)
      }
    } catch (error) {
      setTrainingState('error')
      setTrainingError(
        error instanceof Error ? error.message : 'Training failed. Please try again.'
      )
    }
  }, [canTrain, validClasses, modelName, baseModel, description, trainAndSaveMutation, isNewModel, navigate, existingBaseNames])

  const handleTest = useCallback(async () => {
    if (!testInput.trim() || !activeVersionName) return

    try {
      // Ensure model is loaded before predicting
      await loadMutation.mutateAsync({ model: activeVersionName })

      const result = await predictMutation.mutateAsync({
        model: activeVersionName,
        texts: [testInput.trim()],
      })

      if (result.data && result.data.length > 0) {
        const prediction = result.data[0]
        const newResult: ClassifierTestResult = {
          id: String(Date.now()),
          input: testInput.trim(),
          label: prediction.label.trim(), // API may return label with trailing space
          confidence: prediction.score, // API uses 'score' not 'confidence'
          timestamp: new Date().toISOString(),
        }
        setTestHistory(prev => [newResult, ...prev])
      }
    } catch (error) {
      const errorResult: ClassifierTestResult = {
        id: String(Date.now()),
        input: testInput.trim(),
        label: 'Error',
        confidence: 0,
        timestamp: new Date().toISOString(),
      }
      setTestHistory(prev => [
        { ...errorResult, input: `Error: ${error instanceof Error ? error.message : 'Test failed'}` },
        ...prev,
      ])
    }
    setTestInput('')
  }, [testInput, activeVersionName, loadMutation, predictMutation])

  const handleSetActiveVersion = useCallback(
    async (versionName: string) => {
      try {
        await loadMutation.mutateAsync({ model: versionName })
        setActiveVersionName(versionName)
        setVersions(prev =>
          prev.map(v => ({
            ...v,
            isActive: v.versionedName === versionName,
          }))
        )
      } catch (error) {
        console.error('Failed to load model version:', error)
      }
    },
    [loadMutation]
  )

  const handleDeleteVersion = useCallback(
    async (versionName: string) => {
      const version = versions.find(v => v.versionedName === versionName)
      if (!version) return

      const versionLabel = `Version ${version.versionNumber}`
      const confirmMessage = `Delete ${versionLabel}? This cannot be undone.`
      if (!window.confirm(confirmMessage)) return

      try {
        await deleteMutation.mutateAsync(versionName)

        toast({
          message: `Successfully deleted ${versionLabel}.`,
          icon: 'checkmark-filled',
        })

        if (versionName === activeVersionName) {
          const remaining = versions.filter(v => v.versionedName !== versionName)
          if (remaining.length > 0) {
            setActiveVersionName(remaining[0].versionedName)
          } else {
            setActiveVersionName(null)
          }
        }
      } catch (error) {
        console.error('Failed to delete model version:', error)
        toast({
          message: 'Failed to delete version. Please try again.',
          variant: 'destructive',
          icon: 'alert-triangle',
        })
      }
    },
    [versions, activeVersionName, deleteMutation, toast]
  )

  const handleDeleteModel = useCallback(async () => {
    if (!baseModelName || isNewModel) return

    const confirmMessage = `Delete "${baseModelName}" and all ${versions.length} version${versions.length !== 1 ? 's' : ''}? This cannot be undone.`
    if (!window.confirm(confirmMessage)) return

    try {
      // Delete all versions
      await Promise.all(versions.map(v => deleteMutation.mutateAsync(v.versionedName)))

      toast({
        message: `Successfully deleted ${baseModelName} and all its versions.`,
        icon: 'checkmark-filled',
      })

      // Navigate back to models list
      navigate('/chat/models?tab=training')
    } catch (error) {
      console.error('Failed to delete model:', error)
      toast({
        message: 'Failed to delete some model versions. Please try again.',
        variant: 'destructive',
        icon: 'alert-triangle',
      })
    }
  }, [baseModelName, isNewModel, versions, deleteMutation, toast, navigate])

  // CSV import handlers
  const handleCsvFileSelect = useCallback((file: File) => {
    setCsvFile(file)
  }, [])

  const handleCsvDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setIsDraggingCsv(false)
      const file = e.dataTransfer.files[0]
      if (file && (file.name.endsWith('.csv') || file.type === 'text/csv')) {
        handleCsvFileSelect(file)
      }
    },
    [handleCsvFileSelect]
  )

  const handleCsvImport = useCallback(() => {
    if (!csvFile) return

    const reader = new FileReader()
    reader.onload = e => {
      const text = e.target?.result as string
      if (!text) return

      const lines = text.trim().split('\n').filter(Boolean)
      if (lines.length === 0) return

      // Detect delimiter
      const hasTab = lines[0].includes('\t')
      const delimiter = hasTab ? '\t' : ','

      // Parse all rows
      const allRows = lines.map(line =>
        line.split(delimiter).map(v => v.trim().replace(/^"|"$/g, ''))
      )

      // Determine headers and data rows
      let dataRows: string[][]

      if (csvFirstRowIsHeader && allRows.length > 1) {
        dataRows = allRows.slice(1)
      } else {
        dataRows = allRows
      }

      // Create table rows - expect 2 columns: example (col 1), class (col 2)
      const newRows: ClassifierTableRow[] = []
      for (const row of dataRows) {
        if (row.length >= 2 && row[0].trim() && row[1].trim()) {
          newRows.push({
            id: String(Date.now()) + Math.random(),
            example: row[0].trim(),
            className: row[1].trim(),
          })
        } else if (row.length === 1 && row[0].trim()) {
          // Single column: use active class
          const activeClassItem = classLabels.find(c => c.id === activeClassId)
          newRows.push({
            id: String(Date.now()) + Math.random(),
            className: activeClassItem?.name || '',
            example: row[0].trim(),
          })
        }
      }

      if (newRows.length > 0) {
        setTableRows(prev => {
          const updated = [...prev, ...newRows]
          syncTableToClassLabels(updated)
          return updated
        })
        setInputMode('table')
      }

      // Close modal and reset
      setShowCsvModal(false)
      setCsvFile(null)
    }
    reader.readAsText(csvFile)
  }, [csvFile, csvFirstRowIsHeader, classLabels, activeClassId, syncTableToClassLabels])

  const handleCsvModalClose = useCallback(() => {
    setShowCsvModal(false)
    setCsvFile(null)
    setIsDraggingCsv(false)
  }, [])

  // Sample data handler - import the selected dataset with loading state
  const handleImportSampleData = useCallback(() => {
    if (!selectedSampleDataset) return

    const dataset = CLASSIFIER_SAMPLE_DATASETS.find(d => d.id === selectedSampleDataset)
    if (!dataset?.data) {
      toast({
        message: 'Sample data not available.',
        variant: 'destructive',
        icon: 'alert-triangle',
      })
      return
    }

    setIsImportingSampleData(true)

    // Simulate a short loading state for better UX
    setTimeout(() => {
      // Group data by label to create class labels
      const classMap = new Map<string, string[]>()
      for (const item of dataset.data) {
        const existing = classMap.get(item.label) || []
        existing.push(item.text)
        classMap.set(item.label, existing)
      }

      // Create class labels from the sample data
      const newClassLabels: ClassLabel[] = Array.from(classMap.entries()).map(([name, examples], idx) => ({
        id: String(Date.now()) + idx,
        name,
        examples,
      }))

      // Update state
      setClassLabels(newClassLabels)
      if (newClassLabels.length > 0) {
        setActiveClassId(newClassLabels[0].id)
        setTrainingDataInput(newClassLabels[0].examples.join('\n'))
      }

      // Also update table rows for table view
      const newTableRows: ClassifierTableRow[] = dataset.data.map((item, idx) => ({
        id: String(Date.now()) + idx + Math.random(),
        example: item.text,
        className: item.label,
      }))
      setTableRows(newTableRows)

      // Switch to table view after importing sample data
      setInputMode('table')

      // Close modal and reset
      setShowSampleDataModal(false)
      setSelectedSampleDataset(null)
      setIsImportingSampleData(false)
    }, 600)
  }, [selectedSampleDataset, toast])

  // Training area drag handlers
  const handleTrainingAreaDragEnter = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault()
      e.stopPropagation()
      setIsDraggingTrainingArea(true)
    },
    []
  )

  const handleTrainingAreaDragOver = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault()
      e.stopPropagation()
      e.dataTransfer.dropEffect = 'copy'
    },
    []
  )

  const handleTrainingAreaDragLeave = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault()
      e.stopPropagation()
      const rect = trainingAreaRef.current?.getBoundingClientRect()
      const isLeavingZone =
        rect &&
        (e.clientX <= rect.left ||
          e.clientX >= rect.right ||
          e.clientY <= rect.top ||
          e.clientY >= rect.bottom)
      if (isLeavingZone) {
        setIsDraggingTrainingArea(false)
      }
    },
    []
  )

  const handleTrainingAreaDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault()
      e.stopPropagation()
      setIsDraggingTrainingArea(false)

      const file = e.dataTransfer.files[0]
      if (!file) return

      const isCsv =
        file.name.toLowerCase().endsWith('.csv') || file.type === 'text/csv'

      if (isCsv) {
        setCsvFile(file)
        setShowCsvModal(true)
      } else {
        toast({
          message: 'Only CSV files are supported. Please drop a .csv file.',
          variant: 'destructive',
          icon: 'alert-triangle',
        })
      }
    },
    [toast]
  )

  // Get total example count across all classes
  const totalExamples = useMemo(() => {
    return classLabels.reduce((sum, cl) => sum + cl.examples.length, 0)
  }, [classLabels])

  const pageTitle = isNewModel
    ? 'New classifier model'
    : modelName || 'Classifier model'

  const activeClass = classLabels.find(c => c.id === activeClassId)

  if (isLoadingModels && !isNewModel) {
    return (
      <div className="flex-1 min-h-0 overflow-auto pb-20">
        <div className="flex items-center justify-center h-64">
          <div className="text-muted-foreground">Loading model...</div>
        </div>
      </div>
    )
  }

  return (
    <div className="flex-1 min-h-0 overflow-auto pb-20">
      <div className="flex flex-col gap-4">
        {/* Breadcrumb + Done button */}
        <div className="flex items-center justify-between">
          <nav className="text-sm md:text-base flex items-center gap-1.5">
            <button
              className="text-teal-600 dark:text-teal-400 hover:underline"
              onClick={() => navigate('/chat/models?tab=training')}
            >
              Trained models
            </button>
            <span className="text-muted-foreground px-1">/</span>
            <span className="text-foreground">{pageTitle}</span>
          </nav>
          <div className="flex items-center gap-2">
            {!isNewModel && versions.length > 0 && (
              <Button
                variant="ghost"
                onClick={handleDeleteModel}
                className="text-sm text-destructive/70 hover:text-destructive hover:bg-destructive/5"
              >
                Delete
              </Button>
            )}
            <Button variant="outline" onClick={() => navigate('/chat/models?tab=training')}>
              Done
            </Button>
          </div>
        </div>

        {/* Page title */}
        <h1 className="text-2xl font-medium">{pageTitle}</h1>

        {/* Name and Description row */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-2">
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="model-name" className="text-sm font-medium">
              Model name {isNewModel && <span className="text-destructive">*</span>}
            </Label>
            <Input
              id="model-name"
              placeholder="e.g., sentiment-classifier"
              value={modelName}
              onChange={e => {
                const sanitized = e.target.value
                  .toLowerCase()
                  .replace(/[^a-z0-9-]/g, '-')
                  .replace(/-+/g, '-')
                setModelName(sanitized)
              }}
              className={nameExistsWarning ? 'border-amber-500' : ''}
            />
            {nameExistsWarning ? (
              <p className="text-xs text-amber-600 dark:text-amber-400">
                A model with this name exists. Will be saved as "{generateUniqueModelName(modelName, existingBaseNames)}".
              </p>
            ) : (
              <p className="text-xs text-muted-foreground">
                Lowercase letters, numbers, and hyphens only
              </p>
            )}
          </div>
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="description" className="text-sm font-medium">
              Description
            </Label>
            <Input
              id="description"
              placeholder="e.g., Classifies customer feedback sentiment"
              value={description}
              onChange={e => setDescription(e.target.value)}
            />
          </div>
        </div>

        {/* Training Data Card - Full Width */}
        <div className={`rounded-lg border border-border bg-card p-4 flex flex-col gap-4 relative transition-all duration-300 ${trainingState === 'training' ? 'h-[400px] overflow-hidden' : ''}`}>
          {trainingState === 'training' && <TrainingLoadingOverlay message="Training your classifier..." />}
          {/* Collapsed view - show when has versions and not expanded */}
          {hasVersions && !isTrainingExpanded ? (
            <div className="flex items-center justify-between">
              <div className="flex flex-col gap-1">
                <h3 className="text-sm font-medium">Training data</h3>
                <p className="text-xs text-muted-foreground">
                  Add more training data to improve your model
                </p>
              </div>
              <Button
                variant="secondary"
                onClick={() => setIsTrainingExpanded(true)}
              >
                Retrain
              </Button>
            </div>
          ) : (
            <>
              {/* Base model row with collapse button */}
              <div className="flex items-center justify-between pb-3 border-b border-border">
                <div className="flex flex-col gap-1">
                  <Label htmlFor="base-model" className="text-xs text-muted-foreground">
                    Base model
                  </Label>
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <button
                        type="button"
                        id="base-model"
                        className="w-64 h-9 rounded-md border border-input bg-background px-3 text-left flex items-center justify-between text-sm"
                      >
                        <span>{EMBEDDER_MODELS.find(m => m.id === baseModel)?.label || baseModel}</span>
                        <FontIcon type="chevron-down" className="w-4 h-4 text-muted-foreground" />
                      </button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="start" className="w-64">
                      {EMBEDDER_MODELS.map(model => (
                        <DropdownMenuItem
                          key={model.id}
                          onClick={() => setBaseModel(model.id)}
                          className={baseModel === model.id ? 'bg-primary/10' : ''}
                        >
                          <span className="flex-1">{model.label}</span>
                          <span className="text-xs text-muted-foreground ml-2">{model.dim}d</span>
                        </DropdownMenuItem>
                      ))}
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>
                {hasVersions && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setIsTrainingExpanded(false)}
                    className="h-6 w-6 p-0"
                    title="Collapse"
                  >
                    <FontIcon type="chevron-up" className="w-4 h-4" />
                  </Button>
                )}
              </div>

              {/* Training data header */}
              <div className="flex flex-col gap-1">
                <h3 className="text-sm font-medium">
                  Training data{' '}
                  {isNewModel && <span className="text-destructive">*</span>}
                </h3>
                <p className="text-xs text-muted-foreground">
                  Define classes and add example texts for each class
                </p>
              </div>

              {/* Class labels section */}
              <div className="flex flex-col gap-3">
                <div className="flex items-center justify-between">
                  <Label className="text-sm font-medium">Classes</Label>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleAddClass}
                    className="text-xs gap-1 border-primary/50 text-primary hover:bg-primary/10 hover:border-primary"
                  >
                    <FontIcon type="add" className="w-3.5 h-3.5" />
                    Add class
                  </Button>
                </div>

                {/* Class selection - more prominent styling */}
                <div className="flex flex-wrap gap-2">
                  {classLabels.map(classLabel => {
                    const isActive = activeClassId === classLabel.id
                    return (
                      <div
                        key={classLabel.id}
                        className={`group flex items-center gap-2 px-4 py-2 rounded-lg border-2 cursor-pointer transition-all ${
                          isActive
                            ? 'border-primary bg-primary/10 shadow-sm'
                            : 'border-border hover:border-primary/50 bg-muted/30'
                        }`}
                        onClick={() => setActiveClassId(classLabel.id)}
                      >
                        {isActive && (
                          <FontIcon type="checkmark-filled" className="w-4 h-4 text-primary shrink-0" />
                        )}
                        <Input
                          value={classLabel.name}
                          onChange={e => {
                            e.stopPropagation()
                            handleClassNameChange(classLabel.id, e.target.value)
                          }}
                          onClick={e => {
                            e.stopPropagation()
                            setActiveClassId(classLabel.id)
                          }}
                          placeholder="Class name"
                          className={`border-0 bg-transparent p-0 h-auto w-28 text-sm font-medium focus-visible:ring-0 ${
                            isActive ? 'text-primary' : ''
                          }`}
                        />
                        <Badge variant={isActive ? 'default' : 'secondary'} className="text-xs">
                          {classLabel.examples.length}
                        </Badge>
                        {classLabels.length > 2 && (
                          <button
                            onClick={e => {
                              e.stopPropagation()
                              handleRemoveClass(classLabel.id)
                            }}
                            className="opacity-0 group-hover:opacity-100 text-muted-foreground hover:text-destructive transition-opacity ml-1"
                          >
                            <FontIcon type="close" className="w-3.5 h-3.5" />
                          </button>
                        )}
                      </div>
                    )
                  })}
                </div>
              </div>

              {/* Training data section with mode toggle */}
              <div
                ref={trainingAreaRef}
                className="flex flex-col gap-2 rounded-lg transition-colors relative"
                onDragEnter={handleTrainingAreaDragEnter}
                onDragOver={handleTrainingAreaDragOver}
                onDragLeave={handleTrainingAreaDragLeave}
                onDrop={handleTrainingAreaDrop}
              >
                {/* Drop overlay */}
                {isDraggingTrainingArea && (
                  <div className="absolute inset-0 z-10 flex flex-col items-center justify-center rounded-lg border-2 border-dashed border-primary bg-primary/5 backdrop-blur-[2px]">
                    <div className="flex flex-col items-center gap-3 text-center p-6">
                      <div className="w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center">
                        <FontIcon type="upload" className="w-6 h-6 text-primary" />
                      </div>
                      <div className="text-lg font-medium text-foreground">
                        Drop CSV here
                      </div>
                      <p className="text-sm text-muted-foreground max-w-[300px]">
                        Release to import your CSV file as training data
                      </p>
                    </div>
                  </div>
                )}

                {/* Mode toggle */}
                <div className="flex items-center justify-between">
                  {inputMode === 'text' ? (
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium">Add examples to</span>
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <button
                            type="button"
                            className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-primary text-primary-foreground text-sm font-medium hover:opacity-90 transition-opacity"
                          >
                            {activeClass?.name || '(unnamed class)'}
                            <FontIcon type="chevron-down" className="w-3.5 h-3.5" />
                          </button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="start">
                          {classLabels.map(cl => (
                            <DropdownMenuItem
                              key={cl.id}
                              onClick={() => setActiveClassId(cl.id)}
                              className={activeClassId === cl.id ? 'bg-primary/10' : ''}
                            >
                              <span className="flex-1">{cl.name || '(unnamed class)'}</span>
                              <Badge variant="secondary" className="ml-2 text-xs">
                                {cl.examples.length}
                              </Badge>
                            </DropdownMenuItem>
                          ))}
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </div>
                  ) : (
                    <span className="text-sm font-medium">All training data</span>
                  )}
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => setShowSampleDataModal(true)}
                      className="flex items-center gap-1.5 px-2.5 py-1 text-xs text-muted-foreground hover:text-foreground border border-border rounded-md hover:bg-muted/50 transition-colors"
                    >
                      <FontIcon type="data" className="w-3.5 h-3.5" />
                      Use sample data
                    </button>
                    <button
                      type="button"
                      onClick={() => setShowCsvModal(true)}
                      className="flex items-center gap-1.5 px-2.5 py-1 text-xs text-muted-foreground hover:text-foreground border border-border rounded-md hover:bg-muted/50 transition-colors"
                    >
                      <FontIcon type="upload" className="w-3.5 h-3.5" />
                      Import CSV
                    </button>
                    <div className="flex items-center border border-border rounded-md overflow-hidden">
                      <button
                        type="button"
                        onClick={() => handleModeSwitch('text')}
                        className={`px-3 py-1.5 text-xs font-medium transition-colors ${
                          inputMode === 'text'
                            ? 'bg-primary text-primary-foreground'
                            : 'bg-muted/50 text-muted-foreground hover:bg-muted'
                        }`}
                      >
                        Text
                      </button>
                      <button
                        type="button"
                        onClick={() => handleModeSwitch('table')}
                        className={`px-3 py-1.5 text-xs font-medium transition-colors ${
                          inputMode === 'table'
                            ? 'bg-primary text-primary-foreground'
                            : 'bg-muted/50 text-muted-foreground hover:bg-muted'
                        }`}
                      >
                        Table
                      </button>
                    </div>
                  </div>
                </div>

                {inputMode === 'text' ? (
                  <>
                    <p className="text-xs text-muted-foreground">
                      Add example texts that belong to this class. One example per line, or paste from a spreadsheet.
                    </p>
                    <Textarea
                      id="training-data"
                      placeholder={`Enter examples for "${activeClass?.name || 'this class'}"...\n\nOne example per line, or paste from a spreadsheet`}
                      value={trainingDataInput}
                      onChange={e => handleTrainingDataChange(e.target.value)}
                      onPaste={handleTextareaPaste}
                      onBlur={() => setHasBlurredTrainingData(true)}
                      rows={8}
                      className="font-mono max-h-[50vh] min-h-[200px] resize-y"
                    />
                    <div className="flex items-center justify-between">
                      <p className="text-xs text-muted-foreground">
                        {activeClass && activeClass.examples.length > 0
                          ? `${activeClass.examples.length} examples for this class`
                          : 'No examples yet'}
                        {hasBlurredTrainingData && totalExamples > 0 && totalExamples < 50 && (
                          <span className="text-amber-600 dark:text-amber-400">
                            {' — Please add more entries to increase accuracy'}
                          </span>
                        )}
                      </p>
                      {trainingDataInput.trim() && (
                        <button
                          type="button"
                          onClick={() => {
                            setTrainingDataInput('')
                            if (activeClassId) {
                              setClassLabels(prev =>
                                prev.map(cl =>
                                  cl.id === activeClassId ? { ...cl, examples: [] } : cl
                                )
                              )
                            }
                          }}
                          className="flex items-center gap-1.5 px-2.5 py-1 text-xs text-muted-foreground hover:text-destructive border border-border rounded-md hover:bg-destructive/10 hover:border-destructive/50 transition-colors"
                        >
                          <FontIcon type="trashcan" className="w-3.5 h-3.5" />
                          Clear
                        </button>
                      )}
                    </div>
                  </>
                ) : (
                  <>
                    <p className="text-xs text-muted-foreground">
                      View and edit all training data. Paste from a spreadsheet or drag & drop a CSV with two columns: Example (text), Class (label).
                    </p>
                    <div
                      className="border border-border rounded-md overflow-auto max-h-[50vh] min-h-[200px]"
                      onPaste={handleTablePaste}
                    >
                      <table className="w-full text-sm">
                        <thead className="bg-muted/50 sticky top-0 z-10">
                          <tr>
                            <th className="text-left px-3 py-2 font-medium w-1/2">Example</th>
                            <th className="text-left px-3 py-2 font-medium w-1/2">Class</th>
                            <th className="w-10"></th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-border">
                          {tableRows.map(row => (
                            <tr key={row.id} className="bg-card hover:bg-muted/30">
                              <td className="px-1 py-0 border-r border-border">
                                <input
                                  value={row.example}
                                  onChange={e =>
                                    handleTableCellChange(row.id, 'example', e.target.value)
                                  }
                                  onBlur={() => setHasBlurredTrainingData(true)}
                                  className="w-full px-2 py-2 bg-transparent border-0 outline-none text-sm font-mono focus:bg-primary/5"
                                  placeholder="Example text"
                                />
                              </td>
                              <td className="px-1 py-0">
                                <input
                                  value={row.className}
                                  onChange={e =>
                                    handleTableCellChange(row.id, 'className', e.target.value)
                                  }
                                  onBlur={() => setHasBlurredTrainingData(true)}
                                  className="w-full px-2 py-2 bg-transparent border-0 outline-none text-sm font-medium focus:bg-primary/5"
                                  placeholder="Class name"
                                />
                              </td>
                              <td className="px-2 py-0 border-l border-border">
                                <button
                                  onClick={() => handleRemoveTableRow(row.id)}
                                  className="text-muted-foreground hover:text-destructive p-1"
                                  title="Remove row"
                                >
                                  <FontIcon type="close" className="w-3 h-3" />
                                </button>
                              </td>
                            </tr>
                          ))}
                          {/* Add row button as last row */}
                          <tr className="bg-muted/20">
                            <td colSpan={3} className="px-3 py-2">
                              <button
                                onClick={handleAddTableRow}
                                className="text-xs text-muted-foreground hover:text-primary flex items-center gap-1"
                              >
                                <FontIcon type="add" className="w-3 h-3" />
                                Add row
                              </button>
                            </td>
                          </tr>
                        </tbody>
                      </table>
                    </div>
                    <div className="flex items-center justify-between">
                      <p className="text-xs text-muted-foreground">
                        {tableRows.length > 0
                          ? `${tableRows.length} total examples`
                          : 'No examples yet'}
                        {hasBlurredTrainingData && tableRows.length > 0 && tableRows.length < 50 && (
                          <span className="text-amber-600 dark:text-amber-400">
                            {' — Please add more entries to increase accuracy'}
                          </span>
                        )}
                      </p>
                      {tableRows.length > 0 && (
                        <button
                          type="button"
                          onClick={() => {
                            setTableRows([])
                            setClassLabels(prev =>
                              prev.map(cl => ({ ...cl, examples: [] }))
                            )
                          }}
                          className="flex items-center gap-1.5 px-2.5 py-1 text-xs text-muted-foreground hover:text-destructive border border-border rounded-md hover:bg-destructive/10 hover:border-destructive/50 transition-colors"
                        >
                          <FontIcon type="trashcan" className="w-3.5 h-3.5" />
                          Clear
                        </button>
                      )}
                    </div>
                  </>
                )}
              </div>

              {/* Actions row */}
              <div className="flex items-center gap-3 pt-2 border-t border-border">
                <Button
                  onClick={handleTrain}
                  disabled={!canTrain || trainingState === 'training'}
                >
                  {trainingState === 'training'
                    ? 'Training...'
                    : hasVersions
                      ? `Retrain as v${versions.length + 1}`
                      : 'Train'}
                </Button>
                {/* Validation message */}
                {!canTrain && modelName.trim() && (
                  <p className="text-sm text-muted-foreground">
                    Add at least 2 classes with names and examples to train.
                  </p>
                )}
              </div>

              {/* Error message */}
              {trainingState === 'error' && trainingError && (
                <p className="text-sm text-destructive">{trainingError}</p>
              )}
            </>
          )}
        </div>

        {/* Test Panel */}
        <div
          className={`rounded-lg border border-border bg-card p-4 flex flex-col gap-4 ${
            !canTest ? 'opacity-50' : ''
          }`}
        >
          {/* Success message */}
          {trainingState === 'success' && (
            <div className="flex items-center gap-2 text-primary bg-primary/10 border border-primary/20 rounded-md p-3">
              <FontIcon type="checkmark-filled" className="w-4 h-4" />
              <span className="text-sm font-medium">Model trained successfully</span>
            </div>
          )}

          <div className="flex flex-col gap-1.5">
            <div className="flex items-center gap-2">
              <Label className="text-sm font-medium">Test your model</Label>
              {activeVersionName && (
                <Badge variant="secondary" className="text-xs font-normal">
                  {activeVersionName}
                </Badge>
              )}
            </div>
            <p className="text-xs text-muted-foreground">
              {canTest
                ? 'Enter text to see which class it would be assigned to.'
                : 'Train your model first to enable testing.'}
            </p>
          </div>

          <div className="flex gap-2">
            <Input
              placeholder="Enter text to classify"
              value={testInput}
              onChange={e => setTestInput(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'Enter' && canTest) {
                  handleTest()
                }
              }}
              disabled={!canTest || predictMutation.isPending}
              className="flex-1"
            />
            <Button
              onClick={handleTest}
              variant="secondary"
              disabled={!canTest || predictMutation.isPending}
            >
              {predictMutation.isPending ? 'Classifying...' : 'Classify'}
            </Button>
          </div>

          {testHistory.length > 0 && (
            <div className="flex flex-col gap-1">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium text-muted-foreground">Test history</span>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setTestHistory([])}
                  className="text-xs h-5 px-1.5 text-muted-foreground"
                >
                  Clear
                </Button>
              </div>
              <div className="flex flex-col gap-0.5 max-h-[150px] overflow-y-auto">
                {testHistory.map(result => (
                  <div
                    key={result.id}
                    className="flex items-start gap-2 px-2 py-1 rounded text-sm bg-muted/50"
                  >
                    <FontIcon type="checkmark-filled" className="w-3 h-3 text-primary shrink-0" />
                    <span className="font-medium text-primary w-20 shrink-0 break-words">
                      {result.label}
                    </span>
                    <span className="text-muted-foreground w-10 shrink-0">
                      {(result.confidence * 100).toFixed(0)}%
                    </span>
                    <span className="text-muted-foreground break-words flex-1 min-w-0">
                      {result.input}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Model Versions */}
        <div className="flex flex-col gap-3">
          <h3 className="text-sm font-medium">Model versions</h3>
          {hasVersions ? (
            <div className="rounded-lg border border-border overflow-hidden">
              <table className="w-full text-sm">
                <thead className="bg-muted/50">
                  <tr>
                    <th className="text-left px-4 py-2 font-medium">Version</th>
                    <th className="text-left px-4 py-2 font-medium">Model name</th>
                    <th className="text-left px-4 py-2 font-medium">Created</th>
                    <th className="text-left px-4 py-2 font-medium">Labels</th>
                    <th className="text-left px-4 py-2 font-medium">Status</th>
                    <th className="text-right px-4 py-2 font-medium">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {versions.map(version => {
                    const parsed = parseVersionedModelName(version.versionedName)
                    return (
                      <tr key={version.id} className="bg-card">
                        <td className="px-4 py-3">v{version.versionNumber}</td>
                        <td className="px-4 py-3 font-mono text-xs">
                          {version.versionedName}
                        </td>
                        <td className="px-4 py-3 text-muted-foreground">
                          {parsed.timestamp
                            ? formatModelTimestamp(parsed.timestamp)
                            : new Date(version.createdAt).toLocaleDateString()}
                        </td>
                        <td className="px-4 py-3 text-muted-foreground">
                          {version.labels.length > 0
                            ? version.labels.join(', ')
                            : '—'}
                        </td>
                        <td className="px-4 py-3">
                          {version.isActive ? (
                            <Badge variant="default">Active</Badge>
                          ) : (
                            <span className="text-muted-foreground">—</span>
                          )}
                        </td>
                        <td className="px-4 py-3 text-right">
                          <DropdownMenu>
                            <DropdownMenuTrigger asChild>
                              <Button variant="ghost" size="sm">
                                <FontIcon type="overflow" className="w-4 h-4" />
                              </Button>
                            </DropdownMenuTrigger>
                            <DropdownMenuContent align="end">
                              {!version.isActive && (
                                <DropdownMenuItem
                                  onClick={() =>
                                    handleSetActiveVersion(version.versionedName)
                                  }
                                >
                                  Set as active
                                </DropdownMenuItem>
                              )}
                              <DropdownMenuItem
                                onClick={() =>
                                  handleDeleteVersion(version.versionedName)
                                }
                                className="text-destructive"
                              >
                                Delete
                              </DropdownMenuItem>
                            </DropdownMenuContent>
                          </DropdownMenu>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="rounded-lg border border-dashed border-border p-8 text-center">
              <p className="text-sm text-muted-foreground">
                No versions yet. Train your model to create your first version.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* CSV Import Modal */}
      <Dialog open={showCsvModal} onOpenChange={handleCsvModalClose}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Import from CSV</DialogTitle>
          </DialogHeader>
          <div className="flex flex-col gap-4">
            <p className="text-sm text-muted-foreground">
              Import training data from a CSV file. The file should have two columns: Example, Class.
            </p>
            {/* Drop zone / file display */}
            <div
              className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
                isDraggingCsv
                  ? 'border-primary bg-primary/5'
                  : 'border-border hover:border-muted-foreground/50'
              }`}
              onDragOver={e => {
                e.preventDefault()
                setIsDraggingCsv(true)
              }}
              onDragLeave={() => setIsDraggingCsv(false)}
              onDrop={handleCsvDrop}
            >
              {csvFile ? (
                <div className="flex flex-col items-center gap-2">
                  <FontIcon type="data" className="w-8 h-8 text-primary" />
                  <p className="text-sm font-medium">{csvFile.name}</p>
                  <button
                    onClick={() => setCsvFile(null)}
                    className="text-xs text-muted-foreground hover:text-destructive"
                  >
                    Remove
                  </button>
                </div>
              ) : (
                <div className="flex flex-col items-center gap-3">
                  <FontIcon type="upload" className="w-8 h-8 text-muted-foreground" />
                  <div className="flex flex-col gap-1">
                    <p className="text-sm text-muted-foreground">
                      Drop CSV file here or
                    </p>
                    <button
                      onClick={() => csvFileInputRef.current?.click()}
                      className="text-sm text-primary hover:underline"
                    >
                      browse to upload
                    </button>
                  </div>
                </div>
              )}
              <input
                ref={csvFileInputRef}
                type="file"
                accept=".csv,text/csv"
                className="hidden"
                onChange={e => {
                  const file = e.target.files?.[0]
                  if (file) {
                    handleCsvFileSelect(file)
                  }
                  e.target.value = ''
                }}
              />
            </div>

            {/* First row is header checkbox */}
            <label className="flex items-center gap-2 cursor-pointer">
              <Checkbox
                checked={csvFirstRowIsHeader}
                onCheckedChange={checked => setCsvFirstRowIsHeader(checked === true)}
              />
              <span className="text-sm">First row is a header (excluded)</span>
            </label>
          </div>
          <DialogFooter>
            <button
              type="button"
              className="px-3 py-2 rounded-md text-sm border border-input hover:bg-accent/30"
              onClick={handleCsvModalClose}
            >
              Cancel
            </button>
            <button
              type="button"
              className="px-3 py-2 rounded-md text-sm bg-primary text-primary-foreground hover:opacity-90 disabled:opacity-60"
              onClick={handleCsvImport}
              disabled={!csvFile}
            >
              Import
            </button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Sample Data Modal */}
      <Dialog
        open={showSampleDataModal}
        onOpenChange={open => {
          setShowSampleDataModal(open)
          if (!open) {
            setSelectedSampleDataset(null)
            setIsImportingSampleData(false)
          }
        }}
      >
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Use sample data</DialogTitle>
          </DialogHeader>
          <div className="flex flex-col gap-2">
            <p className="text-sm text-muted-foreground mb-2">
              Choose a sample dataset to get started quickly.
            </p>
            {CLASSIFIER_SAMPLE_DATASETS.map(dataset => {
              const isSelected = selectedSampleDataset === dataset.id
              return (
                <button
                  key={dataset.id}
                  onClick={() => setSelectedSampleDataset(dataset.id)}
                  disabled={isImportingSampleData}
                  className={`flex items-center gap-3 p-3 rounded-lg border transition-colors text-left group ${
                    isSelected
                      ? 'border-primary bg-primary/5'
                      : 'border-border hover:bg-muted/50 hover:border-muted-foreground/50'
                  } ${isImportingSampleData ? 'opacity-50 cursor-not-allowed' : ''}`}
                >
                  <div
                    className={`w-9 h-9 rounded-lg flex items-center justify-center shrink-0 ${
                      isSelected ? 'bg-primary/10' : 'bg-muted group-hover:bg-muted/80'
                    }`}
                  >
                    <FontIcon
                      type="prompt"
                      className={`w-4 h-4 ${
                        isSelected ? 'text-primary' : 'text-muted-foreground'
                      }`}
                    />
                  </div>
                  <div className="flex flex-col gap-0.5 min-w-0 flex-1">
                    <span className="text-sm font-medium">{dataset.name}</span>
                    <span className="text-xs text-muted-foreground">
                      {dataset.description}
                    </span>
                  </div>
                  {isSelected && (
                    <FontIcon type="checkmark-filled" className="w-4 h-4 text-primary shrink-0" />
                  )}
                </button>
              )
            })}
          </div>
          <DialogFooter>
            <button
              type="button"
              className="px-3 py-2 rounded-md text-sm border border-input hover:bg-accent/30"
              onClick={() => {
                setShowSampleDataModal(false)
                setSelectedSampleDataset(null)
              }}
              disabled={isImportingSampleData}
            >
              Cancel
            </button>
            <button
              type="button"
              className="px-3 py-2 rounded-md text-sm bg-primary text-primary-foreground hover:opacity-90 disabled:opacity-60 flex items-center gap-2"
              onClick={handleImportSampleData}
              disabled={!selectedSampleDataset || isImportingSampleData}
            >
              {isImportingSampleData && (
                <span className="w-3.5 h-3.5 border-2 border-current border-t-transparent rounded-full animate-spin" />
              )}
              {isImportingSampleData ? 'Importing...' : 'Import data'}
            </button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}

export default ClassifierModel
