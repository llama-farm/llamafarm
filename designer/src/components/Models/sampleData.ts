/**
 * Sample datasets for anomaly detection demo/testing
 */

export interface SampleDataset {
  id: string
  name: string
  description: string
  type: 'numeric' | 'text'
  columns: number
  data: string
}

// Streaming dataset configuration
export interface StreamingDatasetConfig {
  schema: Record<string, 'numeric' | 'text'>
  recommended_rolling_windows: number[]
  recommended_lag_periods: number[]
  anomaly_injection_interval: number // Inject anomaly every N samples
  baseline: Record<string, { mean: number; std: number }>
}

export interface StreamingSampleDataset {
  id: string
  name: string
  description: string
  type: 'streaming'
  columns: number
  streamingConfig: StreamingDatasetConfig
  // generateSample returns one data point
  generateSample: (index: number, isAnomaly: boolean) => Record<string, number>
}

// Helper to generate sensor data with injected anomalies
function generateSensorSample(
  index: number,
  isAnomaly: boolean
): Record<string, number> {
  // Baseline values with typical industrial sensor ranges
  const baselines = {
    temperature: { mean: 72, std: 2 }, // Fahrenheit
    humidity: { mean: 45, std: 5 }, // Percent
    pressure: { mean: 1013.25, std: 5 }, // hPa
    motor_rpm: { mean: 3000, std: 50 }, // RPM
  }

  // Add time-based drift (simulates daily cycle)
  const timeFactor = Math.sin((index / 50) * Math.PI) * 0.5

  if (isAnomaly) {
    // Generate anomaly: spike one or more values
    const anomalyType = index % 4
    switch (anomalyType) {
      case 0: // Temperature spike (overheating)
        return {
          temperature: baselines.temperature.mean + 15 + Math.random() * 10,
          humidity: baselines.humidity.mean + (Math.random() - 0.5) * baselines.humidity.std * 2,
          pressure: baselines.pressure.mean + (Math.random() - 0.5) * baselines.pressure.std * 2,
          motor_rpm: baselines.motor_rpm.mean + (Math.random() - 0.5) * baselines.motor_rpm.std * 2,
        }
      case 1: // Pressure drop (leak)
        return {
          temperature: baselines.temperature.mean + (Math.random() - 0.5) * baselines.temperature.std * 2,
          humidity: baselines.humidity.mean + (Math.random() - 0.5) * baselines.humidity.std * 2,
          pressure: baselines.pressure.mean - 30 - Math.random() * 20,
          motor_rpm: baselines.motor_rpm.mean + (Math.random() - 0.5) * baselines.motor_rpm.std * 2,
        }
      case 2: // Motor RPM spike (mechanical issue)
        return {
          temperature: baselines.temperature.mean + (Math.random() - 0.5) * baselines.temperature.std * 2,
          humidity: baselines.humidity.mean + (Math.random() - 0.5) * baselines.humidity.std * 2,
          pressure: baselines.pressure.mean + (Math.random() - 0.5) * baselines.pressure.std * 2,
          motor_rpm: baselines.motor_rpm.mean + 500 + Math.random() * 300,
        }
      default: // Multi-sensor anomaly (correlated failure)
        return {
          temperature: baselines.temperature.mean + 10 + Math.random() * 5,
          humidity: baselines.humidity.mean + 15 + Math.random() * 10,
          pressure: baselines.pressure.mean - 20 - Math.random() * 10,
          motor_rpm: baselines.motor_rpm.mean + 200 + Math.random() * 100,
        }
    }
  }

  // Normal sample with natural variation
  return {
    temperature:
      baselines.temperature.mean +
      timeFactor * 2 +
      (Math.random() - 0.5) * baselines.temperature.std * 2,
    humidity:
      baselines.humidity.mean +
      timeFactor * 3 +
      (Math.random() - 0.5) * baselines.humidity.std * 2,
    pressure:
      baselines.pressure.mean +
      timeFactor * 2 +
      (Math.random() - 0.5) * baselines.pressure.std * 2,
    motor_rpm:
      baselines.motor_rpm.mean +
      timeFactor * 20 +
      (Math.random() - 0.5) * baselines.motor_rpm.std * 2,
  }
}

// Streaming datasets
export const STREAMING_DATASETS: StreamingSampleDataset[] = [
  {
    id: 'sensor-stream',
    name: 'Factory sensor stream',
    description: 'Streaming demo with periodic anomalies (4 sensors)',
    type: 'streaming',
    columns: 4,
    streamingConfig: {
      schema: {
        temperature: 'numeric',
        humidity: 'numeric',
        pressure: 'numeric',
        motor_rpm: 'numeric',
      },
      recommended_rolling_windows: [5, 10, 20],
      recommended_lag_periods: [1, 2, 5],
      anomaly_injection_interval: 50, // Inject anomaly every 50 samples
      baseline: {
        temperature: { mean: 72, std: 2 },
        humidity: { mean: 45, std: 5 },
        pressure: { mean: 1013.25, std: 5 },
        motor_rpm: { mean: 3000, std: 50 },
      },
    },
    generateSample: generateSensorSample,
  },
]

// Helper to generate a batch of streaming data
export function generateStreamingBatch(
  dataset: StreamingSampleDataset,
  startIndex: number,
  count: number
): { data: Record<string, number>[]; anomalyIndices: number[] } {
  const data: Record<string, number>[] = []
  const anomalyIndices: number[] = []

  for (let i = 0; i < count; i++) {
    const index = startIndex + i
    const isAnomaly = index > 0 && index % dataset.streamingConfig.anomaly_injection_interval === 0
    if (isAnomaly) {
      anomalyIndices.push(index)
    }
    data.push(dataset.generateSample(index, isAnomaly))
  }

  return { data, anomalyIndices }
}

export const SAMPLE_DATASETS: SampleDataset[] = [
  {
    id: 'fridge-temp',
    name: 'Fridge temperature data',
    description: 'Numeric, 1 column',
    type: 'numeric',
    columns: 1,
    data: `36.2
37.1
35.8
36.5
37.0
36.8
35.9
36.4
37.2
36.1
35.7
36.9
36.3
37.1
36.0
36.6
35.8
37.0
36.4
36.2
35.9
36.7
37.1
36.5
36.0
36.8
35.7
36.3
37.2
36.1
36.9
36.4
35.8
36.6
37.0
36.2
35.9
36.5
36.8
37.1
36.0
36.7
35.7
36.3
36.9
36.1
37.2
36.4
35.8
36.6
36.2
37.0
35.9
36.5
36.8
36.3
35.7
37.1
36.4
36.0
36.9
36.6
35.8
36.2
37.0
36.5
35.9
36.8
36.1
37.2
36.7
36.3
35.7
36.4
37.1
36.0
36.9
35.8
36.6
36.2
37.0
36.5
35.9
36.8
36.1
37.2
36.4
36.7
35.7
36.3
36.9
36.0
37.1
35.8
36.6
36.2
36.5
37.0
35.9
36.8`,
  },
  {
    id: 'biometric',
    name: 'Biometric data',
    description: 'Numeric, 5 columns',
    type: 'numeric',
    columns: 5,
    data: `72, 98.4, 98, 118, 14
68, 98.6, 97, 115, 15
75, 98.2, 99, 120, 14
70, 98.5, 98, 112, 16
74, 98.1, 97, 119, 15
69, 98.7, 98, 116, 14
71, 98.3, 99, 114, 15
73, 98.6, 97, 121, 16
67, 98.4, 98, 117, 14
76, 98.2, 99, 113, 15
70, 98.5, 98, 119, 14
72, 98.1, 97, 115, 16
68, 98.6, 99, 118, 15
74, 98.3, 98, 120, 14
71, 98.4, 97, 114, 15
69, 98.7, 98, 116, 16
73, 98.2, 99, 122, 14
75, 98.5, 97, 113, 15
70, 98.1, 98, 117, 14
72, 98.6, 99, 119, 16
68, 98.4, 98, 115, 15
74, 98.3, 97, 121, 14
71, 98.5, 98, 118, 15
69, 98.2, 99, 114, 16
76, 98.6, 97, 116, 14
70, 98.1, 98, 120, 15
73, 98.4, 99, 113, 14
67, 98.7, 97, 117, 16
72, 98.3, 98, 119, 15
75, 98.5, 99, 115, 14
71, 98.2, 97, 122, 15
68, 98.6, 98, 114, 16
74, 98.4, 99, 118, 14
70, 98.1, 98, 116, 15
73, 98.5, 97, 120, 14
69, 98.3, 98, 113, 16
72, 98.7, 99, 117, 15
76, 98.2, 97, 119, 14
71, 98.4, 98, 115, 15
68, 98.6, 99, 121, 16
74, 98.1, 98, 114, 14
70, 98.5, 97, 118, 15
75, 98.3, 98, 116, 14
72, 98.4, 99, 120, 16
69, 98.2, 97, 113, 15
73, 98.6, 98, 117, 14
71, 98.1, 99, 119, 15
67, 98.5, 98, 115, 16
74, 98.3, 97, 122, 14
70, 98.7, 98, 114, 15
72, 98.4, 99, 118, 14
68, 98.2, 98, 116, 15
75, 98.6, 97, 120, 16
71, 98.1, 98, 113, 14
73, 98.5, 99, 117, 15
69, 98.4, 98, 119, 14
72, 98.3, 97, 115, 16
76, 98.6, 99, 121, 15
70, 98.2, 98, 114, 14
74, 98.5, 97, 118, 15
68, 98.1, 98, 116, 14
71, 98.7, 99, 120, 16
73, 98.4, 98, 113, 15
75, 98.3, 97, 117, 14
70, 98.6, 98, 119, 15
72, 98.2, 99, 115, 16
69, 98.5, 98, 122, 14
74, 98.1, 97, 114, 15
71, 98.4, 98, 118, 14
67, 98.6, 99, 116, 16
73, 98.3, 98, 120, 15
76, 98.5, 97, 113, 14
70, 98.2, 98, 117, 15
72, 98.7, 99, 119, 14
68, 98.4, 98, 115, 16
75, 98.1, 97, 121, 15
71, 98.6, 98, 114, 14
74, 98.3, 99, 118, 15
69, 98.5, 98, 116, 16
73, 98.2, 97, 120, 14
70, 98.4, 98, 113, 15
72, 98.6, 99, 117, 14
76, 98.1, 98, 119, 16
68, 98.5, 97, 115, 15
74, 98.3, 98, 122, 14
71, 98.7, 99, 114, 15
73, 98.4, 98, 118, 14
69, 98.2, 97, 116, 16
75, 98.6, 98, 120, 15
70, 98.1, 99, 113, 14
72, 98.5, 98, 117, 15
67, 98.3, 97, 119, 14
74, 98.4, 98, 115, 16
71, 98.6, 99, 121, 15
73, 98.2, 98, 114, 14
69, 98.5, 97, 118, 15
76, 98.1, 98, 116, 16
70, 98.7, 99, 120, 14
72, 98.4, 98, 113, 15
68, 98.3, 97, 117, 14`,
  },
  {
    id: 'build-status',
    name: 'Build statuses',
    description: 'Text, 1 column',
    type: 'text',
    columns: 1,
    data: `success
success
success
running
success
success
queued
success
building
success
success
success
deploying
success
success
success
running
success
success
pending
success
success
success
building
success
success
success
success
queued
success
running
success
success
success
deploying
success
success
success
success
pending
success
success
building
success
success
success
running
success
success
success
queued
success
success
success
success
deploying
success
success
running
success
success
success
building
success
success
pending
success
success
success
success
queued
success
success
success
running
success
success
success
building
success
success
deploying
success
success
success
success
pending
success
success
success
running
success
success
queued
success
success
building
success
success
success
deploying
success`,
  },
  {
    id: 'support-ticket',
    name: 'Support ticket data',
    description: 'Text, 5 columns',
    type: 'text',
    columns: 5,
    data: `low, billing, email, resolved, 2
medium, technical, chat, open, 1
low, account, email, resolved, 3
low, billing, phone, resolved, 1
medium, technical, email, in_progress, 4
low, general, chat, resolved, 2
high, technical, phone, open, 1
low, billing, email, resolved, 3
medium, account, chat, in_progress, 2
low, shipping, email, resolved, 1
low, technical, email, resolved, 4
medium, billing, phone, open, 2
low, general, email, resolved, 1
low, account, chat, resolved, 3
medium, technical, email, in_progress, 5
low, billing, email, resolved, 2
low, shipping, phone, resolved, 1
high, account, chat, open, 1
low, technical, email, resolved, 3
medium, general, email, in_progress, 2
low, billing, chat, resolved, 4
low, technical, email, resolved, 1
medium, shipping, phone, open, 2
low, account, email, resolved, 3
low, billing, email, resolved, 1
medium, technical, chat, in_progress, 4
low, general, email, resolved, 2
low, billing, phone, resolved, 1
high, technical, email, open, 1
low, account, chat, resolved, 3
medium, billing, email, in_progress, 2
low, shipping, email, resolved, 5
low, technical, phone, resolved, 1
medium, account, chat, open, 3
low, billing, email, resolved, 2
low, general, email, resolved, 1
medium, technical, email, in_progress, 4
low, billing, chat, resolved, 2
low, shipping, phone, resolved, 1
high, account, email, open, 1
low, technical, email, resolved, 3
medium, billing, chat, in_progress, 2
low, account, email, resolved, 4
low, general, phone, resolved, 1
medium, technical, email, open, 2
low, billing, email, resolved, 3
low, shipping, chat, resolved, 1
medium, account, email, in_progress, 5
low, technical, phone, resolved, 2
low, billing, email, resolved, 1
high, general, chat, open, 1
low, account, email, resolved, 3
medium, technical, email, in_progress, 2
low, billing, phone, resolved, 4
low, shipping, email, resolved, 1
medium, account, chat, open, 2
low, technical, email, resolved, 3
low, general, email, resolved, 1
medium, billing, email, in_progress, 4
low, account, chat, resolved, 2
low, technical, phone, resolved, 1
high, shipping, email, open, 1
low, billing, email, resolved, 3
medium, general, chat, in_progress, 2
low, account, email, resolved, 5
low, technical, email, resolved, 1
medium, billing, phone, open, 2
low, shipping, chat, resolved, 3
low, general, email, resolved, 1
medium, technical, email, in_progress, 4
low, account, email, resolved, 2
low, billing, chat, resolved, 1
high, technical, phone, open, 1
low, shipping, email, resolved, 3
medium, account, email, in_progress, 2
low, general, phone, resolved, 4
low, billing, email, resolved, 1
medium, technical, chat, open, 2
low, account, email, resolved, 3
low, shipping, email, resolved, 1
medium, billing, email, in_progress, 5
low, technical, chat, resolved, 2
low, general, email, resolved, 1
high, account, email, open, 1
low, billing, phone, resolved, 3
medium, shipping, chat, in_progress, 2
low, technical, email, resolved, 4
low, account, email, resolved, 1
medium, general, email, open, 2
low, billing, chat, resolved, 3
low, shipping, phone, resolved, 1
medium, technical, email, in_progress, 4
low, account, chat, resolved, 2
low, billing, email, resolved, 1
high, technical, email, open, 1
low, general, email, resolved, 3
medium, account, phone, in_progress, 2
low, shipping, email, resolved, 5
low, billing, chat, resolved, 1
medium, technical, email, open, 2`,
  },
]
