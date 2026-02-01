import { useState } from 'react'
import { Button } from '../ui/button'
import RouteResult from './RouteResult'

const SAMPLE_INTENTS = [
  {
    intent: 'Analyze this image for objects',
    capability: 'Vision Analysis',
    confidence: 0.92,
    targetNode: 'vision-node-01',
    explanation: 'Strong keyword match for "analyze" and "image" with vision processing capability',
  },
  {
    intent: "What's the weather forecast?",
    capability: 'Weather Data Retrieval',
    confidence: 0.88,
    targetNode: 'weather-api-node',
    explanation: 'Direct match for weather-related queries with external API integration',
  },
  {
    intent: 'Generate embeddings for this text',
    capability: 'Text Embedding Generation',
    confidence: 0.95,
    targetNode: 'embedding-node-02',
    explanation: 'Exact capability match for embedding generation tasks',
  },
  {
    intent: 'Run inference on 70B model',
    capability: 'Large Model Inference',
    confidence: 0.91,
    targetNode: 'llama-70b-node',
    explanation: 'Model size match (70B) routes to dedicated large model inference node',
  },
]

const RouterDemo = () => {
  const [isRunning, setIsRunning] = useState(false)
  const [currentIndex, setCurrentIndex] = useState(-1)
  const [currentIntent, setCurrentIntent] = useState('')
  const [result, setResult] = useState<any>(null)

  const startDemo = async () => {
    setIsRunning(true)
    setCurrentIndex(0)
    
    for (let i = 0; i < SAMPLE_INTENTS.length; i++) {
      const sample = SAMPLE_INTENTS[i]
      setCurrentIndex(i)
      setCurrentIntent('')
      setResult(null)
      
      // Type out the intent
      for (let j = 0; j <= sample.intent.length; j++) {
        setCurrentIntent(sample.intent.substring(0, j))
        await new Promise(resolve => setTimeout(resolve, 30))
      }
      
      // Short pause before showing result
      await new Promise(resolve => setTimeout(resolve, 400))
      
      // Show result
      setResult({
        capability: sample.capability,
        confidence: sample.confidence,
        targetNode: sample.targetNode,
        explanation: sample.explanation,
      })
      
      // Pause to view result
      await new Promise(resolve => setTimeout(resolve, 2500))
    }
    
    setIsRunning(false)
    setCurrentIndex(-1)
  }

  const stopDemo = () => {
    setIsRunning(false)
    setCurrentIndex(-1)
    setCurrentIntent('')
    setResult(null)
  }

  return (
    <div className="rounded-lg p-6 bg-gradient-to-br from-teal-500/5 to-blue-500/5 border border-teal-500/20">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-semibold text-foreground">Interactive Demo</h3>
          <p className="text-sm text-muted-foreground">
            Watch how semantic routing works with sample intents
          </p>
        </div>
        <Button
          onClick={isRunning ? stopDemo : startDemo}
          variant={isRunning ? 'destructive' : 'default'}
          size="default"
        >
          {isRunning ? 'Stop Demo' : 'Start Demo'}
        </Button>
      </div>

      {currentIndex >= 0 && (
        <div className="space-y-4">
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <div className="flex gap-1">
              {SAMPLE_INTENTS.map((_, idx) => (
                <div
                  key={idx}
                  className={`h-1 w-8 rounded-full transition-colors ${
                    idx === currentIndex
                      ? 'bg-teal-500'
                      : idx < currentIndex
                        ? 'bg-teal-500/40'
                        : 'bg-muted'
                  }`}
                />
              ))}
            </div>
            <span>
              Step {currentIndex + 1} of {SAMPLE_INTENTS.length}
            </span>
          </div>

          <div className="rounded p-4 bg-background/60 border border-border">
            <p className="text-sm text-muted-foreground mb-2">Intent:</p>
            <p className="text-base text-foreground font-medium">
              {currentIntent}
              {!result && <span className="animate-pulse">|</span>}
            </p>
          </div>

          {result && (
            <div className="animate-in fade-in duration-300">
              <RouteResult
                capability={result.capability}
                confidence={result.confidence}
                targetNode={result.targetNode}
                explanation={result.explanation}
              />
            </div>
          )}
        </div>
      )}

      {currentIndex === -1 && !isRunning && (
        <div className="text-center py-8 text-muted-foreground">
          <p className="text-sm">Click "Start Demo" to see semantic routing in action</p>
        </div>
      )}
    </div>
  )
}

export default RouterDemo
