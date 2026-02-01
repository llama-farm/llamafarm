import { useState, useEffect } from 'react'
import { Button } from '../ui/button'
import { Input } from '../ui/input'
import FontIcon from '../../common/FontIcon'
import RouteResult from './RouteResult'
import RouterDemo from './RouterDemo'
import NodeCard from './NodeCard'

import { useToast } from '../ui/toast'

interface RouteResponse {
  capability: string
  confidence: number
  node_name: string
  explanation?: string
}

interface CapabilitiesResponse {
  capabilities: Array<{
    node_name: string
    capabilities: Array<{
      capability: string
      description?: string
      examples?: string[]
    }>
  }>
}

const Router = () => {
  const [intent, setIntent] = useState('')
  const [isRouting, setIsRouting] = useState(false)
  const [result, setResult] = useState<RouteResponse | null>(null)
  const [capabilities, setCapabilities] = useState<CapabilitiesResponse | null>(null)
  const [isLoadingCapabilities, setIsLoadingCapabilities] = useState(false)
  const [healthStatus, setHealthStatus] = useState<'healthy' | 'unhealthy' | 'unknown'>('unknown')
  
  const { toast } = useToast()

  // Check health on mount
  useEffect(() => {
    checkHealth()
    loadCapabilities()
  }, [])

  const checkHealth = async () => {
    try {
      const response = await fetch('/v1/router/health')
      if (response.ok) {
        setHealthStatus('healthy')
      } else {
        setHealthStatus('unhealthy')
      }
    } catch (error) {
      console.error('Health check failed:', error)
      setHealthStatus('unhealthy')
    }
  }

  const loadCapabilities = async () => {
    setIsLoadingCapabilities(true)
    try {
      const response = await fetch('/v1/router/capabilities')
      if (response.ok) {
        const data = await response.json()
        setCapabilities(data)
      } else {
        toast({
          message: 'Failed to load capabilities',
          variant: 'destructive',
        })
      }
    } catch (error) {
      console.error('Failed to load capabilities:', error)
      toast({
        message: 'Error connecting to router',
        
        variant: 'destructive',
      })
    } finally {
      setIsLoadingCapabilities(false)
    }
  }

  const handleRoute = async () => {
    if (!intent.trim()) {
      toast({
        message: 'Please enter an intent to route',
        
        variant: 'destructive',
      })
      return
    }

    setIsRouting(true)
    try {
      const response = await fetch('/v1/router/route', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ intent: intent.trim() }),
      })

      if (response.ok) {
        const data = await response.json()
        setResult(data)
      } else {
        await response.json().catch(() => ({}))
        toast({
          message: 'Routing failed',
          
          variant: 'destructive',
        })
      }
    } catch (error) {
      console.error('Routing error:', error)
      toast({
        message: 'Error connecting to router',
        
        variant: 'destructive',
      })
    } finally {
      setIsRouting(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !isRouting && intent.trim()) {
      handleRoute()
    }
  }

  return (
    <div className="w-full max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <h1 className="text-3xl font-bold text-foreground">Semantic Router</h1>
          <div className="flex items-center gap-2">
            <div
              className={`flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium ${
                healthStatus === 'healthy'
                  ? 'bg-green-500/10 text-green-600 dark:text-green-400'
                  : healthStatus === 'unhealthy'
                    ? 'bg-red-500/10 text-red-600 dark:text-red-400'
                    : 'bg-muted text-muted-foreground'
              }`}
            >
              <div
                className={`w-2 h-2 rounded-full ${
                  healthStatus === 'healthy'
                    ? 'bg-green-500'
                    : healthStatus === 'unhealthy'
                      ? 'bg-red-500'
                      : 'bg-muted-foreground'
                }`}
              />
              {healthStatus === 'healthy'
                ? 'Router Active'
                : healthStatus === 'unhealthy'
                  ? 'Router Offline'
                  : 'Checking...'}
            </div>
          </div>
        </div>
        <p className="text-muted-foreground">
          Intelligently route user intents to the most capable node in your LlamaFarm cluster.
          The semantic router uses embeddings and similarity matching to understand what users want
          and direct them to the right service.
        </p>
      </div>

      {/* Interactive Demo */}
      <RouterDemo />

      {/* Main Routing Interface */}
      <div className="rounded-lg p-6 bg-card border border-border space-y-4">
        <div>
          <h2 className="text-xl font-semibold text-foreground mb-2">Try It Out</h2>
          <p className="text-sm text-muted-foreground">
            Enter a natural language intent and see how it gets routed
          </p>
        </div>

        <div className="flex gap-3">
          <div className="flex-1 relative">
            <Input
              type="text"
              placeholder="E.g., 'Summarize this document' or 'Process this image'"
              value={intent}
              onChange={(e) => setIntent(e.target.value)}
              onKeyPress={handleKeyPress}
              disabled={isRouting}
              className="pr-10"
            />
            {intent && (
              <button
                onClick={() => setIntent('')}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                aria-label="Clear input"
              >
                <FontIcon type="close" className="w-4 h-4" />
              </button>
            )}
          </div>
          <Button
            onClick={handleRoute}
            disabled={!intent.trim() || isRouting}
            size="default"
          >
            {isRouting ? (
              <>
                <FontIcon type="loading" className="w-4 h-4 mr-2 animate-spin" />
                Routing...
              </>
            ) : (
              'Route'
            )}
          </Button>
        </div>

        <RouteResult
          capability={result?.capability}
          confidence={result?.confidence}
          targetNode={result?.node_name}
          explanation={result?.explanation}
          isLoading={isRouting}
        />
      </div>

      {/* Registered Capabilities */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-semibold text-foreground">Registered Capabilities</h2>
            <p className="text-sm text-muted-foreground">
              Nodes and their advertised capabilities in the cluster
            </p>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={loadCapabilities}
            disabled={isLoadingCapabilities}
          >
            {isLoadingCapabilities ? (
              <>
                <FontIcon type="loading" className="w-4 h-4 mr-2 animate-spin" />
                Refreshing...
              </>
            ) : (
              <>
                <FontIcon type="recently-viewed" className="w-4 h-4 mr-2" />
                Refresh
              </>
            )}
          </Button>
        </div>

        {isLoadingCapabilities ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {[1, 2, 3].map((i) => (
              <div
                key={i}
                className="rounded-lg p-5 bg-card border border-border animate-pulse"
              >
                <div className="h-5 bg-muted rounded w-1/2 mb-3"></div>
                <div className="space-y-2">
                  <div className="h-4 bg-muted rounded w-3/4"></div>
                  <div className="h-4 bg-muted rounded w-2/3"></div>
                </div>
              </div>
            ))}
          </div>
        ) : capabilities && capabilities.capabilities.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {capabilities.capabilities.map((node, idx) => (
              <NodeCard
                key={idx}
                nodeName={node.node_name}
                capabilities={node.capabilities}
              />
            ))}
          </div>
        ) : (
          <div className="rounded-lg p-8 bg-card border border-border text-center">
            <FontIcon type="info" className="w-12 h-12 mx-auto mb-3 text-muted-foreground" />
            <p className="text-muted-foreground">
              No capabilities registered yet. Start nodes with advertised capabilities to see them here.
            </p>
          </div>
        )}
      </div>
    </div>
  )
}

export default Router
