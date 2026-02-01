interface Capability {
  capability: string
  description?: string
  examples?: string[]
}

interface NodeCardProps {
  nodeName: string
  capabilities: Capability[]
}

const NodeCard = ({ nodeName, capabilities }: NodeCardProps) => {
  return (
    <div className="rounded-lg p-5 bg-card border border-border hover:border-teal-500/50 transition-colors">
      <div className="flex items-center gap-2 mb-3">
        <div className="w-2 h-2 rounded-full bg-teal-500"></div>
        <h3 className="text-base font-semibold text-foreground">{nodeName}</h3>
      </div>

      <div className="space-y-2">
        {capabilities.length === 0 ? (
          <p className="text-sm text-muted-foreground italic">No capabilities registered</p>
        ) : (
          capabilities.map((cap, idx) => (
            <div key={idx} className="text-sm">
              <p className="text-foreground font-medium">{cap.capability}</p>
              {cap.description && (
                <p className="text-muted-foreground text-xs mt-1">{cap.description}</p>
              )}
              {cap.examples && cap.examples.length > 0 && (
                <div className="mt-1 pl-3 border-l-2 border-muted">
                  {cap.examples.slice(0, 2).map((example, exIdx) => (
                    <p key={exIdx} className="text-xs text-muted-foreground italic">
                      "{example}"
                    </p>
                  ))}
                </div>
              )}
            </div>
          ))
        )}
      </div>

      <div className="mt-3 pt-3 border-t border-border">
        <p className="text-xs text-muted-foreground">
          {capabilities.length} {capabilities.length === 1 ? 'capability' : 'capabilities'}
        </p>
      </div>
    </div>
  )
}

export default NodeCard
