import { useState, useRef, useEffect } from 'react'
import { Check, Copy } from 'lucide-react'
import type { CapturedRequest } from '../../contexts/DevToolsContext'
import { generateCode, type CodeFormat } from '../../utils/devToolsCodeGen'
import { cn } from '@/lib/utils'

interface DevToolsCodeSnippetsProps {
  request: CapturedRequest
}

const codeFormats: { id: CodeFormat; label: string }[] = [
  { id: 'curl', label: 'cURL' },
  { id: 'python', label: 'Python' },
  { id: 'javascript', label: 'JavaScript' },
]

export default function DevToolsCodeSnippets({ request }: DevToolsCodeSnippetsProps) {
  const [format, setFormat] = useState<CodeFormat>('curl')
  const [copied, setCopied] = useState(false)
  const copyTimeoutRef = useRef<NodeJS.Timeout | null>(null)

  // Cleanup timeout on unmount to prevent memory leak
  useEffect(() => {
    return () => {
      if (copyTimeoutRef.current) {
        clearTimeout(copyTimeoutRef.current)
      }
    }
  }, [])

  const code = generateCode(request, format)

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code)
      setCopied(true)
      // Clear any existing timeout before setting a new one
      if (copyTimeoutRef.current) {
        clearTimeout(copyTimeoutRef.current)
      }
      copyTimeoutRef.current = setTimeout(() => setCopied(false), 2000)
    } catch (err) {
      console.error('Failed to copy:', err)
    }
  }

  return (
    <div className="p-4 h-full flex flex-col gap-3">
      {/* Format selector */}
      <div className="flex items-center justify-between">
        <div className="flex gap-1">
          {codeFormats.map(f => (
            <button
              key={f.id}
              onClick={() => setFormat(f.id)}
              className={cn(
                'px-2 py-1 text-xs rounded transition-colors',
                format === f.id
                  ? 'bg-primary text-primary-foreground'
                  : 'text-muted-foreground hover:text-foreground hover:bg-muted'
              )}
            >
              {f.label}
            </button>
          ))}
        </div>

        {/* Copy button */}
        <button
          onClick={handleCopy}
          className="flex items-center gap-1.5 px-2 py-1 text-xs rounded bg-muted hover:bg-accent transition-colors"
          aria-label={copied ? 'Copied' : 'Copy to clipboard'}
        >
          {copied ? (
            <>
              <Check className="w-3.5 h-3.5 text-green-500" />
              <span className="text-green-500">Copied</span>
            </>
          ) : (
            <>
              <Copy className="w-3.5 h-3.5 text-muted-foreground" />
              <span className="text-muted-foreground">Copy</span>
            </>
          )}
        </button>
      </div>

      {/* Code display */}
      <pre className="block p-3 rounded bg-muted text-foreground font-mono text-xs overflow-auto flex-1 scrollbar-thin whitespace-pre-wrap">
        {code}
      </pre>
    </div>
  )
}
