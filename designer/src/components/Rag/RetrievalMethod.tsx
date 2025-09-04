import { useMemo } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { Button } from '../ui/button'
import FontIcon from '../../common/FontIcon'

function RetrievalMethod() {
  const navigate = useNavigate()
  const { strategyId } = useParams()

  const strategyName = useMemo(() => {
    if (!strategyId) return 'Strategy'
    return strategyId
      .replace(/[-_]/g, ' ')
      .replace(/\b\w/g, c => c.toUpperCase())
  }, [strategyId])

  return (
    <div className="h-full w-full flex flex-col gap-3 pb-20">
      {/* Breadcrumb */}
      <nav className="text-sm md:text-base flex items-center gap-1.5 mb-1">
        <button
          className="text-teal-600 dark:text-teal-400 hover:underline"
          onClick={() => navigate('/chat/rag')}
        >
          RAG
        </button>
        <span className="text-muted-foreground px-1">/</span>
        <button
          className="text-teal-600 dark:text-teal-400 hover:underline"
          onClick={() => navigate(`/chat/rag/${strategyId}`)}
        >
          {strategyName}
        </button>
        <span className="text-muted-foreground px-1">/</span>
        <span className="text-foreground">Retrieval method</span>
      </nav>

      {/* Header */}
      <div className="flex items-center justify-between mb-1">
        <h2 className="text-lg md:text-xl font-medium">Retrieval method</h2>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => navigate(`/chat/rag/${strategyId}`)}
          >
            Back
          </Button>
          <Button size="sm" onClick={() => navigate(-1)}>
            <span className="mr-2 inline-flex">
              <FontIcon type="checkmark-filled" className="w-4 h-4" />
            </span>
            Save
          </Button>
        </div>
      </div>

      {/* Placeholder content */}
      <section className="rounded-lg border border-border bg-card p-4">
        <div className="text-sm text-muted-foreground">
          Page scaffold for configuring retrieval method. Add fields here next.
        </div>
      </section>
    </div>
  )
}

export default RetrievalMethod
