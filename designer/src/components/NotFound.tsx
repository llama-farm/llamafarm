import { useNavigate } from 'react-router-dom'
import { Button } from './ui/button'

const NotFound = () => {
  const navigate = useNavigate()

  return (
    <div className="flex items-center justify-center h-full w-full">
      <div className="text-center px-6 py-10 rounded-xl border border-border bg-card/40 max-w-[560px]">
        <div className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-full bg-primary/15 border border-primary/30">
          <span className="text-primary text-lg">🤔</span>
        </div>
        <div className="text-lg font-medium text-foreground">
          Oops, something isn't right
        </div>
        <div className="mt-1 text-sm text-muted-foreground">
          The page you're looking for doesn't exist or has been moved.
        </div>
        <div className="mt-4">
          <Button
            onClick={() => navigate('/chat/dashboard')}
            className="w-full sm:w-auto"
          >
            Return to dashboard
          </Button>
        </div>
      </div>
    </div>
  )
}

export default NotFound
