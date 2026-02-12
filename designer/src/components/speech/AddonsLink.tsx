import { Package } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '../ui/tooltip'

interface AddonsLinkProps {
  className?: string
}

/**
 * Contextual link to manage add-ons page.
 * Place next to add-on-dependent feature toggles.
 */
export function AddonsLink({ className = '' }: AddonsLinkProps) {
  const navigate = useNavigate()

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <button
            type="button"
            onClick={() => navigate('/addons')}
            className={`text-muted-foreground hover:text-foreground transition-colors ${className}`}
            aria-label="Manage add-ons"
          >
            <Package className="w-4 h-4" />
          </button>
        </TooltipTrigger>
        <TooltipContent>
          <p>This feature is an add-on. Click to manage add-ons</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  )
}
