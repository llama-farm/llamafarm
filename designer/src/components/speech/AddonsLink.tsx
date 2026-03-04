import { Package } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip'

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
          <span
            role="link"
            tabIndex={0}
            onClick={(e) => { e.stopPropagation(); navigate('/addons') }}
            onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.stopPropagation(); navigate('/addons') } }}
            className={`text-muted-foreground hover:text-foreground transition-colors cursor-pointer inline-flex ${className}`}
            aria-label="Manage add-ons"
          >
            <Package className="w-4 h-4" />
          </span>
        </TooltipTrigger>
        <TooltipContent>
          <p>This feature is an add-on. Click to manage add-ons</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  )
}
