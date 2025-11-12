import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '../ui/dropdown-menu'
import FontIcon from '../../common/FontIcon'

interface ModelOption {
  identifier: string
  name: string
  source: 'project' | 'disk'
}

interface ModelSelectorProps {
  currentModelIdentifier: string
  availableProjectModels: ModelOption[]
  availableDeviceModels: ModelOption[]
  onModelChange: (newModelIdentifier: string) => void
  disabled?: boolean
  label?: string
}

export function ModelSelector({
  currentModelIdentifier,
  availableProjectModels,
  availableDeviceModels,
  onModelChange,
  disabled = false,
  label = 'Model',
}: ModelSelectorProps) {
  // Filter out the current model from project and device models
  const filteredProjectModels = availableProjectModels.filter(
    m => m.identifier !== currentModelIdentifier
  )
  const filteredDeviceModels = availableDeviceModels.filter(
    m => m.identifier !== currentModelIdentifier
  )

  const hasProjectModels = filteredProjectModels.length > 0
  const hasDeviceModels = filteredDeviceModels.length > 0

  return (
    <div>
      {label && (
        <label className="text-xs text-muted-foreground mb-1 block">
          {label}
        </label>
      )}
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <button
            type="button"
            disabled={disabled}
            className="w-full h-9 rounded-lg border border-input bg-background px-3 text-left flex items-center justify-between disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <span className="truncate text-sm">
              {currentModelIdentifier || 'Select model'}
            </span>
            <FontIcon type="chevron-down" className="w-4 h-4" />
          </button>
        </DropdownMenuTrigger>
        <DropdownMenuContent className="w-64 max-h-64 overflow-auto">
          {hasProjectModels && (
            <>
              <div className="px-2 py-1.5 text-xs font-medium text-muted-foreground">
                From project
              </div>
              {filteredProjectModels.map(model => (
                <DropdownMenuItem
                  key={`project-${model.identifier}`}
                  className="w-full justify-start text-left"
                  onClick={() => onModelChange(model.identifier)}
                >
                  <div className="flex flex-col">
                    <span className="text-sm">{model.identifier}</span>
                    {model.name !== model.identifier && (
                      <span className="text-xs text-muted-foreground">
                        {model.name}
                      </span>
                    )}
                  </div>
                </DropdownMenuItem>
              ))}
            </>
          )}
          
          {hasProjectModels && hasDeviceModels && (
            <div className="h-px bg-border my-1" />
          )}
          
          {hasDeviceModels && (
            <>
              <div className="px-2 py-1.5 text-xs font-medium text-muted-foreground">
                From disk
              </div>
              {filteredDeviceModels.map(model => (
                <DropdownMenuItem
                  key={`disk-${model.identifier}`}
                  className="w-full justify-start text-left"
                  onClick={() => onModelChange(model.identifier)}
                >
                  <div className="flex flex-col">
                    <span className="text-sm">{model.identifier}</span>
                    {model.name !== model.identifier && (
                      <span className="text-xs text-muted-foreground">
                        {model.name}
                      </span>
                    )}
                  </div>
                </DropdownMenuItem>
              ))}
            </>
          )}
          
          {!hasProjectModels && !hasDeviceModels && (
            <div className="px-2 py-2 text-xs text-muted-foreground text-center">
              No other models available
            </div>
          )}
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  )
}

