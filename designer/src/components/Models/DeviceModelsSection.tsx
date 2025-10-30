import { Button } from '../ui/button'
import FontIcon from '../../common/FontIcon'
import Loader from '../../common/Loader'

interface DeviceModel {
  id: string
  name: string
  modelIdentifier: string
  meta: string
  badges: string[]
}

interface DeviceModelCardProps {
  model: DeviceModel
  onUse: () => void
  onDelete: () => void
  isInUse?: boolean
}

function DeviceModelCard({
  model,
  onUse,
  onDelete,
  isInUse,
}: DeviceModelCardProps) {
  return (
    <div className="w-full h-full bg-card rounded-lg border border-border flex flex-col gap-2 p-4 relative">
      <div className="absolute top-2 right-2">
        <div className="relative">
          <button
            className="w-6 h-6 grid place-items-center rounded-md text-muted-foreground hover:bg-accent/30"
            onClick={onDelete}
            aria-label="Delete from disk"
          >
            <FontIcon type="overflow" className="w-4 h-4" />
          </button>
        </div>
      </div>

      <div className="text-sm text-muted-foreground">
        {model.modelIdentifier}
      </div>

      <div className="flex items-center gap-2">
        <div className="text-lg font-medium">{model.name}</div>
      </div>

      <div className="text-sm text-muted-foreground">{model.meta}</div>

      <div className="mt-auto flex justify-end pt-2">
        <Button
          onClick={onUse}
          size="sm"
          disabled={isInUse}
          className="w-auto px-6"
        >
          {isInUse ? 'Using' : 'Use'}
        </Button>
      </div>
    </div>
  )
}

interface DeviceModelsSectionProps {
  models: DeviceModel[]
  isLoading: boolean
  isRefreshing: boolean
  onUse: (model: DeviceModel) => void
  onDelete: (model: DeviceModel) => void
  onRefresh: () => void
  isModelInUse: (modelId: string) => boolean
}

export function DeviceModelsSection({
  models,
  isLoading,
  isRefreshing,
  onUse,
  onDelete,
  onRefresh,
  isModelInUse,
}: DeviceModelsSectionProps) {
  return (
    <div className="flex flex-col gap-4 mb-8 md:mb-10">
      <div className="flex items-start justify-between">
        <div>
          <h3 className="font-medium">Models on device</h3>
          <div className="h-1" />
          <div className="text-sm text-muted-foreground">
            Models found on your local disk that are ready to use.
          </div>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={onRefresh}
          disabled={isLoading || isRefreshing}
          className="flex items-center gap-2"
        >
          {isLoading || isRefreshing ? (
            <Loader size={14} className="border-primary" />
          ) : (
            <FontIcon type="recently-viewed" className="w-4 h-4" />
          )}
          {isLoading || isRefreshing ? 'Refreshing...' : 'Refresh'}
        </Button>
      </div>
      {isLoading || isRefreshing ? (
        <div className="flex items-center justify-center py-8">
          <Loader size={24} className="border-primary" />
        </div>
      ) : models.length === 0 ? (
        <div className="rounded-xl border border-border bg-card p-8 flex items-center justify-center">
          <div className="text-sm text-muted-foreground text-center">
            No models found on disk. Download models below to get started.
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 items-stretch">
          {models.map(m => (
            <DeviceModelCard
              key={m.id}
              model={m}
              onUse={() => onUse(m)}
              onDelete={() => onDelete(m)}
              isInUse={isModelInUse(m.id)}
            />
          ))}
        </div>
      )}
    </div>
  )
}

export type { DeviceModel }

