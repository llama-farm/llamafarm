import { useState } from 'react'
import { Input } from '../ui/input'
import { Badge } from '../ui/badge'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '../ui/dropdown-menu'
import { useToast } from '../ui/toast'
import FontIcon from '../../common/FontIcon'
import { useVisionModels, useLoadVisionModel, useExportVisionModel } from '../../hooks/useVision'

export function ModelsPanel() {
  const [search, setSearch] = useState('')
  const { data, isLoading, error } = useVisionModels()
  const loadModel = useLoadVisionModel()
  const exportModel = useExportVisionModel()
  const { toast } = useToast()

  const models = data?.models ?? []
  const filtered = models.filter(m =>
    m.name.toLowerCase().includes(search.toLowerCase())
  )

  const handleLoad = (name: string) => {
    loadModel.mutate(
      { name },
      {
        onSuccess: () => toast({ message: `Model "${name}" loaded`, variant: 'default' }),
        onError: (err: Error) => toast({ message: err.message, variant: 'destructive' }),
      }
    )
  }

  const handleExport = (name: string) => {
    exportModel.mutate(
      { name },
      {
        onSuccess: (blob) => {
          const url = URL.createObjectURL(blob)
          const a = document.createElement('a')
          a.href = url
          a.download = `${name}.zip`
          a.click()
          // Delay revocation to give browser time to start download
          setTimeout(() => URL.revokeObjectURL(url), 30_000)
          toast({ message: `Model "${name}" exported`, variant: 'default' })
        },
        onError: (err: Error) => toast({ message: err.message, variant: 'destructive' }),
      }
    )
  }

  if (isLoading) {
    return <div className="text-sm text-muted-foreground py-8 text-center">Loading models...</div>
  }

  if (error) {
    return (
      <div className="text-sm text-destructive py-8 text-center">
        Failed to load models: {(error as Error).message}
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center gap-2">
        <div className="relative flex-1">
          <FontIcon type="search" className="w-4 h-4 text-muted-foreground absolute left-3 top-1/2 -translate-y-1/2" />
          <Input
            placeholder="Search models..."
            value={search}
            onChange={e => setSearch(e.target.value)}
            className="pl-9"
          />
        </div>
      </div>

      {filtered.length === 0 ? (
        <div className="text-center py-12">
          <p className="text-sm text-muted-foreground">
            {models.length === 0 ? 'No vision models saved yet. Train a model to get started.' : 'No models match your search.'}
          </p>
        </div>
      ) : (
        <div className="rounded-lg border border-border overflow-hidden">
          <div className="grid grid-cols-12 items-center bg-secondary text-secondary-foreground text-xs px-3 py-3">
            <div className="col-span-4">Name</div>
            <div className="col-span-3">Task</div>
            <div className="col-span-3">Created</div>
            <div className="col-span-2" />
          </div>
          {filtered.map(model => (
            <div
              key={model.name}
              className="grid grid-cols-12 items-center px-3 py-3 text-sm border-t border-border hover:bg-accent/40"
            >
              <div className="col-span-4 font-medium truncate">{model.name}</div>
              <div className="col-span-3">
                <Badge variant="secondary">{model.task}</Badge>
              </div>
              <div className="col-span-3 text-muted-foreground text-xs">
                {model.created_at ? new Date(model.created_at).toLocaleDateString() : '—'}
              </div>
              <div className="col-span-2 flex justify-end">
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <button className="w-6 h-6 grid place-items-center rounded-md text-muted-foreground hover:bg-accent/30">
                      <FontIcon type="overflow" className="w-4 h-4" />
                    </button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    <DropdownMenuItem onClick={() => handleLoad(model.name)}>
                      Load
                    </DropdownMenuItem>
                    <DropdownMenuItem onClick={() => handleExport(model.name)}>
                      Export
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
