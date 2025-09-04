import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import FontIcon from '../../common/FontIcon'
import { Button } from '../ui/button'
import { Badge } from '../ui/badge'
import SearchInput from '../ui/search-input'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '../ui/dropdown-menu'

type RagStrategy = {
  id: string
  name: string
  description: string
  isDefault: boolean
  datasetsUsing: number
}

function Rag() {
  const navigate = useNavigate()
  const [query, setQuery] = useState('')

  const strategies: RagStrategy[] = [
    {
      id: 'pdf-simple',
      name: 'PDF Simple',
      description: 'Description saying what good for',
      isDefault: true,
      datasetsUsing: 2,
    },
    {
      id: 'pdf-complex',
      name: 'PDF Complex',
      description: 'Description saying what good for',
      isDefault: true,
      datasetsUsing: 0,
    },
    {
      id: 'csv-extract',
      name: 'CSV extract',
      description: 'Description saying what good for',
      isDefault: true,
      datasetsUsing: 0,
    },
    {
      id: 'chat-extract-1',
      name: 'Chat extract',
      description: 'Description saying what good for',
      isDefault: true,
      datasetsUsing: 0,
    },
    {
      id: 'json-extract-1',
      name: 'JSON extract',
      description: 'Description saying what good for',
      isDefault: true,
      datasetsUsing: 0,
    },
    {
      id: 'chat-extract-2',
      name: 'Chat extract',
      description: 'Description saying what good for',
      isDefault: true,
      datasetsUsing: 0,
    },
    {
      id: 'json-extract-2',
      name: 'JSON extract',
      description: 'Description saying what good for',
      isDefault: false,
      datasetsUsing: 2,
    },
  ]

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (!q) return strategies
    return strategies.filter(s =>
      [s.name, s.description].some(v => v.toLowerCase().includes(q))
    )
  }, [query])

  return (
    <div className="w-full flex flex-col gap-4 pb-32">
      <div className="flex items-center justify-between mb-2">
        <h2 className="text-2xl">RAG</h2>
      </div>

      <div className="text-sm text-muted-foreground mb-1">
        simple description – these can be applied to datasets
      </div>

      <div className="rounded-lg border border-border bg-card p-4">
        <div className="flex items-center justify-between mb-2">
          <div className="text-sm font-medium">Default RAG strategies</div>
          <Button size="sm" onClick={() => navigate('/chat/rag#create')}>
            Create new
          </Button>
        </div>

        <div className="mb-3 max-w-xl">
          <SearchInput
            placeholder="Search RAG Strategies"
            value={query}
            onChange={e => setQuery(e.target.value)}
          />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
          {filtered.map(s => (
            <div
              key={s.id}
              className="w-full bg-card rounded-lg border border-border flex flex-col gap-2 p-4 relative hover:bg-accent/20 hover:cursor-pointer transition-colors"
              onClick={() => navigate(`/chat/rag/${s.id}`)}
              role="button"
              tabIndex={0}
              onKeyDown={e => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault()
                  navigate(`/chat/rag/${s.id}`)
                }
              }}
            >
              <div className="absolute top-2 right-2">
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <button
                      className="w-6 h-6 grid place-items-center rounded-md text-muted-foreground hover:bg-accent/30"
                      onClick={e => e.stopPropagation()}
                    >
                      <FontIcon type="overflow" className="w-4 h-4" />
                    </button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent
                    align="end"
                    className="min-w-[10rem] w-[10rem]"
                  >
                    <DropdownMenuItem
                      onClick={() => navigate(`/chat/rag/${s.id}`)}
                    >
                      Configure
                    </DropdownMenuItem>
                    <DropdownMenuItem>Duplicate</DropdownMenuItem>
                    <DropdownMenuItem className="text-destructive focus:text-destructive">
                      Delete
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>

              <div className="text-sm font-medium">{s.name}</div>
              <button
                className="text-xs text-primary text-left hover:underline w-fit"
                onClick={e => e.stopPropagation()}
              >
                {s.description}
              </button>

              <div className="flex items-center gap-2 flex-wrap">
                {s.isDefault ? (
                  <Badge variant="default" size="sm" className="rounded-xl">
                    Default
                  </Badge>
                ) : (
                  <Badge
                    variant="default"
                    size="sm"
                    className="rounded-xl bg-emerald-600 text-emerald-50 dark:bg-emerald-400 dark:text-emerald-900"
                  >
                    Custom
                  </Badge>
                )}
                <Badge variant="secondary" size="sm" className="rounded-xl">
                  {s.datasetsUsing > 0
                    ? `${s.datasetsUsing} datasets using`
                    : '—'}
                </Badge>
              </div>

              <div className="flex justify-end pt-1">
                <Button
                  variant="outline"
                  size="sm"
                  className="px-3"
                  onClick={e => {
                    e.stopPropagation()
                    navigate(`/chat/rag/${s.id}`)
                  }}
                >
                  Configure
                </Button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

export default Rag
