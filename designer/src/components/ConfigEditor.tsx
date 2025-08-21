import { useMemo, useState } from 'react'
import FontIcon from '../common/FontIcon'

type FileId =
  | 'prompts.json'
  | 'training.json'
  | 'data.json'
  | 'config.yaml'
  | 'models.json'
  | 'rag.yaml'

interface OpenTab {
  id: FileId
  label: string
}

const initialFiles: { id: FileId; label: string; group: string }[] = [
  { id: 'prompts.json', label: 'prompts.json', group: 'Config' },
  { id: 'training.json', label: 'training.json', group: 'Config' },
  { id: 'data.json', label: 'data.json', group: 'Data' },
  { id: 'config.yaml', label: 'config.yaml', group: 'Config' },
  { id: 'models.json', label: 'models.json', group: 'Models' },
  { id: 'rag.yaml', label: 'rag.yaml', group: 'RAG' },
]

const dummyContent: Record<FileId, string> = {
  'prompts.json':
    '{\n  "system": "You are LlamaFarm.",\n  "templates": []\n}\n',
  'training.json': '{\n  "dataset": [],\n  "strategy": "grid-search"\n}\n',
  'data.json': '{\n  "sources": ["./data/logs/*.pdf"]\n}\n',
  'config.yaml': 'project: aircraft-mx-flow\nfeatures:\n  - rag\n  - prompts\n',
  'models.json': '{\n  "default": "llama3:8b",\n  "adapters": []\n}\n',
  'rag.yaml': 'retriever:\n  strategy: hybrid\n',
}

function ConfigEditor() {
  const [query, setQuery] = useState('')
  const [openGroups, setOpenGroups] = useState<Record<string, boolean>>({
    Config: true,
    Data: true,
    Models: true,
    RAG: true,
  })
  const [openTabs, setOpenTabs] = useState<OpenTab[]>([
    { id: 'prompts.json', label: 'prompts.json' },
  ])
  const [activeId, setActiveId] = useState<FileId>('prompts.json')
  const [content, setContent] = useState<Record<FileId, string>>(dummyContent)

  const filesByGroup = useMemo(() => {
    const groups: Record<string, { id: FileId; label: string }[]> = {}
    for (const f of initialFiles) {
      if (query && !f.label.toLowerCase().includes(query.toLowerCase()))
        continue
      if (!groups[f.group]) groups[f.group] = []
      groups[f.group].push({ id: f.id, label: f.label })
    }
    return groups
  }, [query])

  const openFile = (id: FileId, label: string) => {
    setActiveId(id)
    setOpenTabs(prev =>
      prev.find(t => t.id === id) ? prev : [...prev, { id, label }]
    )
  }

  const closeTab = (id: FileId) => {
    setOpenTabs(prev => {
      const remaining = prev.filter(t => t.id !== id)
      // Prevent closing the last tab; keep at least one open
      if (remaining.length === 0) return prev
      if (activeId === id) {
        setActiveId(remaining[0].id)
      }
      return remaining
    })
  }

  return (
    <div className="w-full h-[70vh] min-h-[420px] rounded-lg overflow-hidden flex bg-card">
      {/* Sidebar */}
      <div className="w-64 shrink-0 border-r border-border p-3 space-y-3">
        <div className="flex items-center gap-2 bg-card rounded-md px-2 py-1 border border-input">
          <FontIcon
            type="search"
            className="w-4 h-4 text-gray-800 dark:text-white"
          />
          <input
            value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder="Search config"
            className="w-full bg-transparent text-sm focus:outline-none text-foreground"
          />
        </div>

        {Object.entries(filesByGroup).map(([group, files]) => (
          <div key={group} className="text-sm">
            <button
              className="w-full flex items-center justify-between px-2 py-1 rounded hover:bg-accent/20"
              onClick={() => setOpenGroups(g => ({ ...g, [group]: !g[group] }))}
            >
              <span className="text-foreground">{group}</span>
              <FontIcon
                type="chevron-down"
                className={`w-4 h-4 text-foreground transition-transform ${
                  openGroups[group] ? 'rotate-180' : ''
                }`}
              />
            </button>
            {openGroups[group] && (
              <div className="mt-1 space-y-1">
                {files.map(f => (
                  <button
                    key={f.id}
                    className={`w-full text-left px-2 py-1 rounded-md hover:bg-accent/20 ${
                      activeId === f.id ? 'bg-accent/30' : ''
                    }`}
                    onClick={() => openFile(f.id, f.label)}
                  >
                    {f.label}
                  </button>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Main */}
      <div className="flex-1 flex flex-col">
        {/* Tabs */}
        <div className="flex items-center gap-1 px-3 py-2 border-b border-border bg-card">
          {openTabs.map(tab => (
            <div
              key={tab.id}
              className={`flex items-center gap-2 px-2 py-1 rounded-md text-sm cursor-default ${
                activeId === tab.id
                  ? 'bg-accent/30 text-foreground'
                  : 'text-muted-foreground'
              }`}
            >
              <button
                className="outline-none"
                onClick={() => setActiveId(tab.id)}
              >
                {tab.label}
              </button>
              <FontIcon
                isButton
                type="close"
                className="w-4 h-4"
                handleOnClick={() => closeTab(tab.id)}
              />
            </div>
          ))}
        </div>

        {/* Editor area */}
        <div className="flex-1 grid grid-cols-1 lg:grid-cols-2 gap-3 p-3">
          <div className="col-span-1">
            <textarea
              className="w-full h-full resize-none bg-background text-foreground rounded-md p-3 font-mono text-sm"
              spellCheck={false}
              autoComplete="off"
              value={content[activeId] ?? ''}
              onChange={e =>
                setContent(prev => ({ ...prev, [activeId]: e.target.value }))
              }
            />
          </div>
          <div className="hidden lg:block col-span-1">
            <div className="w-full h-full rounded-md p-3 font-mono text-[13px] leading-6 bg-card border border-border overflow-auto">
              <pre className="whitespace-pre-wrap">
                <code>
                  <span className="text-foreground">project:</span>{' '}
                  <span className="text-teal-500 dark:text-teal-300">
                    aircraft-mx-flow
                  </span>
                  {'\n'}
                  <span className="text-foreground">features:</span>
                  {'\n'} -{' '}
                  <span className="text-teal-500 dark:text-teal-300">rag</span>
                  {'\n'} -{' '}
                  <span className="text-teal-500 dark:text-teal-300">
                    prompts
                  </span>
                  {'\n'} -{' '}
                  <span className="text-teal-500 dark:text-teal-300">
                    models
                  </span>
                  {'\n'}
                  <span className="text-foreground">retriever:</span>
                  {'\n'} <span className="text-foreground">strategy:</span>{' '}
                  <span className="text-teal-500 dark:text-teal-300">
                    hybrid
                  </span>
                  {'\n'} <span className="text-foreground">top_k:</span>{' '}
                  <span className="text-teal-500 dark:text-teal-300">8</span>
                  {'\n'}{' '}
                  <span className="text-foreground">score_threshold:</span>{' '}
                  <span className="text-teal-500 dark:text-teal-300">0.72</span>
                </code>
              </pre>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ConfigEditor
