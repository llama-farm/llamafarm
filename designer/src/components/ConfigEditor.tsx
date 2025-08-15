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
    <div className="w-full h-[70vh] min-h-[420px] rounded-lg overflow-hidden flex bg-white dark:bg-blue-500">
      {/* Sidebar */}
      <div className="w-64 shrink-0 border-r border-blue-50 dark:border-blue-700 p-3 space-y-3">
        <div className="flex items-center gap-2 bg-[#FFFFFF] dark:bg-blue-600 rounded-md px-2 py-1 border border-blue-50 dark:border-blue-400/40">
          <FontIcon
            type="search"
            className="w-4 h-4 text-gray-800 dark:text-white"
          />
          <input
            value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder="Search config"
            className="w-full bg-transparent text-sm focus:outline-none text-gray-900 dark:text-white"
          />
        </div>

        {Object.entries(filesByGroup).map(([group, files]) => (
          <div key={group} className="text-sm">
            <button
              className="w-full flex items-center justify-between px-2 py-1 rounded hover:bg-blue-50/50 dark:hover:bg-blue-600/40"
              onClick={() => setOpenGroups(g => ({ ...g, [group]: !g[group] }))}
            >
              <span className="text-gray-800 dark:text-white">{group}</span>
              <FontIcon
                type="chevron-down"
                className={`w-4 h-4 text-gray-800 dark:text-white transition-transform ${
                  openGroups[group] ? 'rotate-180' : ''
                }`}
              />
            </button>
            {openGroups[group] && (
              <div className="mt-1 space-y-1">
                {files.map(f => (
                  <button
                    key={f.id}
                    className={`w-full text-left px-2 py-1 rounded-md hover:bg-blue-50/50 dark:hover:bg-blue-600/40 ${
                      activeId === f.id
                        ? 'bg-blue-50/60 dark:bg-blue-600/50'
                        : ''
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
        <div className="flex items-center gap-1 px-3 py-2 border-b border-blue-50 dark:border-blue-700 bg-[#FFFFFF] dark:bg-blue-600">
          {openTabs.map(tab => (
            <div
              key={tab.id}
              className={`flex items-center gap-2 px-2 py-1 rounded-md text-sm cursor-default ${
                activeId === tab.id
                  ? 'bg-blue-50/60 dark:bg-blue-700 text-gray-900 dark:text-white'
                  : 'text-blue-100'
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
        <div className="flex-1 p-3">
          <textarea
            className="w-full h-full resize-none bg-[#0b122b] text-white rounded-md p-3 font-mono text-sm"
            value={content[activeId] ?? ''}
            onChange={e =>
              setContent(prev => ({ ...prev, [activeId]: e.target.value }))
            }
          />
        </div>
      </div>
    </div>
  )
}

export default ConfigEditor
