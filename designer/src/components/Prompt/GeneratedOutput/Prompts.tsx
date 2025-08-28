import { useState } from 'react'
import FontIcon from '../../../common/FontIcon'
import PromptModal, { PromptModalMode } from './PromptModal'

interface PromptRow {
  version: string
  status: 'Active' | 'Draft'
  preview: string
  settings: string
}

const Prompts = () => {
  const [rows, setRows] = useState<PromptRow[]>([
    {
      version: '1.0',
      status: 'Active',
      preview:
        'You are an experienced aircraft maintenance technician with 15+ years of experience working on military and commercial aircraft. You specialize in...',
      settings: '[ ]',
    },
    {
      version: '1.1',
      status: 'Active',
      preview:
        'You are a senior aircraft maintenance specialist focused on rapid diagnosis and...',
      settings: '[ ]',
    },
    {
      version: '1.2',
      status: 'Active',
      preview:
        'You are an aircraft maintenance worker focused on diagnosis and error handling...',
      settings: '[ ]',
    },
  ])
  const [isOpen, setIsOpen] = useState(false)
  const [mode, setMode] = useState<PromptModalMode>('edit')
  const [editingIndex, setEditingIndex] = useState<number | null>(null)
  const [initialVersion, setInitialVersion] = useState('')
  const [initialText, setInitialText] = useState('')

  const openCreate = () => {
    setMode('create')
    setEditingIndex(null)
    setInitialVersion('')
    setInitialText('')
    setIsOpen(true)
  }

  const openEdit = (idx: number) => {
    const r = rows[idx]
    setMode('edit')
    setEditingIndex(idx)
    setInitialVersion(r.version)
    setInitialText(r.preview)
    setIsOpen(true)
  }

  const handleSave = (version: string, text: string) => {
    if (mode === 'create') {
      setRows(prev => [
        { version, status: 'Draft', preview: text, settings: '[ ]' },
        ...prev,
      ])
    } else if (editingIndex !== null) {
      setRows(prev =>
        prev.map((r, i) =>
          i === editingIndex ? { ...r, version, preview: text } : r
        )
      )
    }
    setIsOpen(false)
  }

  const handleDelete = () => {
    if (editingIndex === null) return
    setRows(prev => prev.filter((_, i) => i !== editingIndex))
    setIsOpen(false)
  }

  return (
    <div className="w-full h-full">
      <div className="w-full flex justify-end mb-2">
        <button
          className="px-3 py-2 rounded-md bg-primary text-primary-foreground text-sm"
          onClick={openCreate}
        >
          New prompt
        </button>
      </div>
      <table className="w-full">
        <thead className="bg-white dark:bg-blue-600 font-normal">
          <tr>
            <th className="text-left w-[10%] py-2 px-3">Version</th>
            <th className="text-left w-[10%] py-2 px-3">Status</th>
            <th className="text-left w-[50%] py-2 px-3">Preview</th>
            <th className="text-left w-[10%] py-2 px-3">Settings</th>
            <th className="text-left w-[10%] py-2 px-3">Actions</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((prompt, index) => (
            <tr
              key={index}
              className={`border-b border-solid border-white dark:border-blue-600 text-sm font-mono leading-4 tracking-[0.32px]${
                index === rows.length - 1 ? 'border-b-0' : 'border-b'
              }`}
            >
              <td className="align-top p-3">{prompt.version}</td>
              <td className="align-top p-3">
                <FontIcon
                  type="checkmark-outline"
                  className="w-6 h-6 text-blue-100 dark:text-green-100"
                />
              </td>
              <td className="align-top p-3">{prompt.preview}</td>
              <td className="align-top p-3">{prompt.settings}</td>
              <td className="flex flex-row gap-4 align-top p-3">
                <FontIcon
                  type="edit"
                  className="w-6 h-6 text-blue-100"
                  isButton
                  handleOnClick={() => openEdit(index)}
                />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <PromptModal
        isOpen={isOpen}
        mode={mode}
        initialVersion={initialVersion}
        initialText={initialText}
        onClose={() => setIsOpen(false)}
        onSave={handleSave}
        onDelete={handleDelete}
      />
    </div>
  )
}

export default Prompts
