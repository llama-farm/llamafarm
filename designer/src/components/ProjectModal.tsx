import { useEffect, useState } from 'react'
import FontIcon from '../common/FontIcon'

export type ProjectModalMode = 'create' | 'edit'

interface ProjectModalProps {
  isOpen: boolean
  mode: ProjectModalMode
  initialName?: string
  initialDescription?: string
  onClose: () => void
  onSave: (name: string, description: string) => void
  onDelete?: () => void
}

const ProjectModal: React.FC<ProjectModalProps> = ({
  isOpen,
  mode,
  initialName = '',
  initialDescription = '',
  onClose,
  onSave,
  onDelete,
}) => {
  const [name, setName] = useState(initialName)
  const [desc, setDesc] = useState(initialDescription)

  useEffect(() => {
    if (isOpen) {
      setName(initialName)
      setDesc(initialDescription)
    }
  }, [isOpen, initialName, initialDescription])

  if (!isOpen) return null

  const title = mode === 'create' ? 'Create new project' : 'Edit project'
  const cta = mode === 'create' ? 'Create' : 'Save'
  const isValid = name.trim().length > 0

  const handleDelete = () => {
    if (!onDelete) return
    const ok = confirm('Are you sure you want to delete this project?')
    if (ok) onDelete()
  }

  return (
    <div className="fixed inset-0 z-[70] flex items-start justify-center pt-16 bg-black/50">
      <div className="w-[720px] max-w-[95vw] rounded-xl overflow-hidden bg-white dark:bg-blue-600 text-gray-900 dark:text-white shadow-xl">
        <div className="flex items-center justify-between px-5 py-4 border-b border-blue-50/50 dark:border-blue-400/30">
          <div className="text-lg">{title}</div>
          <FontIcon
            type="close"
            isButton
            handleOnClick={onClose}
            className="w-5 h-5 text-gray-700 dark:text-white"
          />
        </div>

        <div className="p-5 flex flex-col gap-3">
          <div>
            <label className="text-xs text-blue-100">Project name</label>
            <input
              className="w-full mt-1 bg-transparent rounded-lg py-2 px-3 border border-blue-50 text-gray-800 dark:text-white dark:border-blue-100"
              placeholder="Enter name"
              value={name}
              onChange={e => setName(e.target.value)}
            />
          </div>
          <div>
            <label className="text-xs text-blue-100">Description</label>
            <textarea
              rows={4}
              className="w-full mt-1 bg-transparent rounded-lg py-2 px-3 border border-blue-50 text-gray-800 dark:text-white dark:border-blue-100"
              placeholder="Add a brief description"
              value={desc}
              onChange={e => setDesc(e.target.value)}
            />
          </div>
        </div>

        <div className="px-5 py-4 flex items-center justify-between bg-white dark:bg-blue-600">
          {mode === 'edit' ? (
            <button
              className="px-3 py-2 rounded-md bg-red-600 text-white hover:bg-red-700 text-sm"
              onClick={handleDelete}
            >
              Delete
            </button>
          ) : (
            <div />
          )}
          <div className="flex items-center gap-2">
            <button
              className="px-3 py-2 rounded-md text-sm text-blue-200 dark:text-green-100 hover:underline"
              onClick={onClose}
              type="button"
            >
              Cancel
            </button>
            <button
              className={`px-3 py-2 rounded-md text-sm ${
                isValid
                  ? 'bg-blue-200 text-white hover:opacity-90'
                  : 'opacity-50 cursor-not-allowed bg-blue-200 text-white'
              }`}
              onClick={() => isValid && onSave(name.trim(), desc.trim())}
            >
              {cta}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ProjectModal
