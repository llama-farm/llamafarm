import FontIcon from '../common/FontIcon'

export type Mode = 'designer' | 'code'

interface ModeToggleProps {
  mode: Mode
  onToggle: (mode: Mode) => void
}

function ModeToggle({ mode, onToggle }: ModeToggleProps) {
  const baseBtn = 'w-8 h-7 flex items-center justify-center transition-colors'
  const activeBtn = 'bg-blue-100 text-white dark:bg-blue-400'
  const inactiveBtn =
    'text-gray-100 bg-[#F4F4F4] hover:text-white hover:bg-gray-100 dark:text-blue-100 dark:bg-transparent dark:hover:bg-blue-700'

  return (
    <div className="flex rounded-lg overflow-hidden border-gray-300 dark:border-blue-400/50 dark:border">
      <button
        type="button"
        className={`${baseBtn} ${mode === 'code' ? activeBtn : inactiveBtn}`}
        onClick={() => onToggle('code')}
        aria-pressed={mode === 'code'}
        title="Code view"
      >
        <FontIcon type="code" className="w-4 h-4" />
      </button>
      <button
        type="button"
        className={`${baseBtn} ${mode === 'designer' ? activeBtn : inactiveBtn}`}
        onClick={() => onToggle('designer')}
        aria-pressed={mode === 'designer'}
        title="Designer view"
      >
        <FontIcon type="tools-alt" className="w-4 h-4" />
      </button>
    </div>
  )
}

export default ModeToggle
