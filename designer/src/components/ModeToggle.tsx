import FontIcon from '../common/FontIcon'

export type Mode = 'designer' | 'code'

interface ModeToggleProps {
  mode: Mode
  onToggle: (mode: Mode) => void
}

function ModeToggle({ mode, onToggle }: ModeToggleProps) {
  const baseBtn = 'w-8 h-7 flex items-center justify-center transition-colors'
  const activeBtn = 'bg-primary text-primary-foreground'
  const inactiveBtn =
    'bg-secondary text-secondary-foreground hover:bg-secondary/80'

  return (
    <div className="flex rounded-lg overflow-hidden border border-border">
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
