import { useNavigate } from 'react-router-dom'

type DataCardsProps = {
  filesProcessed: number
  databaseCount: number
  modelsCount: number
}

const DataCards = ({
  filesProcessed,
  databaseCount,
  modelsCount,
}: DataCardsProps) => {
  const navigate = useNavigate()

  const handleKeyDown = (
    event: React.KeyboardEvent,
    callback: () => void
  ) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault()
      callback()
    }
  }

  return (
    <div className="w-full grid grid-cols-1 md:grid-cols-3 gap-4">
      <div
        className="min-h-[103px] flex flex-col rounded-lg p-6 pb-8 bg-card min-w-0 overflow-hidden cursor-pointer hover:bg-accent/20 transition-colors"
        onClick={() => navigate('/chat/data')}
        onKeyDown={(e) => handleKeyDown(e, () => navigate('/chat/data'))}
        role="button"
        aria-label="View data"
        tabIndex={0}
      >
        <div className="text-sm text-foreground">Number of files processed</div>
        <div className="flex flex-row gap-2 items-baseline min-w-0">
          <span className="text-[40px] text-teal-500 dark:text-teal-300">
            {filesProcessed}
          </span>
          <span className="text-sm text-muted-foreground truncate">
            {filesProcessed === 1 ? 'file' : 'files'} uploaded
          </span>
        </div>
      </div>
      <div
        className="min-h-[103px] flex flex-col rounded-lg p-6 pb-8 bg-card min-w-0 overflow-hidden cursor-pointer hover:bg-accent/20 transition-colors"
        onClick={() => navigate('/chat/databases')}
        onKeyDown={(e) => handleKeyDown(e, () => navigate('/chat/databases'))}
        role="button"
        aria-label="View databases"
        tabIndex={0}
      >
        <div className="text-sm text-foreground">Database count</div>
        <div className="flex flex-row gap-2 items-baseline min-w-0">
          <span className="text-[40px] text-teal-500 dark:text-teal-300">
            {databaseCount}
          </span>
          <span className="text-sm text-muted-foreground truncate">
            {databaseCount === 1 ? 'database' : 'databases'}
          </span>
        </div>
      </div>
      <div
        className="min-h-[103px] flex flex-col rounded-lg p-6 pb-8 bg-card min-w-0 overflow-hidden cursor-pointer hover:bg-accent/20 transition-colors"
        onClick={() => navigate('/chat/models')}
        onKeyDown={(e) => handleKeyDown(e, () => navigate('/chat/models'))}
        role="button"
        aria-label="View models"
        tabIndex={0}
      >
        <div className="text-sm text-foreground">Models utilized</div>
        <div className="flex flex-row gap-2 items-baseline min-w-0">
          <span className="text-[40px] text-teal-500 dark:text-teal-300">
            {modelsCount}
          </span>
          <span className="text-sm text-muted-foreground truncate">
            {modelsCount === 1 ? 'inference model' : 'inference models'}
          </span>
        </div>
      </div>
    </div>
  )
}

export default DataCards
