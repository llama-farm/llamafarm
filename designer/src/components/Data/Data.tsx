import { useState, useCallback, useRef, useMemo } from 'react'
import FontIcon from '../../common/FontIcon'
import Loader from '../../common/Loader'
import LoadingSteps from '../../common/LoadingSteps'
import ModeToggle, { Mode } from '../ModeToggle'
import ConfigEditor from '../ConfigEditor'

const Data = () => {
  const [isDragging, setIsDragging] = useState(false)
  const [isDropped, setIsDropped] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [droppedFiles, setDroppedFiles] = useState<File[]>([])
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [searchValue, setSearchValue] = useState('')
  const [mode, setMode] = useState<Mode>('designer')

  const handleDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback(() => {
    setIsDragging(false)
  }, [])

  const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDropped(true)
    setIsLoading(true)

    setTimeout(() => {
      setIsDragging(false)
      setIsDropped(false)
    }, 1000)

    const files = Array.from(e.dataTransfer.files)
    setTimeout(() => {
      setDroppedFiles(prev => [...prev, ...files])
      setIsLoading(false)
    }, 4000)

    // console.log('Dropped files:', files)
  }, [])

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files ? Array.from(e.target.files) : []
    if (files.length === 0) return

    setIsDropped(true)
    setIsLoading(true)

    setTimeout(() => {
      setDroppedFiles(prev => [...prev, ...files])
      setIsDropped(false)
      setIsLoading(false)
    }, 4000)

    // console.log('Selected files:', files)
  }

  const handleSearch = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value
    setSearchValue(value)
  }

  const filteredFiles = useMemo(
    () =>
      droppedFiles.filter(file =>
        file.name.toLowerCase().includes(searchValue.toLowerCase())
      ),
    [droppedFiles, searchValue]
  )

  return (
    <div
      className="h-full w-full flex flex-col gap-2"
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <div className="w-full flex items-center justify-between mb-4">
        <h2 className="text-2xl ">
          {mode === 'designer' ? 'Data' : 'Config editor'}
        </h2>
        <div className="flex items-center gap-3">
          <ModeToggle mode={mode} onToggle={setMode} />
          <button className="opacity-50 cursor-not-allowed text-sm px-3 py-2 rounded-lg border border-input text-muted-foreground">
            Deploy
          </button>
        </div>
      </div>
      <input
        type="file"
        ref={fileInputRef}
        className="hidden"
        multiple
        onChange={handleFileSelect}
      />
      <div className="w-full flex flex-col h-full">
        {mode === 'designer' && (
          <>
            <div className="mb-2">Project data</div>
            <div className="mb-2 flex flex-row gap-2 justify-between items-end">
              <div className="text-sm">Dataset</div>
              <button className="py-2 px-3 bg-primary rounded-lg text-sm text-primary-foreground ">
                Upload data
              </button>
            </div>
          </>
        )}
        {mode !== 'designer' ? (
          <ConfigEditor />
        ) : isDragging ? (
          <div
            className={`w-full h-full flex flex-col items-center justify-center border border-dashed rounded-lg p-4 gap-2 transition-colors border-input`}
          >
            <div className="flex flex-col items-center justify-center gap-4 text-center my-[56px] text-primary">
              {isDropped ? (
                <Loader />
              ) : (
                <FontIcon
                  type="upload"
                  className="w-10 h-10 text-blue-200 dark:text-white"
                />
              )}
              <div className="text-xl text-foreground">Drop data here</div>
            </div>
            <p className="max-w-[527px] text-sm text-muted-foreground text-center mb-10">
              You can upload PDFs, explore various list formats, or draw
              inspiration from other data sources to enhance your project with
              LlaMaFarm.
            </p>
          </div>
        ) : (
          <div>
            {mode === 'designer' && droppedFiles.length <= 0 ? (
              <div className="w-full mb-6 flex items-center justify-center rounded-lg py-4 text-primary text-center bg-primary/10">
                Datasets will appear here when they’re ready
              </div>
            ) : (
              mode === 'designer' && (
                <div className="grid grid-cols-2 gap-2 mb-6">
                  {Array.from({ length: 2 }).map((_, index) => (
                    <div
                      key={index}
                      className="w-full bg-card rounded-lg border border-border flex flex-col gap-2 p-4"
                    >
                      <div className="text-sm">
                        air-craft-maintenance-guides
                      </div>
                      <div className="text-xs text-muted-foreground">
                        Updated on 8/23/25
                      </div>
                      <div className="flex flex-row gap-2">
                        <div className="text-xs text-primary-foreground bg-primary rounded-xl px-3 py-0.5">
                          Embedded
                        </div>
                        <div className="text-xs text-primary-foreground bg-primary rounded-xl px-3 py-0.5">
                          Chunked
                        </div>
                      </div>
                      <div className="text-xs text-muted-foreground">
                        more info here in a line
                      </div>
                      <div className="flex justify-end">
                        <button className="text-sm bg-transparent text-primary hover:opacity-80 rounded-lg px-2 py-1 w-fit">
                          View raw data
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )
            )}
            {mode === 'designer' && <div className="mb-4">Raw data files</div>}
            {isLoading && droppedFiles.length <= 0 ? (
              <div className="w-full flex flex-col items-center justify-center border border-solid rounded-lg p-4 gap-2 transition-colors border-input">
                <div className="flex flex-col items-center justify-center gap-4 text-center my-[40px]">
                  <div className="text-xl text-foreground">
                    Processing your data...
                  </div>
                  <Loader size={72} className="border-primary" />
                  <LoadingSteps />
                </div>
              </div>
            ) : (
              mode === 'designer' &&
              droppedFiles.length <= 0 && (
                <div className="w-full flex flex-col items-center justify-center border border-dashed rounded-lg p-4 gap-2 transition-colors border-input">
                  <div className="flex flex-col items-center justify-center gap-4 text-center my-[56px]">
                    <FontIcon
                      type="upload"
                      className="w-10 h-10 text-foreground"
                    />
                    <div className="text-xl text-foreground">
                      Drop data here to start
                    </div>
                    <button
                      className="text-sm py-2 px-6 border border-solid border-foreground rounded-lg hover:bg-secondary hover:border-secondary hover:text-secondary-foreground"
                      onClick={() => {
                        fileInputRef.current?.click()
                      }}
                    >
                      Or choose files
                    </button>
                  </div>
                  <p className="max-w-[527px] text-sm text-muted-foreground text-center mb-10">
                    You can upload PDFs, explore various list formats, or draw
                    inspiration from other data sources to enhance your project
                    with LlaMaFarm.
                  </p>
                </div>
              )
            )}
            {mode === 'designer' && filteredFiles.length > 0 && (
              <div>
                <div className="w-full flex flex-row gap-2">
                  <div className="w-3/4 flex flex-row gap-2 items-center bg-card rounded-lg pl-2 border border-solid border-input">
                    <FontIcon
                      type="search"
                      className="w-4 h-4 text-foreground"
                    />
                    <input
                      type="search"
                      className="w-full bg-transparent rounded-lg py-2 px-4 text-foreground text-sm focus:outline-none"
                      placeholder="Search files"
                      value={searchValue}
                      onChange={handleSearch}
                    />
                  </div>
                  <div className="w-1/4 text-sm text-foreground flex items-center bg-card rounded-lg px-3 justify-between border border-input">
                    <div>All datasets</div>
                    <FontIcon
                      type="chevron-down"
                      className="w-4 h-4 text-foreground"
                    />
                  </div>
                </div>
                {filteredFiles.map((file, i) => (
                  <div
                    key={i}
                    className="w-full bg-card rounded-lg py-2 px-4 text-sm mt-2 flex justify-between items-center border border-border"
                  >
                    <div className="text-xs text-muted-foreground font-mono leading-4 tracking-[0.32px] truncate whitespace-nowrap overflow-hidden">
                      {file.name}
                    </div>
                    <div className="w-1/2 flex justify-between items-center">
                      <div className="flex items-center gap-2">
                        <div className="text-muted-foreground text-xs">
                          MX-log-data
                        </div>
                        <FontIcon
                          type="chevron-down"
                          className="w-4 h-4 text-muted-foreground"
                        />
                      </div>
                      <div className="flex items-center gap-6">
                        <div className="flex items-center gap-2">
                          <FontIcon
                            type="fade"
                            className="w-4 h-4 text-muted-foreground"
                          />
                          <div className="text-xs text-muted-foreground">
                            Processing
                          </div>
                        </div>
                        <FontIcon
                          type="trashcan"
                          className="w-4 h-4 text-muted-foreground"
                        />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default Data
