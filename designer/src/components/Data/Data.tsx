import { useState, useCallback, useRef, useMemo } from 'react'
import FontIcon from '../../common/FontIcon'
import Loader from '../../common/Loader'
import LoadingSteps from '../../common/LoadingSteps'

const Data = () => {
  const [isDragging, setIsDragging] = useState(false)
  const [isDropped, setIsDropped] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [droppedFiles, setDroppedFiles] = useState<File[]>([])
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [searchValue, setSearchValue] = useState('')

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
      <input
        type="file"
        ref={fileInputRef}
        className="hidden"
        multiple
        onChange={handleFileSelect}
      />
      <div className="w-full flex flex-col h-full">
        <div className="mb-2">Project data</div>
        <div className="mb-2 flex flex-row gap-2 justify-between items-end">
          <div className="text-sm">Dataset</div>
          <button className="py-2 px-3 bg-blue-200 rounded-lg text-sm text-white ">
            Upload data
          </button>
        </div>
        {isDragging ? (
          <div
            className={`w-full h-full flex flex-col items-center justify-center border-[1px] border-dashed rounded-lg p-4 gap-2 transition-colors border-blue-100`}
          >
            <div className="flex flex-col items-center justify-center gap-4 text-center my-[56px] text-blue-200 dark:text-white">
              {isDropped ? (
                <Loader />
              ) : (
                <FontIcon
                  type="upload"
                  className="w-10 h-10 text-blue-200 dark:text-white"
                />
              )}
              <div className="text-xl">Drop data here</div>
            </div>
            <p className="max-w-[527px] text-sm text-blue-100 text-center mb-10">
              You can upload PDFs, explore various list formats, or draw
              inspiration from other data sources to enhance your project with
              LlaMaFarm.
            </p>
          </div>
        ) : (
          <div>
            {droppedFiles.length <= 0 ? (
              <div
                className="w-full mb-6 flex items-center justify-center bg-blue-500 rounded-lg py-4 text-blue-200 text-center"
                style={{ background: 'rgba(1, 123, 247, 0.10)' }}
              >
                Datasets will appear here when they’re ready
              </div>
            ) : (
              <div className="grid grid-cols-2 gap-2 mb-6">
                {Array.from({ length: 2 }).map((_, index) => (
                  <div
                    key={index}
                    className="w-full bg-[#FFFFFF] dark:bg-blue-500 rounded-lg border-[1px] border-blue-50 border-solid dark:border-none flex flex-col gap-2 p-4"
                  >
                    <div className="text-sm">air-craft-maintenance-guides</div>
                    <div className="text-xs text-blue-100">
                      Updated on 8/23/25
                    </div>
                    <div className="flex flex-row gap-2">
                      <div className="text-xs text-white bg-blue-100 dark:bg-blue-200 rounded-xl px-3 py-0.5">
                        Embedded
                      </div>
                      <div className="text-xs text-white bg-blue-100 dark:bg-blue-200 rounded-xl px-3 py-0.5">
                        Chunked
                      </div>
                    </div>
                    <div className="text-xs text-blue-100">
                      more info here in a line
                    </div>
                    <div className="flex justify-end">
                      <button className="text-sm bg-transparent text-blue-200 dark:text-green-100 hover:text-white hover:bg-blue-100 dark:hover:text-black dark:hover:bg-green-100 rounded-lg px-2 py-1 w-fit">
                        View raw data
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
            <div className="mb-4">Raw data files</div>
            {isLoading && droppedFiles.length <= 0 ? (
              <div className="w-full flex flex-col items-center justify-center border-[1px] border-solid rounded-lg p-4 gap-2 transition-colors border-blue-100">
                <div className="flex flex-col items-center justify-center gap-4 text-center my-[40px]">
                  <div className="text-xl">Processing your data...</div>
                  <Loader
                    size={72}
                    className="border-blue-200 dark:border-blue-100"
                  />
                  <LoadingSteps />
                </div>
              </div>
            ) : (
              droppedFiles.length <= 0 && (
                <div className="w-full flex flex-col items-center justify-center border-[1px] border-dashed rounded-lg p-4 gap-2 transition-colors border-blue-100">
                  <div className="flex flex-col items-center justify-center gap-4 text-center my-[56px]">
                    <FontIcon
                      type="upload"
                      className="w-10 h-10 text-gray-800 dark:text-white"
                    />
                    <div className="text-xl">Drop data here to start</div>
                    <button
                      className="text-sm py-2 px-6 border-[1px] border-solid border-gray-800 rounded-lg hover:bg-blue-200 hover:border-blue-200 hover:text-white dark:border-white dark:hover:bg-white dark:hover:text-blue-200"
                      onClick={() => {
                        fileInputRef.current?.click()
                      }}
                    >
                      Or choose files
                    </button>
                  </div>
                  <p className="max-w-[527px] text-sm text-blue-100 text-center mb-10">
                    You can upload PDFs, explore various list formats, or draw
                    inspiration from other data sources to enhance your project
                    with LlaMaFarm.
                  </p>
                </div>
              )
            )}
            {filteredFiles.length > 0 && (
              <div>
                <div className="w-full flex flex-row gap-2">
                  <div className="w-3/4 flex flex-row gap-2 items-center bg-[#FFFFFF] dark:bg-transparent rounded-lg pl-2 border-[1px] border-solid border-blue-50 dark:border-blue-100">
                    <FontIcon
                      type="search"
                      className="w-4 h-4 text-gray-800 dark:text-white"
                    />
                    <input
                      type="search"
                      className="w-full bg-transparent rounded-lg py-2 px-4 text-gray-800 dark:text-white text-sm focus:outline-none"
                      placeholder="Search files"
                      value={searchValue}
                      onChange={handleSearch}
                    />
                  </div>
                  <div className="w-1/4 text-sm text-gray-800 dark:text-white flex items-center bg-[#FFFFFF] dark:bg-blue-500 rounded-lg px-3 justify-between">
                    <div>All datasets</div>
                    <FontIcon
                      type="chevron-down"
                      className="w-4 h-4 text-gray-800 dark:text-white"
                    />
                  </div>
                </div>
                {filteredFiles.map((file, i) => (
                  <div
                    key={i}
                    className="w-full bg-[#FFFFFF] dark:bg-blue-500 rounded-lg py-2 px-4  text-sm mt-2 flex justify-between items-center"
                  >
                    <div className="text-xs text-[#545454] dark:text-white font-mono leading-4 tracking-[0.32px] truncate whitespace-nowrap overflow-hidden">
                      {file.name}
                    </div>
                    <div className="w-1/2 flex justify-between items-center">
                      <div className="flex items-center gap-2">
                        <div className="text-blue-100 text-xs">MX-log-data</div>
                        <FontIcon
                          type="chevron-down"
                          className="w-4 h-4 text-blue-100"
                        />
                      </div>
                      <div className="flex items-center gap-6">
                        <div className="flex items-center gap-2">
                          <FontIcon
                            type="fade"
                            className="w-4 h-4 text-blue-100"
                          />
                          <div className="text-xs text-blue-100">
                            Processing
                          </div>
                        </div>
                        <FontIcon
                          type="trashcan"
                          className="w-4 h-4 text-blue-100"
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
