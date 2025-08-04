import { useState, useCallback, useRef } from 'react'
import FontIcon from '../common/FontIcon'
import Loader from '../common/Loader'
import LoadingSteps from '../common/LoadingSteps'

const Data = () => {
  const [isDragging, setIsDragging] = useState(false)
  const [isDropped, setIsDropped] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [droppedFiles, setDroppedFiles] = useState<File[]>([])
  const fileInputRef = useRef<HTMLInputElement>(null)

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

  return (
    <div
      className="h-full w-full flex flex-col gap-2 py-8 px-8"
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
        <div className="mb-6 flex flex-row gap-2 justify-between items-center">
          <div className="text-sm">Dataset</div>
          <button className="py-2 px-3 bg-blue-200 rounded-lg text-sm">
            Upload data
          </button>
        </div>
        {isDragging ? (
          <div
            className={`w-full h-full flex flex-col items-center justify-center border-[1px] border-dashed rounded-lg p-4 gap-2 transition-colors border-blue-100`}
          >
            <div className="flex flex-col items-center justify-center gap-4 text-center my-[56px]">
              {isDropped ? (
                <Loader />
              ) : (
                <FontIcon type="upload" className="w-10 h-10 text-white" />
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
            <div
              className="w-full mb-6 flex items-center justify-center bg-blue-500 rounded-lg py-4 text-blue-200 text-center"
              style={{ background: 'rgba(1, 123, 247, 0.10)' }}
            >
              Datasets will appear here when they’re ready
            </div>
            <div className="mb-2">Raw data files</div>
            {isLoading && droppedFiles.length <= 0 ? (
              <div className="w-full flex flex-col items-center justify-center border-[1px] border-solid rounded-lg p-4 gap-2 transition-colors border-blue-100">
                <div className="flex flex-col items-center justify-center gap-4 text-center my-[40px]">
                  <div className="text-xl">Processing your data...</div>
                  <Loader size={72} />
                  <LoadingSteps />
                </div>
              </div>
            ) : (
              droppedFiles.length <= 0 && (
                <div className="w-full flex flex-col items-center justify-center border-[1px] border-dashed rounded-lg p-4 gap-2 transition-colors border-blue-100">
                  <div className="flex flex-col items-center justify-center gap-4 text-center my-[56px]">
                    <FontIcon type="upload" className="w-10 h-10 text-white" />
                    <div className="text-xl">Drop data here to start</div>
                    <button
                      className="text-sm py-2 px-6 border-[1px] border-solid border-white rounded-lg hover:bg-white hover:text-blue-200"
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
            {droppedFiles.length > 0 &&
              droppedFiles.map((file, i) => (
                <div className="w-full bg-blue-500 rounded-lg py-2 px-4 text-white text-sm mt-2 flex justify-between items-center">
                  <div
                    className="w-1/2 text-[#CCCCCC] text-xs"
                    style={{
                      fontFamily: 'Roboto Mono',
                      fontWeight: '400',
                      lineHeight: '16px',
                      letterSpacing: '0.32px',
                    }}
                    key={i}
                  >
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
                        <div className="text-xs text-blue-100">Processing</div>
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
    </div>
  )
}

export default Data
