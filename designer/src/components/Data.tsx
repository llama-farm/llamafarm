import FontIcon from '../common/FontIcon'

const Data = () => {
  return (
    <div className="h-full w-full flex flex-col gap-2 pt-8 px-8">
      <div className="w-full flex flex-col">
        <div className="mb-2">Project data</div>
        <div className="mb-6 flex flex-row gap-2 justify-between items-center">
          <div className="text-sm">Dataset</div>
          <button className="py-2 px-3 bg-[#017BF7] rounded-lg text-sm">
            Upload data
          </button>
        </div>
        <div
          className="w-full mb-6 flex items-center justify-center bg-[#131E45] rounded-lg py-4 text-[#017BF7] text-center"
          style={{ background: 'rgba(1, 123, 247, 0.10)' }}
        >
          Datasets will appear here when they’re ready
        </div>
        <div className="mb-2">Raw data files</div>
        <div className="w-full flex flex-col items-center justify-center border-[1px] border-dashed border-[#85B1FF] rounded-lg p-4 gap-2">
          <div className="flex flex-col items-center justify-center gap-4 text-center my-[56px]">
            <FontIcon type="upload" className="w-10 h-10 text-white" />
            <div className="text-xl">Drop data here to start</div>
            <button className="text-sm py-2 px-6 border-[1px] border-solid border-white rounded-lg hover:bg-white hover:text-[#017BF7]">
              Or choose files
            </button>
          </div>
          <p className="max-w-[527px] text-sm text-[#85B1FF] text-center mb-10">
            You can upload PDFs, explore various list formats, or draw
            inspiration from other data sources to enhance your project with
            LlaMaFarm.
          </p>
        </div>
      </div>
    </div>
  )
}

export default Data
