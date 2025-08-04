interface LoaderProps {
  size?: number
}

const Loader = ({ size = 40 }: LoaderProps) => {
  return (
    <div
      className="animate-spin rounded-full aspect-square border-[6px] border-blue-400 border-t-transparent"
      style={{ width: size, height: size }}
    ></div>
  )
}

export default Loader
