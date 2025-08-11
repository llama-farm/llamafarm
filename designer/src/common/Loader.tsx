interface LoaderProps {
  size?: number
  className?: string
}

const Loader = ({ size = 40, className }: LoaderProps) => {
  const defaultClassNameColor = 'border-blue-400 dark:border-blue-100'
  const classNameColor = className ? className : defaultClassNameColor

  return (
    <div
      className={`animate-spin rounded-full aspect-square border-[6px] border-t-transparent dark:border-t-transparent ${classNameColor}`}
      style={{ width: size, height: size }}
    ></div>
  )
}

export default Loader
