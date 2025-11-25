import React from 'react'
import clsx from 'clsx'
import { translate } from '@docusaurus/Translate'
import CopyIcon from '../../../components/icons/CopyIcon'
import type { Props } from '@theme/CodeBlock/CopyButton'
import styles from './styles.module.css'

export default function CopyButton({ code, className }: Props): JSX.Element {
  const [isCopied, setIsCopied] = React.useState(false)
  const copyTimeout = React.useRef<number | undefined>(undefined)
  const handleCopyCode = React.useCallback(() => {
    // Clear any existing timeout before setting a new one
    if (copyTimeout.current) {
      window.clearTimeout(copyTimeout.current)
    }
    
    navigator.clipboard.writeText(code).then(
      () => {
        setIsCopied(true)
        copyTimeout.current = window.setTimeout(() => {
          setIsCopied(false)
        }, 1000)
      },
      error => console.error(error)
    )
  }, [code])

  React.useEffect(() => () => window.clearTimeout(copyTimeout.current), [])

  return (
    <button
      type="button"
      aria-label={
        isCopied
          ? translate({
              id: 'theme.CodeBlock.copied',
              message: 'Copied',
              description: 'The copied button label on code blocks',
            })
          : translate({
              id: 'theme.CodeBlock.copyButtonAriaLabel',
              message: 'Copy code to clipboard',
              description: 'The ARIA label for copy code blocks button',
            })
      }
      title={translate({
        id: 'theme.CodeBlock.copy',
        message: 'Copy',
        description: 'The copy button label on code blocks',
      })}
      className={clsx(
        'clean-btn',
        className,
        styles.copyButton,
        isCopied && styles.copyButtonCopied
      )}
      onClick={handleCopyCode}
    >
      <span className={styles.copyButtonIcons} aria-hidden="true">
        <CopyIcon className={styles.copyButtonIcon} />
        <svg
          viewBox="0 0 24 24"
          className={styles.copyButtonSuccessIcon}
          width="18"
          height="18"
        >
          <path
            fill="currentColor"
            d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"
          />
        </svg>
      </span>
    </button>
  )
}
