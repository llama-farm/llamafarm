import React from 'react';
import clsx from 'clsx';
import useIsBrowser from '@docusaurus/useIsBrowser';
import {translate} from '@docusaurus/Translate';
import SunIcon from '../../components/icons/SunIcon';
import MoonIcon from '../../components/icons/MoonIcon';
import type {Props} from '@theme/ColorModeToggle';
import styles from './styles.module.css';

function ColorModeToggle({
  className,
  buttonClassName,
  value,
  onChange,
}: Props): JSX.Element {
  const isBrowser = useIsBrowser();

  const title = translate(
    {
      message: 'Switch between dark and light mode (currently {mode})',
      id: 'theme.colorToggle.ariaLabel',
      description: 'The ARIA label for the navbar color mode toggle',
    },
    {
      mode:
        value === 'dark'
          ? translate({
              message: 'dark mode',
              id: 'theme.colorToggle.ariaLabel.mode.dark',
              description: 'The name for the dark color mode',
            })
          : translate({
              message: 'light mode',
              id: 'theme.colorToggle.ariaLabel.mode.light',
              description: 'The name for the light color mode',
            }),
    },
  );

  return (
    <div className={clsx(styles.toggle, className)}>
      <button
        className={clsx(
          'clean-btn',
          styles.toggleButton,
          !isBrowser && styles.toggleButtonDisabled,
          buttonClassName,
        )}
        type="button"
        onClick={() => onChange(value === 'dark' ? 'light' : 'dark')}
        disabled={!isBrowser}
        title={title}
        aria-label={title}
        aria-pressed={value === 'dark'}>
        {value === 'dark' ? <SunIcon /> : <MoonIcon />}
      </button>
    </div>
  );
}

export default React.memo(ColorModeToggle);

