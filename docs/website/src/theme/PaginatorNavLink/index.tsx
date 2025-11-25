import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import ArrowLeftIcon from '../../components/icons/ArrowLeftIcon';
import ArrowRightIcon from '../../components/icons/ArrowRightIcon';

interface Props {
  permalink: string;
  title: string;
  subLabel?: string;
  isNext?: boolean;
}

export default function PaginatorNavLink(props: Props): JSX.Element {
  const {permalink, title, subLabel, isNext} = props;
  return (
    <Link
      className={clsx(
        'pagination-nav__link',
        isNext ? 'pagination-nav__link--next' : 'pagination-nav__link--prev',
      )}
      to={permalink}>
      <div style={{flex: 1}}>
        {subLabel && <div className="pagination-nav__sublabel">{subLabel}</div>}
        <div
          className="pagination-nav__label"
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            justifyContent: isNext ? 'flex-end' : 'flex-start',
          }}>
          {!isNext && (
            <ArrowLeftIcon
              style={{
                width: '1.25rem',
                height: '1.25rem',
                flexShrink: 0,
              }}
            />
          )}
          {title}
          {isNext && (
            <ArrowRightIcon
              style={{
                width: '1.25rem',
                height: '1.25rem',
                flexShrink: 0,
              }}
            />
          )}
        </div>
      </div>
    </Link>
  );
}

